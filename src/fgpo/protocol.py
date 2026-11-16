"""Method identities for the r3 finite-horizon success study.

The protocol is part of checkpoint identity: v1 rewards and pending advantages
must never be resumed as r3 updates. Author recipes remain explicitly separate
from comparators using the common observation, rollout and optimizer contract.
"""

PROTOCOL = "fgpo-r3-success-v2"
OWN_METHODS = (
    "fgpo", "fgpo_no_full_filtration", "fgpo_no_transfer",
    "fgpo_obs_only", "fgpo_no_support_weight",
)
SHARED_BASELINES = {"gigpo_shared": "gigpo", "bipace_q_shared": "bipace_q"}
SHARED_FROZEN = "native_qwen_shared"
SHARED_METHODS = OWN_METHODS + tuple(SHARED_BASELINES) + (SHARED_FROZEN,)
AUTHOR_METHODS = ("gigpo", "bipace_q", "self", "skill_disco", "native_qwen")
METHODS = SHARED_METHODS + AUTHOR_METHODS
FROZEN_METHODS = ("native_qwen", SHARED_FROZEN, "skill_disco")
RETIRED_METHODS = ("fgpo_no_befp", "fgpo_no_dag_loo")

# These are the horizons and complete splits implemented by the official
# environment adapters. Keep this lightweight module independent of torch.
OFFICIAL_HORIZONS = {"alfworld": 50, "appworld": 100}
OFFICIAL_EVALUATION_SPLITS = {
    "alfworld": ("valid_unseen",),
    "appworld": ("test_normal", "test_challenge"),
}


def _method_config(config, method):
    options = dict(config)
    options.update(config.get("method_configs", {}).get(method, {}))
    return options


def require_protocol(config):
    if config.get("protocol") != PROTOCOL:
        raise ValueError(
            f"r3 requires protocol {PROTOCOL!r} and a fresh study directory; "
            "v1 checkpoints retain different rewards and credit semantics."
        )
    methods = (config["method"],) if "method" in config else tuple(config.get("methods", ()))
    for method in methods:
        if method in RETIRED_METHODS:
            raise ValueError("The BEFP and DAG-LOO variants were retired; select an r3 ablation explicitly.")
        if method not in METHODS:
            raise ValueError(f"Unknown r3 study method {method!r}")
    for options in [config, *(_method_config(config, method) for method in methods)]:
        for benchmark, horizon in options.get("horizons", {}).items():
            if benchmark not in OFFICIAL_HORIZONS or horizon != OFFICIAL_HORIZONS[benchmark]:
                raise ValueError("Configured horizons must match the implemented official limits: ALFWorld 50, AppWorld 100")
        for benchmark, splits in options.get("evaluation_splits", {}).items():
            if benchmark not in OFFICIAL_EVALUATION_SPLITS or tuple(splits) != OFFICIAL_EVALUATION_SPLITS[benchmark]:
                raise ValueError("The r3 study requires the complete official evaluation splits in their declared order")
    for method in methods:
        options = _method_config(config, method)
        if method in SHARED_METHODS and float(options.get("gamma", 1.)) != 1.:
            raise ValueError("Shared r3 methods optimize undiscounted terminal success and require gamma=1")
        if method not in FROZEN_METHODS and method != "self" and options.get("ppo_epochs", 1) != 1:
            raise ValueError("The implemented PPO update performs exactly one epoch per collected round")
        fixed = ({"self_batch_size": 16, "lora_rank": 96, "lora_alpha": 128,
                  "lora_dropout": .05, "teacher_ema_decay": .99} if method == "self" else
                 {"skill_refinements": 3, "skill_operation_batch_size": 32,
                  "skill_induction_output_tokens": 8192} if method == "skill_disco" else {})
        for key, value in fixed.items():
            if options.get(key, value) != value:
                raise ValueError(f"The implemented {method} recipe requires {key}={value}")
        if method == "skill_disco" and "skill_induction_tasks" in options:
            counts = options["skill_induction_tasks"]
            expected_counts = {"alfworld": 200, "appworld": 90}
            selected = ({config["benchmark"]: counts} if not isinstance(counts, dict) else counts)
            if any(benchmark not in expected_counts or value != expected_counts[benchmark]
                   for benchmark, value in selected.items()):
                raise ValueError("The implemented Skill-DisCo induction uses 200 ALFWorld or 90 AppWorld tasks")


def effective_training_identity(config):
    """Canonical semantics of one resolved study cell, excluding resource knobs.

    Checkpoints can contain pending rollouts, reference values and partially
    accumulated gradients. Their collection and update definitions must agree
    even when a changed minibatch size happens to produce the same denominator.
    GPU placement, caches, offload, transfer buckets and head/representation
    microbatches remain execution choices and are deliberately omitted.
    """
    require_protocol(config)
    method, benchmark = config["method"], config["benchmark"]
    options = _method_config(config, method)
    if benchmark not in OFFICIAL_HORIZONS:
        raise ValueError(f"Unknown official benchmark {benchmark!r}")
    default_output = options.get("output_tokens_by_benchmark", {}).get(
        benchmark, 256 if benchmark == "alfworld" else 1024)
    identity = {
        "protocol": options["protocol"], "method": method, "benchmark": benchmark,
        "seed": int(options["seed"]), "model_path": str(options["model_path"]),
        "horizon": OFFICIAL_HORIZONS[benchmark],
        "evaluation_splits": OFFICIAL_EVALUATION_SPLITS[benchmark],
        "max_context_tokens": int(options.get("max_context_tokens", 32768)),
        "output_tokens": int(options.get("output_tokens", default_output)),
        "rollout": "shared_bounded_actor_prefix" if method in SHARED_METHODS else "author_native",
        "history_length": 2 if method in ("fgpo_no_full_filtration", "gigpo", "bipace_q") else None,
    }
    if method in ("gigpo", "bipace_q") and benchmark == "alfworld":
        identity["author_prompt_length"] = int(options.get("author_prompt_length", 2048))
    if method == "skill_disco":
        identity.update(skill_induction_tasks=200 if benchmark == "alfworld" else 90,
                        skill_refinements=3, skill_operation_batch_size=32,
                        skill_induction_output_tokens=8192)
    if method in FROZEN_METHODS:
        return identity

    shared = method in SHARED_METHODS
    defaults = ({"lr": 3e-4, "weight_decay": 0., "rounds": 60} if method == "self" else
                {"lr": 5e-7, "weight_decay": .01, "rounds": 150} if method == "bipace_q" else
                {"lr": 1e-6, "weight_decay": .01, "rounds": 60 if shared else 150})
    identity.update(
        rounds=int(options.get("rounds", defaults["rounds"])),
        learning_rate=float(options.get("learning_rate", defaults["lr"])),
        weight_decay=float(options.get("weight_decay", defaults["weight_decay"])),
        optimizer_betas=tuple(float(value) for value in options.get("optimizer_betas", (.9, .999))),
        optimizer_eps=float(options.get("optimizer_eps", 1e-8)),
    )
    if method == "self":
        identity.update(self_collection_trajectories=int(options.get("self_collection_trajectories", 16)),
                        self_batch_size=16, lora_rank=96, lora_alpha=128,
                        lora_dropout=.05, teacher_ema_decay=.99)
        return identity

    identity.update(
        groups=int(options.get("groups", 16)), trajectories=int(options.get("trajectories", 8)),
        reward_scale=float(options.get("reward_scale", 10.)),
        gamma=float(options.get("gamma", 1. if shared else .95)),
        syntax_reward_penalty=0. if shared else float(options.get("syntax_reward_penalty", .1)),
        ppo_epochs=1, ppo_minibatch_rows=int(options.get("ppo_minibatch_rows", 128 if method == "bipace_q" else 256)),
        loss_normalization=options.get("loss_normalization", "trajectory" if shared else "token"),
        constant_denominator=shared, ppo_clip_ratio=float(options.get("ppo_clip_ratio", .2)),
        ppo_dual_clip=float(options.get("ppo_dual_clip", 3.)),
        entropy_coef=float(options.get("entropy_coef", .001)), kl_coef=float(options.get("kl_coef", .01)),
        max_grad_norm=float(options.get("max_grad_norm", 1.)),
    )
    if method in OWN_METHODS:
        identity.update(credit_radius=float(options.get("credit_radius", .10)),
                        credit_prior_weight=float(options.get("credit_prior_weight", 1.)),
                        credit_transfer_max=float(options.get("credit_transfer_max", 1.)))
    return identity


def require_training_identity(saved, expected):
    """Require identical effective training/evaluation semantics on resume."""
    before, after = effective_training_identity(saved), effective_training_identity(expected)
    different = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    if different:
        raise ValueError("Resume training identity differs at " + ", ".join(different))
