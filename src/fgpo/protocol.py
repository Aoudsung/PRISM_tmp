"""Published comparisons and claim-scoped own ablations; independent of torch."""
from .baseline_specs import PUBLISHED_RECIPES, PUBLISHED_SOURCES, published_recipe


PROTOCOL = "fgpo-r3-published-v4"
ABLATION_METHODS = ("fgpo_obs_only", "fgpo_no_transfer", "fgpo_no_support_weight")
OWN_METHODS = ("fgpo",) + ABLATION_METHODS
PUBLISHED_METHODS = tuple(PUBLISHED_RECIPES)
COMPARISON_METHODS = ("fgpo",) + PUBLISHED_METHODS
METHODS = COMPARISON_METHODS + ABLATION_METHODS
RETIRED_METHODS = ("fgpo_no_befp", "fgpo_no_dag_loo", "fgpo_no_full_filtration",
                   "gigpo_shared", "bipace_q_shared", "native_qwen_shared",
                   "bipace_q", "self", "skill_disco", "native_qwen")
METHOD_ROLES = {method: ("proposed_method" if method == "fgpo" else
                         "own_ablation" if method in ABLATION_METHODS else
                         "published_baseline") for method in METHODS}

OFFICIAL_HORIZONS = {"alfworld": 50, "appworld": 100}
OFFICIAL_EVALUATION_SPLITS = {
    "alfworld": ("valid_unseen",),
    "appworld": ("test_normal", "test_challenge"),
}
SUITE_METHODS = {"comparison": COMPARISON_METHODS, "ablation": ABLATION_METHODS}
SUITE_BENCHMARKS = {"comparison": ("alfworld", "appworld"), "ablation": ("alfworld",)}

_OWN_DEFAULTS = {
    "rounds": 60, "groups": 16, "trajectories": 8,
    "learning_rate": 1e-6, "weight_decay": .01, "optimizer_betas": (.9, .999),
    "optimizer_eps": 1e-8, "gamma": 1., "reward_scale": 10.,
    "syntax_reward_penalty": 0., "ppo_minibatch_rows": None, "ppo_epochs": 1,
    "loss_normalization": "trajectory", "ppo_clip_ratio": .2, "ppo_dual_clip": 3.,
    "entropy_coef": .001, "kl_coef": .01, "max_grad_norm": 1.,
    "temperature": 1., "eval_temperature": .4, "top_p": 1., "eval_top_p": 1.,
    "history_length": None, "shuffle": False, "credit_transfer_max": .5,
}
_RECIPE_ALIASES = {
    "learning_rate": "lr", "optimizer_betas": "betas", "optimizer_eps": "adam_eps",
    "groups": "tasks_per_round", "trajectories": "group_size",
    "history_length": "history", "output_tokens": "max_new_tokens",
    "ppo_minibatch_rows": "minibatch", "ppo_clip_ratio": "clip_ratio",
    "ppo_dual_clip": "dual_clip", "entropy_coef": "entropy_beta", "kl_coef": "kl_beta",
}


def resolve_method_config(config, method=None, benchmark=None):
    """Resolve source recipes without leaking own-method defaults into them.

    Resource knobs remain common. Published semantic defaults take precedence
    over common own-method values; explicit per-method changes are checked by
    require_protocol before any environment or model work starts.
    """
    method = config["method"] if method is None else method
    benchmark = config["benchmark"] if benchmark is None else benchmark
    options = {**_OWN_DEFAULTS, **config, "method": method, "benchmark": benchmark}
    if method in PUBLISHED_METHODS:
        recipe = published_recipe(method, benchmark, config["model_path"])
        options.update(recipe)
        options.update({key: recipe[source] for key, source in _RECIPE_ALIASES.items()})
        options["loss_normalization"] = "token"
    options.update(config.get("method_configs", {}).get(method, {}))
    if method in OWN_METHODS:
        limits = options.get("output_tokens_by_benchmark", {"alfworld": 512, "appworld": 1024})
        options["output_tokens"] = int(options.get("output_tokens", limits[benchmark]))
    return options


def _same_value(before, after):
    if isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)):
        return tuple(before) == tuple(after)
    return before == after


def require_protocol(config):
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"This study requires {PROTOCOL!r} and a fresh output directory; "
                         "prior protocols have different methods and collection semantics.")
    methods = (config["method"],) if "method" in config else tuple(config.get("methods", ()))
    if any(method not in METHODS for method in methods):
        raise ValueError("Select the published comparison or a named own-method ablation from this protocol")
    benchmarks = (config["benchmark"],) if "benchmark" in config else tuple(config.get("benchmarks", ()))
    for benchmark, horizon in config.get("horizons", {}).items():
        if benchmark not in OFFICIAL_HORIZONS or horizon != OFFICIAL_HORIZONS[benchmark]:
            raise ValueError("Official horizons are ALFWorld 50 and AppWorld 100")
    for benchmark, splits in config.get("evaluation_splits", {}).items():
        if benchmark not in OFFICIAL_EVALUATION_SPLITS or tuple(splits) != OFFICIAL_EVALUATION_SPLITS[benchmark]:
            raise ValueError("Evaluation uses each declared complete official split")
    for method in methods:
        for benchmark in benchmarks:
            if benchmark not in OFFICIAL_HORIZONS:
                raise ValueError(f"Unknown official benchmark {benchmark!r}")
            options = resolve_method_config(config, method, benchmark)
            if options["ppo_epochs"] != 1 or options["shuffle"]:
                raise ValueError("This implementation uses one PPO epoch with the released stable row order")
            # PPO old/current/reference probabilities all use the unit-temperature law.
            if (options["temperature"] != 1. or options["top_p"] != 1.
                    or options["eval_temperature"] != .4 or options["eval_top_p"] != 1.):
                raise ValueError("Formal sampling uses train temperature 1, evaluation .4, and top_p 1")
            if method in OWN_METHODS:
                expected_tokens = 512 if benchmark == "alfworld" else 1024
                if (options["gamma"] != 1. or options["syntax_reward_penalty"] != 0.
                        or options["loss_normalization"] != "trajectory" or options["history_length"] is not None
                        or int(options["trajectories"]) != 8 or options["ppo_minibatch_rows"] is not None
                        or int(options["output_tokens"]) != expected_tokens):
                    raise ValueError("Own v4 methods require undiscounted success, K=8 atomic mixed groups, 512/1024 output tokens and trajectory loss")
                continue
            recipe = published_recipe(method, benchmark, config["model_path"])
            required = {**recipe, **{key: recipe[source] for key, source in _RECIPE_ALIASES.items()},
                        "loss_normalization": "token"}
            changed = [key for key, value in required.items() if not _same_value(options.get(key), value)]
            if changed:
                raise ValueError(f"Published {method} source recipe differs at {', '.join(changed)}")


def effective_training_identity(config):
    """Semantics of saved rollouts and pending updates; excludes resource knobs."""
    require_protocol(config)
    method, benchmark = config["method"], config["benchmark"]
    options = resolve_method_config(config)
    identity = {
        "protocol": PROTOCOL, "method": method, "method_role": METHOD_ROLES[method],
        "benchmark": benchmark, "seed": int(options["seed"]), "model_path": str(options["model_path"]),
        "horizon": OFFICIAL_HORIZONS[benchmark],
        "evaluation_splits": OFFICIAL_EVALUATION_SPLITS[benchmark],
        "max_context_tokens": int(options.get("max_context_tokens", 32768)),
        "rollout": "bounded_actor_prefix" if method in OWN_METHODS else "released_native_prompt",
        "constant_denominator": method in OWN_METHODS,
        "runtime": "pytorch_full_finetune_cpu_fp32_adamw_published_virtual_batch_v4",
    }
    for key in ("rounds", "groups", "trajectories", "output_tokens", "learning_rate", "weight_decay",
                "optimizer_eps", "gamma", "reward_scale", "syntax_reward_penalty", "history_length",
                "ppo_epochs", "ppo_minibatch_rows", "loss_normalization", "ppo_clip_ratio", "ppo_dual_clip",
                "entropy_coef", "kl_coef", "max_grad_norm", "temperature", "eval_temperature", "top_p",
                "eval_top_p", "shuffle"):
        identity[key] = options[key]
    identity["optimizer_betas"] = tuple(options["optimizer_betas"])
    if method in OWN_METHODS:
        identity.update(credit_radius=float(options.get("credit_radius", .10)),
                        credit_prior_weight=float(options.get("credit_prior_weight", 1.)),
                        credit_transfer_max=float(options.get("credit_transfer_max", .5)),
                        own_batching="atomic_mixed_group",
                        group_denominator=int(options["trajectories"]),
                        homogeneous_group_bypass=True,
                        positive_peer_continuation_gate=method != "fgpo_no_transfer",
                        local_observation_matching="exact_token_id" if benchmark == "alfworld" else "dual_view_cosine",
                        alfworld_frame="three_block_step_unquoted_commands_strict_tags_v4" if benchmark == "alfworld" else None)
    else:
        identity["published_recipe"] = published_recipe(method, benchmark, options["model_path"])
        identity["published_recipe"]["betas"] = tuple(identity["published_recipe"]["betas"])
        identity["source_commit"] = PUBLISHED_SOURCES[method]["repo_commit"]
    return identity


def require_training_identity(saved, expected):
    before, after = effective_training_identity(saved), effective_training_identity(expected)
    different = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    if different:
        raise ValueError("Resume training identity differs at " + ", ".join(different))
