"""Published comparisons and claim-scoped own ablations; independent of torch."""
import json
from pathlib import Path

from .baseline_specs import (PUBLISHED_RECIPES, PUBLISHED_SOURCES, UPSTREAM_COMMIT, published_recipe,
                             supports_published_benchmark)
from .public import EVAL_HORIZONS, EXACT_OBSERVATION_BENCHMARKS, HORIZONS as OFFICIAL_HORIZONS, OUTPUT_LIMITS
from .textcraft import (TEXTCRAFT_1P5B_FORMAT_STUDY, TEXTCRAFT_CONTEXT_TOKENS, TEXTCRAFT_COUNTS,
                        TEXTCRAFT_EVAL_HORIZON, TEXTCRAFT_FORMAT_EXAMPLE, TEXTCRAFT_PUBLISHED_OUTPUT_TOKENS,
                        TEXTCRAFT_SOURCE_COMMIT, TEXTCRAFT_TRAIN_GROUPS, TEXTCRAFT_TRAIN_ROUNDS)
from .webshop import PAIRED_TRAIN_ROUNDS, TRAIN_GROUPS, TRAIN_ROUNDS, VALIDATION_SESSIONS, VALIDATION_WORKERS


PROTOCOL = "fgpo-r8-scale-invariant-v1"
REFERENCE_HIDDEN_SIZE = 3584
TEXTCRAFT_RESPONSE192_STUDY = "textcraft-v4-response192-v1"
TEXTCRAFT_RESPONSE192_TOKENS = 192
WEBSHOP_PAIRED_STUDY = "webshop-150-paired-v1"
WEBSHOP_JOINT_STUDY = "webshop-150-v4-joint-v1"
WEBSHOP_JOINT_FROM_BASE_STUDY = "webshop-150-joint-from-base-v1"
WEBSHOP_1P5B_ADAPTED_STUDY = "webshop-150-1p5b-joint-adapted-from-base-v1"
WEBSHOP_R5_150_STUDY = "webshop-150-r5-from-base-v1"
WEBSHOP_R6_150_STUDY = "webshop-150-r6-from-base-v1"
WEBSHOP_R7_150_STUDY = "webshop-150-r7-from-base-v1"
WEBSHOP_R8_150_STUDY = "webshop-150-r8-from-base-v1"
WEBSHOP_JOINT_FROM_BASE_STUDIES = (WEBSHOP_JOINT_FROM_BASE_STUDY, WEBSHOP_1P5B_ADAPTED_STUDY)
WEBSHOP_FROM_BASE_STUDIES = (
    *WEBSHOP_JOINT_FROM_BASE_STUDIES,
    WEBSHOP_R5_150_STUDY,
    WEBSHOP_R6_150_STUDY,
    WEBSHOP_R7_150_STUDY,
    WEBSHOP_R8_150_STUDY,
)
WEBSHOP_JOINT_SEED = 1
WEBSHOP_JOINT_OVERRIDES = {"entropy_coef": .002, "credit_transfer_max": .25}
WEBSHOP_RADIUS_STUDY = "webshop-150-v4-radius-v1"
WEBSHOP_RADII = (.05, .10)
WEBSHOP_RADIUS_OVERRIDES = {"entropy_coef": .002, "credit_prior_weight": .5, "credit_transfer_max": .125}
WEBSHOP_CONTINUATION_STUDIES = (WEBSHOP_PAIRED_STUDY, WEBSHOP_JOINT_STUDY, WEBSHOP_RADIUS_STUDY)
WEBSHOP_150_STUDIES = (*WEBSHOP_CONTINUATION_STUDIES, *WEBSHOP_FROM_BASE_STUDIES)
WEBSHOP_FIXED_MENU = "WebShop_public_search_menu_matches_native_commands"
EXACT_FIRST_HIT_METHOD = "fgpo_exact_first_hit"
WEBSHOP_PAIRED_METHODS = ("fgpo", EXACT_FIRST_HIT_METHOD)
ABLATION_METHODS = ("fgpo_obs_only", "fgpo_no_transfer", "fgpo_no_support_weight")
OWN_METHODS = ("fgpo",) + ABLATION_METHODS + (EXACT_FIRST_HIT_METHOD,)
PUBLISHED_METHODS = tuple(PUBLISHED_RECIPES)
COMPARISON_METHODS = ("fgpo",) + PUBLISHED_METHODS
METHODS = COMPARISON_METHODS + ABLATION_METHODS + (EXACT_FIRST_HIT_METHOD,)
RETIRED_METHODS = ("fgpo_no_befp", "fgpo_no_dag_loo", "fgpo_no_full_filtration",
                   "gigpo_shared", "bipace_q_shared", "native_qwen_shared",
                   "bipace_q", "self", "skill_disco", "native_qwen")
METHOD_ROLES = {method: ("proposed_method" if method in WEBSHOP_PAIRED_METHODS else
                         "own_ablation" if method in ABLATION_METHODS else
                         "published_baseline") for method in METHODS}

OFFICIAL_EVALUATION_SPLITS = {
    "alfworld": ("valid_unseen",),
    "appworld": ("test_normal", "test_challenge"),
    "webshop": ("test",),
    "textcraft": ("val",),
}
SUITE_METHODS = {"comparison": COMPARISON_METHODS, "ablation": ABLATION_METHODS}
SUITE_BENCHMARKS = {"comparison": tuple(OFFICIAL_HORIZONS), "ablation": ("alfworld", "webshop", "textcraft")}

_OWN_DEFAULTS = {
    "rounds": 60, "groups": 16, "trajectories": 8,
    "learning_rate": 1e-6, "weight_decay": .01, "optimizer_betas": (.9, .999),
    "optimizer_eps": 1e-8, "gamma": 1., "reward_scale": 10.,
    "syntax_reward_penalty": 0., "ppo_minibatch_rows": 256, "ppo_epochs": 1,
    "loss_normalization": "token", "entropy_normalization": "token",
    "ppo_clip_ratio": .2, "ppo_dual_clip": 3.,
    "entropy_coef": .001, "kl_coef": .01, "max_grad_norm": 1.,
    "temperature": 1., "eval_temperature": .4, "top_p": 1., "eval_top_p": 1.,
    "history_length": None, "shuffle": True,
    "credit_radius": .08, "credit_time_radius": .10,
    "credit_prior_weight": 1., "credit_transfer_max": 1.,
}


def scale_learning_rate(base_lr: float, hidden_size: int | None) -> float:
    """Keep constant step size <=7B and scale inversely with residual width above d=3584."""
    lr = float(base_lr)
    if hidden_size is None or int(hidden_size) <= 0:
        return lr
    return lr * min(1.0, float(REFERENCE_HIDDEN_SIZE) / float(hidden_size))


def _model_hidden_size(model_path) -> int | None:
    if not model_path:
        return None
    cfg_file = Path(str(model_path)) / "config.json"
    if not cfg_file.is_file():
        return None
    try:
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("hidden_size")
    return int(value) if isinstance(value, int) and value > 0 else None
_RECIPE_ALIASES = {
    "learning_rate": "lr", "optimizer_betas": "betas", "optimizer_eps": "adam_eps",
    "groups": "tasks_per_round", "trajectories": "group_size",
    "history_length": "history", "output_tokens": "max_new_tokens",
    "ppo_minibatch_rows": "minibatch", "ppo_clip_ratio": "clip_ratio",
    "ppo_dual_clip": "dual_clip", "entropy_coef": "entropy_beta", "kl_coef": "kl_beta",
}


def method_benchmark_pairs(methods, benchmarks):
    """Keep the declared iteration order and select each available recipe."""
    return tuple((benchmark, method) for benchmark in benchmarks for method in methods
                 if (method != EXACT_FIRST_HIT_METHOD or benchmark == "webshop")
                 and (benchmark != "webshop" or method in (*WEBSHOP_PAIRED_METHODS, "fgpo_obs_only", "fgpo_no_transfer"))
                 and (benchmark != "textcraft" or method in (*OWN_METHODS, "graphgpo", "g2po"))
                 and (method not in PUBLISHED_METHODS or supports_published_benchmark(method, benchmark)))


def suite_methods(config, suite):
    if suite == "comparison":
        study = config.get("study_protocol")
        if study == WEBSHOP_PAIRED_STUDY:
            return WEBSHOP_PAIRED_METHODS
        if study in WEBSHOP_150_STUDIES:
            return ("fgpo",)
    return SUITE_METHODS[suite]


def study_identity(config):
    return {"study_protocol": config["study_protocol"]} if config.get("study_protocol") is not None else {}


def resolve_method_config(config, method=None, benchmark=None):
    """Resolve source recipes without leaking own-method defaults into them.

    Resource knobs remain common. Published semantic defaults take precedence
    over common own-method values; explicit per-method changes are checked by
    require_protocol before any environment or model work starts.
    """
    method = config["method"] if method is None else method
    benchmark = config["benchmark"] if benchmark is None else benchmark
    defaults = (dict(_OWN_DEFAULTS, rounds=TEXTCRAFT_TRAIN_ROUNDS, max_context_tokens=TEXTCRAFT_CONTEXT_TOKENS)
                if benchmark == "textcraft" else _OWN_DEFAULTS)
    options = {**defaults, **config, "method": method, "benchmark": benchmark}
    if method in PUBLISHED_METHODS:
        recipe = published_recipe(method, benchmark, config["model_path"])
        options.update(recipe)
        options.update({key: recipe[source] for key, source in _RECIPE_ALIASES.items()})
        options["loss_normalization"] = "token"
    elif method == EXACT_FIRST_HIT_METHOD:
        options.update(credit_radius=.10, credit_time_radius=.10, credit_prior_weight=1., credit_transfer_max=1.)
    options.update(config.get("method_configs", {}).get(method, {}))
    if method in OWN_METHODS:
        if options.get("learning_rate") == "auto" or (
                "learning_rate" not in config and "learning_rate" not in config.get("method_configs", {}).get(method, {})):
            options["learning_rate"] = scale_learning_rate(
                _OWN_DEFAULTS["learning_rate"], _model_hidden_size(options.get("model_path"))
            )
        limits = {**OUTPUT_LIMITS, **options.get("output_tokens_by_benchmark", {})}
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
    study = config.get("study_protocol")
    textcraft_response192 = study == TEXTCRAFT_RESPONSE192_STUDY
    textcraft_format = study == TEXTCRAFT_1P5B_FORMAT_STUDY
    if textcraft_response192 and (
            config.get("suite", "comparison") != "comparison" or benchmarks != ("textcraft",)
            or methods != ("fgpo",) or config.get("seed", 0) != 0
            or tuple(config.get("seeds", (0,))) != (0,)):
        raise ValueError("TextCraft response192 requires the F-GPO comparison at seed 0")
    if textcraft_format and (
            config.get("suite", "comparison") != "comparison" or benchmarks != ("textcraft",)
            or methods != ("fgpo",) or config.get("seed", 1) != 1
            or tuple(config.get("seeds", (1,))) != (1,)):
        raise ValueError("TextCraft 1.5B format requires the F-GPO comparison at seed 1")
    webshop_150 = study in WEBSHOP_150_STUDIES
    joint = study in (WEBSHOP_JOINT_STUDY, *WEBSHOP_JOINT_FROM_BASE_STUDIES)
    adapted_1p5b = study == WEBSHOP_1P5B_ADAPTED_STUDY
    r5_webshop = study == WEBSHOP_R5_150_STUDY
    r6_webshop = study == WEBSHOP_R6_150_STUDY
    r7_webshop = study == WEBSHOP_R7_150_STUDY
    r8_webshop = study == WEBSHOP_R8_150_STUDY
    radius_study = study == WEBSHOP_RADIUS_STUDY
    if study is not None and not webshop_150 and not textcraft_response192 and not textcraft_format:
        raise ValueError(f"Unknown study protocol {study!r}")
    if webshop_150:
        if (config.get("suite", "comparison") != "comparison" or benchmarks != ("webshop",)
                or not methods or not set(methods).issubset(suite_methods(config, "comparison"))
                or WEBSHOP_FIXED_MENU not in config.get("adaptations", ())):
            raise ValueError("The 150-round study requires the repaired WebShop menu and its declared methods")
        if study in (WEBSHOP_JOINT_STUDY, WEBSHOP_RADIUS_STUDY) and (
                config.get("seed", WEBSHOP_JOINT_SEED) != WEBSHOP_JOINT_SEED
                or tuple(config.get("seeds", (WEBSHOP_JOINT_SEED,))) != (WEBSHOP_JOINT_SEED,)):
            raise ValueError(f"{study} uses seed 1")
        if study in WEBSHOP_FROM_BASE_STUDIES and not set(config.get("seeds", ())).issubset((0, 1, 2)):
            raise ValueError(f"{study} uses seeds 0–2")
    elif EXACT_FIRST_HIT_METHOD in methods:
        raise ValueError("Exact first-hit credit is reserved for the 150-round paired WebShop study")
    for benchmark, horizon in config.get("horizons", {}).items():
        if benchmark not in OFFICIAL_HORIZONS or horizon != OFFICIAL_HORIZONS[benchmark]:
            raise ValueError(f"Invalid official horizon for {benchmark!r}")
    for benchmark, splits in config.get("evaluation_splits", {}).items():
        if benchmark not in OFFICIAL_EVALUATION_SPLITS or tuple(splits) != OFFICIAL_EVALUATION_SPLITS[benchmark]:
            raise ValueError("Evaluation uses each declared complete official split")
    for benchmark in benchmarks:
        if benchmark not in OFFICIAL_HORIZONS:
            raise ValueError(f"Unknown official benchmark {benchmark!r}")
    pairs = method_benchmark_pairs(methods, benchmarks)
    if methods and benchmarks and not pairs:
        raise ValueError("Selected methods have no published recipe for the selected benchmarks")
    for benchmark, method in pairs:
        options = resolve_method_config(config, method, benchmark)
        effective_study = options.get("study_protocol")
        from_base_studies = (*WEBSHOP_FROM_BASE_STUDIES, TEXTCRAFT_1P5B_FORMAT_STUDY)
        if study != effective_study and (study in from_base_studies or effective_study in from_base_studies):
            raise ValueError("Select the study protocol at the experiment level")
        if benchmark == "textcraft" and method in PUBLISHED_METHODS:
            output_limit = TEXTCRAFT_PUBLISHED_OUTPUT_TOKENS
        elif textcraft_response192:
            output_limit = TEXTCRAFT_RESPONSE192_TOKENS
        else:
            output_limit = OUTPUT_LIMITS[benchmark]
        # PPO old/current/reference probabilities all use the unit-temperature law.
        if (options["temperature"] != 1. or options["top_p"] != 1.
                or options["eval_temperature"] != .4 or options["eval_top_p"] != 1.):
            raise ValueError("Formal sampling uses train temperature 1, evaluation .4, and top_p 1")
        if benchmark == "textcraft" and (
                options["rounds"] != TEXTCRAFT_TRAIN_ROUNDS or options["groups"] != TEXTCRAFT_TRAIN_GROUPS
                or options["trajectories"] != 8 or options.get("seed", 0) not in (0, 1, 2)
                or options["reward_scale"] != 10. or options["output_tokens"] != output_limit
                or options["max_context_tokens"] != TEXTCRAFT_CONTEXT_TOKENS):
            raise ValueError(f"TextCraft uses 50 rounds, 16 eight-slot groups, seeds 0–2, an 8192-token context and {output_limit}-token responses")
        if textcraft_response192:
            required = {"learning_rate": 2e-6, "entropy_coef": .0005, "kl_coef": .02,
                        "credit_prior_weight": .5, "credit_transfer_max": .5}
            changed = [key for key, value in required.items() if options.get(key) != value]
            if changed:
                raise ValueError(f"{study} training recipe differs at " + ", ".join(changed))
        if textcraft_format:
            required = {**_OWN_DEFAULTS, "rounds": TEXTCRAFT_TRAIN_ROUNDS, "learning_rate": 2e-6,
                        "credit_radius": .10, "credit_time_radius": .10,
                        "credit_prior_weight": .5, "credit_transfer_max": .5}
            changed = [key for key, value in required.items() if not _same_value(options.get(key), value)]
            if changed:
                raise ValueError(f"{study} training recipe differs at " + ", ".join(changed))
        if method in OWN_METHODS:
            if (options["gamma"] != 1. or options["syntax_reward_penalty"] != 0.
                    or options["loss_normalization"] != "token"
                    or options.get("entropy_normalization") != "token"
                    or options["history_length"] is not None
                    or int(options["trajectories"]) != 8
                    or int(options.get("ppo_minibatch_rows") or 0) < 1
                    or int(options.get("ppo_epochs", 0)) != 1
                    or int(options["output_tokens"]) != output_limit):
                raise ValueError("Own r8 methods require undiscounted success, K=8, 1-epoch token-mean PPO minibatches, official output limits, and token-mean entropy")
            rounds = PAIRED_TRAIN_ROUNDS if webshop_150 else TRAIN_ROUNDS
            if benchmark == "webshop" and (
                    options["rounds"] != rounds or options["groups"] != TRAIN_GROUPS
                    or options.get("seed", 0) not in (0, 1, 2) or options["reward_scale"] != 10.
                    or options.get("max_context_tokens", 32768) != 32768):
                raise ValueError(f"WebShop F-GPO uses {rounds} rounds, 16 groups, seeds 0–2, a 32768-token context and a full-success reward of 10")
            if webshop_150:
                expected_lr = scale_learning_rate(
                    _OWN_DEFAULTS["learning_rate"], _model_hidden_size(options.get("model_path"))
                )
                if r8_webshop:
                    required = {**_OWN_DEFAULTS, "rounds": PAIRED_TRAIN_ROUNDS, "learning_rate": expected_lr}
                else:
                    required = {**_OWN_DEFAULTS, "rounds": PAIRED_TRAIN_ROUNDS, "credit_radius": .10,
                                "credit_prior_weight": _OWN_DEFAULTS["credit_prior_weight"] if (r6_webshop or r7_webshop) else 1.,
                                "credit_transfer_max": 1. if method == EXACT_FIRST_HIT_METHOD else .5}
                if joint:
                    required.update(WEBSHOP_JOINT_OVERRIDES)
                if adapted_1p5b:
                    required.update(credit_radius=.05, credit_prior_weight=.5)
                if r5_webshop or r6_webshop or r7_webshop:
                    radius = options.get("credit_radius")
                    if radius not in WEBSHOP_RADII:
                        raise ValueError(f"{study} requires the selected WebShop radius (.05 or .10)")
                    required.update(credit_radius=radius)
                if radius_study:
                    radius = options.get("credit_radius")
                    if radius not in WEBSHOP_RADII:
                        raise ValueError(f"{study} requires the selected WebShop radius (.05 or .10)")
                    required.update(WEBSHOP_RADIUS_OVERRIDES, credit_radius=radius)
                changed = [key for key, value in required.items()
                           if not _same_value(options.get(key) if (adapted_1p5b or r5_webshop or r6_webshop or r7_webshop or r8_webshop) else options.get(key, value), value)]
                if changed:
                    raise ValueError(f"{study} training recipe differs at " + ", ".join(changed))
            continue
        if options["ppo_epochs"] != 1 or options["shuffle"]:
            raise ValueError("Published baseline implementations use one PPO epoch with the released stable row order")
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
        "eval_horizon": EVAL_HORIZONS[benchmark],
        "evaluation_splits": OFFICIAL_EVALUATION_SPLITS[benchmark],
        "max_context_tokens": int(options.get("max_context_tokens", 32768)),
        "rollout": "canonical_tag_preserving_pomdp_filtration_r8" if method in OWN_METHODS else "released_native_prompt",
        "constant_denominator": False,
        "entropy_normalization": options.get("entropy_normalization", "token") if method in OWN_METHODS else "token",
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
        identity.update(credit_radius=float(options.get("credit_radius", .08)),
                        credit_time_radius=float(options.get("credit_time_radius", .10)),
                        credit_prior_weight=float(options.get("credit_prior_weight", 1.)),
                        credit_transfer_max=float(options.get("credit_transfer_max", 1.)),
                        representation_layer_rule="floor_5L_over_7",
                        credit_estimator=("exact_first_hit_v4" if method == EXACT_FIRST_HIT_METHOD
                                          else "r8_symmetric_empirical_bayes_cross_depth"),
                        own_batching="balanced_mixed_token_mean_zero_mask_sdpa_r8",
                        group_denominator=None,
                        homogeneous_group_bypass=True,
                        positive_peer_continuation_gate=(method == EXACT_FIRST_HIT_METHOD),
                        local_observation_matching=("exact_token_id" if method == EXACT_FIRST_HIT_METHOD
                                                    or benchmark in EXACT_OBSERVATION_BENCHMARKS
                                                    else "dual_view_cosine"),
                        alfworld_frame="canonical_tag_preserving_affordance_and_public_effects_r8" if benchmark == "alfworld" else None)
        if benchmark == "webshop":
            identity["webshop_recipe"] = {
                "source_commit": UPSTREAM_COMMIT, "observation_mode": "text_rich", "use_small": True,
                "human_goals": False, "validation_workers": VALIDATION_WORKERS,
                "validation_session_population": VALIDATION_SESSIONS,
                "task_selection": "released_worker_seed_and_session_index",
                "terminal_outcome": "native_purchase_score_equals_one",
            }
            if options.get("study_protocol") in WEBSHOP_150_STUDIES:
                identity["study_protocol"] = options["study_protocol"]
                identity["webshop_menu"] = WEBSHOP_FIXED_MENU
            if method == EXACT_FIRST_HIT_METHOD:
                identity["credit_matching"] = {
                    "kernel_distance": "max(history_cosine_distance,absolute_step_distance_divided_by_horizon)",
                    "exact_observation_required": True,
                    "time_window_steps": 1,
                    "peer_weight": "maximum_over_alive_steps_one_vote_per_other_trajectory",
                    "prior": "other_trajectory_terminal_success_mean",
                    "stop": "first_strictly_later_alive_step_with_positive_support_else_termination",
                    "stop_value": "weighted_mean_of_other_trajectory_terminal_success",
                    "transfer": "stop_support_over_support_plus_one_if_other_successful_peer_is_alive_in_time_window_else_zero",
                    "terminal_transfer": 0.,
                }
    else:
        identity["published_recipe"] = published_recipe(method, benchmark, options["model_path"])
        # This description records the existing schedule and leaves cell identity stable.
        identity["published_recipe"].pop("task_schedule_migration", None)
        identity["published_recipe"]["betas"] = tuple(identity["published_recipe"]["betas"])
        identity["source_commit"] = PUBLISHED_SOURCES[method]["repo_commit"]
    if benchmark == "textcraft":
        if options.get("study_protocol") in (TEXTCRAFT_RESPONSE192_STUDY, TEXTCRAFT_1P5B_FORMAT_STUDY):
            identity["study_protocol"] = options["study_protocol"]
        identity["textcraft_recipe"] = {
            "source_commit": TEXTCRAFT_SOURCE_COMMIT, "split_counts": dict(TEXTCRAFT_COUNTS),
            "observation": "native_feedback_and_visible_inventory",
            "action": "single_tagged_native_command_or_empty_native_rejection",
            "terminal_outcome": "native_worker_won", "environment_seed": "seed_times_1000003_plus_episode_ordinal",
        }
        if method in OWN_METHODS:
            identity["textcraft_recipe"].update(
                observation_representation="exact_visible_inventory_ids",
                user_suffix="canonical_tag_preserving_pinned_recipes_r8",
                train_horizon=OFFICIAL_HORIZONS["textcraft"],
                eval_horizon=TEXTCRAFT_EVAL_HORIZON,
                strict_think_closure=True,
            )
        if options.get("study_protocol") == TEXTCRAFT_1P5B_FORMAT_STUDY:
            identity["textcraft_recipe"]["public_system_prompt_suffix"] = "\n" + TEXTCRAFT_FORMAT_EXAMPLE
    return identity


def require_training_identity(saved, expected):
    before, after = effective_training_identity(saved), effective_training_identity(expected)
    different = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    if different:
        raise ValueError("Resume training identity differs at " + ", ".join(different))
