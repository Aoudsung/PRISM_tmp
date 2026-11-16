"""Published comparator recipes and provenance; safe before CUDA isolation.

The selected implementation is the authors' released recipe, including its
documented differences from the paper equations.  AppWorld is a native-code
benchmark transfer; none of these papers reported an AppWorld experiment.
"""
from __future__ import annotations


UPSTREAM_REPOSITORY = "https://github.com/langfengQ/verl-agent"
UPSTREAM_COMMIT = "20bd331bdbc9026a5668e11362178e10ab7400c8"
_SOURCE = f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/"

_COMMON = {
    "trainable": True,
    "lr": 1e-6,
    "weight_decay": .01,
    "betas": (.9, .999),
    "adam_eps": 1e-8,
    "tasks_per_round": 16,
    "group_size": 8,
    "history": 2,
    "reward_scale": 10.,
    "syntax_reward_penalty": .1,
    "mode": "mean_std_norm",
    "epsilon": 1e-6,
    "max_new_tokens": 512,
    "temperature": 1.,
    "eval_temperature": .4,
    "top_p": 1.,
    "eval_top_p": 1.,
    "ppo_epochs": 1,
    "clip_ratio": .2,
    "dual_clip": 3.,
    "kl_beta": .01,
    "entropy_beta": .001,
    "loss_aggregation": "token-mean",
    "max_grad_norm": 1.,
    "shuffle": False,
    "episode_statistics": "steps",
    "episode_std_unbiased": True,
    "step_std_unbiased": True,
    "enable_similarity": False,
    "source_rollout_n": 1,
    "source_dynamic_batching": False,
    "source_balance_batch": True,
    "source_adjust_mode": "copy",
    "source_row_order": "group_trajectory_step",
}

# Defaults target the study's Qwen2.5-7B model.  The official GraphGPO 1.5B
# script uses a different minibatch; published_recipe resolves that selection.
PUBLISHED_RECIPES = {
    "gigpo": {
        **_COMMON, "rounds": 150, "minibatch": 256, "gamma": .95,
        "step_advantage_w": 1., "episode_advantage_w": 1.,
        "author_prompt_length": 2048, "prompt_truncation": "error",
        "source_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "source_virtual_world_size": 2, "source_actor_microbatch": 32,
        "source_rollout_logprob_microbatch": 32, "source_ref_logprob_microbatch": 32,
        "source_copy_before_advantage": True,
    },
    "hgpo": {
        **_COMMON, "rounds": 160, "minibatch": 256, "gamma": .95,
        "length_weight_alpha": 1., "base_group": False,
        "step_std_unbiased": False, "author_prompt_length": 4096,
        "prompt_truncation": "left",
        "source_model": "Qwen/Qwen2.5-7B-Instruct",
        "source_virtual_world_size": 4, "source_actor_microbatch": 8,
        "source_rollout_logprob_microbatch": 8, "source_ref_logprob_microbatch": 8,
        "source_copy_before_advantage": False,
    },
    "graphgpo": {
        **_COMMON, "rounds": 150, "minibatch": 64, "gamma": .10,
        "step_advantage_w": 1., "episode_advantage_w": 1.,
        "normalize_distance": False, "author_prompt_length": 2048,
        "prompt_truncation": "error",
        "source_model": "Qwen/Qwen2.5-7B-Instruct",
        "source_virtual_world_size": 4, "source_actor_microbatch": 4,
        "source_rollout_logprob_microbatch": 16, "source_ref_logprob_microbatch": 16,
        "source_copy_before_advantage": False,
    },
}

_SHARED_FILES = {
    "ppo_defaults": _SOURCE + "verl/trainer/config/ppo_trainer.yaml",
    "optimizer": _SOURCE + "verl/workers/fsdp_workers.py",
    "actor_loss": _SOURCE + "verl/workers/actor/dp_actor.py",
    "alfworld_prompt": _SOURCE + "agent_system/environments/prompts/alfworld.py",
    "alfworld_projection": _SOURCE + "agent_system/environments/env_package/alfworld/projection.py",
    "memory": _SOURCE + "agent_system/memory/memory.py",
    "batch_adjustment": _SOURCE + "agent_system/multi_turn_rollout/utils.py",
    "rank_balance": _SOURCE + "verl/utils/seqlen_balancing.py",
    "rank_dispatch": _SOURCE + "verl/single_controller/base/decorator.py",
    "trajectory_row_order": _SOURCE + "agent_system/multi_turn_rollout/rollout_loop.py",
}

PUBLISHED_SOURCES = {
    "gigpo": {
        "name": "GiGPO", "venue": "NeurIPS 2025",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html",
        "paper_version": "2505.10978v3",
        "repo": UPSTREAM_REPOSITORY, "repo_commit": UPSTREAM_COMMIT,
        "implementation": "released-source-recipe",
        "source_files": {**_SHARED_FILES,
            "credit": _SOURCE + "gigpo/core_gigpo.py",
            "recipe": _SOURCE + "examples/gigpo_trainer/run_alfworld.sh",
            "reward_order": _SOURCE + "verl/trainer/ppo/ray_trainer.py"},
        "credit_functions": ["compute_gigpo_outcome_advantage", "episode_norm_reward",
                             "build_step_group", "step_norm_reward"],
        "paper_code_differences": [
            "Released episode_norm_reward defaults to statistics over step rows, "
            "repeating each trajectory reward; paper Eq. 3 describes trajectory statistics.",
            "The trainer subtracts the current row's invalid-action penalty after "
            "discounted-return computation and from that row's episode score.",
            "The released ALFWorld full-finetuning script specifies Qwen2.5-1.5B; "
            "the study explicitly transfers its recipe to Qwen2.5-7B, retaining "
            "the source's virtual 2-rank, 32-row microbatch weighting.",
        ],
    },
    "hgpo": {
        "name": "HGPO", "venue": "ICLR 2026",
        "paper": "https://openreview.net/forum?id=T8Dev99qnz",
        "paper_version": "2602.22817v1",
        "repo": UPSTREAM_REPOSITORY, "repo_commit": UPSTREAM_COMMIT,
        "implementation": "released-source-recipe",
        "source_files": {**_SHARED_FILES,
            "credit": _SOURCE + "recipe/hgpo/core_hgpo.py",
            "recipe": _SOURCE + "recipe/hgpo/run_qwen2.5_7b_alfworld_train.sh",
            "config": _SOURCE + "recipe/hgpo/config/hgpo_trainer.yaml",
            "reward_order": _SOURCE + "recipe/hgpo/hgpo_ray_trainer.py"},
        "credit_functions": ["hgpo_advantage_estimate", "compute_step_discounted_returns"],
        "paper_code_differences": [
            "Released groups use observation-only suffixes of length k=1..K+1; "
            "the actor memory also contains actions.",
            "Released aggregation excludes zero advantages, renormalizes the "
            "remaining weights, and uses (k+1)^alpha for suffix length k; "
            "paper Eq. 7 sums all levels indexed from zero.",
            "Released within-group standard deviation is population std; "
            "base_group=False omits an episode-level term.",
        ],
    },
    "graphgpo": {
        "name": "GraphGPO", "venue": "ICML 2026",
        "paper": "https://proceedings.mlr.press/v306/cheng26e.html",
        "paper_version": "2605.26684v1; PMLR 306:18861-18886",
        "repo": UPSTREAM_REPOSITORY, "repo_commit": UPSTREAM_COMMIT,
        "implementation": "released-source-recipe",
        "source_files": {**_SHARED_FILES,
            "credit": _SOURCE + "recipe/GraphGPO/core_graph.py",
            "normalization": _SOURCE + "gigpo/core_gigpo.py",
            "recipe": _SOURCE + "recipe/GraphGPO/run_qwen2.5_7b_alfworld_train.sh",
            "recipe_1_5b": _SOURCE + "recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train.sh",
            "config": _SOURCE + "recipe/GraphGPO/config/graphgpo_trainer.yaml",
            "public_state": _SOURCE + "recipe/GraphGPO/alfworld_envs.py",
            "state_prompt": _SOURCE + "recipe/GraphGPO/graphgpo_env_manager.py",
            "success_node": _SOURCE + "recipe/GraphGPO/rollout_loop.py",
            "batch_adjustment": _SOURCE + "recipe/GraphGPO/utils.py",
            "trajectory_row_order": _SOURCE + "recipe/GraphGPO/rollout_loop.py",
            "reward_order": _SOURCE + "recipe/GraphGPO/graphgpo_ray_trainer.py"},
        "credit_functions": ["ComplexGraph.from_data", "all_to_target_shortest_paths",
                             "compute_graph_path_returns", "compute_graphgpo_outcome_advantage"],
        "paper_code_differences": [
            "Released unit-cost graph return is 10*gamma**d(next); paper Eq. 4 "
            "uses an additional transition-cost exponent.",
            "Released episode normalization uses the same step-row statistics "
            "and current-row invalid-action penalty as GiGPO.",
            "ALFWorld actor observations append the released public action/feedback "
            "history summary; graph keys additionally include sorted admissible commands.",
        ],
    },
}


def published_recipe(method, benchmark, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Resolve published settings and the explicitly named benchmark transfer."""
    recipe = dict(PUBLISHED_RECIPES[method])
    if method == "graphgpo" and "1.5b" in str(model_name).lower():
        recipe.update(minibatch=256, source_model="Qwen/Qwen2.5-1.5B-Instruct",
                      source_virtual_world_size=2, source_actor_microbatch=32,
                      source_rollout_logprob_microbatch=32, source_ref_logprob_microbatch=32)
    elif method == "hgpo" and "1.5b" in str(model_name).lower():
        recipe.update(source_model="Qwen/Qwen2.5-1.5B-Instruct",
                      source_virtual_world_size=2, source_actor_microbatch=16,
                      source_rollout_logprob_microbatch=16, source_ref_logprob_microbatch=16)
    recipe["backbone_transfer"] = (
        "released Qwen2.5-1.5B full-finetuning recipe transferred to study Qwen2.5-7B"
        if method == "gigpo" and "7b" in str(model_name).lower() else "source backbone family")
    if benchmark == "appworld":
        recipe.update(max_new_tokens=1024, author_prompt_length=None,
                      prompt_truncation="complete_messages",
                      benchmark_transfer="AppWorld native Python; pinned task/API bootstrap and recent memory")
    elif benchmark == "alfworld":
        recipe["benchmark_transfer"] = "official ALFWorld text interface"
    else:
        raise ValueError(f"No published-baseline benchmark adapter for {benchmark!r}")
    return recipe
