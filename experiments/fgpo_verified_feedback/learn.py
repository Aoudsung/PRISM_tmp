"""Matched WebShop updates and no-hint evaluation for verified feedback."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
from statistics import fmean
import sys
from time import monotonic

import torch
import torch.nn.functional as F

import fgpo
from fgpo import rollout
from fgpo.appworld_batch import RolloutGroupSpec
from fgpo.checkpoint import seed_everything
from fgpo.credit import advantages as v4_advantages
from fgpo.envs import EnvironmentPool
from fgpo.metrics import webshop_score
from fgpo.optimization import OffloadedAdamW, ppo_update
from fgpo.records import CostLedger
from fgpo.reference import prepare_round as v4_prepare_round
from fgpo.rollout import collect_evaluation
from fgpo.runner import _bind_compute_pair_cpu_affinity, isolate_gpus, write_json
from update import PilotEngine, load_reference


TRAIN = {
    "archive_null": ("archive", "v4", False),
    "archive_success": ("archive", "v4", True),
    "archive_success_history": ("archive", "history", True),
    "fresh_null_history": ("fresh", "history", False),
    "fresh_success_history": ("fresh", "history", True),
}
ARMS = ("anchor_v4", "anchor_v5", *TRAIN)
PAIRS = (
    ("archive_success", "archive_null"),
    ("archive_success_history", "archive_success"),
    ("fresh_success_history", "fresh_null_history"),
    ("anchor_v5", "anchor_v4"),
    ("fresh_success_history", "anchor_v5"),
    ("fresh_null_history", "anchor_v5"),
    ("archive_success_history", "anchor_v4"),
)


class ObservedAdamW(OffloadedAdamW):
    def step(self, max_norm=1.):
        self.last_mask = tuple(name for name, parameter in self.params if parameter.grad is not None)
        return super().step(max_norm=max_norm)


class LearningEngine(PilotEngine):
    def _make_learning_state(self):
        config = self.config
        return ObservedAdamW(
            self.policy.model,
            lr=config["learning_rate"], weight_decay=config["weight_decay"],
            betas=tuple(config["optimizer_betas"]), eps=config["optimizer_eps"],
            bucket_mib=config["optimizer_bucket_mib"], pin_memory=config["optimizer_pin_memory"],
        ), None, None


def load_groups(root, tasks, corpus):
    scheduled = []
    for task in tasks:
        index = task["index"]
        path = root / "samples" / f"{index:03d}.pt"
        record = torch.load(path, map_location="cpu", weights_only=False)
        eligible = record["eligible"]
        if corpus == "archive":
            group = record["Q"] if eligible else record["P"]
            note_index = record["diagnostics"]["note_index"]
        else:
            group = record["group"]
            note_index = record["note_index"]
        trajectories = group.trajectories
        if (record["index"] != index or group.task_id != task["task_id"]
                or group.initial_seed != task["environment_seed"]
                or [t.slot for t in trajectories] != list(range(8))
                or [t.sampling_seed for t in trajectories] != task["slot_seeds"]
                or eligible != all(not t.official_success for t in trajectories[:4])
                or (note_index is not None) != eligible
                or (eligible and note_index < 1)):
            raise ValueError(f"Frozen group disagrees with its manifest at {path}")
        for trajectory in trajectories:
            webshop_score(trajectory.official_metrics)
            if trajectory.official_success is not trajectory.official_metrics["success"]:
                raise ValueError(f"Outcome disagrees with the native receipt at {path}")
        scheduled.append((task, group, eligible, note_index))
    return scheduled


def summarize_credit(diagnostics):
    rows = list(diagnostics.values())
    steps = sum(row["step_count"] for row in rows)
    matched = sum(row["matched_steps"] for row in rows)
    return {
        "ordinary_mixed_groups": len(rows), "steps": steps, "matched_steps": matched,
        "matched_step_fraction": matched / steps if steps else None,
        "mean_support_mass": (math.fsum(row["mean_support_mass"] * row["step_count"]
                                        for row in rows) / steps if steps else None),
        "loo_outcome_brier": (math.fsum(row["loo_outcome_brier"] * row["step_count"]
                                       for row in rows) / steps if steps else None),
        "loo_task_prior_brier": (math.fsum(row["loo_task_prior_brier"] * row["step_count"]
                                          for row in rows) / steps if steps else None),
        "matched_loo_outcome_brier": (math.fsum(row["matched_loo_outcome_brier"] * row["matched_steps"]
                                               for row in rows if row["matched_steps"]) / matched
                                     if matched else None),
    }


def ordinary_credit(engine, scheduled, cache_path, kernel):
    start = monotonic()
    trajectories = [t for _, group, eligible, _ in scheduled
                    if not eligible and mixed(group) for t in group.trajectories]
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cache, True, monotonic() - start

    reference_data, values, diagnostics = None, {}, {}
    if trajectories:
        if kernel == "history":
            from history_credit import advantages, prepare_round
        else:
            advantages, prepare_round = v4_advantages, v4_prepare_round
        policy, config = engine.policy, engine.config
        reference = load_reference(policy.model, Path(config["model_path"]))
        # The anchor is already finetuned. Save and restore it around pretrained scoring.
        with policy.phase("reference.single_round_activation"), reference.activate(policy.model):
            policy._reference_active = True
            try:
                reference_data = prepare_round(policy, trajectories, config=config, round_index=0)
            finally:
                policy._reference_active = False
        del reference
        with policy.phase("credit.peer_values"):
            values = advantages(trajectories, reference_data, config=config, diagnostics=diagnostics)
        reference_data.release_features()
    cache = dict(reference=reference_data, advantages=values, diagnostics=diagnostics,
                 prepared_seconds=monotonic() - start)
    temporary = cache_path.with_suffix(".incoming")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    return cache, False, monotonic() - start


def marked_responses(group, note_index):
    reached = [t for t in group.trajectories[4:] if len(t.steps) >= note_index]
    rows = []
    for trajectory in reached:
        if not trajectory.official_success:
            continue
        step = trajectory.steps[note_index - 1]
        span = step.response_span
        segment = next(s for s in trajectory.segments if s.segment_id == span.segment_id)
        tokens = segment.token_ids[:span.stop]
        if (step.index != note_index - 1 or span.start != step.prompt_length or span.start < 1
                or not step.response_ids or tokens[span.start:] != step.response_ids):
            raise ValueError(f"Marked response is not aligned with its ordinary prefix at {step.step_id}")
        rows.append((step, tokens))
    return len(reached), rows


def supervised_tick(policy, optimizer, rows, config, active):
    denominator = 8
    chunk_tokens = config["head_chunk_tokens"]
    head_device = policy.model.get_output_embeddings().weight.device
    nll = torch.zeros((), device=head_device, dtype=torch.float64)
    optimizer.zero_grad()
    policy.model.train()
    try:
        for step, tokens in rows:
            with policy.phase("success_nll.forward"):
                hidden, _ = policy.packed_segment_hidden(tokens, hidden_layer=None,
                                                         role="success_nll.ordinary_prefix")
                response = hidden[0, step.prompt_length - 1:].contiguous()
                del hidden
                targets = torch.tensor(step.response_ids, dtype=torch.long, device=head_device)
                response_gradient = torch.empty_like(response)
            with policy.phase("success_nll.backward"):
                # Stream vocabulary chunks and traverse the causal backbone once per response.
                for start in range(0, len(targets), chunk_tokens):
                    end = min(start + chunk_tokens, len(targets))
                    states = response[start:end].detach().requires_grad_(True)
                    logits = policy.head(states, role="success_nll.head")
                    token_nll = F.cross_entropy(logits, targets[start:end], reduction="sum")
                    nll.add_(token_nll.detach().double())
                    (token_nll / denominator * int(active)).backward()
                    response_gradient[start:end].copy_(states.grad)
                    del logits, states, token_nll
                response.backward(response_gradient)
                del response, response_gradient, targets
        true_nll = nll.item() / denominator
        if not math.isfinite(true_nll):
            raise FloatingPointError("Non-finite verified-success NLL")
        with policy.phase("success_nll.adamw"):
            norm = optimizer.step(max_norm=config["max_grad_norm"])
        scale = min(1., config["max_grad_norm"] / (norm + 1e-6))
        return true_nll, norm, norm * scale
    finally:
        optimizer.zero_grad()
        policy.model.eval()


def mixed(group):
    return 0 < sum(t.official_success for t in group.trajectories) < len(group.trajectories)


@contextmanager
def trace_attention(operations, tick):
    if operations is None or tick in operations:
        yield
        return
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        yield
    operations[tick] = sorted({event.key for event in trace.key_averages()
                               if "scaled_dot_product" in event.key})


def update(engine, scheduled, cache_path, kernel, active):
    start = monotonic()
    policy, optimizer, config = engine.policy, engine.optimizer, engine.config
    before = engine.ledger.snapshot()
    cache, reused, credit_seconds = ordinary_credit(engine, scheduled, cache_path, kernel)
    attention_operations = {} if torch.are_deterministic_algorithms_enabled() else None
    group_rows = []
    with (engine.output_dir / "groups.jsonl").open("w", buffering=1) as output:
        for task, group, eligible, note in scheduled:
            started = monotonic()
            engine.ledger.begin_group(group.group_id)
            seed = task["slot_seeds"][0]
            seed_everything(seed, engine.gpu_ids)
            trajectory_successes = [t.official_success for t in group.trajectories]
            trajectory_scores = [t.official_metrics["score"] for t in group.trajectories]
            row = dict(index=task["index"], task_id=group.task_id, group_id=group.group_id,
                       training_seed=seed, slot_seeds=task["slot_seeds"], eligible=eligible,
                       note_index=note, trajectory_successes=trajectory_successes,
                       trajectory_native_scores=trajectory_scores, tick="skip",
                       full_scores=sum(trajectory_successes), marked_reached=0,
                       marked_successes=[], supervised_tokens=0, true_nll_over_8=None,
                       preclip_gradient_norm=0., postclip_gradient_norm=0., gradient_parameter_names=[])
            if eligible:
                reached, responses = marked_responses(group, note)
                row.update(marked_reached=reached,
                           marked_successes=[step.step_id for step, _ in responses],
                           supervised_tokens=sum(len(step.response_ids) for step, _ in responses))
                if responses:
                    with (engine.ledger.context(group_id=group.group_id, round_index=0),
                          trace_attention(attention_operations, "supervised")):
                        nll, preclip, postclip = supervised_tick(policy, optimizer, responses, config, active)
                    row.update(tick="supervised", true_nll_over_8=nll, preclip_gradient_norm=preclip,
                               postclip_gradient_norm=postclip, supervised_loss_scale=int(active))
            elif mixed(group):
                with trace_attention(attention_operations, "ppo"):
                    cursor = ppo_update(
                        policy, group.trajectories, cache["reference"], cache["advantages"], optimizer,
                        trajectory_groups=[group], round_index=0, chunk_tokens=config["head_chunk_tokens"],
                        constant_denominator=True, loss_normalization=config["loss_normalization"],
                        clip_ratio=config["ppo_clip_ratio"], dual_clip=config["ppo_dual_clip"],
                        entropy_coef=config["entropy_coef"], kl_coef=config["kl_coef"],
                        max_grad_norm=config["max_grad_norm"],
                    )
                preclip = cursor.metrics["gradient_norm"]
                row.update(tick="ppo", preclip_gradient_norm=preclip,
                           postclip_gradient_norm=preclip * cursor.metrics["batch_clip_scales"][0],
                           ppo=cursor.metrics, credit=cache["diagnostics"][group.group_id])
            if row["tick"] != "skip":
                row["gradient_parameter_names"] = optimizer.last_mask
            row.update(optimizer_steps_total=optimizer.steps, seconds=monotonic() - started)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            group_rows.append(row)
            print(f"training index={row['index']} tick={row['tick']} seconds={row['seconds']:.1f}", flush=True)
    engine.ledger.begin_group()
    optimizer.state.clear()
    optimizer._transfer_slots.clear()
    after = engine.ledger.snapshot()
    return {
        "scheduled_groups": len(group_rows), "passes_over_frozen_collection": 1,
        "eligible_groups": sum(row["eligible"] for row in group_rows),
        "all_256_trajectory_full_scores": sum(row["full_scores"] for row in group_rows),
        "marked_reaches": sum(row["marked_reached"] for row in group_rows),
        "marked_successes": sum(len(row["marked_successes"]) for row in group_rows),
        "optimizer_steps": optimizer.steps,
        "ppo_steps": sum(row["tick"] == "ppo" for row in group_rows),
        "supervised_ticks": sum(row["tick"] == "supervised" for row in group_rows),
        "supervised_tokens": sum(row["supervised_tokens"] for row in group_rows),
        "credit": summarize_credit(cache["diagnostics"]),
        "fixed_reference_cache": str(cache_path), "fixed_reference_reused": reused,
        "fixed_reference_prepare_seconds": cache["prepared_seconds"],
        "fixed_reference_this_arm_seconds": credit_seconds,
        "groups_seconds": math.fsum(row["seconds"] for row in group_rows),
        "training_seconds": monotonic() - start,
        "training_sdpa_operations_first_real_tick": attention_operations,
        "training_cost": {key: after[key] - before[key] for key in (
            "input_tokens", "output_tokens", "forward_tokens", "env_steps",
        )},
    }


def evaluate(engine, panels, arm):
    config, policy = engine.config, engine.policy
    policy.model.eval()
    policy.sampling_temperature = config["eval_temperature"]
    pool = EnvironmentPool(config, "webshop", engine.output_dir / "environments",
                           size=config["environment_pool_size"])
    results = {}
    try:
        with (engine.output_dir / "evaluation.jsonl").open("w", buffering=1) as output:
            engine._refresh_collection_replica(role="evaluation")
            with policy.use_collection_replica(engine._collection_replica):
                for panel, tasks in panels.items():
                    started = monotonic()
                    task_by_id = {task["task_id"]: task for task in tasks}
                    rows = []

                    def publish(spec, episode):
                        native_score = webshop_score(episode.official_evaluation)
                        row = {**task_by_id[spec.task_id], **asdict(episode), "arm": arm, "panel": panel,
                               "official_success": episode.official_evaluation["success"],
                               "native_score": native_score,
                               "batch_seconds_to_completion": monotonic() - batch_started}
                        output.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.append(row)

                    for offset in range(0, len(tasks), pool.size):
                        specs = tuple(RolloutGroupSpec(
                            f"v5-no-hint:{panel}:{task['index']}", task["task_id"], task["environment_seed"],
                            (task["sampling_seed"],),
                        ) for task in tasks[offset:offset + pool.size])
                        batch_started = monotonic()
                        with policy.phase("evaluation.rollout", details={"panel": panel, "offset": offset}):
                            collect_evaluation(policy, pool, specs, benchmark="webshop",
                                               on_episode_complete=publish)
                    if {row["task_id"] for row in rows} != set(task_by_id) or len(rows) != len(tasks):
                        raise RuntimeError(f"The no-hint evaluation missed manifest tasks on {panel}")
                    results[panel] = {
                        "tasks": len(rows), "official_successes": sum(row["official_success"] for row in rows),
                        "official_success_rate": fmean(row["official_success"] for row in rows),
                        "mean_native_score": fmean(row["native_score"] for row in rows),
                        "generated_tokens": sum(row["output_tokens"] for row in rows),
                        "mean_generated_tokens": fmean(row["output_tokens"] for row in rows),
                        "mean_steps": fmean(row["steps"] for row in rows), "seconds": monotonic() - started,
                    }
                    print(f"evaluation panel={panel} {json.dumps(results[panel])}", flush=True)
    finally:
        engine._collection_replica = None
        pool.close()
    return results


def evaluate_filter_only(engine, tasks, original_public, output):
    """Isolate the inaccurate-command removal from the V5 thought instruction."""
    started = monotonic()
    spec = importlib.util.spec_from_file_location("fgpo._filter_only_original_public", original_public)
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)

    def project(tokenizer, benchmark, raw_observation, *, action_schema, goal, initial=False):
        available = [command for command in action_schema["available_actions"]
                     if command.casefold() != "click[search]"]
        return public.project_public_observation(tokenizer, benchmark, raw_observation,
            action_schema={**action_schema, "available_actions": available}, goal=goal, initial=initial)

    def pin(tokenizer, system_prompt, initial, *, benchmark):
        return public.pinned_prompt(tokenizer, system_prompt, initial)

    original_project, original_pin = rollout.project_public_observation, rollout.pinned_prompt
    prior_output, prior_ledger = engine.output_dir, engine.ledger
    output.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(output / "costs.jsonl", gpu_ids=engine.gpu_ids)
    try:
        rollout.project_public_observation, rollout.pinned_prompt = project, pin
        engine.output_dir, engine.ledger, engine.policy.ledger = output, ledger, ledger
        result = evaluate(engine, {"confirm32": tasks}, "filter_only")
    finally:
        rollout.project_public_observation, rollout.pinned_prompt = original_project, original_pin
        engine.output_dir, engine.ledger, engine.policy.ledger = prior_output, prior_ledger, prior_ledger
    receipt = {
        "arm": "filter_only", "exit_code": 0, "primary_endpoint": False,
        "registered_after": "initial fresh WebShop samples showed increased response token counts",
        "protocol": "V4 WebShop system and observation suffix with click[search] omitted from the advertised list",
        "frozen_actor": engine.anchor, "actor_device_map": engine.policy.device_map,
        "evaluation_temperature": engine.config["eval_temperature"],
        "evaluation_top_p": engine.config["eval_top_p"], "hint": False,
        "projection_source": str(original_public), "training": None,
        "evaluation": result, "wall_seconds": monotonic() - started,
    }
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def fresh_placement(root):
    # The successful archival arm's observed split fits the shared occupancy on A1.
    load = next(row["details"] for row in read_jsonl(root / "archive_success" / "costs.jsonl")
                if row["role"] == "model.load")
    capacity = load["gpu_memory_preallocation"]["placement_budget_gib"]
    return {int(gpu): math.floor(gib) for gpu, gib in capacity.items()}, load["device_map"]


def execution_mode(policy):
    return {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "model_attention": policy.metadata["attention"],
        "sdpa_enabled_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }


def compare_gradient_prefix(first, second):
    rows = []
    first_supervised_index = None
    all_schedules_match = True
    all_masks_match = True
    for a, b in zip(first, second, strict=True):
        all_schedules_match &= (a["index"], a["group_id"], a["optimizer_steps_total"], a["tick"],
                                a["marked_successes"], a["supervised_tokens"]) == (
            b["index"], b["group_id"], b["optimizer_steps_total"], b["tick"],
            b["marked_successes"], b["supervised_tokens"])
        all_masks_match &= a["gradient_parameter_names"] == b["gradient_parameter_names"]
        if a["tick"] == "supervised" and first_supervised_index is None:
            first_supervised_index = a["index"]
        if first_supervised_index is None and a["tick"] == "ppo":
            delta = a["preclip_gradient_norm"] - b["preclip_gradient_norm"]
            rows.append({"index": a["index"], "first_norm": a["preclip_gradient_norm"],
                         "second_norm": b["preclip_gradient_norm"], "difference": delta})
    return {
        "first_supervised_index": first_supervised_index,
        "optimizer_schedules_equal": all_schedules_match, "gradient_tensor_masks_equal": all_masks_match,
        "prior_ppo_gradient_norms_equal": all(row["difference"] == 0 for row in rows) if rows else None,
        "prior_ppo_gradient_norms": rows,
    }


def summarize(root, archive, panels):
    paths = {arm: root / arm / "receipt.json" for arm in ARMS}
    if not all(path.is_file() for path in paths.values()):
        return
    receipts = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    rows = {arm: {(row["panel"], row["task_id"]): row
                  for row in read_jsonl(root / arm / "evaluation.jsonl")} for arm in ARMS}
    prior = archive / "updates" / "initial"
    prior_receipt = json.loads((prior / "receipt.json").read_text())
    receipts["anchor_v4"]["evaluation"]["dev48"] = prior_receipt["evaluation"]
    rows["anchor_v4"].update({("dev48", row["task_id"]): row
                             for row in read_jsonl(prior / "evaluation.jsonl")})
    comparisons = {}
    with (root / "paired_evaluation.jsonl").open("w", buffering=1) as output:
        for panel, tasks in panels.items():
            for task in tasks:
                arms = {}
                for arm in ARMS:
                    row = rows[arm][panel, task["task_id"]]
                    if row["sampling_seed"] != task["sampling_seed"]:
                        raise ValueError(f"No-hint evaluation differs in seed at {panel}/{task['task_id']}")
                    arms[arm] = {key: row.get(key) for key in (
                        "official_success", "native_score", "steps", "input_tokens", "output_tokens",
                        "batch_seconds_to_completion",
                    )}
                output.write(json.dumps({**task, "panel": panel, "arms": arms}, allow_nan=False) + "\n")
            comparisons[panel] = {}
            for a, b in PAIRS:
                pairs = [(rows[a][panel, t["task_id"]], rows[b][panel, t["task_id"]]) for t in tasks]
                comparisons[panel][f"{a}_minus_{b}"] = {
                    "pairs": len(pairs),
                    "full_success_gains": sum(x["official_success"] and not y["official_success"] for x, y in pairs),
                    "full_success_losses": sum(y["official_success"] and not x["official_success"] for x, y in pairs),
                    "success_rate_difference": fmean(int(x["official_success"]) - int(y["official_success"])
                                                     for x, y in pairs),
                    "native_score_difference": fmean(x["native_score"] - y["native_score"] for x, y in pairs),
                    "eval_seconds_difference": (receipts[a]["evaluation"][panel]["seconds"]
                                                - receipts[b]["evaluation"][panel]["seconds"]),
                }
    filter_root = root / "filter_only"
    supplementary = None
    if (filter_root / "receipt.json").is_file():
        supplementary = json.loads((filter_root / "receipt.json").read_text())
        filtered = {row["task_id"]: row for row in read_jsonl(filter_root / "evaluation.jsonl")}
        supplementary["paired_anchor_gaps"] = {}
        for anchor in ("anchor_v4", "anchor_v5"):
            pairs = [(row, rows[anchor]["confirm32", task_id]) for task_id, row in filtered.items()]
            if any(x["sampling_seed"] != y["sampling_seed"] for x, y in pairs):
                raise ValueError(f"Supplementary filter uses different seeds from {anchor}")
            supplementary["paired_anchor_gaps"][f"filter_only_minus_{anchor}"] = {
                "pairs": len(pairs),
                "full_success_gains": sum(x["official_success"] and not y["official_success"] for x, y in pairs),
                "full_success_losses": sum(y["official_success"] and not x["official_success"] for x, y in pairs),
                "native_score_difference": fmean(x["native_score"] - y["native_score"] for x, y in pairs),
                "generated_tokens_difference": sum(x["output_tokens"] - y["output_tokens"] for x, y in pairs),
                "mean_steps_difference": fmean(x["steps"] - y["steps"] for x, y in pairs),
                "eval_seconds_difference": (supplementary["evaluation"]["confirm32"]["seconds"]
                                            - receipts[anchor]["evaluation"]["confirm32"]["seconds"]),
            }
    write_json(root / "summary.json", {
        "primary_panel": "confirm32", "secondary_panel": "dev48",
        "archived_anchor_dev_source": str(prior / "receipt.json"),
        "arms": receipts, "paired_differences": comparisons,
        "supplementary_filter_only": supplementary,
        "matched_training": {f"{a}_minus_{b}": compare_gradient_prefix(
            read_jsonl(root / a / "groups.jsonl"), read_jsonl(root / b / "groups.jsonl"))
            for a, b in (PAIRS[0], PAIRS[2])},
    })


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    start = monotonic()
    fresh_training = args.arm.startswith("fresh_")
    fresh_actor_input = fresh_training or args.arm == "anchor_v5"
    if fresh_actor_input:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if fresh_training:
        torch.use_deterministic_algorithms(True)
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent = json.loads((args.archive / "manifest.json").read_text())
    anchor, physical_ids = parent["source_checkpoint"], parent["physical_gpu_ids"]
    if socket.gethostname() != parent["host"] or physical_ids != [3, 4]:
        raise ValueError("The update requires the original Host A 3+4 allocation")
    tasks = manifest["tasks"]
    panels = {"confirm32": manifest["confirmation_tasks"], "dev48": parent["evaluation_tasks"]}
    if ([task["index"] for task in tasks] != list(range(32))
            or len(panels["confirm32"]) != 32 or len(panels["dev48"]) != 48):
        raise ValueError("The update requires 32 frozen groups, confirm32, and dev48")
    config = json.loads(args.config.read_text())
    logical_ids, _ = isolate_gpus(physical_ids, config["excluded_gpu_ids"])
    _bind_compute_pair_cpu_affinity(config, physical_ids)
    config.update(method="fgpo", benchmark="webshop", seed=0, gpu_ids=logical_ids,
                  allocated_physical_gpu_ids=physical_ids, compute_physical_gpu_ids=physical_ids,
                  compute_logical_gpu_ids=logical_ids, model_placement_explicit=True)
    root = args.root / "updates"
    output = root / args.arm
    output.mkdir(parents=True, exist_ok=True)
    placement, expected_map = None, None
    if fresh_actor_input:
        placement, expected_map = fresh_placement(root)
        config["gpu_free_memory_gib"] = placement
    engine = LearningEngine(config, output, anchor)
    config = engine.config
    try:
        actual_map = engine.policy.device_map
        if expected_map is not None and actual_map != expected_map:
            raise RuntimeError("Fresh experiment could not reproduce the specified archival actor layout")
        training, saved = None, None
        if args.arm in TRAIN:
            corpus, kernel, active = TRAIN[args.arm]
            scheduled = load_groups(args.archive if corpus == "archive" else args.root, tasks, corpus)
            cache_path = root / f"fixed-{corpus}-{kernel}.pt"
            training = update(engine, scheduled, cache_path, kernel, active)
            training.update(corpus=corpus, kernel=kernel, supervised_loss_scale=int(active),
                            execution_mode=execution_mode(engine.policy))
            write_json(output / "training.json", training)
        if fresh_training:
            torch.use_deterministic_algorithms(False)
        evaluation = evaluate(engine, {"confirm32": panels["confirm32"]} if args.arm == "anchor_v4" else panels,
                              args.arm)
        if args.arm == "fresh_success_history":
            pending = output / "model.incoming"
            saved = output / "model"
            engine.policy.save_model(pending)
            pending.rename(saved)
        seconds = monotonic() - start
        receipt = {
            "arm": args.arm, "exit_code": 0, "actor_anchor": anchor,
            "source": str(Path(fgpo.__file__).resolve().parent), "config": str(args.config),
            "manifest": str(manifest_path), "physical_gpu_ids": physical_ids,
            "command": [sys.executable, *sys.argv], "allocated_gpu_seconds": len(physical_ids) * seconds,
            "total_seconds": seconds, "reference": config["model_path"], "optimizer_initial_step": 0,
            "ppo_config": {key: config[key] for key in (
                "learning_rate", "weight_decay", "optimizer_betas", "optimizer_eps", "reward_scale",
                "loss_normalization", "ppo_clip_ratio", "ppo_dual_clip", "entropy_coef", "kl_coef",
                "max_grad_norm",
            )},
            "evaluation_temperature": config["eval_temperature"],
            "evaluation_top_p": config["eval_top_p"], "hint": False,
            "evaluation_execution_mode": execution_mode(engine.policy),
            "actor_device_map": actual_map, "placement_budget_cap_gib": placement,
            "cost_ledger": str(output / "costs.jsonl"),
            "training": training, "evaluation": evaluation, "saved_model": str(saved) if saved else None,
            "peak_reserved_gpu_bytes": [torch.cuda.max_memory_reserved(i) for i in logical_ids],
        }
        write_json(output / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        if args.arm == "anchor_v5":
            original_public = args.config.parent / "src" / "fgpo" / "public.py"
            evaluate_filter_only(engine, panels["confirm32"], original_public, root / "filter_only")
        summarize(root, args.archive, panels)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
