"""Frozen WebShop pilot corpus, matched V4 mini-updates and no-hint evaluation."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import socket
from statistics import fmean
import sys
from time import monotonic

import torch
from safetensors import safe_open

import fgpo
from fgpo.appworld_batch import RolloutGroupSpec
from fgpo.credit import advantages
from fgpo.engine import Engine, _mixed_groups
from fgpo.envs import EnvironmentPool
from fgpo.optimization import OffloadedAdamW, WeightReference, ppo_update
from fgpo.reference import prepare_round
from fgpo.rollout import collect_evaluation
from fgpo.runner import _bind_compute_pair_cpu_affinity, isolate_gpus, write_json


ARMS = ("initial", "ordinary", "proposed")


class GroupWeightedAdamW(OffloadedAdamW):
    group_weight = 1.0

    def _update_slice(self, entry, gradient, initial):
        # The deployed optimizer has already clipped this FP32 gradient bucket.
        gradient.mul_(self.group_weight)
        return super()._update_slice(entry, gradient, initial)


class PilotEngine(Engine):
    def __init__(self, config, output, anchor):
        self.anchor = str(anchor)
        super().__init__(config, output)

    def _make_policy(self, path=None):
        return super()._make_policy(self.anchor)

    def _make_learning_state(self):
        config = self.config
        return GroupWeightedAdamW(
            self.policy.model,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=tuple(config["optimizer_betas"]),
            eps=config["optimizer_eps"],
            bucket_mib=config["optimizer_bucket_mib"],
            pin_memory=config["optimizer_pin_memory"],
        ), None, None


def load_reference(model, path):
    weights = {}
    by_shard = defaultdict(list)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    for name, parameter in model.named_parameters():
        by_shard[weight_map[name]].append((name, parameter.dtype, parameter.shape))
    for shard, entries in by_shard.items():
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            for name, dtype, shape in entries:
                tensor = handle.get_tensor(name)
                if tensor.shape != shape:
                    raise ValueError(f"Initial reference parameter shape differs at {name}")
                weights[name] = tensor.to(dtype=dtype)
    return WeightReference(state=weights, fixed=True)


def select_group(pilot, task, arm):
    index = task["index"]
    record = torch.load(pilot / "samples" / f"{index:03d}.pt",
                        map_location="cpu", weights_only=False)
    eligible = not any(t.official_success for t in record["P"].trajectories[:4])
    selected = task["mixture_arm"] if arm == "proposed" and eligible else "P"
    group = record[selected]
    if (record["index"] != index or record["eligible"] != eligible
            or (eligible and record["mixture_arm"] != task["mixture_arm"])
            or group is None or group.task_id != task["task_id"]
            or group.initial_seed != task["environment_seed"]
            or [t.sampling_seed for t in group.trajectories] != task["slot_seeds"]):
        raise ValueError(f"Collected group disagrees with manifest entry {index}")
    weight = float(record[f"w_{selected}"]) if arm == "proposed" and eligible else 1.0
    if not math.isfinite(weight) or not 0.0 <= weight <= 2.0:
        raise ValueError(f"Invalid whole-group weight at manifest entry {index}")
    return task, group, eligible, selected, weight


def prepare_credit(policy, config, groups, *, reference_loaded=False):
    started = monotonic()
    mixed_groups = _mixed_groups(groups)
    trajectories = [t for group in groups for t in group.trajectories]
    reference_data = None
    if mixed_groups:
        mixed_trajectories = [t for group in mixed_groups for t in group.trajectories]
        activation = (nullcontext() if reference_loaded else
                      load_reference(policy.model, Path(config["model_path"])).activate(policy.model))
        with policy.phase("reference.single_round_activation"), activation:
            policy._reference_active = True
            try:
                reference_data = prepare_round(policy, mixed_trajectories, config=config, round_index=0)
            finally:
                policy._reference_active = False
    with policy.phase("credit.peer_values"):
        group_advantages = advantages(trajectories, reference_data, config=config)
    if reference_data is not None:
        reference_data.release_features()
    return {"reference": reference_data, "advantages": group_advantages, "prepared_seconds": monotonic() - started}


def update(engine, pilot, tasks, arm, *, credit=None):
    start = monotonic()
    config, policy, optimizer = engine.config, engine.policy, engine.optimizer
    scheduled = [select_group(pilot, task, arm) for task in tasks]
    groups = [item[1] for item in scheduled]
    mixed_groups = _mixed_groups(groups)
    mixed_ids = {group.group_id for group in mixed_groups}
    if credit is None:
        credit = prepare_credit(policy, config, groups)
    reference_data, group_advantages = credit["reference"], credit["advantages"]

    rows = []
    with (engine.output_dir / "groups.jsonl").open("w", buffering=1) as output:
        for task, group, eligible, selected, weight in scheduled:
            mixed = group.group_id in mixed_ids
            row = {
                "index": task["index"], "task_id": group.task_id, "group_id": group.group_id,
                "eligible": eligible, "planned_arm": task["mixture_arm"], "selected_arm": selected,
                "weight": weight, "successes": sum(t.official_success for t in group.trajectories),
                "mixed": mixed, "optimizer_step": False, "optimizer_steps_total": optimizer.steps,
                "preclip_gradient_norm": 0.0, "clip_scale": None, "clipped_gradient_norm": 0.0,
                "weighted_gradient_norm": 0.0, "variance_trace_contribution": 0.0, "finite": True,
            }
            if mixed:
                optimizer.group_weight = weight
                try:
                    cursor = ppo_update(
                        policy, group.trajectories, reference_data, group_advantages, optimizer,
                        trajectory_groups=[group], round_index=0,
                        chunk_tokens=config["head_chunk_tokens"], constant_denominator=True,
                        loss_normalization=config["loss_normalization"],
                        clip_ratio=config["ppo_clip_ratio"], dual_clip=config["ppo_dual_clip"],
                        entropy_coef=config["entropy_coef"], kl_coef=config["kl_coef"],
                        max_grad_norm=config["max_grad_norm"],
                    )
                except FloatingPointError:
                    output.write(json.dumps({**row, "finite": False, "preclip_gradient_norm": None,
                                             "clipped_gradient_norm": None,
                                             "weighted_gradient_norm": None,
                                             "variance_trace_contribution": None}) + "\n")
                    raise
                grad_norm = cursor.metrics["gradient_norm"]
                scale = cursor.metrics["batch_clip_scales"][0]
                clipped = grad_norm * scale
                row.update(
                    optimizer_step=True, optimizer_steps_total=optimizer.steps,
                    preclip_gradient_norm=grad_norm, clip_scale=scale,
                    clipped_gradient_norm=clipped, weighted_gradient_norm=weight * clipped,
                    variance_trace_contribution=(weight * weight - weight) * clipped**2,
                    ppo=cursor.metrics,
                )
            output.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
    result = {
        "scheduled_groups": len(rows), "mixed_groups": len(mixed_groups),
        "optimizer_steps": optimizer.steps, "all_gradients_finite": all(row["finite"] for row in rows),
        "mean_clipped_gradient_norm": fmean(row["clipped_gradient_norm"] for row in rows),
        "mean_weighted_gradient_norm": fmean(row["weighted_gradient_norm"] for row in rows),
        "mean_variance_trace_contribution": fmean(row["variance_trace_contribution"] for row in rows),
        "seconds": monotonic() - start,
    }
    write_json(engine.output_dir / "update.json", result)
    optimizer.state.clear()
    return result


def evaluate(engine, tasks, arm):
    start = monotonic()
    config, policy = engine.config, engine.policy
    policy.model.eval()
    policy.sampling_temperature = config["eval_temperature"]
    pool = EnvironmentPool(config, "webshop", engine.output_dir / "environments",
                           size=config["environment_pool_size"])
    task_by_id = {task["task_id"]: task for task in tasks}
    rows = []
    try:
        with (engine.output_dir / "evaluation.jsonl").open("w", buffering=1) as output:
            def publish(spec, episode):
                task = task_by_id[spec.task_id]
                row = {**task, **asdict(episode), "arm": arm,
                       "official_success": episode.official_evaluation["success"],
                       "native_score": episode.official_evaluation["score"]}
                output.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)

            engine._refresh_collection_replica(role="evaluation")
            with policy.use_collection_replica(engine._collection_replica):
                for start_index in range(0, len(tasks), pool.size):
                    batch = tasks[start_index:start_index + pool.size]
                    specs = tuple(RolloutGroupSpec(
                        f"pilot-evaluation:{task['index']}", task["task_id"], task["environment_seed"],
                        (task["sampling_seed"],),
                    ) for task in batch)
                    with policy.phase("evaluation.rollout", details={"start_index": start_index}):
                        collect_evaluation(policy, pool, specs, benchmark="webshop",
                                           on_episode_complete=publish)
    finally:
        engine._collection_replica = None
        pool.close()
    if {row["task_id"] for row in rows} != set(task_by_id) or len(rows) != len(tasks):
        raise RuntimeError("The no-hint evaluation did not score every manifest entry once")
    return {
        "tasks": len(rows), "official_successes": sum(row["official_success"] for row in rows),
        "official_success_rate": fmean(row["official_success"] for row in rows),
        "mean_native_score": fmean(row["native_score"] for row in rows),
        "mean_steps": fmean(row["steps"] for row in rows),
        "seconds": monotonic() - start,
    }


def summarize(output):
    receipts = {arm: output / arm / "receipt.json" for arm in ARMS}
    if not all(path.is_file() for path in receipts.values()):
        return
    rows = {}
    for arm in ARMS:
        path = output / arm / "evaluation.jsonl"
        rows[arm] = {row["task_id"]: row for row in map(json.loads, path.read_text().splitlines())}
    comparisons = {}
    for first, second in (("ordinary", "initial"), ("proposed", "initial"), ("proposed", "ordinary")):
        pairs = [(row, rows[second][task_id]) for task_id, row in rows[first].items()]
        comparisons[f"{first}_minus_{second}"] = {
            "pairs": len(pairs),
            "full_success_gains": sum(a["official_success"] and not b["official_success"] for a, b in pairs),
            "full_success_losses": sum(b["official_success"] and not a["official_success"] for a, b in pairs),
            "success_rate_difference": fmean(int(a["official_success"]) - int(b["official_success"]) for a, b in pairs),
            "native_score_difference": fmean(a["native_score"] - b["native_score"] for a, b in pairs),
            "steps_difference": fmean(a["steps"] - b["steps"] for a, b in pairs),
        }
    write_json(output / "summary.json", {
        "arms": {arm: json.loads(path.read_text()) for arm, path in receipts.items()},
        "paired_differences": comparisons,
    })


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    start = monotonic()
    manifest = json.loads((args.pilot / "manifest.json").read_text())
    anchor = Path(manifest["source_checkpoint"])
    physical_ids = manifest["physical_gpu_ids"]
    if socket.gethostname() != manifest["host"] or physical_ids != [3, 4]:
        raise ValueError("The pilot requires its manifest's Host A 3+4 allocation")
    config = json.loads(args.config.read_text())
    logical_ids, _ = isolate_gpus(physical_ids, config["excluded_gpu_ids"])
    _bind_compute_pair_cpu_affinity(config, physical_ids)
    config.update(method="fgpo", benchmark="webshop", seed=0, gpu_ids=logical_ids,
                  allocated_physical_gpu_ids=physical_ids, compute_physical_gpu_ids=physical_ids,
                  compute_logical_gpu_ids=logical_ids, model_placement_explicit=True)
    if len(manifest["tasks"]) != 32 or len(manifest["evaluation_tasks"]) != 48:
        raise ValueError("The pilot manifest must contain 32 update and 48 no-hint evaluation tasks")
    root = args.pilot / "updates"
    output = root / args.arm
    output.mkdir(parents=True, exist_ok=True)
    engine = PilotEngine(config, output, anchor)
    try:
        result = None if args.arm == "initial" else update(engine, args.pilot, manifest["tasks"], args.arm)
        evaluation = evaluate(engine, manifest["evaluation_tasks"], args.arm)
        seconds = monotonic() - start
        receipt = {
            "arm": args.arm, "exit_code": 0, "total_seconds": seconds,
            "allocated_gpu_seconds": len(physical_ids) * seconds,
            "command": [sys.executable, *sys.argv],
            "formal_source": str(Path(fgpo.__file__).resolve().parent),
            "formal_config": str(args.config), "actor_anchor": str(anchor),
            "reference": config["model_path"], "manifest": str(args.pilot / "manifest.json"),
            "manifest_update_groups": len(manifest["tasks"]), "formal_groups_per_round": config["groups"],
            "passes_over_frozen_collection": int(args.arm != "initial"),
            "optimizer_initial_step": 0,
            "ppo_config": {key: engine.config[key] for key in (
                "learning_rate", "weight_decay", "optimizer_betas", "optimizer_eps", "reward_scale",
                "loss_normalization", "ppo_clip_ratio", "ppo_dual_clip", "entropy_coef", "kl_coef",
                "max_grad_norm",
            )},
            "evaluation_config": {
                "split": "held_out_native_train_ids", "temperature": engine.config["eval_temperature"],
                "top_p": engine.config["eval_top_p"], "horizon": config["horizons"]["webshop"],
                "response_tokens": engine.config["output_tokens"], "hint": False,
            },
            "physical_gpu_ids": physical_ids,
            "actor_device_map": engine.policy.device_map,
            "peak_reserved_gpu_bytes": [torch.cuda.max_memory_reserved(i) for i in logical_ids],
            "training": result, "evaluation": evaluation,
        }
        write_json(output / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        summarize(root)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
