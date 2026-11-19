"""Collect fresh official WebShop 4+4 groups under the V5 public contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from fgpo.envs import EnvironmentPool
from fgpo.policy import CausalPolicy
from fgpo.public import OUTPUT_LIMITS
from fgpo.records import CostLedger, GroupRecord
from pilot_collect_v5 import (
    BENCHMARK,
    atomic_write,
    collect_four,
    make_review,
)


def collect_task(policy, environments, executor, task):
    group_id = f"verified-feedback-v5:{task['index']:03d}"
    started = perf_counter()
    first, first_costs, _, first_details = collect_four(
        policy, environments, executor, task, group_id, range(4), "first")
    eligible = all(not trajectory.official_success for trajectory in first)
    review, review_costs, review_seconds = None, (), 0.0
    if eligible:
        review_started = perf_counter()
        policy.ledger.begin_group(group_id)
        with policy.ledger.aggregate_model_calls():
            review = make_review(policy, first[task["donor_slot"]], task["review_seed"])
        review_costs = tuple(policy.ledger.records)
        review_seconds = perf_counter() - review_started

    second_scope = "guided" if eligible else "ordinary"
    second, second_costs, marked, second_details = collect_four(
        policy, environments, executor, task, group_id, range(4, 8), second_scope, review)
    trajectories = first + second
    group = GroupRecord(group_id, task["task_id"], BENCHMARK, task["environment_seed"],
                        trajectories, first_costs + review_costs + second_costs)
    note_index = review.turn if review else None
    marked_by_slot = {row["slot"]: row for row in marked}
    density_fields = ("slot", "generated_tokens", "logp", "logq", "sampled_terminator")
    slots = [
        {
            "slot": trajectory.slot,
            "sampling_seed": trajectory.sampling_seed,
            "wave": 1 if trajectory.slot < 4 else 2,
            "behavior": "Q" if trajectory.slot in marked_by_slot else "P",
            "marked_reached": trajectory.slot in marked_by_slot,
            "official_success": trajectory.official_success,
            "native_score": trajectory.official_metrics["score"],
            "official_evaluation": dict(trajectory.official_metrics),
        }
        for trajectory in trajectories
    ]
    diagnostics = {
        "index": task["index"],
        "task_id": task["task_id"],
        "public_contract": "fgpo-v5",
        "eligible": eligible,
        "donor_slot": task["donor_slot"],
        "note_index": note_index,
        "second_wave": "Q" if eligible else "P",
        "first4": [slot["official_success"] for slot in slots[:4]],
        "second4": [slot["official_success"] for slot in slots[4:]],
        "marked_reach": [slot["marked_reached"] for slot in slots],
        "slots": slots,
        "marked_density": [
            {field: row[field] for field in density_fields} for row in marked
        ],
        "first4_cost": first_details,
        "second4_cost": second_details,
        "review_cost": {
            "seconds": review_seconds,
            "input_tokens": review.input_tokens if review else 0,
            "output_tokens": review.output_tokens if review else 0,
        },
        "wall_seconds": perf_counter() - started,
    }
    return {
        "index": task["index"],
        "eligible": eligible,
        "group": group,
        "note_index": note_index,
        "hint_ids": review.hint_ids if review else (),
        "review": asdict(review) if review else None,
        "marked": marked,
        "diagnostics": diagnostics,
    }


@torch.inference_mode()
def run(root, config_path, shard):
    root = root.resolve()
    config = json.loads(config_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    rows = sorted(manifest["tasks"], key=lambda row: row["index"])
    if ([row["index"] for row in rows] != list(range(32))
            or len({row["task_id"] for row in rows}) != 32):
        raise ValueError("The manifest must specify 32 distinct fixed training tasks.")
    for row in rows:
        if not (row["task_id"].startswith(f"train/{row['environment_seed']}/")
                and 0 <= row["donor_slot"] < 4
                and len(set(row["slot_seeds"])) == len(row["slot_seeds"]) == 8):
            raise ValueError(f"Invalid training manifest entry at index {row['index']}.")
    if torch.cuda.device_count() != 1:
        raise ValueError("Each WebShop collection shard requires exactly one visible CUDA device.")

    started = perf_counter()
    samples = root / "samples"
    samples.mkdir(exist_ok=True)
    ledger = CostLedger(gpu_ids=(0,))
    policy = CausalPolicy(
        str(Path(manifest["source_checkpoint"]).resolve(strict=True)),
        gpu_ids=(0,), ledger=ledger, trainable=False, benchmark=BENCHMARK,
        max_context_tokens=int(config["max_context_tokens"]),
        max_new_tokens=OUTPUT_LIMITS[BENCHMARK],
        cache_initial_capacity=int(config["cache_initial_capacity"]),
        working_memory_gib=float(config["working_memory_gib"]),
    )
    policy.sampling_temperature = 1.0
    environments = EnvironmentPool(config, BENCHMARK, root / f"workers-{shard}", size=4)
    print(f"sampler ready shard={shard} physical_gpu={os.environ['CUDA_VISIBLE_DEVICES']} "
          f"gpu_name={torch.cuda.get_device_name(0)} cold_load_seconds={perf_counter() - started:.1f}",
          flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for task in rows[shard::2]:
                index = task["index"]
                record = samples / f"{index:03d}.pt"
                companion = samples / f"{index:03d}.json"
                if record.exists() and companion.exists():
                    continue
                if record.exists():
                    sample = torch.load(record, map_location="cpu", weights_only=False)
                else:
                    print(f"start task index={index}", flush=True)
                    sample = collect_task(policy, environments, executor, task)
                    atomic_write(record, lambda stream: torch.save(sample, stream))
                details = sample["diagnostics"]
                encoded = (json.dumps(details, allow_nan=False) + "\n").encode("utf-8")
                atomic_write(companion, lambda stream: stream.write(encoded))
                print(f"done task index={index} eligible={int(details['eligible'])} "
                      f"note_index={details['note_index']} "
                      f"marked_reached={sum(details['marked_reach'])} "
                      f"first4_successes={sum(details['first4'])} "
                      f"second4_successes={sum(details['second4'])} "
                      f"seconds={details['wall_seconds']:.1f}", flush=True)
    finally:
        environments.close()
    print(f"sampler complete shard={shard} seconds={perf_counter() - started:.1f} "
          f"peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.2f}", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    run(args.root, args.config, args.shard)
