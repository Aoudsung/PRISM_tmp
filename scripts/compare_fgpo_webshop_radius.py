"""Compare V4 radii on saved mixed training groups using the fixed reference."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fgpo.credit import CrossTrajectoryCredit
from fgpo.policy import CausalPolicy
from fgpo.protocol import PROTOCOL
from fgpo.records import CostLedger
from fgpo.reference import ReferenceRound, ReferenceStep
from fgpo.runner import write_json


@torch.no_grad()
def reference_features(policy, trajectories, *, include_obs):
    result = ReferenceRound()
    users = {}
    for trajectory in trajectories:
        previous = tuple(trajectory.initial_observation_ids)
        by_segment = {}
        for row in trajectory.steps:
            by_segment.setdefault(row.response_span.segment_id, []).append(row)
            if include_obs:
                key = previous or policy.encode("[Empty stdout]")
                users.setdefault(key, []).append(row.step_id)
                previous = tuple(row.observation_repr_ids)
        for segment in trajectory.segments:
            rows = by_segment.get(segment.segment_id, ())
            if not rows:
                continue
            _, hidden = policy.packed_segment_hidden(
                segment.token_ids[:segment.behavior_closed_length], hidden_layer=-8,
                hidden_positions=[row.prompt_length - 1 for row in rows], stop_at_hidden_layer=True,
            )
            vectors = F.normalize(hidden[0].float(), dim=-1).cpu()
            for row, vector in zip(rows, vectors, strict=True):
                result.steps[row.step_id] = ReferenceStep((), z_hist=vector)
    if include_obs:
        for step_ids, vector in zip(users.values(), policy.observation_embeddings(users), strict=True):
            for step_id in step_ids:
                result.steps[step_id].z_obs = vector
    return result


def score(trajectories, reference, config, radius, prior_weight):
    credit = CrossTrajectoryCredit(trajectories, reference, config={
        **config, "credit_radius": radius, "credit_prior_weight": prior_weight,
    })
    if config["benchmark"] == "textcraft":
        credit.advantages()
        diagnostics = credit.diagnostics
        steps = diagnostics["step_count"]
        return {
            "steps": steps, "supported_steps": diagnostics["matched_steps"],
            "step_squared_error": steps * diagnostics["loo_outcome_brier"],
        }

    weights = credit._same_depth_weights()
    support = weights.sum(dim=-1)
    kappa = credit.prior_weight
    baseline = (weights @ credit.outcomes + kappa * credit.prior[None, :]) / (support + kappa)
    errors = (baseline / credit.reward_scale - credit.outcomes[None, :] / credit.reward_scale).square()
    errors = errors[credit.times, credit.owners]
    tokens = torch.tensor([len(row.response_ids) for row in credit.rows], dtype=torch.float64)
    matched = support[credit.times, credit.owners] > 0
    return {
        "steps": len(credit.rows), "supported_steps": int(matched.sum()),
        "response_tokens": int(tokens.sum()), "supported_response_tokens": int(tokens[matched].sum()),
        "step_squared_error": float(errors.sum()), "response_token_squared_error": float(errors @ tokens),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("webshop", "textcraft"), default="webshop")
    parser.add_argument("--snapshot", type=Path, required=True, action="append")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    webshop = args.benchmark == "webshop"
    if webshop:
        from fgpo.protocol import WEBSHOP_FIXED_MENU, WEBSHOP_RADII, WEBSHOP_RADIUS_OVERRIDES

        radii = WEBSHOP_RADII
        prior_weight = WEBSHOP_RADIUS_OVERRIDES["credit_prior_weight"]
    else:
        radii = (.05, .10)
        prior_weight = 1.

    groups, sources = {}, []
    for path in args.snapshot:
        trainer = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
        state, config = trainer["state"], trainer["config"]
        if (config["protocol"], config["benchmark"], config["method"]) != (PROTOCOL, args.benchmark, "fgpo"):
            raise ValueError(f"Expected saved {args.benchmark} V4 F-GPO training groups: {path}")
        if webshop and WEBSHOP_FIXED_MENU not in config.get("adaptations", ()):
            raise ValueError(f"Expected the native repaired WebShop menu: {path}")
        if not webshop and config.get("credit_prior_weight", 1.) != prior_weight:
            raise ValueError(f"Expected standard TextCraft kappa: {path}")
        training_tasks = set() if webshop else {row["task_id"] for row in state.task_order}
        origin_path = path.with_suffix(".origin")
        sources.append({"retained_state": str(path), "original_state": (
            origin_path.read_text().strip() if origin_path.is_file() else str(path)),
            "seed": config["seed"], "round": state.round + 1, "phase": state.phase,
            "group_cursor": state.group_cursor,
            "study_protocol": config.get("study_protocol"), "available_groups": len(state.groups)})
        for group in state.groups:
            trajectories = group.trajectories
            successes = sum(t.official_success for t in trajectories)
            if len(trajectories) != config["trajectories"] or not 0 < successes < len(trajectories):
                continue
            if any(not (t.task_id.startswith("train/") if webshop else t.task_id in training_tasks)
                   for t in trajectories):
                raise ValueError("The radius comparison requires official training tasks")
            first = trajectories[0]
            groups.setdefault(first.group_id, (trajectories, config, state.reference_data))
    if not groups:
        if not webshop:
            write_json(args.output, {
                "benchmark": args.benchmark, "usable": False, "mixed_groups": 0,
                "reason": "No complete mixed groups in the preserved training snapshots",
                "training_sources": sources,
            })
            return
        raise ValueError("No preserved mixed training groups are available")

    policy = None
    results = {f"{radius:.2f}": [] for radius in radii}
    for group_id, (trajectories, config, reference) in groups.items():
        if reference is None or any(
            (step := reference.steps.get(row.step_id)) is None or step.z_hist is None or (webshop and step.z_obs is None)
            for t in trajectories for row in t.steps
        ):
            if policy is None:
                policy = CausalPolicy(
                    config["model_path"], gpu_ids=(0,), ledger=CostLedger(gpu_ids=(0,)),
                    benchmark=args.benchmark, trainable=False, free_memory_gib={0: 79},
                    max_context_tokens=config.get("max_context_tokens", 32768),
                    representation_batch_size=config.get("representation_batch_size", 8),
                    representation_token_budget=config.get("representation_token_budget", 8192),
                )
                weights = torch.load(args.reference, map_location="cpu", weights_only=True, mmap=True)
                policy.model.load_state_dict(weights)
                del weights
            reference = reference_features(policy, trajectories, include_obs=webshop)
            feature_source = "reconstructed from sealed fixed reference"
        else:
            feature_source = "saved frozen reference features"
        for radius in radii:
            results[f"{radius:.2f}"].append({
                "group_id": group_id, "task_id": trajectories[0].task_id, "feature_source": feature_source,
                "trajectories": len(trajectories), "successes": sum(t.official_success for t in trajectories),
                **score(trajectories, reference, config, radius, prior_weight),
            })
        print(f"Compared {group_id} using {feature_source}", flush=True)

    comparison = {}
    for radius, rows in results.items():
        steps = sum(row["steps"] for row in rows)
        supported = sum(row["supported_steps"] for row in rows)
        comparison[radius] = {
            "brier": math.fsum(row["step_squared_error"] for row in rows) / steps,
            "coverage": supported / steps, "supported_steps": supported, "eligible_steps": steps,
            "groups": rows,
        }
        if webshop:
            tokens = sum(row["response_tokens"] for row in rows)
            supported_tokens = sum(row["supported_response_tokens"] for row in rows)
            comparison[radius].update(
                response_token_brier=math.fsum(row["response_token_squared_error"] for row in rows) / tokens,
                response_token_coverage=supported_tokens / tokens,
                supported_response_tokens=supported_tokens, eligible_response_tokens=tokens,
            )
    report = {
        "benchmark": args.benchmark, "usable": True, "compared_radii": radii,
        "results": comparison,
        "credit_prior_weight": prior_weight, "mixed_groups": len(groups),
        "training_rounds": sorted({(source["seed"], source["round"]) for source in sources}),
        "training_sources": sources, "fixed_reference": str(args.reference),
    }
    if webshop:
        report.update(chosen_radius=min(radii, key=lambda radius: (comparison[f"{radius:.2f}"]["brier"], -radius)),
                      selection_metric="pooled_action_step_brier")
    write_json(args.output, report)


if __name__ == "__main__":
    main()
