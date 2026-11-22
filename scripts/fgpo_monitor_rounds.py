"""Project completed round records for the dashboard's read-only probes."""


def _aggregate_credit_groups(credit_groups):
    """Aggregate mixed-group credit diagnostics by step_count without sending per-group dicts."""
    if not isinstance(credit_groups, dict) or not credit_groups:
        return None
    mapping = (
        ("support_frac", "matched_step_fraction"),
        ("transfer_frac", "transfer_step_fraction"),
        ("mean_transfer_weight", "mean_transfer_weight"),
        ("loo_brier", "loo_outcome_brier"),
        ("prior_brier", "loo_task_prior_brier"),
        ("matched_loo_brier", "matched_loo_outcome_brier"),
    )
    sums = {out_k: 0.0 for out_k, _ in mapping}
    weights = {out_k: 0.0 for out_k, _ in mapping}
    for grp in credit_groups.values():
        if not isinstance(grp, dict) or grp.get("homogeneous_group"):
            continue
        raw_steps = grp.get("step_count")
        if raw_steps is None or raw_steps <= 0:
            continue
        w = float(raw_steps)
        for out_k, src_k in mapping:
            val = grp.get(src_k)
            if val is not None:
                sums[out_k] += float(val) * w
                weights[out_k] += w
    aggregated = {
        out_k: round(sums[out_k] / weights[out_k], 5)
        for out_k, _ in mapping
        if weights[out_k] > 0.0
    }
    return aggregated or None


def project_training_round(row, benchmark):
    fields = ("round", "iter", "trajectories", "successes", "optimizer_steps")
    result = {key: row.get(key) for key in fields}

    raw_metrics = row.get("metrics")
    if isinstance(raw_metrics, dict):
        metric_keys = (
            "loss", "entropy", "reference_kl", "gradient_norm",
            "ppo_clip_fraction", "steps", "mixed_groups",
        )
        result["metrics"] = {k: raw_metrics[k] for k in metric_keys if k in raw_metrics}
    else:
        result["metrics"] = None

    raw_costs = row.get("costs") or {}
    if isinstance(raw_costs, dict):
        costs = {}
        for ck in ("wall_seconds", "allocated_device_seconds"):
            cv = raw_costs.get(ck)
            if cv is not None:
                costs[ck] = round(float(cv), 2)
        if costs:
            result["costs"] = costs

    diagnostics = row.get("diagnostics") or {}
    if (benchmark == "textcraft" and row.get("protocol") == "fgpo-r3-published-v4"
            and not diagnostics.get("admissible_observed_steps")):
        # Legacy TextCraft rounds counted missing native labels as invalid.
        diagnostics = {**diagnostics, "admissible_valid_fraction": None}
    diag_keys = (
        "all_zero_groups", "mixed_groups", "all_one_groups", "total_groups",
        "admissible_observed_steps", "admissible_valid_fraction", "syntax_valid_fraction",
    )
    result["diagnostics"] = {key: diagnostics.get(key) for key in diag_keys}

    credit = _aggregate_credit_groups(diagnostics.get("credit_groups"))
    if credit:
        result["credit"] = credit

    if not (row.get("method") or "").startswith("fgpo_tree_"):
        return result

    tree_fields = (
        "root_count", "rooted_success", "total_groups", "split_count", "discordant_split_count",
        "unique_response_edges", "unique_response_tokens", "native_sampled_steps", "native_replay_steps",
        "syntax_valid_fraction", "admissible_valid_fraction", "optimizer_steps_in_round",
    )
    optimization_fields = (
        "ppo_loss_per_weighted_token", "reference_capped_k3_per_weighted_token", "entropy_per_weighted_token",
        "ppo_clip_fraction_by_actual_response_token", "preclip_gradient_norms", "gradient_clip_scales",
        "token_weight_sum",
    )
    optimization = row.get("optimization") or {}
    result["optimization"] = {key: optimization[key] for key in optimization_fields if key in optimization}
    tree = {key: diagnostics[key] for key in tree_fields if key in diagnostics}
    batches = row.get("optimizer_batches") or []
    tree["batch_denominators"] = [batch["denominator"] for batch in batches if "denominator" in batch]
    splits = [task["split_credit"] for batch in batches
              for task in (batch.get("diagnostics") or {}).get("per_task", ()) if task.get("split_credit")]
    if splits:
        shifts = [split["mean_abs_branch_advantage_change_vs_other_root"] for split in splits]
        example = splits[0]
        tree["mean_abs_branch_credit_shift"] = sum(shifts) / len(shifts)
        tree["shared_prefix_response_tokens"] = sum(split["shared_prefix_response_tokens"] for split in splits)
        tree["first_split"] = {
            "leaf_success": example["leaf_success"],
            "advantages": [edge["advantage"] for edge in example["branch_edges"]],
            "masses": [edge["mass"] for edge in example["branch_edges"]],
            "shared_prefix_advantage": example["shared_prefix_advantage"],
        }
    result["tree"] = tree
    return result
