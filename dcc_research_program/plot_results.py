"""Render the finite-model study, preserving both favorable and unfavorable cases."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent
data = json.loads((base / "results.json").read_text())
methods = {
    "complete_program_evsi": ("Complete-program EVSI", "#075985"),
    "model_information": ("Model information gain", "#b45309"),
    "uniform_programs": ("Uniform programs", "#64748b"),
    "predictive_entropy": ("Predictive entropy", "#a21caf"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), sharey=True)
for ax, run in zip(axes, data["learned_finite_model"]):
    for method, (label, color) in methods.items():
        rows = [r for r in run["summary"] if r["method"] == method]
        x = np.array([r["physical_actions"] for r in rows])
        y = np.array([r["mean_deployment_response_value"] for r in rows])
        se = np.array([r["seed_standard_error"] for r in rows])
        style = "--" if method == "model_information" else "-"
        ax.plot(x, y, style, label=label, color=color, linewidth=2, marker="o", markersize=3)
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.10, linewidth=0)
    programs = run["fields"] * (run["fields"] - 1) // 2
    ax.set_title(f"{run['fields']} public fields / {programs} candidate programs", fontsize=11)
    ax.set_xlabel("Environment actions (4 per measurement block)", fontsize=9)
    ax.axhline(0.8, color="#64748b", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
    ax.set_xlim(0, 132)
    ax.set_ylim(0.49, 0.82)
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
axes[0].set_ylabel("Expected local response value")
fig.suptitle("Learning a hidden two-bit control relation from short public experiments", y=0.98, fontsize=15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
fig.text(0.5, 0.03,
         "512 seeds; bands: 95% normal intervals. Known finite model class and program grammar; all full-task training returns are zero.\n"
         "Local response ceiling = 0.8. These are controlled mechanism diagnostics, not native benchmark results.",
         ha="center", va="bottom", fontsize=9, color="#475569")
fig.subplots_adjust(left=0.065, right=0.99, top=0.76, bottom=0.22, wspace=0.13)
fig.savefig(base / "finite_model_results.png", dpi=200)
fig.savefig(base / "finite_model_results.svg")

comparisons = []
for run in data["learned_finite_model"]:
    raw = run["raw_seed_values"]
    target = np.array(raw["complete_program_evsi"]["32"])
    for other in ("uniform_programs", "model_information", "predictive_entropy"):
        diff = target - np.array(raw[other]["32"])
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        comparisons.append({"fields": run["fields"], "comparison": "complete_program_evsi minus " + other,
                            "mean_difference": float(diff.mean()), "paired_standard_error": se,
                            "normal_95_interval": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)]})
(base / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
print(json.dumps(comparisons, indent=2))

growth_path = base / "growth_results.json"
if growth_path.exists():
    growth_data = json.loads(growth_path.read_text())
    growth_methods = {
        "action_diff_growth": ("Action-diff witness growth + EVSI", "#075985", "-"),
        "label_shuffled_growth": ("Label-shuffled proposer + EVSI", "#dc2626", "--"),
        "prediction_error_growth": ("Prediction-error growth + EVSI", "#b45309", "-."),
        "uniform_prior_growth": ("Uniform prior growth + EVSI", "#64748b", ":"),
        "lsharp_paid_intervention": ("L# separating (paid intervention)", "#7c3aed", "-"),
        "lsharp_free_oracle": ("L# separating (free oracle ref)", "#0f766e", ":"),
    }
    runs = growth_data["program_growth_runs"]
    fig_g, axes_g = plt.subplots(2, len(runs), figsize=(14.2, 7.6), sharey="row")
    if len(runs) == 1:
        axes_g = np.array([[axes_g[0]], [axes_g[1]]])
    for col, run in enumerate(runs):
        ax_val = axes_g[0, col]
        ax_hit = axes_g[1, col]
        for method, (label, color, style) in growth_methods.items():
            rows = [r for r in run["summary"] if r["method"] == method]
            x = np.array([r["physical_actions"] for r in rows])
            y_val = np.array([r["mean_deployment_response_value"] for r in rows])
            se_val = np.array([r["seed_standard_error"] for r in rows])
            y_hit = np.array([r["relevant_program_hit_rate"] for r in rows])
            se_hit = np.array([r["hit_rate_standard_error"] for r in rows])
            ax_val.plot(x, y_val, style, label=label, color=color, linewidth=1.9, marker="o", markersize=3)
            ax_val.fill_between(x, y_val - 1.96 * se_val, y_val + 1.96 * se_val, color=color, alpha=0.08, linewidth=0)
            ax_hit.plot(x, y_hit, style, label=label, color=color, linewidth=1.9, marker="s", markersize=3)
            ax_hit.fill_between(x, np.clip(y_hit - 1.96 * se_hit, 0, 1), np.clip(y_hit + 1.96 * se_hit, 0, 1),
                                color=color, alpha=0.08, linewidth=0)
        programs = run["fields"] * (run["fields"] - 1) // 2
        ax_val.set_title(f"{run['fields']} fields / {programs} total programs (budget={run['proposal_budget_per_block']}/blk)", fontsize=10)
        ax_val.axhline(0.8, color="#64748b", linestyle=":", linewidth=1)
        ax_val.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
        ax_val.set_ylim(0.49, 0.82)
        ax_val.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax_hit.set_xlabel("Environment actions (4 per measurement block)", fontsize=9)
        ax_hit.set_ylim(-0.02, 1.04)
        ax_hit.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    axes_g[0, 0].set_ylabel("Expected local response value")
    axes_g[1, 0].set_ylabel("Relevant program hit rate")
    fig_g.suptitle("Bounded-budget program discovery: action-difference witnesses vs controls", y=0.98, fontsize=14)
    handles, labels = axes_g[0, 0].get_legend_handles_labels()
    fig_g.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=9)
    fig_g.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.09, wspace=0.13, hspace=0.24)
    fig_g.savefig(base / "program_growth_results.png", dpi=200)
    fig_g.savefig(base / "program_growth_results.svg")

    growth_comparisons = []
    for run in runs:
        max_b = str(run["max_blocks"])
        raw = run["raw_seed_values"]
        raw_hits = run["raw_hit_rates"]
        target = np.array(raw["action_diff_growth"][max_b])
        target_hits = np.array(raw_hits["action_diff_growth"][max_b])
        for other in ("label_shuffled_growth", "prediction_error_growth", "uniform_prior_growth", "lsharp_paid_intervention"):
            diff = target - np.array(raw[other][max_b])
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            h_diff = target_hits - np.array(raw_hits[other][max_b])
            h_se = float(h_diff.std(ddof=1) / np.sqrt(len(h_diff)))
            growth_comparisons.append({
                "fields": run["fields"],
                "blocks": run["max_blocks"],
                "comparison": "action_diff_growth minus " + other,
                "response_value_mean_difference": float(diff.mean()),
                "response_value_paired_se": se,
                "response_value_normal_95_interval": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                "hit_rate_mean_difference": float(h_diff.mean()),
                "hit_rate_paired_se": h_se,
                "hit_rate_normal_95_interval": [float(h_diff.mean() - 1.96 * h_se), float(h_diff.mean() + 1.96 * h_se)],
            })
    (base / "growth_paired_comparisons.json").write_text(json.dumps(growth_comparisons, indent=2) + "\n")
    print(json.dumps(growth_comparisons, indent=2))
