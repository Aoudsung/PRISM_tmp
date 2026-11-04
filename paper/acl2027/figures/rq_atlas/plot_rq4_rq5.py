"""RQ4/RQ5 figures, rendered from the collected experiment evidence."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, FOCUS, METHODS, METHOD_COLORS,
    METHOD_LABELS, MODEL_IDS, MODEL_LABELS, colorbar, heatmap, main_index,
    save_figure,
)


SHORT_MODELS = dict(zip(MODEL_IDS, ["Qwen7", "Llama8", "Qwen27"]))
LINESTYLES = {"prism_evolve": "-", "ablation_unweighted": "--", "step_dpo": "-."}
MARKERS = {"prism_evolve": "o", "ablation_unweighted": "^", "step_dpo": "s"}


def _legend(fig, y=.90):
    handles = [Line2D([], [], color=METHOD_COLORS[m], ls=LINESTYLES[m],
                      marker=MARKERS[m], lw=1.7, ms=4, label=METHOD_LABELS[m])
               for m in FOCUS]
    handles.append(Line2D([], [], color=COLORS["context"], lw=1, label="Other methods"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, y), ncol=4)


def _group_headers(ax, labels, centers):
    for label, center in zip(labels, centers):
        ax.text(center, 1.075, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.4, fontweight="bold")


def stage_evolution(metrics, comparisons, out_dir):
    stages = {(r["model"], r["benchmark"], r["method"], r["stage_k"]): r["values"]
              for r in metrics["stage_results"]}
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.975, bottom=.13, top=.81, wspace=.18, hspace=.38)
    x = np.arange(4)
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for method in [m for m in METHODS if m not in FOCUS] + list(reversed(FOCUS)):
                values = [stages[model, benchmark, method, int(k)]["mean_utility"] for k in x]
                means = np.array([v["mean"] for v in values])
                if method in FOCUS:
                    sd = np.array([v["std"] for v in values])
                    ax.fill_between(x, means-sd, means+sd, color=METHOD_COLORS[method], alpha=.075, lw=0)
                    ax.plot(x, means, color=METHOD_COLORS[method], lw=1.7,
                            ls=LINESTYLES[method], marker=MARKERS[method], ms=3.8, zorder=4)
                else:
                    ax.plot(x, means, color=COLORS["context"], lw=.75, alpha=.6)
            ax.set_title(f"{BENCH_LABELS[benchmark]} | {MODEL_LABELS[model]}", fontsize=8.6)
            ax.set_xticks(x)
            ax.set_xlim(-.12, 3.12)
            ax.set_ylim(0, .88)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel("Development utility")
            if row == 1:
                ax.set_xlabel("Training stage")
    _legend(fig)
    return save_figure(fig, out_dir, "rq4_01_stage_evolution", "RQ4 | Utility across training stages",
                       "All 14 methods; curves are three-seed means. Bands show sample SD for SPAR, Unweighted and Step-DPO.",
                       ["metrics.stage_results"])


def feedback_utilization(metrics, comparisons, out_dir):
    idx = main_index(metrics)
    settings = [(b, m) for b in BENCHMARKS for m in MODEL_IDS]
    values = [[idx[m, b, method]["final_dev"]["delta_use"]["mean"]
               for b, m in settings] for method in METHODS]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_axes([.19, .43, .71, .405])
    im = heatmap(ax, values, [METHOD_LABELS[m] for m in METHODS],
                 [SHORT_MODELS[m] for b, m in settings], vmin=0, vmax=.75, fmt=".4f", textsize=7.3)
    ax.axvline(2.5, color="white", lw=3)
    _group_headers(ax, [BENCH_LABELS[b] for b in BENCHMARKS], [.25, .75])
    colorbar(fig, im, ax, r"Mean $\Delta_{\mathrm{use}}$", fraction=.025, pad=.025)

    lower = fig.add_axes([.23, .12, .65, .215])
    changes = {(r["model"], r["benchmark"]): r for r in comparisons["stage_changes"]}
    for pos, (benchmark, model) in enumerate(settings):
        contrast = changes[model, benchmark]["delta_use"]
        start, end = contrast["reference"], contrast["ours"]
        lower.plot([start, end], [pos, pos], color="#CBD5E1", lw=1.7, zorder=1)
        lower.scatter([start], [pos], s=25, marker="o", facecolors="white",
                      edgecolors=COLORS["muted"], linewidths=.9, zorder=3)
        lower.scatter([end], [pos], s=29, color=COLORS["spar"], zorder=4)
        lower.annotate(f"{start:.4f}", (start, pos), xytext=(-7, 0), textcoords="offset points",
                       ha="right", va="center", fontsize=7.2)
        lower.annotate(f"{end:.4f}", (end, pos), xytext=(7, 0), textcoords="offset points",
                       ha="left", va="center", fontsize=7.2, fontweight="bold")
    lower.set_yticks(range(6), [f"{SHORT_MODELS[m]} / {'AppW' if b=='appworld' else 'PB-XL'}"
                                for b, m in settings])
    lower.invert_yaxis()
    lower.set_xlim(0, .81)
    lower.set_xticks([0, .2, .4, .6, .8])
    lower.set_xlabel(r"SPAR feedback utilization $\Delta_{\mathrm{use}}$")
    lower.grid(axis="x")
    lower.set_title("SPAR: Stage 0 to Stage 3", loc="left", fontsize=9, pad=7)
    lower.legend(handles=[Line2D([], [], color=COLORS["muted"], marker="o", mfc="white", ls="", label="Stage 0"),
                          Line2D([], [], color=COLORS["spar"], marker="o", ls="", label="Stage 3")],
                 loc="lower right", bbox_to_anchor=(1.03, 1.02), ncol=2, fontsize=7)
    return save_figure(fig, out_dir, "rq4_02_feedback_utilization", "RQ4 | Masked-feedback utilization",
                       "Fixed 24-task development diagnostic; first information feedback is masked. Top: three-seed means. Bottom: SPAR stage endpoints.",
                       ["metrics.main_results.final_dev.delta_use", "comparisons.stage_changes"])


def trajectory_outcomes(metrics, comparisons, out_dir):
    idx = {(r["model"], r["benchmark"], r["method"]): r for r in comparisons["trajectory_outcomes"]}
    fig, axes = plt.subplots(2, 3, figsize=(9, 8.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.13, top=.815, wspace=.15, hspace=.28)
    y = np.arange(len(METHODS))
    categories = [("success_without_recorded_failure", COLORS["spar"], "Success / no failure"),
                  ("recovered_success", "#0A9396", "Recovered success"),
                  ("final_failure", "#C83E4D", "Final failure")]
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            records = [idx[model, benchmark, method] for method in METHODS]
            left = np.zeros(len(METHODS))
            for key, color, label in categories:
                widths = np.array([r[key]/r["seed_task_evaluations"]*100 for r in records])
                ax.barh(y, widths, left=left, height=.69, color=color,
                        edgecolor="white", lw=.55, zorder=3)
                for j, width in enumerate(widths):
                    if width >= 19:
                        ax.text(left[j]+width/2, j, f"{width:.0f}", ha="center", va="center",
                                fontsize=6.3, color="white")
                left += widths
            count = records[0]["seed_task_evaluations"]
            ax.set_title(f"{BENCH_LABELS[benchmark]} | {MODEL_LABELS[model]}\nN = {count}", fontsize=8.3, pad=7)
            ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS])
            ax.tick_params(axis="y", length=0, labelsize=7.3)
            ax.set_ylim(13.65, -.65)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x")
            if row == 1:
                ax.set_xlabel("Seed-task evaluations (%)")
    fig.legend(handles=[Patch(fc=c, label=l) for _, c, l in categories], loc="upper center",
               bbox_to_anchor=(.56, .90), ncol=3, fontsize=7.6)
    return save_figure(fig, out_dir, "rq4_03_trajectory_outcomes", "RQ4 | Trajectory outcomes across all methods",
                       "Counts pooled over three seeds. Success / no failure means success without a recorded intermediate failure; recovered success follows a failure.",
                       ["comparisons.trajectory_outcomes"])


def conditional_recovery(metrics, comparisons, out_dir):
    idx = {(r["model"], r["benchmark"], r["method"]): r for r in comparisons["trajectory_outcomes"]}
    settings = [(b, m) for b in BENCHMARKS for m in MODEL_IDS]
    values, labels = [], []
    for method in METHODS:
        records = [idx[m, b, method] for b, m in settings]
        values.append([r["pooled_recovery_rate"] for r in records])
        labels.append([f"{r['recovered_success']}/{r['evaluations_with_recorded_failure']}"
                       if r["evaluations_with_recorded_failure"] else "—" for r in records])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(left=.19, right=.91, bottom=.17, top=.82)
    im = heatmap(ax, values, [METHOD_LABELS[m] for m in METHODS],
                 [SHORT_MODELS[m] for b, m in settings], labels=labels,
                 vmin=0, vmax=1, textsize=7.7)
    ax.axvline(2.5, color="white", lw=3)
    _group_headers(ax, [BENCH_LABELS[b] for b in BENCHMARKS], [.25, .75])
    colorbar(fig, im, ax, "Pooled recovery rate", fraction=.025)
    return save_figure(fig, out_dir, "rq4_04_conditional_recovery", "RQ4 | Recovery conditional on an intermediate failure",
                       "Each cell is recovered successes / evaluations with a recorded failure, pooled across seeds. Color encodes the ratio; dash means a zero denominator.",
                       ["comparisons.trajectory_outcomes"])


def budget_scaling(metrics, comparisons, out_dir):
    idx = {(r["model"], r["method"], r["budget"]): r["values"] for r in metrics["budgets"]}
    budgets = sorted({r["budget"] for r in metrics["budgets"]})
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.6), sharex=True, sharey="row")
    fig.subplots_adjust(left=.09, right=.975, bottom=.15, top=.81, wspace=.18, hspace=.35)
    for col, model in enumerate(MODEL_IDS):
        for row, (field, scale) in enumerate([( "mean_utility", 1), ("success_rate", 100)]):
            ax = axes[row, col]
            for method in [m for m in METHODS if m not in FOCUS] + list(reversed(FOCUS)):
                vals = [idx[model, method, b][field]*scale for b in budgets]
                if method in FOCUS:
                    ax.plot(budgets, vals, color=METHOD_COLORS[method], ls=LINESTYLES[method],
                            marker=MARKERS[method], lw=1.75, ms=3.8, zorder=4)
                else:
                    ax.plot(budgets, vals, color=COLORS["context"], lw=.75, alpha=.6)
            if row == 0:
                ax.set_title(MODEL_LABELS[model])
                ax.set_ylim(0, .88)
            else:
                ax.set_ylim(0, 104)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_xlabel(r"Budget multiplier $B/B_0$")
            ax.set_xticks(budgets, [f"{b:.1f}x" for b in budgets])
            ax.set_xlim(.55, 1.45)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel("Development utility" if row==0 else "Success (%)")
    _legend(fig)
    return save_figure(fig, out_dir, "rq5_01_budget_scaling", "RQ5 | Utility and success across available budgets",
                       "AppWorld development set; all 14 methods. Values are three-seed means from the saved budget sweep.",
                       ["metrics.budgets"])


def budget_matrix(metrics, comparisons, out_dir):
    idx = {(r["model"], r["method"], r["budget"]): r["values"] for r in metrics["budgets"]}
    budgets = sorted({r["budget"] for r in metrics["budgets"]})
    settings = [(m, b) for m in MODEL_IDS for b in budgets]
    values = [[idx[m, method, b]["mean_utility"] for m, b in settings] for method in METHODS]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(left=.19, right=.91, bottom=.16, top=.82)
    im = heatmap(ax, values, [METHOD_LABELS[m] for m in METHODS],
                 [f"{b:.1f}x" for m, b in settings], vmin=0, vmax=.85, fmt=".4f", textsize=7.0)
    for split in [2.5, 5.5]:
        ax.axvline(split, color="white", lw=3)
    _group_headers(ax, [MODEL_LABELS[m] for m in MODEL_IDS], [1/6, .5, 5/6])
    ax.set_xlabel(r"Budget multiplier $B/B_0$")
    colorbar(fig, im, ax, "Mean utility", fraction=.025)
    return save_figure(fig, out_dir, "rq5_02_budget_matrix", "RQ5 | Complete budget utility matrix",
                       "AppWorld development set; 3 backbones x 14 methods x 3 budgets. Cells are three-seed mean utility.",
                       ["metrics.budgets"])


def split_schedule(metrics, comparisons, out_dir):
    idx = {(r["model"], r["schedule"]): r["values"] for r in metrics["split_steps"]}
    schedules = ["fixed_T0", "uniform_T0_1", "uniform_T0_2"]
    labels = ["Root\n$T=0$", "Uniform\n$T\in\{0,1\}$", "Uniform\n$T\in\{0,1,2\}$"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.8), sharey="row")
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.84, wspace=.2, hspace=.52)
    for col, model in enumerate(MODEL_IDS):
        for row, (split, field) in enumerate([("final_test", "mean_utility"), ("final_dev", "delta_use")]):
            ax = axes[row, col]
            means = [idx[model, s][split][field]["mean"] for s in schedules]
            stds = [idx[model, s][split][field]["std"] for s in schedules]
            ax.bar(range(3), means, width=.56,
                   color=[COLORS["spar"], "#94A3B8", "#CBD5E1"], edgecolor="none", zorder=3)
            ax.errorbar(range(3), means, yerr=stds, fmt="none", ecolor="#5D6D75",
                        elinewidth=.8, capsize=2, capthick=.8, zorder=4)
            for x, value, sd in zip(range(3), means, stds):
                ax.text(x, value+sd+.023, f"{value:.4f}", ha="center", va="bottom", fontsize=7.3,
                        fontweight="bold" if x==0 else "normal")
            ax.set_xticks(range(3), labels, fontsize=7)
            ax.set_ylim(0, .88)
            ax.grid(axis="y")
            if row == 0:
                ax.set_title(MODEL_LABELS[model])
            if col == 0:
                ax.set_ylabel("Test utility" if row==0 else r"Development $\Delta_{\mathrm{use}}$")
    return save_figure(fig, out_dir, "rq5_03_split_schedule", "RQ5 | Where to split the training trajectory",
                       "AppWorld; bar heights are three-seed means with sample SD. Uniform schedules sample the split location T from the displayed set.",
                       ["metrics.split_steps"])


def build(metrics, comparisons, out_dir):
    return [fn(metrics, comparisons, out_dir) for fn in [
        stage_evolution, feedback_utilization, trajectory_outcomes,
        conditional_recovery, budget_scaling, budget_matrix, split_schedule,
    ]]
