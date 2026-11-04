"""Compact RQ1/RQ2 figures in the existing paper's visual style."""

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, FOCUS, METHODS, METHOD_COLORS,
    METHOD_LABELS, MODEL_IDS, MODEL_LABELS, main_index, save_figure,
)

GROUPS = {
    "peer": ["frozen_base", "arew", "gigpo", "infopo", "worldevolver", "pos_envprobe",
             "step_dpo", "ablation_unweighted", "prism_evolve"],
    "control": ["frozen_base", "outcome_pg", "outcome_value_baseline", "outcome_rloo",
                "info_rl_neighbor", "unstratified_2branch", "ablation_unweighted", "prism_evolve"],
}
MODEL_SUFFIXES = ["qwen25_7b", "llama31_8b", "qwen38_27b"]
MODEL_XLABELS = ["Qwen2.5-7B\n(7.6B)", "Llama-3.1-8B\n(8.0B)", "Qwen3.8-27B\n(27.0B)"]


def _grouped_figure(metrics, out_dir, field, scale, group):
    index = main_index(metrics)
    methods = GROUPS[group]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.85), sharey=True,
                             gridspec_kw={"wspace": .12})
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.75)
    x = np.arange(len(MODEL_IDS))
    width = .84 / len(methods)
    offsets = (np.arange(len(methods)) - (len(methods)-1)/2) * width
    success = field == "success_rate"
    for panel, (ax, benchmark) in enumerate(zip(axes, BENCHMARKS)):
        for offset, method in zip(offsets, methods):
            means = [scale * index[model, benchmark, method]["final_test"][field]["mean"]
                     for model in MODEL_IDS]
            stds = [scale * index[model, benchmark, method]["final_test"][field]["std"]
                    for model in MODEL_IDS]
            bars = ax.bar(x+offset, means, width*.90, yerr=stds, capsize=1.4,
                          color=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABELS[method],
                          error_kw={"lw": .55, "ecolor": "#64748B", "capthick": .55}, zorder=3)
            if method == "prism_evolve":
                for model_index, bar in enumerate(bars):
                    top = max(scale * (index[MODEL_IDS[model_index], benchmark, m]["final_test"][field]["mean"] +
                                      index[MODEL_IDS[model_index], benchmark, m]["final_test"][field]["std"])
                              for m in methods[-3:])
                    label = f"{means[model_index]:.1f}%" if success else f"{means[model_index]:.3f}"
                    ax.text(bar.get_x()+bar.get_width()/2, top+(2 if success else .022), label,
                            ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        title = "(a) AppWorld Held-Out Test (168 Tasks)" if panel == 0 else "(b) PlanBench-XL Retail Test (90 Queries)"
        ax.set_title(title)
        ax.set_xticks(x, MODEL_XLABELS, fontsize=7.4, linespacing=1.25)
        ax.set_ylim(0, 118 if success else .90)
        ax.set_yticks([0, 25, 50, 75, 100] if success else [0, .2, .4, .6, .8])
        ax.grid(axis="y")
    axes[0].set_ylabel("Terminal Success (%) ↑" if success else r"Budget-Aware Utility $\mathbb{E}[U_B]\uparrow$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.005),
               ncol=5 if group == "peer" else 4, fontsize=6.7,
               columnspacing=.95, handletextpad=.35, labelspacing=.55)
    metric_name = "test_success" if success else "test_utility"
    number = "02" if success else "01"
    return save_figure(
        fig, out_dir, f"rq1_{number}_{metric_name}_{group}",
        f"RQ1 | {group.title()} comparison: {'test success' if success else 'test utility'}",
        "Bars show three-seed means; error bars show sample SD. Only SPAR means are directly labelled.",
        ["metrics/main_results"],
    )


def _panel_layout(height=3.2, left=.17):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=left, right=.985, bottom=.18, top=.80,
                        wspace=.23, hspace=.48)
    for row, benchmark in enumerate(BENCHMARKS):
        fig.text(.017, .66 if row == 0 else .30, BENCH_LABELS[benchmark],
                 rotation=90, ha="center", va="center", fontsize=7.2, fontweight="bold")
        for col, model in enumerate(MODEL_IDS):
            axes[row, col].set_title(MODEL_LABELS[model], pad=4)
    return fig, axes


def _seed_variation(metrics, out_dir, model, suffix):
    index = main_index(metrics)
    seeds = metrics["config"]["seeds"]
    seed_index = {(r["model"], r["benchmark"], r["method"], r["seed"]): r
                  for r in metrics["seed_results"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharex=True, sharey=True,
                             gridspec_kw={"wspace": .12})
    fig.subplots_adjust(left=.17, right=.985, bottom=.18, top=.81)
    markers = ("o", "^", "s")
    offsets = np.linspace(-.15, .15, len(seeds))
    for panel, (ax, benchmark) in enumerate(zip(axes, BENCHMARKS)):
        ax.axhspan(len(METHODS)-1.5, len(METHODS)-.5, color="#E0F4F8", alpha=.65, zorder=0)
        for y, method in enumerate(METHODS):
            summary = index[model, benchmark, method]["final_test"]["mean_utility"]
            color = METHOD_COLORS[method]
            ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="|",
                        color=color, markersize=5, capsize=1.3, capthick=.55,
                        elinewidth=.7, zorder=2)
            for seed, marker, offset in zip(seeds, markers, offsets):
                value = seed_index[model, benchmark, method, seed]["final_test"]["mean_utility"]
                ax.scatter(value, y+offset, marker=marker, s=10, color=color,
                           edgecolor="white", linewidth=.25, zorder=3)
        ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
        ax.set_xlim(.18, .84)
        ax.set_xticks([.2, .4, .6, .8])
        ax.grid(axis="x")
        ax.tick_params(axis="y", labelsize=6.8, length=0, pad=3)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel(r"Test utility $\mathbb{E}[U_B]$")
        ax.set_title(f"({'a' if panel == 0 else 'b'}) {BENCH_LABELS[benchmark]} | {MODEL_LABELS[model]}")
    axes[0].set_ylim(len(METHODS)-.5, -.5)
    for label in axes[0].get_yticklabels():
        if label.get_text() == "SPAR":
            label.set_fontweight("bold")
    handles = [Line2D([], [], marker=marker, ls="none", color="#64748B", markersize=3.5,
                      label=f"Seed {seed}") for seed, marker in zip(seeds, markers)]
    handles.append(Line2D([], [], marker="|", color="#64748B", lw=.7, markersize=5,
                          label="Mean ± sample SD"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.58, .995), ncol=4,
               fontsize=7, columnspacing=1.2, handletextpad=.35)
    return save_figure(
        fig, out_dir, f"rq1_03_seed_variation_{suffix}", f"RQ1 | Individual seeds: {MODEL_LABELS[model]}",
        "Fourteen methods, two benchmarks, three seeds. Points use fixed seed shapes; method colors match the comparison figures.",
        ["metrics/seed_results", "metrics/main_results"],
    )


def _mechanism_controls(metrics, out_dir):
    index = main_index(metrics)
    controls = ["outcome_pg", "outcome_value_baseline", "outcome_rloo", "info_rl_neighbor",
                "unstratified_2branch", "ablation_unweighted", "step_dpo"]
    gains = {(model, benchmark): [100 * (
        index[model, benchmark, "prism_evolve"]["final_test"]["mean_utility"]["mean"] -
        index[model, benchmark, method]["final_test"]["mean_utility"]["mean"])
        for method in controls] for model in MODEL_IDS for benchmark in BENCHMARKS}
    maximum = math.ceil(max(max(v) for v in gains.values())/5)*5
    fig, axes = _panel_layout()
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            values = gains[model, benchmark]
            y = np.arange(len(controls))
            ax.barh(y, values, height=.38, color=COLORS["spar"], zorder=3)
            for yp, value in zip(y, values):
                ax.text(value+.55, yp, f"{value:+.1f}", va="center", fontsize=6.6)
            ax.set_yticks(y, [METHOD_LABELS[m] for m in controls])
            ax.tick_params(axis="y", labelsize=6.5, length=0, pad=3)
            ax.tick_params(axis="x", labelsize=6.8)
            ax.set_xlim(0, maximum+5)
            ax.set_xticks([0, 20, 40])
            ax.grid(axis="x")
            if row == 1:
                ax.set_xlabel("SPAR gain (pts)")
    axes[0, 0].set_ylim(len(controls)-.5, -.5)
    fig.text(.58, .985, "SPAR minus reference mean utility", ha="center", va="top", fontsize=7.2)
    return save_figure(
        fig, out_dir, "rq2_01_mechanism_controls", "RQ2 | Utility gains over mechanism controls",
        "Bars are 100 × (SPAR mean test utility − reference mean test utility); the reference method is shown on each row.",
        ["metrics/main_results"],
    )


def _aligned_weighting(comparisons, out_dir):
    lookup = {(r["model"], r["benchmark"]): r for r in comparisons["method_comparisons"]
              if r["reference_role"] == "unweighted"}
    settings = [(model, benchmark) for model in MODEL_IDS for benchmark in BENCHMARKS]
    names = [f"{MODEL_LABELS[m]} / {BENCH_LABELS[b]}" for m, b in settings]
    specs = [
        ("(a) Test utility", "Utility gain (pts)", "utility", "difference_pts"),
        ("(b) Interaction cost", "Cost reduction (%)", "cost", "reduction_percent"),
        ("(c) Feedback utilization", r"$\Delta_{\mathrm{use}}$ gain (pts)", "dev_delta_use", "difference_pts"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    fig.subplots_adjust(left=.25, right=.985, bottom=.22, top=.77, wspace=.20)
    y = np.arange(len(settings))
    for ax, (title, xlabel, key, field) in zip(axes, specs):
        values = [lookup[setting][key][field] for setting in settings]
        ax.barh(y, values, height=.43, color=COLORS["spar"], zorder=3)
        for yp, value in zip(y, values):
            text = f"{value:.2f}%" if key == "cost" else f"{value:+.2f}"
            ax.text(value+max(values)*.035, yp, text, va="center", fontsize=6.8)
        ax.set_xlim(0, max(values)*1.34)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y, names)
        ax.tick_params(axis="y", length=0, labelsize=6.5, pad=3)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x")
    axes[0].set_ylim(len(settings)-.5, -.5)
    fig.text(.62, .97, "SPAR vs. Unweighted", ha="center", va="top", fontsize=7.4, fontweight="bold")
    return save_figure(
        fig, out_dir, "rq2_02_aligned_weighting", "RQ2 | Aligned weighting versus unweighted branching",
        "Contrasts of saved means. Cost reduction uses Unweighted as denominator; Δuse uses the fixed 24-task development diagnostic.",
        ["comparisons/method_comparisons/reference_role=unweighted"],
    )


def _cost_utility(metrics, out_dir):
    index = main_index(metrics)
    points = [(r["final_test"]["mean_cumulative_cost"]["mean"],
               r["final_test"]["mean_utility"]["mean"]) for r in metrics["main_results"]]
    xmin = max(0, math.floor((min(x for x, _ in points)-.06)*20)/20)
    xmax = math.ceil((max(x for x, _ in points)+.04)*20)/20
    ymin = max(0, math.floor((min(y for _, y in points)-.04)*10)/10)
    ymax = math.ceil((max(y for _, y in points)+.08)*10)/10
    fig, axes = _panel_layout(left=.11)
    markers = {"prism_evolve": "o", "ablation_unweighted": "^", "step_dpo": "s"}
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for method in [m for m in METHODS if m not in FOCUS]+["step_dpo", "ablation_unweighted", "prism_evolve"]:
                data = index[model, benchmark, method]["final_test"]
                x, y = data["mean_cumulative_cost"]["mean"], data["mean_utility"]["mean"]
                highlighted = method in FOCUS
                ax.scatter(x, y, s=23 if method == "prism_evolve" else (21 if highlighted else 9),
                           color=METHOD_COLORS[method] if highlighted else COLORS["context"],
                           marker=markers.get(method, "o"), edgecolor="white", linewidth=.35,
                           alpha=1 if highlighted else .8, zorder=4 if highlighted else 2)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([.2, .35, .5])
            ax.set_yticks([.2, .4, .6, .8])
            ax.tick_params(labelsize=6.8)
            ax.grid(True)
            if row == 1:
                ax.set_xlabel("Mean macro-action cost")
            if col == 0:
                ax.set_ylabel(r"Test utility $\mathbb{E}[U_B]$")
    handles = [Line2D([], [], marker=markers[m], ls="none", color=METHOD_COLORS[m],
                      markersize=4, label=METHOD_LABELS[m]) for m in FOCUS]
    handles.append(Line2D([], [], marker="o", ls="none", color=COLORS["context"],
                          markersize=3, label="Other methods"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, 1.005), ncol=4,
               fontsize=6.8, columnspacing=1.0, handletextpad=.3)
    return save_figure(
        fig, out_dir, "rq2_03_cost_utility", "RQ2 | Test utility and interaction cost",
        "Each point is one method's three-seed mean. The six panels share axes; colors and marker shapes identify the highlighted methods.",
        ["metrics/main_results"],
    )


def build(metrics, comparisons, out_dir):
    figures = [_grouped_figure(metrics, out_dir, field, scale, group)
               for field, scale in (("mean_utility", 1), ("success_rate", 100))
               for group in ("peer", "control")]
    figures.extend(_seed_variation(metrics, out_dir, model, suffix)
                   for model, suffix in zip(MODEL_IDS, MODEL_SUFFIXES))
    figures.extend([_mechanism_controls(metrics, out_dir), _aligned_weighting(comparisons, out_dir),
                    _cost_utility(metrics, out_dir)])
    return figures
