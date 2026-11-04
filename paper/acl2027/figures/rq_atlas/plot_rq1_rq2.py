"""RQ1/RQ2 views from the complete saved experiment evidence."""

import math

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, FOCUS, METHODS, METHOD_COLORS,
    METHOD_LABELS, MODEL_IDS, MODEL_LABELS, heatmap, main_index, save_figure,
)


def _setting_columns():
    return [(model, benchmark) for benchmark in BENCHMARKS for model in MODEL_IDS]


def _matrix_figure(metrics, out_dir, field, scale, stem, title, bar_label, vmax, fmt):
    index = main_index(metrics)
    settings = _setting_columns()
    values = [[scale * index[model, benchmark, method]["final_test"][field]["mean"]
               for model, benchmark in settings] for method in METHODS]
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    fig.subplots_adjust(left=.18, right=.975, bottom=.12, top=.79)
    heatmap(ax, values, [METHOD_LABELS[m] for m in METHODS],
                 [MODEL_LABELS[m] for m, _ in settings], vmin=0, vmax=vmax,
                 fmt=fmt, textsize=8.1, cmap=ListedColormap(["#FFFFFF"]))
    spar_row = METHODS.index("prism_evolve")
    ax.axhspan(spar_row-.5, spar_row+.5, color="#E0F4F8", zorder=1)
    ax.axvline(2.5, color="white", lw=6, zorder=2)
    ax.axhline(-.5, color=COLORS["ink"], lw=.75, clip_on=False)
    ax.axhline(len(METHODS)-.5, color=COLORS["ink"], lw=.75, clip_on=False)
    ax.axhline(6.5, color="#CBD5E1", lw=.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=7)
    ax.tick_params(axis="x", which="both", length=0)
    ax.text(-.015, 1.035, "Method", transform=ax.transAxes, ha="right", va="center",
            fontsize=8.4, fontweight="bold")
    for x, benchmark in ((.25, "appworld"), (.75, "planbench_xl")):
        ax.text(x, 1.13, BENCH_LABELS[benchmark], transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    return save_figure(
        fig, out_dir, stem, title,
        f"{bar_label}. Test split; mean across seeds 42, 43, 44. AppWorld: 168 tasks; PlanBench-XL: 90 queries.",
        ["metrics/main_results"],
    )


def _panel_layout(height):
    fig, axes = plt.subplots(2, 3, figsize=(8.8, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.975, bottom=.13, top=.835,
                        wspace=.16, hspace=.32)
    for row, benchmark in enumerate(BENCHMARKS):
        fig.text(.027, .665 if row == 0 else .305, BENCH_LABELS[benchmark],
                 rotation=90, ha="center", va="center", fontsize=8.4, fontweight="bold")
        for col, model in enumerate(MODEL_IDS):
            axes[row, col].set_title(MODEL_LABELS[model])
    return fig, axes


def _seed_variation(metrics, out_dir):
    index = main_index(metrics)
    seeds = metrics["config"]["seeds"]
    seed_index = {(r["model"], r["benchmark"], r["method"], r["seed"]): r
                  for r in metrics["seed_results"]}
    fig, axes = _panel_layout(8.1)
    markers = ("o", "^", "s")
    offsets = np.linspace(-.17, .17, len(seeds))
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for y, method in enumerate(METHODS):
                summary = index[model, benchmark, method]["final_test"]["mean_utility"]
                color = COLORS["spar"] if method == "prism_evolve" else "#64748B"
                ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="|",
                            color=color, markersize=8, capsize=2, capthick=.6,
                            elinewidth=1.15 if method == "prism_evolve" else .75, zorder=2)
                for seed, marker, offset in zip(seeds, markers, offsets):
                    value = seed_index[model, benchmark, method, seed]["final_test"]["mean_utility"]
                    ax.scatter(value, y + offset, marker=marker, s=17,
                               color=color, edgecolor="white", linewidth=.35, zorder=3)
            ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
            ax.set_xlim(0, .85)
            ax.set_xticks([0, .2, .4, .6, .8])
            ax.grid(axis="x")
            ax.tick_params(axis="y", labelsize=7.4, length=0)
            if row == 1:
                ax.set_xlabel(r"Test utility $\mathbb{E}[U_B]$")
    axes[0, 0].set_ylim(len(METHODS) - .5, -.5)
    handles = [Line2D([], [], marker=marker, ls="none", color=COLORS["muted"],
                      markersize=4, label=f"Seed {seed}")
               for seed, marker in zip(seeds, markers)]
    handles.append(Line2D([], [], marker="|", color=COLORS["ink"], lw=.8,
                          markersize=7, label="Mean ± sample SD"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .918), ncol=4)
    return save_figure(
        fig, out_dir, "rq1_03_seed_variation", "RQ1 | Individual seeds and test-utility variation",
        "All 252 seed-level results. Points show individual seeds; horizontal intervals show the stored three-seed sample SD. SPAR is highlighted.",
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
    maximum = math.ceil(max(max(v) for v in gains.values()) / 5) * 5
    fig, axes = _panel_layout(6.8)
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            values = gains[model, benchmark]
            y = np.arange(len(controls))
            ax.barh(y, values, height=.42, color=COLORS["spar"], zorder=3)
            for yp, value in zip(y, values):
                ax.text(value + .55, yp, f"{value:+.1f}", va="center", fontsize=7.5)
            ax.set_yticks(y, [METHOD_LABELS[m] for m in controls])
            ax.tick_params(axis="y", labelsize=7.4, length=0)
            ax.set_xlim(0, maximum + 5)
            ax.grid(axis="x")
            if row == 1:
                ax.set_xlabel("SPAR gain (pts)")
    axes[0, 0].set_ylim(len(controls) - .5, -.5)
    return save_figure(
        fig, out_dir, "rq2_01_mechanism_controls", "RQ2 | Utility gains over mechanism controls",
        "Each bar is 100 × (SPAR mean test utility − reference mean test utility). All contrasts use the same saved model and benchmark setting.",
        ["metrics/main_results"],
    )


def _aligned_weighting(comparisons, out_dir):
    lookup = {(r["model"], r["benchmark"]): r for r in comparisons["method_comparisons"]
              if r["reference_role"] == "unweighted"}
    settings = [(model, benchmark) for model in MODEL_IDS for benchmark in BENCHMARKS]
    names = [f"{MODEL_LABELS[m]} / {BENCH_LABELS[b]}" for m, b in settings]
    specs = [
        ("(a) Test utility", "Utility gain (pts)", "utility", "difference_pts"),
        ("(b) Mean interaction cost", "Cost reduction (%)", "cost", "reduction_percent"),
        ("(c) Masked-feedback gain", r"$\Delta_{\mathrm{use}}$ gain (pts)", "dev_delta_use", "difference_pts"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.6), sharey=True)
    fig.subplots_adjust(left=.235, right=.975, bottom=.19, top=.79, wspace=.20)
    y = np.arange(len(settings))
    for ax, (title, xlabel, key, field) in zip(axes, specs):
        values = [lookup[setting][key][field] for setting in settings]
        ax.barh(y, values, height=.47, color=COLORS["spar"], zorder=3)
        for yp, value in zip(y, values):
            text = f"{value:.2f}%" if key == "cost" else f"{value:+.2f}"
            ax.text(value + max(values) * .035, yp, text, va="center", fontsize=7.7)
        ax.set_xlim(0, max(values) * 1.30)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y, names)
        ax.tick_params(axis="y", length=0, labelsize=7.5)
        ax.grid(axis="x")
    axes[0].set_ylim(len(settings)-.5, -.5)
    return save_figure(
        fig, out_dir, "rq2_02_aligned_weighting", "RQ2 | Aligned weighting versus unweighted branching",
        "Contrasts of saved means. Cost reduction uses Unweighted as the denominator; feedback gain uses the fixed 24-task development diagnostic.",
        ["comparisons/method_comparisons/reference_role=unweighted"],
    )


def _cost_utility(metrics, out_dir):
    index = main_index(metrics)
    points = [(r["final_test"]["mean_cumulative_cost"]["mean"],
               r["final_test"]["mean_utility"]["mean"]) for r in metrics["main_results"]]
    xmin = max(0, math.floor((min(x for x, _ in points)-.06) * 20)/20)
    xmax = math.ceil((max(x for x, _ in points)+.04) * 20)/20
    ymax = math.ceil((max(y for _, y in points)+.08) * 10)/10
    fig, axes = _panel_layout(6.7)
    markers = {"prism_evolve": "o", "ablation_unweighted": "^", "step_dpo": "s"}
    offsets = {"prism_evolve": (4, 6, "left"), "ablation_unweighted": (4, -10, "left"),
               "step_dpo": (-10, 4, "right")}
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for method in [m for m in METHODS if m not in FOCUS] + FOCUS:
                data = index[model, benchmark, method]["final_test"]
                x, y = data["mean_cumulative_cost"]["mean"], data["mean_utility"]["mean"]
                highlighted = method in FOCUS
                ax.scatter(x, y, s=34 if highlighted else 19,
                           color=METHOD_COLORS.get(method, COLORS["context"]),
                           marker=markers.get(method, "o"), edgecolor="white", linewidth=.45,
                           alpha=1 if highlighted else .8, zorder=4 if highlighted else 2)
                if highlighted:
                    dx, dy, align = offsets[method]
                    ax.annotate(METHOD_LABELS[method], (x, y), xytext=(dx, dy),
                                textcoords="offset points", ha=align, va="center", fontsize=7,
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .15})
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, ymax)
            ax.grid(True)
            if row == 1:
                ax.set_xlabel("Mean macro-action cost")
            if col == 0:
                ax.set_ylabel(r"Test utility $\mathbb{E}[U_B]$")
    handles = [Line2D([], [], marker=markers[m], ls="none", color=METHOD_COLORS[m],
                      markersize=5, label=METHOD_LABELS[m]) for m in FOCUS]
    handles.append(Line2D([], [], marker="o", ls="none", color=COLORS["context"],
                          markersize=4, label="Other methods"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .918), ncol=4)
    return save_figure(
        fig, out_dir, "rq2_03_cost_utility", "RQ2 | Test utility and interaction cost",
        "Each point is one method's three-seed mean. The six panels share axes; higher utility and lower macro-action cost are preferred.",
        ["metrics/main_results"],
    )


def build(metrics, comparisons, out_dir):
    return [
        _matrix_figure(metrics, out_dir, "mean_utility", 1, "rq1_01_test_utility",
                       "RQ1 | Test utility across all methods", r"Mean $\mathbb{E}[U_B]$", .75, ".4f"),
        _matrix_figure(metrics, out_dir, "success_rate", 100, "rq1_02_test_success",
                       "RQ1 | Test success across all methods", "Mean success rate (%)", 100, ".1f"),
        _seed_variation(metrics, out_dir),
        _mechanism_controls(metrics, out_dir),
        _aligned_weighting(comparisons, out_dir),
        _cost_utility(metrics, out_dir),
    ]
