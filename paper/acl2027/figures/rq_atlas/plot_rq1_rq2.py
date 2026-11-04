"""RQ1/RQ2 views following the approved focused-comparison design."""

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, METHODS, METHOD_COLORS, METHOD_LABELS,
    MODEL_IDS, MODEL_LABELS, main_index, save_figure,
)

MODEL_SUFFIXES = ["qwen25_7b", "llama31_8b", "qwen38_27b"]


def _result_table(metrics, out_dir, field, scale, stem, title):
    index = main_index(metrics)
    settings = [(model, benchmark) for model in MODEL_IDS for benchmark in BENCHMARKS]
    success = field == "success_rate"
    precision = 2 if success else 4
    fig = plt.figure(figsize=(7.6, 3.15))
    ax = fig.add_axes([.03, .025, .955, .865])
    ax.set_xlim(0, 7.5)
    ax.set_ylim(len(METHODS)-.25, -2.65)
    ax.axis("off")
    ax.axhspan(len(METHODS)-1.5, len(METHODS)-.5, color="#EAF5F7", zorder=0)
    ax.hlines([-.55, len(METHODS)-.5], 0, 7.5, color=COLORS["ink"], lw=.7)
    ax.hlines(6.5, 0, 7.5, color="#CBD5E1", lw=.45)
    ax.hlines(len(METHODS)-1.5, 0, 7.5, color="#94A3B8", lw=.5)
    ax.text(.02, -1.1, "Method", ha="left", va="center", fontsize=7.4, fontweight="bold")
    for model_index, model in enumerate(MODEL_IDS):
        centre = 1.5 + 2*model_index + 1
        ax.text(centre, -2.0, MODEL_LABELS[model], ha="center", va="center",
                fontsize=8.4, fontweight="bold")
        ax.hlines(-1.55, 1.55+2*model_index, 3.45+2*model_index,
                  color="#94A3B8", lw=.45)
    for column, (_, benchmark) in enumerate(settings):
        ax.text(2+column, -1.1, BENCH_LABELS[benchmark], ha="center", va="center", fontsize=6.8)
    best = {setting: max(index[setting[0], setting[1], method]["final_test"][field]["mean"]
                         for method in METHODS) for setting in settings}
    for row, method in enumerate(METHODS):
        ax.text(.02, row, METHOD_LABELS[method], ha="left", va="center", fontsize=7,
                fontweight="bold" if method == "prism_evolve" else "normal")
        for column, (model, benchmark) in enumerate(settings):
            value = index[model, benchmark, method]["final_test"][field]
            is_best = math.isclose(value["mean"], best[model, benchmark], abs_tol=1e-12)
            anchor = 1.98+column
            ax.text(anchor, row, f"{scale*value['mean']:.{precision}f}", ha="right", va="center",
                    fontsize=7.1, fontweight="bold" if is_best else "normal")
            ax.text(anchor+.025, row, f"±{scale*value['std']:.{precision}f}",
                    ha="left", va="center", fontsize=5.8, color="#64748B")
    fig.text(.03, .985, title, ha="left", va="top", fontsize=8.4, fontweight="bold")
    fig.text(.985, .985, "Mean ± sample SD (%)" if success else "Mean ± sample SD",
             ha="right", va="top", fontsize=7.1, color="#64748B")
    return save_figure(fig, out_dir, stem, f"RQ1 | {title}",
                       "Complete 14-method table with three-seed means and sample SD. Model blocks each contain AppWorld and PlanBench-XL; best means are bold.",
                       ["metrics/main_results"])


def _model_panels(height=3.1, left=.15):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=left, right=.985, bottom=.17, top=.85, wspace=.23, hspace=.20)
    for col, model in enumerate(MODEL_IDS):
        axes[0, col].set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}", pad=6.5)
    for row, benchmark in enumerate(BENCHMARKS):
        fig.text(.012, .68 if row == 0 else .32, BENCH_LABELS[benchmark], rotation=90,
                 ha="left", va="center", fontsize=7.3, fontweight="bold")
    return fig, axes


def _utility_gain_overview(comparisons, out_dir):
    roles = ["frozen", "unweighted", "strongest_named_peer"]
    lookup = {(r["model"], r["benchmark"], r["reference_role"]): r
              for r in comparisons["method_comparisons"]}
    fig, axes = _model_panels(2.85, left=.15)
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for y, role in enumerate(roles):
                item = lookup[model, benchmark, role]
                gain = item["utility"]["difference_pts"]
                ax.hlines(y, 0, gain, color="#CBD5E1", lw=.8, zorder=1)
                ax.plot(gain, y, "o", color=COLORS["spar"], ms=3.6, zorder=3)
                label = f"+{gain:.2f}"
                if role == "strongest_named_peer":
                    label += f" ({METHOD_LABELS[item['reference_method']]})"
                ax.annotate(label, (gain, y), xytext=(4, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6.4)
            ax.set_yticks(range(3), ["Frozen Base", "Unweighted", "Best peer"])
            ax.tick_params(axis="y", labelsize=6.6, length=0, pad=3)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_xlim(0, 55)
            ax.set_xticks([0, 20, 40])
            ax.grid(axis="x")
    axes[0, 0].set_ylim(2.5, -.5)
    fig.text(.985, .99, "SPAR − reference", ha="right", va="top", fontsize=7.1)
    fig.supxlabel("Mean test utility gain (pts)", y=.035, fontsize=7.8)
    return save_figure(fig, out_dir, "rq1_03_utility_gain_overview", "RQ1 | Focused utility gains",
                       "Mean-based utility differences multiplied by 100. The strongest named peer is selected within each saved model/benchmark setting; its name is displayed.",
                       ["comparisons/method_comparisons"])


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
        ax.axhspan(len(METHODS)-1.5, len(METHODS)-.5, color="#EAF5F7", zorder=0)
        for y, method in enumerate(METHODS):
            summary = index[model, benchmark, method]["final_test"]["mean_utility"]
            color = METHOD_COLORS[method]
            ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="|",
                        color=color, markersize=5, capsize=1.3, capthick=.55, elinewidth=.75,
                        linestyle="--" if method == "ablation_unweighted" else "-", zorder=2)
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
        ax.set_title(f"({'a' if panel == 0 else 'b'}) {BENCH_LABELS[benchmark]}")
    axes[0].set_ylim(len(METHODS)-.5, -.5)
    for label in axes[0].get_yticklabels():
        if label.get_text() == "SPAR":
            label.set_fontweight("bold")
    handles = [Line2D([], [], marker=marker, ls="none", color="#64748B", markersize=3.5,
                      label=f"Seed {seed}") for seed, marker in zip(seeds, markers)]
    handles.append(Line2D([], [], marker="|", color="#64748B", lw=.7, markersize=5, label="Mean ± SD"))
    fig.text(.17, .99, MODEL_LABELS[model], ha="left", va="top", fontsize=8.4, fontweight="bold")
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.994, 1.005), ncol=4,
               fontsize=7, columnspacing=1.1, handletextpad=.35)
    return save_figure(fig, out_dir, f"rq1_04_seed_variation_{suffix}",
                       f"RQ1 | Individual seeds: {MODEL_LABELS[model]}",
                       "All 14 methods and three seeds; each horizontal interval is the stored sample SD around the mean. Raw seed points use fixed marker shapes.",
                       ["metrics/seed_results", "metrics/main_results"])


def _mechanism_controls(metrics, out_dir):
    index = main_index(metrics)
    controls = ["outcome_pg", "outcome_value_baseline", "outcome_rloo", "info_rl_neighbor",
                "unstratified_2branch", "ablation_unweighted", "step_dpo"]
    positions = [0, 1, 2, 3, 4, 5.65, 6.65]
    gains = {(model, benchmark): [100 * (
        index[model, benchmark, "prism_evolve"]["final_test"]["mean_utility"]["mean"] -
        index[model, benchmark, method]["final_test"]["mean_utility"]["mean"])
        for method in controls] for model in MODEL_IDS for benchmark in BENCHMARKS}
    fig, axes = _model_panels(3.15, left=.17)
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            ax.axhline(4.85, color="#CBD5E1", lw=.5)
            for y, value in zip(positions, gains[model, benchmark]):
                ax.hlines(y, 0, value, color="#CBD5E1", lw=.8, zorder=1)
                ax.plot(value, y, "o", color=COLORS["spar"], ms=3.1, zorder=3)
                ax.annotate(f"+{value:.1f}", (value, y), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=6.4)
            ax.set_yticks(positions, [METHOD_LABELS[m] for m in controls])
            ax.tick_params(axis="y", labelsize=6.4, length=0, pad=3)
            ax.tick_params(axis="x", labelsize=6.8)
            ax.set_xlim(0, 40)
            ax.set_xticks([0, 20, 40])
            ax.grid(axis="x")
    axes[0, 0].set_ylim(7.15, -.5)
    fig.text(.985, .99, "SPAR − control", ha="right", va="top", fontsize=7.1)
    fig.supxlabel("Mean test utility gain (pts)", y=.035, fontsize=7.8)
    return save_figure(fig, out_dir, "rq2_01_mechanism_controls", "RQ2 | Mechanism-control gains",
                       "The first five rows compare return/sampling controls; the final two rows compare weighting/preference controls. Values are mean utility differences ×100.",
                       ["metrics/main_results"])


def _aligned_weighting(comparisons, out_dir):
    lookup = {(r["model"], r["benchmark"]): r for r in comparisons["method_comparisons"]
              if r["reference_role"] == "unweighted"}
    settings = [(model, benchmark) for model in MODEL_IDS for benchmark in BENCHMARKS]
    names = [f"{MODEL_LABELS[m]} / {BENCH_LABELS[b]}" for m, b in settings]
    specs = [("(a) Test utility", "Utility gain (pts)", "utility", "difference_pts"),
             ("(b) Interaction cost", "Cost reduction (%)", "cost", "reduction_percent"),
             ("(c) Feedback utilization", r"$\Delta_{\mathrm{use}}$ gain (pts)", "dev_delta_use", "difference_pts")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.65), sharey=True)
    fig.subplots_adjust(left=.25, right=.985, bottom=.21, top=.79, wspace=.20)
    y = np.arange(len(settings))
    for ax, (title, xlabel, key, field) in zip(axes, specs):
        values = [lookup[setting][key][field] for setting in settings]
        ax.hlines(y, 0, values, color="#CBD5E1", lw=.85, zorder=1)
        ax.plot(values, y, "o", color=COLORS["spar"], ms=3.6, zorder=3)
        for yp, value in zip(y, values):
            label = f"{value:.2f}%" if key == "cost" else f"+{value:.2f}"
            ax.text(value+max(values)*.04, yp, label, va="center", fontsize=6.6)
        ax.set_xlim(0, max(values)*1.37)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y, names)
        ax.tick_params(axis="y", length=0, labelsize=6.5, pad=3)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x")
    axes[0].set_ylim(len(settings)-.5, -.5)
    fig.text(.62, .985, "SPAR vs. Unweighted", ha="center", va="top", fontsize=7.4, fontweight="bold")
    return save_figure(fig, out_dir, "rq2_02_aligned_weighting", "RQ2 | Aligned weighting versus Unweighted",
                       "Mean-based contrasts. Relative cost reduction uses Unweighted as denominator; feedback values use the fixed 24-task development diagnostic.",
                       ["comparisons/method_comparisons/reference_role=unweighted"])


def _cost_utility(metrics, out_dir):
    index = main_index(metrics)
    points = [(r["final_test"]["mean_cumulative_cost"]["mean"], r["final_test"]["mean_utility"]["mean"])
              for r in metrics["main_results"]]
    xmin = max(0, math.floor((min(x for x, _ in points)-.05)*20)/20)
    xmax = math.ceil((max(x for x, _ in points)+.035)*20)/20
    ymin = max(0, math.floor((min(y for _, y in points)-.04)*10)/10)
    ymax = math.ceil((max(y for _, y in points)+.075)*10)/10
    fig, axes = _model_panels(3.1, left=.10)
    selected = ["step_dpo", "ablation_unweighted", "prism_evolve"]
    markers = {"prism_evolve": "o", "ablation_unweighted": "^", "step_dpo": "s"}
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for method in [m for m in METHODS if m not in selected]+selected:
                values = index[model, benchmark, method]["final_test"]
                focused = method in selected
                ax.scatter(values["mean_cumulative_cost"]["mean"], values["mean_utility"]["mean"],
                           s=24 if method == "prism_evolve" else (21 if focused else 8),
                           color=METHOD_COLORS[method] if focused else "#CBD5E1",
                           marker=markers.get(method, "o"), edgecolor="white", linewidth=.4,
                           zorder=5 if method == "prism_evolve" else (4 if focused else 2))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([.2, .35, .5])
            ax.set_yticks([.2, .4, .6, .8])
            ax.tick_params(labelsize=7)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel("Test utility")
    handles = [Line2D([], [], marker=markers[m], ls="none", color=METHOD_COLORS[m],
                      markersize=4, label=METHOD_LABELS[m]) for m in reversed(selected)]
    handles.append(Line2D([], [], marker="o", ls="none", color="#CBD5E1", markersize=3, label="Other methods"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.994, 1.005), ncol=4,
               fontsize=7.1, columnspacing=1.1, handletextpad=.35)
    fig.supxlabel("Mean macro-action cost", y=.035, fontsize=7.8)
    return save_figure(fig, out_dir, "rq2_03_cost_utility", "RQ2 | Utility and interaction cost",
                       "Each point is one method's three-seed mean; panels share the cost and utility scales. SPAR, Unweighted and Step-DPO are highlighted.",
                       ["metrics/main_results"])


def build(metrics, comparisons, out_dir):
    figures = [
        _result_table(metrics, out_dir, "mean_utility", 1, "rq1_01_test_utility_full", "Test utility"),
        _result_table(metrics, out_dir, "success_rate", 100, "rq1_02_test_success_full", "Test success (%)"),
        _utility_gain_overview(comparisons, out_dir),
    ]
    figures.extend(_seed_variation(metrics, out_dir, model, suffix)
                   for model, suffix in zip(MODEL_IDS, MODEL_SUFFIXES))
    figures.extend([_mechanism_controls(metrics, out_dir), _aligned_weighting(comparisons, out_dir),
                    _cost_utility(metrics, out_dir)])
    return figures
