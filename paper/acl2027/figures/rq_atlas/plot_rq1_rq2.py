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


def _draw_result_panel(ax, index, field, scale, panel_title, right_note):
    settings = [(model, benchmark) for model in MODEL_IDS for benchmark in BENCHMARKS]
    success = field == "success_rate"
    precision = 2 if success else 4
    ax.set_xlim(0, 7.5)
    ax.set_ylim(len(METHODS)-.25, -2.95)
    ax.axis("off")
    ax.text(.02, -2.65, panel_title, ha="left", va="center", fontsize=8.1, fontweight="bold", color=COLORS["ink"])
    ax.text(7.48, -2.65, right_note, ha="right", va="center", fontsize=6.9, color="#64748B")
    ax.axhspan(len(METHODS)-1.5, len(METHODS)-.5, color="#E6F3F6", zorder=0)
    ax.hlines([-.55, len(METHODS)-.5], 0, 7.5, color=COLORS["ink"], lw=.75)
    ax.hlines(.5, 0, 7.5, color="#CBD5E1", lw=.45, ls="--")
    ax.hlines(6.5, 0, 7.5, color="#CBD5E1", lw=.45)
    ax.hlines(len(METHODS)-1.5, 0, 7.5, color="#94A3B8", lw=.55)
    ax.text(.02, -1.1, "Method", ha="left", va="center", fontsize=7.2, fontweight="bold")
    for model_index, model in enumerate(MODEL_IDS):
        centre = 1.5 + 2*model_index + 1
        ax.text(centre, -1.95, MODEL_LABELS[model], ha="center", va="center",
                fontsize=8.0, fontweight="bold")
        ax.hlines(-1.52, 1.55+2*model_index, 3.45+2*model_index,
                  color="#94A3B8", lw=.45)
    for column, (_, benchmark) in enumerate(settings):
        ax.text(2+column, -1.1, BENCH_LABELS[benchmark], ha="center", va="center", fontsize=6.7)
    sorted_vals = {
        setting: sorted((index[setting[0], setting[1], m]["final_test"][field]["mean"] for m in METHODS), reverse=True)
        for setting in settings
    }
    best = {setting: vals[0] for setting, vals in sorted_vals.items()}
    second = {setting: vals[1] for setting, vals in sorted_vals.items()}
    for row, method in enumerate(METHODS):
        is_spar = method == "prism_evolve"
        ax.text(.02, row, METHOD_LABELS[method], ha="left", va="center", fontsize=6.9,
                fontweight="bold" if is_spar else "normal",
                color=COLORS["spar"] if is_spar else COLORS["ink"])
        for column, (model, benchmark) in enumerate(settings):
            value = index[model, benchmark, method]["final_test"][field]
            is_best = math.isclose(value["mean"], best[model, benchmark], abs_tol=1e-12)
            is_second = (not is_best) and math.isclose(value["mean"], second[model, benchmark], abs_tol=1e-12)
            anchor = 1.98+column
            txt_str = f"{scale*value['mean']:.{precision}f}"
            ax.text(anchor, row, txt_str, ha="right", va="center",
                    fontsize=6.9, fontweight="bold" if (is_best or is_second) else "normal",
                    color=COLORS["spar"] if is_best else ("#334155" if is_second else COLORS["ink"]))
            if is_second:
                ax.hlines(row + .32, anchor - .42, anchor, color="#64748B", lw=.45)
            ax.text(anchor+.025, row, f"±{scale*value['std']:.{precision}f}",
                    ha="left", va="center", fontsize=5.7, color="#64748B")


def _result_table_combined(metrics, out_dir):
    index = main_index(metrics)
    fig = plt.figure(figsize=(7.6, 5.75))
    ax_top = fig.add_axes([.03, .515, .955, .465])
    ax_bot = fig.add_axes([.03, .02, .955, .465])
    _draw_result_panel(ax_top, index, "mean_utility", 1, "(a) Test utility", "Mean ± sample SD")
    _draw_result_panel(ax_bot, index, "success_rate", 100, "(b) Test success (%)", "Mean ± sample SD (%)")
    return save_figure(fig, out_dir, "rq1_01_test_results_full", "RQ1 | Test utility and success (%)",
                       "Complete 14-method table with three-seed means and sample SD for both test utility and success rate. Model blocks each contain AppWorld and PlanBench-XL; best means are bold and second-best underlined.",
                       ["metrics/main_results"])


def _model_panels(height=3.1, left=.15, bottom=.18, top=.84):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=left, right=.985, bottom=bottom, top=top, wspace=.23, hspace=.22)
    for col, model in enumerate(MODEL_IDS):
        axes[0, col].set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}", pad=6.5)
    for row, benchmark in enumerate(BENCHMARKS):
        fig.text(.012, .67 if row == 0 else .32, BENCH_LABELS[benchmark], rotation=90,
                 ha="left", va="center", fontsize=7.3, fontweight="bold")
    return fig, axes


def _utility_gain_overview(comparisons, out_dir):
    roles = ["frozen", "unweighted", "strongest_named_peer"]
    role_styles = {
        "frozen": (COLORS["reference"], "s"),
        "unweighted": (COLORS["unweighted"], "^"),
        "strongest_named_peer": (COLORS["spar"], "o"),
    }
    lookup = {(r["model"], r["benchmark"], r["reference_role"]): r
              for r in comparisons["method_comparisons"]}
    fig, axes = _model_panels(2.95, left=.15, bottom=.19, top=.83)
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for y, role in enumerate(roles):
                item = lookup[model, benchmark, role]
                gain = item["utility"]["difference_pts"]
                r_color, r_marker = role_styles[role]
                ax.barh(y, gain, height=.36, color=r_color, alpha=.14, edgecolor="none", zorder=1)
                ax.hlines(y, 0, gain, color=r_color, lw=1.4, zorder=2)
                ax.plot(gain, y, r_marker, color=r_color, ms=4.0,
                        markeredgecolor="white", markeredgewidth=.35, zorder=3)
                label = f"+{gain:.2f}"
                if role == "strongest_named_peer":
                    label += f" ({METHOD_LABELS[item['reference_method']]})"
                ax.annotate(label, (gain, y), xytext=(4.5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6.3,
                            fontweight="bold" if role == "strongest_named_peer" else "normal",
                            color=COLORS["spar"] if role == "strongest_named_peer" else COLORS["ink"])
            ax.set_yticks(range(3), ["Frozen Base", "Unweighted", "Best peer"])
            ax.tick_params(axis="y", labelsize=6.6, length=0, pad=3)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_xlim(0, 62)
            ax.set_xticks([0, 20, 40, 60])
            ax.grid(axis="x")
    axes[0, 0].set_ylim(2.55, -.55)
    handles = [
        Line2D([], [], marker="s", color=COLORS["reference"], lw=1.3, ms=3.8, label="vs. Frozen Base"),
        Line2D([], [], marker="^", color=COLORS["unweighted"], lw=1.3, ms=3.8, label="vs. Unweighted"),
        Line2D([], [], marker="o", color=COLORS["spar"], lw=1.3, ms=3.8, label="vs. Best peer"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.992, 1.005), ncol=3,
               fontsize=6.9, columnspacing=1.1, handletextpad=.35)
    fig.supxlabel("Mean test utility gain: SPAR − reference (pts)", y=.03, fontsize=7.8)
    return save_figure(fig, out_dir, "rq1_03_utility_gain_overview", "RQ1 | Focused utility gains",
                       "Mean-based utility differences multiplied by 100. The strongest named peer is selected within each saved model/benchmark setting; its name is displayed.",
                       ["comparisons/method_comparisons"])


def _seed_variation(metrics, out_dir, model, suffix):
    index = main_index(metrics)
    seeds = metrics["config"]["seeds"]
    seed_index = {(r["model"], r["benchmark"], r["method"], r["seed"]): r
                  for r in metrics["seed_results"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), sharex=True, sharey=True,
                             gridspec_kw={"wspace": .14})
    fig.subplots_adjust(left=.17, right=.985, bottom=.21, top=.81)
    markers = ("o", "^", "s")
    offsets = np.linspace(-.15, .15, len(seeds))
    for panel, (ax, benchmark) in enumerate(zip(axes, BENCHMARKS)):
        ax.axhspan(len(METHODS)-1.5, len(METHODS)-.5, color="#E6F3F6", zorder=0)
        for y_div, ls_div in [(.5, ":"), (6.5, "--"), (len(METHODS)-1.5, "-")]:
            ax.axhline(y_div, color="#CBD5E1", lw=.5, ls=ls_div, zorder=1)
        spar_mean = index[model, benchmark, "prism_evolve"]["final_test"]["mean_utility"]["mean"]
        ax.axvline(spar_mean, color=COLORS["spar"], lw=.75, ls="--", alpha=.5, zorder=1)
        for y, method in enumerate(METHODS):
            summary = index[model, benchmark, method]["final_test"]["mean_utility"]
            color = METHOD_COLORS[method]
            ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="|",
                        color=color, markersize=5.2, capsize=1.4, capthick=.6, elinewidth=.85,
                        linestyle="--" if method == "ablation_unweighted" else "-", zorder=2)
            for seed, marker, offset in zip(seeds, markers, offsets):
                value = seed_index[model, benchmark, method, seed]["final_test"]["mean_utility"]
                ax.scatter(value, y+offset, marker=marker, s=11, color=color,
                           edgecolor="white", linewidth=.25, zorder=3)
        ax.annotate(f"{spar_mean:.3f}", (spar_mean, len(METHODS)-1), xytext=(-4, -5.5),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.2, fontweight="bold", color=COLORS["spar"])
        ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
        ax.set_xlim(.18, .85)
        ax.set_xticks([.2, .4, .6, .8])
        ax.grid(axis="x")
        ax.tick_params(axis="y", labelsize=6.8, length=0, pad=3)
        ax.tick_params(axis="x", labelsize=7, pad=3.5)
        ax.set_xlabel(r"Test utility $\mathbb{E}[U_B]$", labelpad=7.5)
        ax.set_title(f"({'a' if panel == 0 else 'b'}) {BENCH_LABELS[benchmark]}")
    axes[0].set_ylim(len(METHODS)-.45, -.55)
    for label in axes[0].get_yticklabels():
        if label.get_text() == "SPAR":
            label.set_fontweight("bold")
            label.set_color(COLORS["spar"])
    handles = [Line2D([], [], marker=marker, ls="none", color="#64748B", markersize=3.5,
                      label=f"Seed {seed}") for seed, marker in zip(seeds, markers)]
    handles.append(Line2D([], [], marker="|", color="#64748B", lw=.7, markersize=5, label="Mean ± SD"))
    handles.append(Line2D([], [], color=COLORS["spar"], lw=.8, ls="--", label="SPAR mean"))
    fig.text(.17, .99, MODEL_LABELS[model], ha="left", va="top", fontsize=8.4, fontweight="bold")
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.994, 1.005), ncol=5,
               fontsize=6.9, columnspacing=1.0, handletextpad=.35)
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
    fig, axes = _model_panels(3.2, left=.17, bottom=.18, top=.83)
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            ax.axhline(4.85, color="#94A3B8", lw=.55, ls="--")
            for idx_c, (y, value) in enumerate(zip(positions, gains[model, benchmark])):
                is_weight = idx_c >= 5
                c_col = COLORS["spar"] if is_weight else COLORS["probe"]
                c_mk = "D" if is_weight else "o"
                ax.barh(y, value, height=.42, color=c_col, alpha=.14, edgecolor="none", zorder=1)
                ax.hlines(y, 0, value, color=c_col, lw=1.35, zorder=2)
                ax.plot(value, y, c_mk, color=c_col, ms=3.4,
                        markeredgecolor="white", markeredgewidth=.3, zorder=3)
                ax.annotate(f"+{value:.1f}", (value, y), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=6.3,
                            fontweight="bold" if is_weight else "normal",
                            color=COLORS["spar"] if is_weight else COLORS["ink"])
            ax.set_yticks(positions, [METHOD_LABELS[m] for m in controls])
            ax.tick_params(axis="y", labelsize=6.4, length=0, pad=3)
            ax.tick_params(axis="x", labelsize=6.8)
            ax.set_xlim(0, 42)
            ax.set_xticks([0, 20, 40])
            ax.grid(axis="x")
    axes[0, 0].set_ylim(7.2, -.55)
    handles = [
        Line2D([], [], marker="o", color=COLORS["probe"], lw=1.3, ms=3.6, label="Sampling / credit controls"),
        Line2D([], [], marker="D", color=COLORS["spar"], lw=1.3, ms=3.6, label="Weighting / preference controls"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.992, 1.005), ncol=2,
               fontsize=6.9, columnspacing=1.2, handletextpad=.35)
    fig.supxlabel("Mean test utility gain: SPAR − control (pts)", y=.03, fontsize=7.8)
    return save_figure(fig, out_dir, "rq2_01_mechanism_controls", "RQ2 | Mechanism-control gains",
                       "The first five rows compare return/sampling controls; the final two rows compare weighting/preference controls. Values are mean utility differences ×100.",
                       ["metrics/main_results"])


def _aligned_weighting(comparisons, out_dir):
    lookup = {(r["model"], r["benchmark"]): r for r in comparisons["method_comparisons"]
              if r["reference_role"] == "unweighted"}
    settings = [(model, benchmark) for model in MODEL_IDS for benchmark in BENCHMARKS]
    names = [f"{MODEL_LABELS[m]} / {BENCH_LABELS[b]}" for m, b in settings]
    specs = [("(a) Test utility", "Utility gain (pts)", "utility", "difference_pts", COLORS["spar"], "o"),
             ("(b) Interaction cost", "Cost reduction (%)", "cost", "reduction_percent", COLORS["probe"], "D"),
             ("(c) Feedback utilization", r"$\Delta_{\mathrm{use}}$ gain (pts)", "dev_delta_use", "difference_pts", COLORS["spar"], "s")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75), sharey=True)
    fig.subplots_adjust(left=.25, right=.985, bottom=.22, top=.79, wspace=.20)
    y = np.arange(len(settings))
    for ax, (title, xlabel, key, field, col_c, mk) in zip(axes, specs):
        values = [lookup[setting][key][field] for setting in settings]
        for y_div in (1.5, 3.5):
            ax.axhline(y_div, color="#E2E8EE", lw=.6, ls="--", zorder=0)
        ax.barh(y, values, height=.40, color=col_c, alpha=.14, edgecolor="none", zorder=1)
        ax.hlines(y, 0, values, color=col_c, lw=1.4, zorder=2)
        ax.plot(values, y, mk, color=col_c, ms=3.8, markeredgecolor="white", markeredgewidth=.35, zorder=3)
        for yp, value in zip(y, values):
            label = f"−{value:.2f}%" if key == "cost" else f"+{value:.2f}"
            ax.text(value+max(values)*.045, yp, label, va="center", fontsize=6.5,
                    fontweight="bold", color=col_c if key != "cost" else COLORS["ink"])
        ax.set_xlim(0, max(values)*1.42)
        ax.set_title(title)
        ax.set_xlabel(xlabel, labelpad=6.8)
        ax.set_yticks(y, names)
        ax.tick_params(axis="y", length=0, labelsize=6.5, pad=3)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x")
    axes[0].set_ylim(len(settings)-.45, -.55)
    fig.text(.62, .985, "SPAR vs. Unweighted (Aligned Weighting Effect)", ha="center", va="top", fontsize=7.6, fontweight="bold")
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
    fig, axes = _model_panels(3.15, left=.10, bottom=.18, top=.83)
    selected = ["frozen_base", "step_dpo", "ablation_unweighted", "prism_evolve"]
    markers = {"prism_evolve": "o", "ablation_unweighted": "^", "step_dpo": "D", "frozen_base": "s"}
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for method in [m for m in METHODS if m not in selected]+selected:
                values = index[model, benchmark, method]["final_test"]
                focused = method in selected
                cx = values["mean_cumulative_cost"]["mean"]
                cy = values["mean_utility"]["mean"]
                c_col = METHOD_COLORS[method] if focused else "#CBD5E1"
                if focused:
                    ax.errorbar(cx, cy, xerr=values["mean_cumulative_cost"]["std"],
                                yerr=values["mean_utility"]["std"], fmt="none",
                                ecolor=c_col, elinewidth=.7, capsize=1.3, alpha=.65, zorder=3)
                ax.scatter(cx, cy,
                           s=26 if method == "prism_evolve" else (21 if focused else 9),
                           color=c_col,
                           marker=markers.get(method, "o"), edgecolor="white", linewidth=.4,
                           zorder=6 if method == "prism_evolve" else (4 if focused else 2))
            unw = index[model, benchmark, "ablation_unweighted"]["final_test"]
            spr = index[model, benchmark, "prism_evolve"]["final_test"]
            ax.annotate("", xy=(spr["mean_cumulative_cost"]["mean"], spr["mean_utility"]["mean"]),
                        xytext=(unw["mean_cumulative_cost"]["mean"], unw["mean_utility"]["mean"]),
                        arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.22",
                                        color=COLORS["spar"], lw=.85, ls="--", shrinkA=3.5, shrinkB=3.5),
                        zorder=5)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([.2, .35, .5])
            ax.set_yticks([.2, .4, .6, .8])
            ax.tick_params(labelsize=7)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel("Test utility")
    handles = [Line2D([], [], marker=markers[m], ls="none", color=METHOD_COLORS[m],
                      markersize=4.2, label=METHOD_LABELS[m]) for m in reversed(selected)]
    handles.append(Line2D([], [], marker="o", ls="none", color="#CBD5E1", markersize=3.2, label="Other methods"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.994, 1.005), ncol=5,
               fontsize=6.9, columnspacing=1.0, handletextpad=.35)
    fig.supxlabel("Mean macro-action cost (lower is more efficient)", y=.03, fontsize=7.8)
    return save_figure(fig, out_dir, "rq2_03_cost_utility", "RQ2 | Utility and interaction cost",
                       "Each point is one method's three-seed mean with SD error crosshairs on focused methods; dashed arrows show the Unweighted-to-SPAR shift.",
                       ["metrics/main_results"])


def build(metrics, comparisons, out_dir):
    figures = [
        _result_table_combined(metrics, out_dir),
        _utility_gain_overview(comparisons, out_dir),
    ]
    figures.extend(_seed_variation(metrics, out_dir, model, suffix)
                   for model, suffix in zip(MODEL_IDS, MODEL_SUFFIXES))
    figures.extend([_mechanism_controls(metrics, out_dir), _aligned_weighting(comparisons, out_dir),
                    _cost_utility(metrics, out_dir)])
    return figures
