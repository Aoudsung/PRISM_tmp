"""Training and resource views of the saved RQ evidence."""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, FOCUS, METHODS, METHOD_COLORS, METHOD_LABELS,
    MODEL_COLORS, MODEL_IDS, MODEL_LABELS, save_figure,
)


def _training_index(metrics):
    index = defaultdict(list)
    for row in metrics["training_progress"]:
        index[(row["model"], row["benchmark"], row["method"], row["stage_k"])].append(row["values"])
    stages = sorted({row["stage_k"] for row in metrics["training_progress"]})
    return index, stages


def _stats(index, stages, model, benchmark, method, field):
    samples = [np.asarray([r[field] for r in index[(model, benchmark, method, stage)]]) for stage in stages]
    return np.asarray([v.mean() for v in samples]), np.asarray([v.std(ddof=1) for v in samples])


def _returns(index, stages, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.14, top=.79, wspace=.28, hspace=.66)
    all_bounds = []
    named = ["prism_evolve", "ablation_unweighted", "step_dpo", "pos_envprobe", "worldevolver",
             "infopo", "gigpo", "arew", "frozen_base"]
    markers = dict(zip(named, ["o", "^", "v", "s", "p", "D", "x", "+", None]))
    styles = dict(zip(named, ["-", "--", "-.", "-", "-", "--", ":", "-.", "--"]))
    draw_order = [m for m in METHODS if m not in named] + list(reversed(named))
    for r, benchmark in enumerate(BENCHMARKS):
        for c, model in enumerate(MODEL_IDS):
            ax = axes[r, c]
            for method in draw_order:
                avg, sd = _stats(index, stages, model, benchmark, method, "mean_batch_return")
                all_bounds.extend([float((avg-sd).min()), float((avg+sd).max())])
                if method in named:
                    col = METHOD_COLORS[method]
                    if method in FOCUS:
                        ax.fill_between(stages, avg-sd, avg+sd, color=col, alpha=.07, linewidth=0)
                    ax.plot(stages, avg, color=col, marker=markers[method], ls=styles[method],
                            ms=2.7, lw=1.4 if method == "prism_evolve" else .85, zorder=4)
                else:
                    ax.plot(stages, avg, color=COLORS["context"], lw=.55, alpha=.65, zorder=2)
            ax.set_title(f"{BENCH_LABELS[benchmark]} | {MODEL_LABELS[model]}", fontsize=7, pad=4)
            ax.set_xticks(stages)
            ax.set_xlim(stages[0]-.15, stages[-1]+.15)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(labelsize=6.5, pad=2)
            ax.grid(axis="y")
            if c == 0:
                ax.set_ylabel("Batch return", fontsize=7)
            if r == 1:
                ax.set_xlabel("Stage", fontsize=7)
    axes[0, 0].set_ylim(min(0, min(all_bounds))-.015, max(all_bounds)+.035)
    handles = [Line2D([], [], color=METHOD_COLORS[m], marker=markers[m], ls=styles[m], lw=1, ms=2.7,
                      label=METHOD_LABELS[m]) for m in named]
    handles.append(Line2D([], [], color=COLORS["context"], lw=.7, label="Internal controls"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.015), ncol=5,
               fontsize=6.5, labelspacing=.45, columnspacing=.9, handletextpad=.35, handlelength=1.7)
    return save_figure(
        fig, out_dir, "supp_01_training_returns", "Training | Batch returns across stages",
        "Lines are three-seed means; bands show sample SD for highlighted methods. Recorded zero-update values are retained.",
        ["training_progress"],
    )


def _losses(index, stages, benchmark, out_dir, family):
    methods = METHODS[:7] if family == "peers" else METHODS[7:]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.2), sharex=True)
    fig.subplots_adjust(left=.055, right=.99, bottom=.15, top=.88, wspace=.36, hspace=.58)
    markers = ["o", "s", "D"]
    for ax, method in zip(axes.flat, methods):
        zero = True
        for model, marker in zip(MODEL_IDS, markers):
            avg, sd = _stats(index, stages, model, benchmark, method, "mean_loss")
            zero = zero and bool(np.all(avg == 0) and np.all(sd == 0))
            col = MODEL_COLORS[model]
            ax.fill_between(stages, avg-sd, avg+sd, color=col, alpha=.09, linewidth=0)
            ax.plot(stages, avg, color=col, marker=marker, ms=2.5, lw=1)
        ax.set_title(METHOD_LABELS[method], fontsize=7.2, pad=4)
        ax.set_xticks(stages)
        ax.set_xlim(stages[0]-.12, stages[-1]+.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        ax.tick_params(labelsize=6.5, pad=2)
        ax.grid(axis="y")
        if zero:
            ax.text(.97, .88, "Loss = 0", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color=COLORS["muted"])
    for ax in axes[-1, :3]:
        ax.set_xlabel("Stage", fontsize=6.7)
    axes[-1, -1].axis("off")
    handles = [Line2D([], [], color=MODEL_COLORS[m], marker=marker, lw=1, ms=2.7,
                      label=MODEL_LABELS[m]) for m, marker in zip(MODEL_IDS, markers)]
    axes[-1, -1].text(0, .97, BENCH_LABELS[benchmark], transform=axes[-1, -1].transAxes,
                      fontsize=7.5, fontweight="bold", va="top")
    axes[-1, -1].text(0, .77, "Loss (mean ± SD)",
                      transform=axes[-1, -1].transAxes, fontsize=6.5, color=COLORS["muted"], va="top")
    axes[-1, -1].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -.05), fontsize=6.2,
                       borderaxespad=0, labelspacing=.3, handletextpad=.4)
    stem = ("supp_02_appworld_losses_" if benchmark == "appworld" else "supp_03_planbench_losses_") + family
    return save_figure(
        fig, out_dir, stem, f"Training | {BENCH_LABELS[benchmark]} loss trajectories | {family}",
        "Points and bands show mean and sample SD across seeds 42, 43, 44. Each method uses its own loss scale.",
        ["training_progress"],
    )


def _ledger_matrix(index, field, scale=1):
    columns = [(benchmark, model) for benchmark in BENCHMARKS for model in MODEL_IDS]
    return [[index[(model, benchmark, method)][field]/scale for benchmark, model in columns]
            for method in METHODS]


def _ledger_index(metrics):
    return {(r["model"], r["benchmark"], r["method"]): r["values"] for r in metrics["cost_ledgers"]}


def _ledger_table(metrics, out_dir, kind):
    index = _ledger_index(metrics)
    if kind == "work":
        fields = ("env_steps", "total_tokens")
        label = "Environment steps / accounted tokens (k), mean per seed"
        stem = "supp_04_resource_work_table"
        title = "Resources | Environment steps and accounted tokens"
    elif kind == "model_gradient":
        fields = ("model_inference_sec", "gradient_update_sec")
        label = "Amortized model time / gradient-update time (s), mean per seed"
        stem = "supp_05_resource_time_model_gradient"
        title = "Resources | Model and gradient-update time"
    else:
        fields = ("wall_clock_sec", "snapshot_restore_sec")
        label = "Wall-clock time / snapshot-restore time (s), mean per seed"
        stem = "supp_05_resource_time_wall_restore"
        title = "Resources | Wall-clock and snapshot-restore time"
    fig = plt.figure(figsize=(7.6, 2.9 if kind == "work" else 3.0))
    ax = fig.add_axes([.025, .025, .95, .95])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [.24, .375, .51, .66, .795, .93]
    ax.text(.01, .995, label, fontsize=7.4, fontweight="bold", va="top")
    ax.plot([.01, .99], [.944, .944], color=COLORS["ink"], lw=.7)
    ax.text(.375, .904, "AppWorld", ha="center", va="center", fontsize=7.4, fontweight="bold")
    ax.text(.795, .904, "PlanBench-XL", ha="center", va="center", fontsize=7.4, fontweight="bold")
    ax.text(.01, .846, "Method", fontsize=6.9, fontweight="bold", va="center")
    for x, model in zip(centers, MODEL_IDS*2):
        ax.text(x, .846, MODEL_LABELS[model], ha="center", va="center", fontsize=6.4)
    ax.plot([.01, .99], [.807, .807], color=COLORS["ink"], lw=.5)
    rows = np.linspace(.770, .055, len(METHODS))
    spacing = rows[0]-rows[1]
    columns = [(b, m) for b in BENCHMARKS for m in MODEL_IDS]
    for y, method in zip(rows, METHODS):
        bold = method == "prism_evolve"
        if bold:
            ax.add_patch(Rectangle((.005, y-spacing/2), .99, spacing, facecolor="#E9F3F5", edgecolor="none"))
            ax.plot([.01, .99], [y+spacing/2, y+spacing/2], color="#94A3B8", lw=.5)
        weight = "bold" if bold else "normal"
        ax.text(.01, y, METHOD_LABELS[method], fontsize=6.8, va="center", fontweight=weight)
        for x, (benchmark, model) in zip(centers, columns):
            values = index[(model, benchmark, method)]
            if kind == "work":
                text = f"{values[fields[0]]:.1f} / {values[fields[1]]/1000:.1f}"
            elif kind == "model_gradient":
                a, b = values[fields[0]], values[fields[1]]
                text = f"{a:.1f} / {b:.2f}"
            else:
                a, b = values[fields[0]], values[fields[1]]
                wall = "0" if a == 0 else "<0.01" if a < .01 else f"{a:.2f}"
                restore = "0" if b == 0 else f"{b:.3f}"
                text = f"{wall} / {restore}"
            ax.text(x, y, text, ha="center", va="center", fontsize=6.7, fontweight=weight)
    ax.plot([.01, .99], [rows[-1]-spacing/2, rows[-1]-spacing/2], color=COLORS["ink"], lw=.7)
    return save_figure(fig, out_dir, stem, title,
        "Training totals per seed, averaged across seeds 42, 43, 44. Model time includes cached encoding amortization.",
        ["cost_ledgers", "backbone_metadata"])


def _resource_counts(metrics, out_dir):
    index = {(r["model"], r["benchmark"], r["method"]): r["values"] for r in metrics["cost_ledgers"]}
    paired = all(v["snapshot_saves"] == v["snapshot_restores"] for v in index.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    fig.subplots_adjust(left=.14, right=.99, bottom=.15, top=.87, wspace=.45)
    internal = {"outcome_pg", "outcome_value_baseline", "outcome_rloo", "info_rl_neighbor", "unstratified_2branch"}
    for ax, field, heading in zip(axes, ["gradient_steps", "snapshot_saves"], ["Gradient updates", "Snapshot pairs"]):
        matrix = np.asarray(_ledger_matrix(index, field), dtype=float)
        assert np.all(matrix == matrix[:, :1]), "These views summarize equal counts across all six settings."
        assert paired
        values = matrix[:, 0]
        y = np.arange(len(METHODS))
        ax.barh(y, values, height=.5, color=[COLORS["context"] if m in internal else METHOD_COLORS[m] for m in METHODS])
        ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS])
        ax.invert_yaxis()
        maximum = float(values.max())
        ax.set_xlim(0, maximum*1.17)
        ax.set_xticks(np.linspace(0, maximum, 4))
        ax.set_xlabel("Per-seed count")
        ax.set_title(f"{heading} | all six settings", fontsize=7.5, pad=6)
        ax.tick_params(labelsize=6.5, pad=2)
        ax.grid(axis="x")
        for row, value in enumerate(values):
            ax.text(value+maximum*.025, row, f"{value:.0f}", va="center", fontsize=6.5)
    return save_figure(fig, out_dir, "supp_04_resource_work_counts", "Resources | Update and snapshot counts",
        "Counts are identical across all six model/benchmark settings. Snapshot saves equal restores.", ["cost_ledgers"])


def build(metrics, comparisons, out_dir):
    index, stages = _training_index(metrics)
    records = [_returns(index, stages, out_dir)]
    for benchmark in BENCHMARKS:
        for family in ["peers", "controls"]:
            records.append(_losses(index, stages, benchmark, out_dir, family))
    records += [_ledger_table(metrics, out_dir, "work"), _resource_counts(metrics, out_dir),
                _ledger_table(metrics, out_dir, "model_gradient"), _ledger_table(metrics, out_dir, "wall_restore")]
    return records
