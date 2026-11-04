"""Training curves and resource tables in the approved reference-guided style."""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, METHODS, METHOD_LABELS,
    MODEL_IDS, MODEL_LABELS, save_figure,
)


def _training_index(metrics):
    index = defaultdict(list)
    for row in metrics["training_progress"]:
        index[(row["model"], row["benchmark"], row["method"], row["stage_k"])].append(row["values"])
    return index, sorted({r["stage_k"] for r in metrics["training_progress"]})


def _stats(index, stages, model, benchmark, method, field):
    samples = [np.asarray([r[field] for r in index[(model, benchmark, method, stage)]]) for stage in stages]
    return np.asarray([v.mean() for v in samples]), np.asarray([v.std(ddof=1) for v in samples])


def _style(method):
    if method == "prism_evolve":
        return {"color": COLORS["spar"], "marker": "o", "ls": "-", "lw": 1.7}
    if method == "ablation_unweighted":
        return {"color": COLORS["unweighted"], "marker": "^", "ls": "--", "lw": 1.4}
    return {"color": COLORS["context"], "marker": "s", "ls": ":", "lw": 1.0}


def _returns(index, stages, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.99, bottom=.13, top=.83, wspace=.23, hspace=.30)
    focus = ["prism_evolve", "ablation_unweighted", "frozen_base"]
    order = [m for m in METHODS if m not in focus] + list(reversed(focus))
    limits = []
    for row, benchmark in enumerate(BENCHMARKS):
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            for method in order:
                avg, sd = _stats(index, stages, model, benchmark, method, "mean_batch_return")
                limits.extend([float((avg-sd).min()), float((avg+sd).max())])
                if method in focus:
                    style = _style(method)
                    ax.fill_between(stages, avg-sd, avg+sd, color=style["color"], alpha=.09, linewidth=0)
                    ax.plot(stages, avg, ms=3.0, zorder=4, **style)
                else:
                    ax.fill_between(stages, avg-sd, avg+sd, color=COLORS["context"], alpha=.018, linewidth=0)
                    ax.plot(stages, avg, color=COLORS["context"], lw=.55, alpha=.5, zorder=1)
            if row == 0:
                ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}", pad=6)
            ax.set_xticks(stages)
            ax.set_xlim(stages[0]-.12, stages[-1]+.12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(labelsize=7, pad=2)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel(f"{BENCH_LABELS[benchmark]}\nBatch return")
            if row == 1:
                ax.set_xlabel("Training stage")
    axes[0, 0].set_ylim(min(0, min(limits))-.015, max(limits)+.035)
    handles = [Line2D([], [], ms=3, label=METHOD_LABELS[m], **_style(m)) for m in focus]
    handles += [Line2D([], [], color=COLORS["context"], lw=.6, label="Other methods"),
                Patch(facecolor=COLORS["context"], alpha=.12, edgecolor="none", label="± SD")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.015), ncol=5,
               fontsize=7.0, handletextpad=.4, columnspacing=1.0)
    return save_figure(fig, out_dir, "supp_01_training_returns", "Training | Batch returns",
        "All 14 methods retained. Lines are seed means; bands are sample SD across seeds 42, 43, 44.",
        ["training_progress"])


def _is_zero_loss(index, stages, benchmark, method):
    return all(v["mean_loss"] == 0 for model in MODEL_IDS for stage in stages
               for v in index[(model, benchmark, method, stage)])


def _loss_page(index, stages, benchmark, methods, page, out_dir):
    rows = len(methods)
    fig, axes = plt.subplots(rows, 3, figsize=(7.6, 1.05+.90*rows), squeeze=False, sharex=True)
    fig.subplots_adjust(left=.10, right=.99, bottom=.13, top=.85, wspace=.23, hspace=.42)
    for row, method in enumerate(methods):
        bounds = []
        for col, model in enumerate(MODEL_IDS):
            ax = axes[row, col]
            avg, sd = _stats(index, stages, model, benchmark, method, "mean_loss")
            bounds.extend([float((avg-sd).min()), float((avg+sd).max())])
            style = _style(method)
            ax.fill_between(stages, avg-sd, avg+sd, color=style["color"], alpha=.11, linewidth=0)
            ax.plot(stages, avg, ms=3, **style)
            if row == 0:
                ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}", pad=6)
            ax.set_xticks(stages)
            ax.set_xlim(stages[0]-.12, stages[-1]+.12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            ax.tick_params(labelsize=7, pad=2)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel(f"{METHOD_LABELS[method]}\nLoss")
            else:
                ax.tick_params(labelleft=False)
            if row == rows-1:
                ax.set_xlabel("Training stage")
        pad = max((max(bounds)-min(bounds))*.08, abs(max(bounds))*.04, .001)
        for ax in axes[row]:
            ax.set_ylim(min(bounds)-pad, max(bounds)+pad)
    fig.text(.10, .98, BENCH_LABELS[benchmark], fontsize=8.4, fontweight="bold", va="top")
    handles = [Line2D([], [], color=COLORS["context"], marker="s", lw=1, ms=3, label="Mean"),
               Patch(facecolor=COLORS["context"], alpha=.15, edgecolor="none", label="± SD (3 seeds)")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.995, 1.015), ncol=2,
               fontsize=7.2, handletextpad=.4, columnspacing=1.2)
    prefix = "supp_02_appworld_losses" if benchmark == "appworld" else "supp_03_planbench_losses"
    return save_figure(fig, out_dir, f"{prefix}_methods_{page:02d}",
        f"Training | {BENCH_LABELS[benchmark]} loss | {', '.join(METHOD_LABELS[m] for m in methods)}",
        "Each method has an independent y-scale shared across its three model panels. Means and sample SD use three seeds.",
        ["training_progress"])


def _zero_loss_table(index, stages, zero_methods, out_dir):
    rows = [(benchmark, method) for benchmark in BENCHMARKS for method in zero_methods[benchmark]]
    fig = plt.figure(figsize=(7.6, 1.5))
    ax = fig.add_axes([.025, .05, .95, .9])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.01, .98, f"Recorded loss at training stages {stages[0]}-{stages[-1]} (mean ± SD)",
            fontsize=7.6, fontweight="bold", va="top")
    ax.plot([.01, .99], [.84, .84], lw=.65, color=COLORS["ink"])
    centers = [.08, .31, .54, .735, .93]
    headers = ["Benchmark", "Method"] + [MODEL_LABELS[m] for m in MODEL_IDS]
    for x, label in zip(centers, headers):
        ax.text(x, .75, label, ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.plot([.01, .99], [.67, .67], lw=.5, color=COLORS["ink"])
    ys = np.linspace(.59, .075, len(rows))
    for y, (benchmark, method) in zip(ys, rows):
        ax.text(centers[0], y, BENCH_LABELS[benchmark], ha="center", va="center", fontsize=7.1)
        ax.text(centers[1], y, METHOD_LABELS[method], ha="center", va="center", fontsize=7.1)
        for x, model in zip(centers[2:], MODEL_IDS):
            avg, sd = _stats(index, stages, model, benchmark, method, "mean_loss")
            assert np.all(avg == 0) and np.all(sd == 0)
            ax.text(x, y, "0.0000 ± 0.0000", ha="center", va="center", fontsize=7.1)
    ax.plot([.01, .99], [.02, .02], lw=.65, color=COLORS["ink"])
    return save_figure(fig, out_dir, "supp_03_recorded_zero_losses", "Training | Recorded zero losses",
        "For these named benchmark/method groups, every stage and seed record has zero loss. This table records loss values only.",
        ["training_progress"])


def _ledger_index(metrics):
    return {(r["model"], r["benchmark"], r["method"]): r["values"] for r in metrics["cost_ledgers"]}


def _ledger_matrix(index, field):
    return np.asarray([[index[(model, benchmark, method)][field]
                        for benchmark in BENCHMARKS for model in MODEL_IDS] for method in METHODS])


def _number(value, digits):
    if value == 0:
        return "0"
    threshold = 10**(-digits)
    return f"<{threshold:.{digits}f}" if value < threshold else f"{value:.{digits}f}"


def _resource_table(metrics, benchmark, kind, out_dir):
    index = _ledger_index(metrics)
    if kind == "work":
        fields = [("env_steps", 1, "Env. steps", 1), ("total_tokens", 1000, "Tokens (k)", 1)]
        family, title = "supp_04_work", "Training work per seed"
    elif kind == "model_gradient":
        fields = [("model_inference_sec", 1, "Model (s)", 1), ("gradient_update_sec", 1, "Gradient (s)", 2)]
        family, title = "supp_05_time_model_gradient", "Model and gradient-update time per seed"
    else:
        fields = [("wall_clock_sec", 1, "Wall (s)", 2), ("snapshot_restore_sec", 1, "Restore (s)", 3)]
        family, title = "supp_05_time_wall_restore", "Wall-clock and snapshot-restore time per seed"
    fig = plt.figure(figsize=(7.6, 2.8))
    ax = fig.add_axes([.025, .025, .95, .95])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.01, .99, BENCH_LABELS[benchmark], fontsize=8.4, fontweight="bold", va="top")
    ax.text(.99, .99, title, fontsize=7.4, va="top", ha="right")
    ax.plot([.01, .99], [.94, .94], lw=.7, color=COLORS["ink"])
    centers = [.29, .41, .56, .68, .83, .95]
    for x, model in zip([.35, .62, .89], MODEL_IDS):
        ax.text(x, .888, MODEL_LABELS[model], ha="center", va="center", fontsize=7.3, fontweight="bold")
    ax.text(.01, .82, "Method", fontsize=7.1, fontweight="bold", va="center")
    for x, (_, _, label, _) in zip(centers, fields*3):
        ax.text(x, .82, label, ha="center", va="center", fontsize=6.9)
    ax.plot([.01, .99], [.782, .782], lw=.5, color=COLORS["ink"])
    ys = np.linspace(.745, .05, len(METHODS))
    spacing = ys[0]-ys[1]
    for y, method in zip(ys, METHODS):
        bold = method == "prism_evolve"
        if bold:
            ax.add_patch(Rectangle((.005, y-spacing/2), .99, spacing,
                                   facecolor="#E9F3F5", edgecolor="none"))
            ax.plot([.01, .99], [y+spacing/2, y+spacing/2], color="#94A3B8", lw=.5)
        weight = "bold" if bold else "normal"
        ax.text(.01, y, METHOD_LABELS[method], fontsize=7.0, va="center", fontweight=weight)
        for col, model in enumerate(MODEL_IDS):
            for field_index, (field, scale, _, digits) in enumerate(fields):
                value = index[(model, benchmark, method)][field]/scale
                ax.text(centers[col*2+field_index]+.035, y, _number(value, digits), ha="right", va="center",
                        fontsize=7.0, fontweight=weight)
    ax.plot([.01, .99], [ys[-1]-spacing/2, ys[-1]-spacing/2], lw=.7, color=COLORS["ink"])
    return save_figure(fig, out_dir, f"{family}_{benchmark}", f"Resources | {BENCH_LABELS[benchmark]} | {title}",
        "Per-seed training totals, averaged over three seeds. Model time is the recorded amortized encoding measure; timing fields remain separate.",
        ["cost_ledgers", "backbone_metadata"])


def _resource_counts(metrics, out_dir):
    index = _ledger_index(metrics)
    assert all(v["snapshot_saves"] == v["snapshot_restores"] for v in index.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    fig.subplots_adjust(left=.14, right=.99, bottom=.15, top=.87, wspace=.45)
    for panel, (ax, field, title) in enumerate(zip(axes, ["gradient_steps", "snapshot_saves"],
                                                ["Gradient updates", "Snapshot pairs"])):
        matrix = _ledger_matrix(index, field)
        assert np.all(matrix == matrix[:, :1]), "Counts in this view are shared across all six settings."
        values = matrix[:, 0]
        y = np.arange(len(METHODS))
        ax.barh(y, values, height=.5, color=[_style(m)["color"] for m in METHODS])
        ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS])
        ax.invert_yaxis()
        maximum = float(values.max())
        ax.set_xlim(0, maximum*1.17)
        ax.set_xticks(np.linspace(0, maximum, 4))
        ax.set_xlabel("Per-seed count")
        ax.set_title(f"({chr(97+panel)}) {title}", pad=6)
        ax.tick_params(labelsize=6.8, pad=2)
        ax.grid(axis="x")
        for row, value in enumerate(values):
            ax.text(value+maximum*.025, row, f"{value:.0f}", va="center", fontsize=6.8)
    return save_figure(fig, out_dir, "supp_04_resource_work_counts", "Resources | Update and snapshot counts",
        "Counts are identical across all six model/benchmark settings. Snapshot saves equal restores.",
        ["cost_ledgers"])


def build(metrics, comparisons, out_dir):
    index, stages = _training_index(metrics)
    records = [_returns(index, stages, out_dir)]
    zero_methods = {}
    for benchmark in BENCHMARKS:
        zero_methods[benchmark] = [m for m in METHODS if _is_zero_loss(index, stages, benchmark, m)]
        methods = [m for m in METHODS if m not in zero_methods[benchmark]]
        for start in range(0, len(methods), 3):
            records.append(_loss_page(index, stages, benchmark, methods[start:start+3], start//3+1, out_dir))
    if any(zero_methods.values()):
        records.append(_zero_loss_table(index, stages, zero_methods, out_dir))
    for kind in ["work", "model_gradient", "wall_restore"]:
        for benchmark in BENCHMARKS:
            records.append(_resource_table(metrics, benchmark, kind, out_dir))
    records.append(_resource_counts(metrics, out_dir))
    return records
