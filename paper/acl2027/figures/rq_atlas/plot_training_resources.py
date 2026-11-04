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

INTERNAL_CONTROLS = {"outcome_pg", "outcome_value_baseline", "outcome_rloo",
                     "info_rl_neighbor", "unstratified_2branch"}


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


def _loss_style(method):
    if method == "prism_evolve":
        return {"color": COLORS["spar"], "marker": "o", "ls": "-", "lw": 1.75}
    if method == "ablation_unweighted":
        return {"color": COLORS["unweighted"], "marker": "^", "ls": "--", "lw": 1.45}
    if method in INTERNAL_CONTROLS:
        return {"color": COLORS["probe"], "marker": "D", "ls": "-", "lw": 1.4}
    return {"color": "#334155", "marker": "s", "ls": "-", "lw": 1.35}


def _returns(index, stages, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.99, bottom=.14, top=.83, wspace=.23, hspace=.30)
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
                    ax.plot(stages, avg, ms=3.2, markeredgecolor="white", markeredgewidth=.3, zorder=4, **style)
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
    handles = [Line2D([], [], ms=3.2, label=METHOD_LABELS[m], **_style(m)) for m in focus]
    handles += [Line2D([], [], color=COLORS["context"], lw=.6, label="Other methods"),
                Patch(facecolor=COLORS["context"], alpha=.14, edgecolor="none", label="± SD")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.015), ncol=5,
               fontsize=7.0, handletextpad=.4, columnspacing=1.0)
    return save_figure(fig, out_dir, "supp_01_training_returns", "Training | Batch returns",
        "All 14 methods retained. Lines are seed means; bands are sample SD across seeds 42, 43, 44.",
        ["training_progress"])


def _is_zero_loss(index, stages, benchmark, method):
    return all(v["mean_loss"] == 0 for model in MODEL_IDS for stage in stages
               for v in index[(model, benchmark, method, stage)])


def _loss_overview(index, stages, benchmark, methods, out_dir, zero_methods=None):
    fig, axes = plt.subplots(3, 4, figsize=(7.6, 4.6), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.12, top=.87,
                        wspace=.55, hspace=.62)
    model_styles = [
        {"marker": "o", "ls": "-"},
        {"marker": "s", "ls": "--"},
        {"marker": "D", "ls": "-."},
    ]
    offsets = [-.045, 0, .045]
    statistics = []
    for panel, (ax, method) in enumerate(zip(axes.flat, methods)):
        color = _loss_style(method)["color"]
        bounds = []
        for model, model_style, offset in zip(MODEL_IDS, model_styles, offsets):
            avg, sd = _stats(index, stages, model, benchmark, method, "mean_loss")
            bounds.extend([float((avg-sd).min()), float((avg+sd).max())])
            # Fixed offsets within each categorical stage keep intervals distinguishable.
            x = np.asarray(stages, dtype=float) + offset
            ax.errorbar(x, avg, yerr=sd, fmt="none", ecolor=color, alpha=.42,
                        elinewidth=.6, capsize=1.35, capthick=.6, zorder=1)
            ax.plot(x, avg, color=color, lw=1.3, ms=3.4,
                    markerfacecolor="white" if model_style["marker"] == "s" else color,
                    markeredgecolor=color, markeredgewidth=.55, zorder=3, **model_style)
            statistics.extend({"model": model, "method": method, "stage_k": int(k),
                               "mean": float(mean), "std": float(std)}
                              for k, mean, std in zip(stages, avg, sd))
        pad = max((max(bounds)-min(bounds))*.09, abs(max(bounds))*.045, .001)
        ax.set_ylim(min(bounds)-pad, max(bounds)+pad)
        ax.set_title(f"({chr(97+panel)}) {METHOD_LABELS[method]}", pad=6)
        ax.set_xticks(stages)
        ax.set_xlim(stages[0]-.16, stages[-1]+.16)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        ax.tick_params(labelsize=7, pad=2)
        ax.grid(axis="y")
        ax.tick_params(labelbottom=True)
    for ax in list(axes.flat)[len(methods):]:
        ax.axis("off")
    if zero_methods:
        ax = axes.flat[-1]
        ax.set_title("(l) Zero-loss methods", pad=6)
        ax.text(.02, .85, "Method", transform=ax.transAxes, fontsize=6.5, fontweight="bold")
        ax.text(.98, .85, "All models", transform=ax.transAxes, fontsize=6.5,
                fontweight="bold", ha="right")
        ax.plot([.02, .98], [.77, .77], transform=ax.transAxes, color="#94A3B8", lw=.5)
        for y, method in zip([.61, .40, .19], zero_methods):
            for model in MODEL_IDS:
                avg, sd = _stats(index, stages, model, benchmark, method, "mean_loss")
                assert np.all(avg == 0) and np.all(sd == 0)
            ax.text(.02, y, METHOD_LABELS[method], transform=ax.transAxes, fontsize=6.5)
            ax.text(.98, y, "0 ± 0", transform=ax.transAxes, fontsize=6.8, ha="right")
    fig.text(.085, .985, BENCH_LABELS[benchmark], fontsize=8.4, fontweight="bold", va="top")
    handles = [Line2D([], [], color="#334155", lw=1.3, ms=3.4,
                      markerfacecolor="white" if style["marker"] == "s" else "#334155",
                      label=MODEL_LABELS[model], **style)
               for model, style in zip(MODEL_IDS, model_styles)]
    handles.append(Line2D([], [], color="#64748B", ls="", marker="|", ms=6, label="± SD"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.995, 1.005), ncol=4,
               fontsize=7.1, handletextpad=.4, columnspacing=1.1)
    fig.supxlabel("Training stage", y=.025, fontsize=7.8)
    fig.supylabel("Recorded training loss", x=.008, fontsize=7.8)
    prefix = "supp_02_appworld_losses" if benchmark == "appworld" else "supp_03_planbench_losses"
    record = save_figure(fig, out_dir, f"{prefix}_combined",
        f"Training | {BENCH_LABELS[benchmark]} losses across three models",
        "One panel per nonzero-loss method; each panel retains the original loss units and a shared scale "
        "for its three models. Points are three-seed means; thin intervals are sample SD. Models use "
        "fixed line styles and circle/square/diamond markers; color retains method identity. "
        "Stages have horizontal offsets -0.045/0/+0.045 solely to separate markers and intervals. "
        "The final cell preserves methods whose recorded losses are zero for all models, stages and seeds.",
        ["training_progress"])
    record.update(panel_count=len(methods), curve_count=len(methods)*len(MODEL_IDS),
                  mean_point_count=len(statistics), loss_statistics=statistics,
                  zero_loss_methods=list(zero_methods or []))
    return record


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


def _draw_resource_panel(ax, index, benchmark, fields, panel_title):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.01, .988, panel_title, fontsize=7.8, fontweight="bold", va="top", color=COLORS["ink"])
    ax.plot([.01, .99], [.932, .932], lw=.7, color=COLORS["ink"])
    group_centers = [.312, .582, .852]
    centers = [.225, .312, .399, .495, .582, .669, .765, .852, .939]
    for x, model in zip(group_centers, MODEL_IDS):
        ax.text(x, .885, MODEL_LABELS[model], ha="center", va="center", fontsize=7.1, fontweight="bold")
        ax.plot([x-.115, x+.115], [.848, .848], lw=.4, color="#94A3B8")
    ax.text(.01, .810, "Method", fontsize=6.7, fontweight="bold", va="center")
    for x, (_, _, label, _) in zip(centers, fields * 3):
        ax.text(x, .810, label, ha="center", va="center", fontsize=6.35, fontweight="bold", color="#334155")
    ax.plot([.01, .99], [.772, .772], lw=.5, color=COLORS["ink"])
    ys = np.linspace(.736, .048, len(METHODS))
    spacing = ys[0] - ys[1]
    for row_idx, (y, method) in enumerate(zip(ys, METHODS)):
        bold = method == "prism_evolve"
        if row_idx in (0, 6):
            ax.plot([.01, .99], [y-spacing/2, y-spacing/2], color="#CBD5E1", lw=.45,
                    ls="--" if row_idx == 0 else "-")
        if bold:
            ax.add_patch(Rectangle((.005, y-spacing/2), .99, spacing,
                                   facecolor="#E6F3F6", edgecolor="none"))
            ax.plot([.01, .99], [y+spacing/2, y+spacing/2], color="#94A3B8", lw=.5)
        weight = "bold" if bold else "normal"
        txt_col = COLORS["spar"] if bold else COLORS["ink"]
        ax.text(.01, y, METHOD_LABELS[method], fontsize=6.65, va="center", fontweight=weight, color=txt_col)
        for col, model in enumerate(MODEL_IDS):
            for field_index, (field, scale, _, digits) in enumerate(fields):
                value = index[(model, benchmark, method)][field] / scale
                ax.text(centers[col*3 + field_index] + .028, y, _number(value, digits),
                        ha="right", va="center", fontsize=6.5, fontweight=weight, color=txt_col)
    ax.plot([.01, .99], [ys[-1]-spacing/2, ys[-1]-spacing/2], lw=.7, color=COLORS["ink"])


def _resource_table_combined(metrics, benchmark, out_dir):
    index = _ledger_index(metrics)
    fields_a = [
        ("env_steps", 1, "Env. steps", 1),
        ("total_tokens", 1000, "Tokens (k)", 1),
        ("wall_clock_sec", 1, "Wall (s)", 2),
    ]
    fields_b = [
        ("model_inference_sec", 1, "Model (s)", 1),
        ("gradient_update_sec", 1, "Grad. (s)", 2),
        ("snapshot_restore_sec", 1, "Restore (s)", 3),
    ]
    fig = plt.figure(figsize=(7.6, 5.55))
    ax_a = fig.add_axes([.025, .515, .95, .455])
    ax_b = fig.add_axes([.025, .025, .95, .455])
    _draw_resource_panel(
        ax_a, index, benchmark, fields_a,
        f"(a) {BENCH_LABELS[benchmark]} — Training interaction work and total wall-clock time per seed",
    )
    _draw_resource_panel(
        ax_b, index, benchmark, fields_b,
        f"(b) {BENCH_LABELS[benchmark]} — Model encoding, gradient-update, and snapshot-restore time per seed",
    )
    stem = "supp_04_resources_appworld" if benchmark == "appworld" else "supp_05_resources_planbench_xl"
    return save_figure(
        fig, out_dir, stem,
        f"Resources | {BENCH_LABELS[benchmark]} | Complete training work and timing ledger",
        "Per-seed training totals averaged over three seeds. Panel (a) reports interaction steps, tokens, and wall-clock time; Panel (b) reports amortized model encoding, gradient update, and snapshot restore time.",
        ["cost_ledgers", "backbone_metadata"],
    )


def _bar_color(method):
    if method == "prism_evolve":
        return COLORS["spar"]
    if method == "ablation_unweighted":
        return COLORS["unweighted"]
    if method in INTERNAL_CONTROLS:
        return COLORS["probe"]
    if method == "frozen_base":
        return "#CBD5E1"
    return "#64748B"


def _resource_counts(metrics, out_dir):
    index = _ledger_index(metrics)
    assert all(v["snapshot_saves"] == v["snapshot_restores"] for v in index.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    fig.subplots_adjust(left=.14, right=.99, bottom=.16, top=.83, wspace=.45)
    for panel, (ax, field, title) in enumerate(zip(axes, ["gradient_steps", "snapshot_saves"],
                                                ["Gradient updates", "Snapshot pairs"])):
        matrix = _ledger_matrix(index, field)
        assert np.all(matrix == matrix[:, :1]), "Counts in this view are shared across all six settings."
        values = matrix[:, 0]
        y = np.arange(len(METHODS))
        for y_div, ls_div in [(.5, ":"), (6.5, "--"), (12.5, "-")]:
            ax.axhline(y_div, color="#CBD5E1", lw=.45, ls=ls_div, zorder=0)
        ax.barh(y, values, height=.54, color=[_bar_color(m) for m in METHODS], zorder=2)
        ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS])
        ax.invert_yaxis()
        maximum = float(values.max())
        ax.set_xlim(0, maximum*1.19)
        ax.set_xticks(np.linspace(0, maximum, 4))
        ax.set_xlabel("Per-seed count")
        ax.set_title(f"({chr(97+panel)}) {title}", pad=6)
        ax.tick_params(labelsize=6.8, pad=2)
        ax.grid(axis="x")
        for row, (method, value) in enumerate(zip(METHODS, values)):
            is_spar = method == "prism_evolve"
            ax.text(value+maximum*.025, row, f"{value:.0f}", va="center", fontsize=6.7,
                    fontweight="bold" if is_spar else "normal",
                    color=COLORS["spar"] if is_spar else COLORS["ink"])
        for label in ax.get_yticklabels():
            if label.get_text() == "SPAR":
                label.set_fontweight("bold")
                label.set_color(COLORS["spar"])
    handles = [
        Patch(facecolor="#64748B", label="External baselines"),
        Patch(facecolor=COLORS["probe"], label="Sampling / credit controls"),
        Patch(facecolor=COLORS["unweighted"], label="Unweighted"),
        Patch(facecolor=COLORS["spar"], label="SPAR"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.992, 1.005), ncol=4,
               fontsize=6.9, columnspacing=1.0, handletextpad=.35)
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
        records.append(_loss_overview(index, stages, benchmark, methods, out_dir,
                                      zero_methods=zero_methods[benchmark]))
    for benchmark in BENCHMARKS:
        records.append(_resource_table_combined(metrics, benchmark, out_dir))
    records.append(_resource_counts(metrics, out_dir))
    return records
