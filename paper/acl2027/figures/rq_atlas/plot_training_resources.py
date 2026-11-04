"""Training and resource views of the saved RQ evidence."""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, FOCUS, METHODS, METHOD_COLORS, METHOD_LABELS,
    MODEL_COLORS, MODEL_IDS, MODEL_LABELS, colorbar, heatmap, save_figure,
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
    fig, axes = plt.subplots(2, 3, figsize=(8.9, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.975, bottom=.145, top=.835, wspace=.26, hspace=.48)
    all_bounds = []
    markers = {"prism_evolve": "o", "ablation_unweighted": "s", "step_dpo": "^"}
    draw_order = [m for m in METHODS if m not in FOCUS] + list(reversed(FOCUS))
    for r, benchmark in enumerate(BENCHMARKS):
        for c, model in enumerate(MODEL_IDS):
            ax = axes[r, c]
            for method in draw_order:
                avg, sd = _stats(index, stages, model, benchmark, method, "mean_batch_return")
                all_bounds.extend([float((avg-sd).min()), float((avg+sd).max())])
                if method in FOCUS:
                    col = METHOD_COLORS[method]
                    ax.fill_between(stages, avg-sd, avg+sd, color=col, alpha=.09, linewidth=0)
                    ax.plot(stages, avg, color=col, marker=markers[method], ms=3.5, lw=1.5, zorder=4)
                else:
                    ax.plot(stages, avg, color=COLORS["context"], lw=.65, alpha=.7, zorder=2)
            ax.set_title(f"{BENCH_LABELS[benchmark]} | {MODEL_LABELS[model]}", fontsize=8.8)
            ax.set_xticks(stages)
            ax.set_xlim(stages[0]-.15, stages[-1]+.15)
            ax.grid(axis="y")
            if c == 0:
                ax.set_ylabel("Mean training batch return")
            if r == 1:
                ax.set_xlabel("Training stage")
    axes[0, 0].set_ylim(min(0, min(all_bounds))-.015, max(all_bounds)+.035)
    handles = [Line2D([], [], color=METHOD_COLORS[m], marker=markers[m], lw=1.5, ms=3.5,
                      label=METHOD_LABELS[m]) for m in FOCUS]
    handles.append(Line2D([], [], color=COLORS["context"], lw=1, label="Other methods"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .91), ncol=4)
    return save_figure(
        fig, out_dir, "supp_01_training_returns", "Training | Batch returns across stages",
        "Lines are three-seed means; bands show sample SD for highlighted methods. Recorded zero-update values are retained.",
        ["training_progress"],
    )


def _losses(index, stages, benchmark, out_dir):
    fig, axes = plt.subplots(5, 3, figsize=(8.9, 9.4), sharex=True)
    fig.subplots_adjust(left=.075, right=.975, bottom=.125, top=.855, wspace=.32, hspace=.87)
    markers = ["o", "s", "^"]
    for ax, method in zip(axes.flat, METHODS):
        zero = True
        for model, marker in zip(MODEL_IDS, markers):
            avg, sd = _stats(index, stages, model, benchmark, method, "mean_loss")
            zero = zero and bool(np.all(avg == 0) and np.all(sd == 0))
            col = MODEL_COLORS[model]
            ax.fill_between(stages, avg-sd, avg+sd, color=col, alpha=.10, linewidth=0)
            ax.plot(stages, avg, color=col, marker=marker, ms=3.1, lw=1.15)
        ax.set_title(METHOD_LABELS[method], fontsize=8.6, pad=5)
        ax.set_xticks(stages)
        ax.set_xlim(stages[0]-.12, stages[-1]+.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        ax.tick_params(labelsize=7, pad=3)
        ax.grid(axis="y")
        if zero:
            ax.text(.97, .88, "Recorded loss = 0", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.7, color=COLORS["muted"])
    for ax in axes[-1, :2]:
        ax.set_xlabel("Training stage", fontsize=7.5)
    axes[-1, -1].axis("off")
    handles = [Line2D([], [], color=MODEL_COLORS[m], marker=marker, lw=1.2, ms=3.5,
                      label=MODEL_LABELS[m]) for m, marker in zip(MODEL_IDS, markers)]
    axes[-1, -1].legend(handles=handles, loc="upper left", fontsize=7.7, borderaxespad=0,
                       labelspacing=.9)
    stem = "supp_02_appworld_losses" if benchmark == "appworld" else "supp_03_planbench_losses"
    return save_figure(
        fig, out_dir, stem, f"Training | {BENCH_LABELS[benchmark]} loss trajectories",
        "Points and bands show mean and sample SD across seeds 42, 43, 44. Each method uses its own loss scale.",
        ["training_progress"],
    )


def _ledger_matrix(index, field, scale=1):
    columns = [(benchmark, model) for benchmark in BENCHMARKS for model in MODEL_IDS]
    return [[index[(model, benchmark, method)][field]/scale for benchmark, model in columns]
            for method in METHODS]


def _resource_panels(metrics, out_dir, timing=False):
    index = {(r["model"], r["benchmark"], r["method"]): r["values"] for r in metrics["cost_ledgers"]}
    paired = all(v["snapshot_saves"] == v["snapshot_restores"] for v in index.values())
    columns = ["Q7\nAW", "L8\nAW", "Q27\nAW", "Q7\nPB", "L8\nPB", "Q27\nPB"]
    if timing:
        fields = [
            ("model_inference_sec", 1, "Amortized model time (s)", "Seconds", ".1f"),
            ("gradient_update_sec", 1, "Gradient-update time (s)", "Seconds", ".2f"),
            ("wall_clock_sec", 1, "Wall-clock time (s)", "Seconds", ".2f"),
            ("snapshot_restore_sec", 1, "Snapshot-restore time (s)", "Seconds", ".3f"),
        ]
        stem, title = "supp_05_resource_time", "Resources | Recorded training time"
        note = "Per-seed means over 3 seeds. Model time includes cached encoding amortization. AW/PB: AppWorld/PlanBench-XL; Q7/L8/Q27: the three backbones."
    else:
        fields = [
            ("env_steps", 1, "Environment steps", "Steps", ".1f"),
            ("total_tokens", 1000, "Accounted tokens (k)", "k tokens", ".1f"),
            ("gradient_steps", 1, "Gradient updates", "Updates", ".0f"),
            ("snapshot_saves", 1, "Snapshot pairs" if paired else "Snapshot saves / restores", "Pairs" if paired else "Saves", ".0f"),
        ]
        stem, title = "supp_04_resource_work", "Resources | Interaction and update counts"
        note = "Per-seed means (3 seeds). Lower panels: counts shared by all six settings. Q7/L8/Q27: Qwen2.5-7B/Llama-3.1-8B/Qwen3.8-27B; AW/PB: AppWorld/PlanBench-XL."
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.5))
    fig.subplots_adjust(left=.14, right=.96, bottom=.145, top=.85, wspace=.48, hspace=.36)
    for ax, (field, scale, heading, unit, fmt) in zip(axes.flat, fields):
        matrix = np.asarray(_ledger_matrix(index, field, scale), dtype=float)
        if not timing and field in ("gradient_steps", "snapshot_saves"):
            assert np.all(matrix == matrix[:, :1]), "Show all setting columns when these counts differ."
            if field == "snapshot_saves":
                assert paired
            values = matrix[:, 0]
            y = np.arange(len(METHODS))
            ax.barh(y, values, height=.55,
                    color=[COLORS["spar"] if m == "prism_evolve" else "#94A3B8" for m in METHODS])
            ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS], fontsize=7)
            ax.invert_yaxis()
            maximum = float(values.max())
            ax.set_xlim(0, maximum*1.17)
            ax.set_xticks(np.linspace(0, maximum, 4))
            ax.set_xlabel(f"{unit} per seed")
            ax.set_title(heading)
            ax.grid(axis="x")
            for row, value in enumerate(values):
                ax.text(value+maximum*.025, row, f"{value:.0f}", va="center", fontsize=7)
            continue
        labels = None
        if field == "snapshot_saves" and not paired:
            labels = [[f"{index[(m,b,method)]['snapshot_saves']:.0f}/{index[(m,b,method)]['snapshot_restores']:.0f}"
                       for b in BENCHMARKS for m in MODEL_IDS] for method in METHODS]
        elif timing:
            labels = [[("0" if not v else "<0.01" if field == "wall_clock_sec" and v < .01 else format(v, fmt))
                       for v in row] for row in matrix]
        im = heatmap(ax, matrix, [METHOD_LABELS[m] for m in METHODS], columns,
                     vmin=0, vmax=float(np.max(matrix)), fmt=fmt, labels=labels, textsize=6.8)
        ax.set_title(heading, fontsize=9.2)
        ax.tick_params(axis="x", labelsize=7, pad=4)
        ax.tick_params(axis="y", labelsize=7, pad=4)
        ax.axvline(2.5, color="white", lw=2.2)
        colorbar(fig, im, ax, unit, fraction=.035, pad=.025)
    return save_figure(fig, out_dir, stem, title, note, ["cost_ledgers", "backbone_metadata"])


def build(metrics, comparisons, out_dir):
    index, stages = _training_index(metrics)
    return [_returns(index, stages, out_dir),
            _losses(index, stages, "appworld", out_dir),
            _losses(index, stages, "planbench_xl", out_dir),
            _resource_panels(metrics, out_dir),
            _resource_panels(metrics, out_dir, timing=True)]
