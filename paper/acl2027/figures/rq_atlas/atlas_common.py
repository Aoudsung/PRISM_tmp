"""Shared typography, data identities and export helpers for the RQ atlas."""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

MODEL_IDS = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3.8-27B"]
MODEL_LABELS = dict(zip(MODEL_IDS, ["Qwen2.5-7B", "Llama-3.1-8B", "Qwen3.8-27B"]))
BENCHMARKS = ["appworld", "planbench_xl"]
BENCH_LABELS = {"appworld": "AppWorld", "planbench_xl": "PlanBench-XL"}
METHODS = ["frozen_base", "arew", "gigpo", "infopo", "worldevolver", "pos_envprobe", "step_dpo",
           "outcome_pg", "outcome_value_baseline", "outcome_rloo", "info_rl_neighbor",
           "unstratified_2branch", "ablation_unweighted", "prism_evolve"]
METHOD_LABELS = dict(zip(METHODS, ["Frozen Base", "AREW / T3", "GiGPO", "InfoPO", "WorldEvolver",
    "PoS / EnvProbe", "Step-DPO", "Outcome-PG", "Outcome-Value", "Outcome-RLOO", "Info-RL Neighbor",
    "Unstratified", "Unweighted", "SPAR"]))
FOCUS = ["prism_evolve", "ablation_unweighted", "step_dpo"]
COLORS = {"spar": "#005F73", "unweighted": "#CA3E47", "reference": "#64748B",
          "probe": "#0088A8", "execute": "#D97706", "peer": "#6D597A",
          "context": "#94A3B8", "ink": "#0F172A", "muted": "#334155", "grid": "#E2E8EE"}
METHOD_COLORS = {method: "#94A3B8" for method in METHODS}
METHOD_COLORS.update(prism_evolve=COLORS["spar"], ablation_unweighted=COLORS["unweighted"],
                     step_dpo="#6D597A", pos_envprobe=COLORS["probe"], frozen_base="#64748B")
MODEL_COLORS = dict(zip(MODEL_IDS, ["#005F73", "#64748B", "#94A3B8"]))
SEQUENTIAL = LinearSegmentedColormap.from_list("atlas_teal", ["#FFFFFF", "#EEF5F7", "#D7E9EC", "#A9CDD3"])

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.2,
    "axes.titlesize": 8.4, "axes.titleweight": "bold", "axes.titlepad": 7.5,
    "axes.labelsize": 7.8, "axes.labelpad": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#94A3B8", "axes.linewidth": 0.6, "axes.axisbelow": True,
    "axes.labelcolor": COLORS["ink"], "text.color": COLORS["ink"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "grid.color": COLORS["grid"], "grid.linewidth": 0.45,
    "legend.frameon": False, "legend.fontsize": 7.4,
    "legend.handlelength": 1.8, "legend.columnspacing": 1.4,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def load_evidence(directory):
    directory = Path(directory)
    return (json.loads((directory / "metrics.json").read_text()),
            json.loads((directory / "comparisons.json").read_text()))


def main_index(metrics):
    return {(r["model"], r["benchmark"], r["method"]): r for r in metrics["main_results"]}


def heatmap(ax, values, row_labels, column_labels, *, vmin=None, vmax=None,
            fmt=".3f", labels=None, cmap=SEQUENTIAL, textsize=7.2):
    values = np.asarray(values, dtype=float)
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), value in np.ndenumerate(values):
        label = labels[row][col] if labels is not None else (format(value, fmt) if np.isfinite(value) else "—")
        ax.text(col, row, label, ha="center", va="center", fontsize=textsize, color=COLORS["ink"],
                fontweight="bold" if row_labels[row] == "SPAR" else "normal")
    for label in ax.get_yticklabels():
        if label.get_text() == "SPAR":
            label.set_fontweight("bold")
    if "SPAR" in row_labels:
        ax.axhline(list(row_labels).index("SPAR")-.5, color="#94A3B8", lw=.65)
    return im


def colorbar(fig, im, ax, label, *, fraction=.028, pad=.025):
    bar = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad, shrink=.92)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.set_label(label, fontsize=7.5)
    return bar


def save_figure(fig, out_dir, stem, title, note, sources):
    """Match the existing paper figures; title/notes stay in PDF metadata and the index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.title.set_fontsize(8.4)
        ax.xaxis.label.set_fontsize(7.8)
        ax.yaxis.label.set_fontsize(7.8)
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if "\n" in lbl.get_text():
                lbl.set_linespacing(1.85)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.04, metadata={"Title": title, "Subject": note})
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    return {"stem": stem, "title": title, "note": note, "pdf": pdf.name, "png": png.name,
            "source_groups": sources}
