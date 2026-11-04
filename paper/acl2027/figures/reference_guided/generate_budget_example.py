"""Reference-guided RQ5 example; run on mgt01 with the existing evidence export."""

import argparse
import json
from pathlib import Path
import sys

PALETTE = {
    "spar": "#005F73",
    "probe": "#0088A8",
    "execute": "#D97706",
    "unweighted": "#CA3E47",
    "context": "#94A3B8",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--atlas-code", type=Path,
                        default=Path(__file__).resolve().parents[1] / "rq_atlas")
    args = parser.parse_args()
    sys.path.insert(0, str(args.atlas_code))

    # Preserve the paper's established serif typography and thin axes.
    from atlas_common import MODEL_IDS, MODEL_LABELS, save_figure
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metrics = json.loads((args.data / "metrics.json").read_text())
    rows = metrics["budgets"]
    idx = {(r["model"], r["method"], r["budget"]): r["values"] for r in rows}
    budgets = sorted({r["budget"] for r in rows})
    methods = ["prism_evolve", "ablation_unweighted", "frozen_base"]
    styles = {
        "prism_evolve": dict(color=PALETTE["spar"], marker="o", ls="-", lw=1.8, ms=4.0),
        "ablation_unweighted": dict(color=PALETTE["unweighted"], marker="^", ls="--", lw=1.35, ms=3.8),
        "frozen_base": dict(color=PALETTE["context"], marker="s", ls=":", lw=1.1, ms=3.2),
    }
    labels = {"prism_evolve": "SPAR", "ablation_unweighted": "Unweighted", "frozen_base": "Frozen Base"}
    metrics_to_plot = [("mean_utility", 1, "Dev utility"), ("success_rate", 100, "Dev success (%)")]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.1), sharex=True, sharey="row")
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.85, wspace=.18, hspace=.19)
    plotted, tight_gains = [], {}
    for col, model in enumerate(MODEL_IDS):
        for row, (field, scale, ylabel) in enumerate(metrics_to_plot):
            ax = axes[row, col]
            ax.axvline(1.0, color="#CBD5E1", lw=.7, ls=(0, (2, 3)), zorder=0)
            ax.grid(axis="y", color="#E2E8EE", lw=.45)
            for method in reversed(methods):
                values = [idx[model, method, b][field] * scale for b in budgets]
                ax.plot(budgets, values, **styles[method], markeredgecolor="white",
                        markeredgewidth=.4,
                        zorder={"prism_evolve": 4, "ablation_unweighted": 5, "frozen_base": 3}[method])
                plotted.extend(dict(model=model, method=method, budget=b, metric=field,
                                    raw_value=idx[model, method, b][field], display_value=y)
                               for b, y in zip(budgets, values))
            ax.set_xlim(.55, 1.45)
            ax.set_xticks(budgets, ["0.6×", "1.0×", "1.4×"])
            if row == 0:
                ax.set_ylim(.18, .87)
                ax.set_yticks([.2, .4, .6, .8])
                ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}", pad=6.5)
                gain = 100 * (idx[model, "prism_evolve", budgets[0]][field]
                              - idx[model, "ablation_unweighted", budgets[0]][field])
                tight_gains[model] = gain
                ax.text(.04, .94, f"Tight budget: +{gain:.2f} pts", transform=ax.transAxes,
                        ha="left", va="top", fontsize=6.9, color="#334155")
            else:
                ax.set_ylim(20, 103)
                ax.set_yticks([25, 50, 75, 100])
            if col == 0:
                ax.set_ylabel(ylabel)

    fig.text(.075, .98, "AppWorld", va="top", fontsize=8.4, fontweight="bold")
    handles = [Line2D([], [], **styles[m], label=labels[m]) for m in methods]
    handles.append(Line2D([], [], color="#CBD5E1", ls=(0, (2, 3)), lw=.8, label="Default budget"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.994, .999),
               ncol=4, fontsize=7.1, columnspacing=1.2, handletextpad=.4)
    fig.supxlabel(r"Budget multiplier $B/B_0$", y=.06, fontsize=7.8)
    fig.text(.075, .015, "Tight-budget labels: SPAR − Unweighted utility at 0.6× budget (×100).",
             ha="left", fontsize=6.6, color="#475569")
    note = ("AppWorld development results; saved three-seed means. Columns identify backbones; "
            "rows show utility and success with shared scales within each metric. "
            "Labels report SPAR minus Unweighted utility at the lowest budget, multiplied by 100. "
            "The dashed vertical line identifies the default budget; Frozen Base uses its recorded "
            "value at each budget. The complete 14-method results remain in rq_atlas.")
    record = save_figure(fig, args.out, "rq5_budget_response_example",
                         "RQ5 | Budget response: focused comparison", note, ["metrics.budgets"])
    record.update(palette=PALETTE, points=plotted, tight_budget_gains_pts=tight_gains,
                  reference_designs=[
                      {"file": "PlanBench-XL_2606.22388.pdf", "pdf_page": 10, "figure": "4",
                       "use": "explicit budget reference and restrained comparison"},
                      {"file": "2505.13353v5.pdf", "pdf_page": 7, "figure": "2",
                       "use": "shared axes and orthogonal row/column facets"},
                  ])
    (args.out / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"points": len(plotted), "tight_gains_pts": tight_gains}, ensure_ascii=False))


if __name__ == "__main__":
    main()
