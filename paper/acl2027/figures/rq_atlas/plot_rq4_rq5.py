"""Compact RQ4/RQ5 figures following the existing paper figure layouts."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from atlas_common import (
    BENCHMARKS, BENCH_LABELS, COLORS, METHODS, METHOD_COLORS,
    METHOD_LABELS, MODEL_COLORS, MODEL_IDS, MODEL_LABELS, main_index, save_figure,
)

SHORT_MODELS = dict(zip(MODEL_IDS, ["Qwen7", "Llama8", "Qwen27"]))
NAMED = ["prism_evolve", "ablation_unweighted", "step_dpo", "pos_envprobe",
         "infopo", "worldevolver", "gigpo", "arew", "frozen_base"]
STYLES = {"prism_evolve": ("-", "o", 2.1, 4.5),
          "ablation_unweighted": ("--", "^", 1.6, 4.0),
          "step_dpo": ("-.", "v", 1.5, 3.8),
          "pos_envprobe": ("-", "s", 1.4, 3.6),
          "infopo": ("--", "D", 1.4, 3.6),
          "worldevolver": ("-.", "P", 1.25, 3.5),
          "gigpo": (":", "x", 1.35, 3.8),
          "arew": (":", "+", 1.2, 3.8),
          "frozen_base": ("--", None, 1.1, 0)}


def _method_legend(fig):
    handles = [Line2D([], [], color=METHOD_COLORS[m], ls=STYLES[m][0],
                      marker=STYLES[m][1], lw=STYLES[m][2], ms=3.6,
                      label=METHOD_LABELS[m]) for m in NAMED]
    handles.append(Line2D([], [], color="#B7C2CA", lw=.8, label="Other controls"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
               ncol=5, fontsize=6.45, columnspacing=.85, handletextpad=.3,
               handlelength=1.5, labelspacing=.45)


def _draw_curves(ax, series, x, *, show_sd=False):
    for method in [m for m in METHODS if m not in NAMED] + list(reversed(NAMED)):
        means, stds = series[method]
        if method in NAMED:
            ls, marker, lw, ms = STYLES[method]
            color = METHOD_COLORS[method]
            if method=="prism_evolve" and show_sd:
                ax.fill_between(x, np.asarray(means)-stds, np.asarray(means)+stds,
                                color=color, alpha=.10, lw=0)
            ax.plot(x, means, color=color, ls=ls, marker=marker, lw=lw, ms=ms,
                    zorder=5 if method=="prism_evolve" else 3)
        else:
            ax.plot(x, means, color="#B7C2CA", lw=.7, alpha=.65, zorder=1)


def stage_evolution(metrics, comparisons, out_dir):
    stages = {(r["model"], r["benchmark"], r["method"], r["stage_k"]): r["values"]
              for r in metrics["stage_results"]}
    records = []
    x = np.arange(4)
    for benchmark in BENCHMARKS:
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.85), sharey=True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.69, wspace=.23)
        for col, (ax, model) in enumerate(zip(axes, MODEL_IDS)):
            series = {}
            for method in METHODS:
                vals = [stages[model, benchmark, method, int(k)]["mean_utility"] for k in x]
                series[method] = (np.array([v["mean"] for v in vals]), np.array([v["std"] for v in vals]))
            _draw_curves(ax, series, x, show_sd=True)
            ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}")
            ax.set_xticks(x, ["0", "1", "2", "3"])
            ax.set_xlim(-.12, 3.12)
            ax.set_ylim(.18, .84)
            ax.set_yticks([.2, .4, .6, .8])
            ax.grid(True)
            ax.set_xlabel("Self-Evolution Stage $k$")
            if col==0:
                ax.set_ylabel(f"{BENCH_LABELS[benchmark]} dev utility")
            value = series["prism_evolve"][0][-1]
            ax.annotate(f"{value:.4f}", (3, value), xytext=(-2, 6), textcoords="offset points",
                        ha="right", va="bottom", fontsize=6.5, fontweight="bold")
        _method_legend(fig)
        records.append(save_figure(fig, out_dir, f"rq4_01_stage_evolution_{benchmark}",
            f"RQ4 | {BENCH_LABELS[benchmark]} utility across stages",
            "All 14 methods; three-seed mean utility. SPAR band shows sample SD.", ["metrics.stage_results"]))
    return records


def feedback_utilization(metrics, comparisons, out_dir):
    idx = main_index(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True, gridspec_kw={"wspace": .18})
    fig.subplots_adjust(left=.155, right=.985, bottom=.18, top=.79)
    y = np.arange(len(METHODS))
    display = ["SPAR (Ours)" if m=="prism_evolve" else METHOD_LABELS[m] for m in METHODS]
    for panel, (ax, benchmark) in enumerate(zip(axes, BENCHMARKS)):
        arrays = [[idx[model, benchmark, method]["final_dev"]["delta_use"]["mean"]
                   for method in METHODS] for model in MODEL_IDS]
        for row in range(len(METHODS)):
            vals = [a[row] for a in arrays]
            ax.plot([min(vals), max(vals)], [row, row], color="#CBD5E1", lw=1.7, zorder=2)
        for model, values, marker, size in zip(MODEL_IDS, arrays, ["o", "s", "D"], [23, 25, 29]):
            ax.scatter(values, y, color=MODEL_COLORS[model], s=size, marker=marker,
                       label=MODEL_LABELS[model], zorder=4)
        ax.axhspan(12.55, 13.45, color="#E0F4F8", alpha=.65, zorder=1)
        ax.axhspan(11.55, 12.45, color="#FEE2E2", alpha=.45, zorder=1)
        ax.set_title(f"({chr(97+panel)}) {BENCH_LABELS[benchmark]}")
        ax.set_yticks(y, display, fontsize=7.0)
        ax.tick_params(axis="y", length=2.3)
        ax.set_ylim(-.65, len(METHODS)-.35)
        ax.set_xlim(.10, .74)
        ax.set_xticks([.1, .2, .3, .4, .5, .6, .7])
        ax.set_xlabel(r"$\Delta_{\mathrm{use}}=\mathbb{E}[U_{\mathrm{visible}}-U_{\mathrm{masked}}]$")
        ax.grid(axis="x")
        for label in ax.get_yticklabels():
            if label.get_text()=="SPAR (Ours)":
                label.set_fontweight("bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.56, .99), ncol=3,
               fontsize=7.1, columnspacing=1.2, handletextpad=.25)
    return save_figure(fig, out_dir, "rq4_02_feedback_utilization", "RQ4 | Masked-feedback utilization",
                       "Fixed 24-task development diagnostic; three-seed means for all methods. Lines span the three model values.",
                       ["metrics.main_results.final_dev.delta_use"])


def feedback_stage_change(metrics, comparisons, out_dir):
    changes = {(r["model"], r["benchmark"]): r for r in comparisons["stage_changes"]}
    settings = [(b, m) for b in BENCHMARKS for m in MODEL_IDS]
    fig, ax = plt.subplots(figsize=(7.6, 2.1))
    fig.subplots_adjust(left=.22, right=.97, bottom=.24, top=.81)
    for row, (benchmark, model) in enumerate(settings):
        contrast = changes[model, benchmark]["delta_use"]
        start, end = contrast["reference"], contrast["ours"]
        ax.plot([start, end], [row, row], color="#CBD5E1", lw=1.8, zorder=1)
        ax.scatter([start], [row], s=23, facecolors="white", edgecolors="#64748B", linewidths=.8, zorder=3)
        ax.scatter([end], [row], s=25, color=COLORS["spar"], zorder=4)
        ax.annotate(f"{start:.4f}", (start, row), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.6)
        ax.annotate(f"{end:.4f}", (end, row), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.6, fontweight="bold")
    ax.set_yticks(range(6), [f"{SHORT_MODELS[m]} / {'AppWorld' if b=='appworld' else 'PB-XL'}" for b,m in settings], fontsize=7)
    ax.set_ylim(5.55, -.55)
    ax.set_xlim(.06, .78)
    ax.set_xticks([.1, .2, .3, .4, .5, .6, .7])
    ax.grid(axis="x")
    ax.set_xlabel(r"SPAR Masked-Feedback Gain $\Delta_{\mathrm{use}}$")
    ax.axhline(2.5, color="#E2E8EE", lw=.6)
    fig.legend(handles=[Line2D([], [], color="#64748B", marker="o", mfc="white", ls="", label="Stage 0"),
                        Line2D([], [], color=COLORS["spar"], marker="o", ls="", label="Stage 3")],
               loc="upper center", bbox_to_anchor=(.59, 1.0), ncol=2, fontsize=7.1)
    return save_figure(fig, out_dir, "rq4_02_feedback_stage_change", "RQ4 | SPAR feedback utilization, stage 0 to 3",
        "Fixed 24-task development diagnostic; endpoints are three-seed means.", ["comparisons.stage_changes"])


def trajectory_outcomes(metrics, comparisons, out_dir):
    idx = {(r["model"], r["benchmark"], r["method"]): r for r in comparisons["trajectory_outcomes"]}
    categories = [("success_without_recorded_failure", "#005F73", "Success / no failure"),
                  ("recovered_success", "#0A9396", "Recovered success"),
                  ("final_failure", "#C83E4D", "Final failure")]
    records = []
    y = np.arange(len(METHODS))
    for benchmark in BENCHMARKS:
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.95), sharex=True, sharey=True)
        fig.subplots_adjust(left=.16, right=.985, bottom=.18, top=.78, wspace=.18)
        for col, (ax, model) in enumerate(zip(axes, MODEL_IDS)):
            rows = [idx[model, benchmark, method] for method in METHODS]
            left = np.zeros(len(METHODS))
            for key, color, label in categories:
                widths = np.array([r[key]/r["seed_task_evaluations"]*100 for r in rows])
                ax.barh(y, widths, left=left, height=.67, color=color, edgecolor="white", lw=.5)
                for j, width in enumerate(widths):
                    if width>=22:
                        ax.text(left[j]+width/2, j, f"{width:.1f}%", ha="center", va="center",
                                fontsize=5.7, color="white")
                left += widths
            n = rows[0]["seed_task_evaluations"]
            ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]} ($N={n}$)", fontsize=8.1)
            ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS], fontsize=6.7)
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(-.65, 13.65)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.set_xlabel("Test Evaluations (%)")
            ax.spines["left"].set_visible(False)
        fig.text(.16, .982, BENCH_LABELS[benchmark], ha="left", va="top", fontsize=7.8, fontweight="bold")
        fig.legend(handles=[Patch(fc=c, label=l) for _,c,l in categories], loc="upper right",
                   bbox_to_anchor=(.99, .995), ncol=3, fontsize=6.8, columnspacing=.9, handletextpad=.3)
        records.append(save_figure(fig, out_dir, f"rq4_03_trajectory_outcomes_{benchmark}",
            f"RQ4 | {BENCH_LABELS[benchmark]} trajectory outcomes",
            "Counts pooled over three seeds. Success / no failure means successful evaluation without a recorded intermediate failure.",
            ["comparisons.trajectory_outcomes"]))
    return records


def _white_table(fig, rows, columns, group_labels, out_dir, stem, title, note, sources, width_first=.18, corner_label=""):
    ax = fig.add_axes([.035, .075, .94, .755])
    ax.axis("off")
    ncol = len(columns)
    widths = [width_first]+[(1-width_first)/(ncol-1)]*(ncol-1)
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                     colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = .04
        cell.set_facecolor("#FFFFFF")
        cell.get_text().set_fontsize(6.8)
        cell.get_text().set_color(COLORS["ink"])
        if col==0:
            cell.get_text().set_ha("left")
        if row==0:
            cell.set_facecolor("#F1F3F5")
            cell.get_text().set_fontweight("bold")
        if row==len(rows):
            cell.set_facecolor("#E0F4F8")
            cell.get_text().set_fontweight("bold")
    for height, width, color in [(1,.75,"#334155"), (0,.75,"#334155"),
                                 (1-1/(len(rows)+1),.5,"#64748B"),
                                 (1/(len(rows)+1),.45,"#94A3B8")]:
        ax.plot([0,1],[height,height],transform=ax.transAxes,color=color,lw=width,clip_on=False)
    block_width = (1-width_first)/len(group_labels)
    if corner_label:
        ax.text(0, 1.065, corner_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    for col, label in enumerate(group_labels):
        ax.text(width_first+block_width*(col+.5), 1.065, label,
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    return save_figure(fig, out_dir, stem, title, note, sources)


def conditional_recovery(metrics, comparisons, out_dir):
    idx = {(r["model"],r["benchmark"],r["method"]):r for r in comparisons["trajectory_outcomes"]}
    settings = [(b,m) for b in BENCHMARKS for m in MODEL_IDS]
    rows = []
    for method in METHODS:
        counts = [idx[m,b,method] for b,m in settings]
        rows.append([METHOD_LABELS[method]]+[f"{r['recovered_success']}/{r['evaluations_with_recorded_failure']}"
            if r["evaluations_with_recorded_failure"] else "—" for r in counts])
    fig = plt.figure(figsize=(7.6, 2.9))
    return _white_table(fig, rows, ["Method"]+[SHORT_MODELS[m] for b,m in settings],
        [BENCH_LABELS[b] for b in BENCHMARKS], out_dir, "rq4_04_conditional_recovery",
        "RQ4 | Recovery conditional on an intermediate failure",
        "Cells show recovered successes / evaluations with recorded failure, pooled over seeds; dash is zero denominator.",
        ["comparisons.trajectory_outcomes"], width_first=.21, corner_label="Recovered / failure cases")


def budget_scaling(metrics, comparisons, out_dir):
    idx = {(r["model"],r["method"],r["budget"]):r["values"] for r in metrics["budgets"]}
    budgets = sorted({r["budget"] for r in metrics["budgets"]})
    records = []
    for field, scale, suffix in [("mean_utility",1,"utility"),("success_rate",100,"success")]:
        fig, axes = plt.subplots(1,3,figsize=(7.6,2.85),sharey=True)
        fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.69,wspace=.23)
        for col, (ax,model) in enumerate(zip(axes,MODEL_IDS)):
            series = {method:(np.array([idx[model,method,b][field]*scale for b in budgets]),None) for method in METHODS}
            _draw_curves(ax,series,budgets)
            ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}")
            ax.set_xticks(budgets,["0.6x","1.0x","1.4x"])
            ax.set_xlim(.55,1.45)
            ax.set_xlabel(r"Budget Multiplier $B/B_0$")
            ax.grid(True)
            if field=="mean_utility":
                ax.set_ylim(.18,.84)
                ax.set_yticks([.2,.4,.6,.8])
                if col==0:ax.set_ylabel("AppWorld dev utility")
            else:
                ax.set_ylim(20,104)
                ax.set_yticks([25,50,75,100])
                if col==0:ax.set_ylabel("AppWorld dev success (%)")
        _method_legend(fig)
        records.append(save_figure(fig,out_dir,f"rq5_01_budget_scaling_{suffix}",
            f"RQ5 | AppWorld budget {suffix}","All 14 methods; saved three-seed means over development tasks.",["metrics.budgets"]))
    return records


def budget_matrix(metrics, comparisons, out_dir):
    idx = {(r["model"],r["method"],r["budget"]):r["values"] for r in metrics["budgets"]}
    budgets = sorted({r["budget"] for r in metrics["budgets"]})
    settings = [(m,b) for m in MODEL_IDS for b in budgets]
    rows = [[METHOD_LABELS[method]]+[f"{idx[m,method,b]['mean_utility']:.4f}" for m,b in settings] for method in METHODS]
    fig=plt.figure(figsize=(7.6,3.0))
    return _white_table(fig,rows,["Method"]+[f"{b:.1f}x" for m,b in settings],
        [MODEL_LABELS[m] for m in MODEL_IDS],out_dir,"rq5_02_budget_matrix","RQ5 | Complete AppWorld budget utility matrix",
        "AppWorld development set; 3 backbones x 14 methods x 3 budgets. Cells are three-seed mean utility.",
        ["metrics.budgets"],width_first=.18,corner_label="AppWorld dev")


def split_schedule(metrics, comparisons, out_dir):
    idx={(r["model"],r["schedule"]):r["values"] for r in metrics["split_steps"]}
    schedules=["fixed_T0","uniform_T0_1","uniform_T0_2"]
    labels=["Root\n$T=0$","Uniform\n$T\\in\\{0,1\\}$","Uniform\n$T\\in\\{0,1,2\\}$"]
    fig,axes=plt.subplots(2,3,figsize=(7.6,3.2),sharex=True,sharey="row")
    fig.subplots_adjust(left=.075,right=.985,bottom=.21,top=.86,wspace=.23,hspace=.32)
    for col,model in enumerate(MODEL_IDS):
        for row,(split,field) in enumerate([("final_test","mean_utility"),("final_dev","delta_use")]):
            ax=axes[row,col]
            means=[idx[model,s][split][field]["mean"] for s in schedules]
            stds=[idx[model,s][split][field]["std"] for s in schedules]
            ax.bar(range(3),means,width=.56,color=["#005F73","#94A3B8","#CBD5E1"],zorder=3)
            ax.errorbar(range(3),means,yerr=stds,fmt="none",ecolor="#64748B",elinewidth=.65,capsize=1.8,capthick=.65,zorder=4)
            for x,value,sd in zip(range(3),means,stds):
                ax.text(x,value+sd+.03,f"{value:.4f}",ha="center",va="bottom",fontsize=6.2,
                        fontweight="bold" if x==0 else "normal")
            ax.set_xticks(range(3),labels,fontsize=6.4)
            ax.set_ylim(0,.91)
            ax.set_yticks([0,.4,.8])
            ax.grid(axis="y")
            if row==0:ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}")
            if col==0:ax.set_ylabel("Test utility" if row==0 else r"Dev $\Delta_{\mathrm{use}}$")
    return save_figure(fig,out_dir,"rq5_03_split_schedule","RQ5 | Training split location",
        "AppWorld; mean +/- sample SD over three seeds. Uniform schedules sample split location T from the displayed set.",
        ["metrics.split_steps"])


def build(metrics,comparisons,out_dir):
    records=stage_evolution(metrics,comparisons,out_dir)
    records.extend([feedback_utilization(metrics,comparisons,out_dir),feedback_stage_change(metrics,comparisons,out_dir)])
    records.extend(trajectory_outcomes(metrics,comparisons,out_dir))
    records.append(conditional_recovery(metrics,comparisons,out_dir))
    records.extend(budget_scaling(metrics,comparisons,out_dir))
    records.extend([budget_matrix(metrics,comparisons,out_dir),split_schedule(metrics,comparisons,out_dir)])
    return records
