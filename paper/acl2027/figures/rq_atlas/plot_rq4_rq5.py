"""RQ4/RQ5 figures following the approved budget-response reference."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from atlas_common import BENCHMARKS, BENCH_LABELS, COLORS, METHODS, METHOD_LABELS, MODEL_IDS, MODEL_LABELS, main_index, save_figure

SHORT=dict(zip(MODEL_IDS,["Qwen7","Llama8","Qwen27"]))
FOCUS=["prism_evolve","ablation_unweighted","frozen_base"]
STYLES={"prism_evolve":dict(color=COLORS["spar"],marker="o",ls="-",lw=1.8,ms=4),
        "ablation_unweighted":dict(color=COLORS["unweighted"],marker="^",ls="--",lw=1.35,ms=3.8),
        "frozen_base":dict(color=COLORS["context"],marker="s",ls=":",lw=1.1,ms=3.2)}


def legend(fig,other=False,budget=False):
    handles=[Line2D([],[],**STYLES[m],label=METHOD_LABELS[m]) for m in FOCUS]
    if other:handles.append(Line2D([],[],color="#CBD5E1",lw=.7,label="Other methods"))
    if budget:handles.append(Line2D([],[],color="#CBD5E1",lw=.7,ls=(0,(2,3)),label="Default budget"))
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.994,.999),ncol=len(handles),
               fontsize=7.1,columnspacing=1.2,handletextpad=.4)


def white_table(rows,columns,groups,out_dir,stem,title,note,sources,corner="",first=.19,scores=None,height=3.0):
    fig=plt.figure(figsize=(7.6,height));ax=fig.add_axes([.035,.075,.94,.755]);ax.axis("off")
    widths=[first]+[(1-first)/(len(columns)-1)]*(len(columns)-1)
    strings=[[v if not isinstance(v,tuple) else "" for v in row] for row in rows]
    tab=ax.table(cellText=strings,colLabels=columns,cellLoc="center",colLoc="center",colWidths=widths,bbox=[0,0,1,1])
    tab.auto_set_font_size(False)
    best=np.nanmax(np.asarray(scores,dtype=float),axis=0) if scores is not None else None
    varying=np.nanmax(np.asarray(scores,dtype=float),axis=0)!=np.nanmin(np.asarray(scores,dtype=float),axis=0) if scores is not None else None
    for (r,c),cell in tab.get_celld().items():
        cell.set_linewidth(0);cell.PAD=.04;cell.set_facecolor("white");cell.get_text().set_fontsize(6.8)
        cell.get_text().set_color(COLORS["ink"])
        if c==0:cell.get_text().set_ha("left")
        if r==0:cell.set_facecolor("#F1F3F5");cell.get_text().set_fontweight("bold")
        if r>0 and rows[r-1][0]=="SPAR":
            cell.set_facecolor("#E6F3F6")
            if c==0:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color(COLORS["spar"])
        if r>0 and c>0:
            bold=best is not None and varying[c-1] and scores[r-1][c-1]==best[c-1]
            cell.get_text().set_fontweight("bold" if bold else "normal")
            if bold:cell.get_text().set_color(COLORS["spar"])
            value=rows[r-1][c]
            if isinstance(value,tuple):
                x=sum(widths[:c])+.5*widths[c];y=1-(r+.5)/(len(rows)+1)
                ax.text(x-.002,y,f"{value[0]:.4f}",transform=ax.transAxes,ha="right",va="center",
                        fontsize=6.8,fontweight="bold" if bold else "normal",
                        color=COLORS["spar"] if bold else COLORS["ink"])
                ax.text(x+.002,y,f"±{value[1]:.4f}",transform=ax.transAxes,ha="left",va="center",fontsize=5.9,color="#64748B")
    n_r=len(rows)+1
    for y,lw,color in [(1,.75,"#334155"),(0,.75,"#334155"),(1-1/n_r,.5,"#64748B"),
                       (1-2/n_r,.4,"#CBD5E1"),(1-8/n_r,.45,"#CBD5E1"),(1/n_r,.5,"#94A3B8")]:
        ax.plot([0,1],[y,y],transform=ax.transAxes,color=color,lw=lw,clip_on=False)
    if corner:ax.text(0,1.065,corner,transform=ax.transAxes,ha="left",va="bottom",fontsize=6.8)
    block=(1-first)/len(groups)
    for i,g in enumerate(groups):
        ax.text(first+block*(i+.5),1.065,g,transform=ax.transAxes,ha="center",va="bottom",fontsize=8.4,fontweight="bold")
    return save_figure(fig,out_dir,stem,title,note,sources)


def stage_evolution(metrics,comparisons,out_dir):
    idx={(r["model"],r["benchmark"],r["method"],r["stage_k"]):r["values"] for r in metrics["stage_results"]}
    vals=[r["values"]["mean_utility"] for r in metrics["stage_results"] if r["method"]=="prism_evolve"]
    low=max(0,float(np.floor(min(v["mean"]-v["std"] for v in vals)*10)/10))
    high=float(np.ceil(max(v["mean"]+v["std"] for v in vals)*10)/10)+.04
    x=np.arange(4);records=[]
    for bench in BENCHMARKS:
        fig,axes=plt.subplots(1,3,figsize=(7.6,2.75),sharex=True,sharey=True)
        fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.79,wspace=.18)
        for col,(ax,model) in enumerate(zip(axes,MODEL_IDS)):
            for method in [m for m in METHODS if m not in FOCUS]+list(reversed(FOCUS)):
                v=[idx[model,bench,method,int(k)]["mean_utility"] for k in x];means=np.array([r["mean"] for r in v])
                if method in FOCUS:
                    if method in ("prism_evolve","ablation_unweighted"):
                        sd=np.array([r["std"] for r in v])
                        ax.fill_between(x,means-sd,means+sd,color=STYLES[method]["color"],
                                        alpha=.11 if method=="prism_evolve" else .07,lw=0)
                    ax.plot(x,means,**STYLES[method],markeredgecolor="white",markeredgewidth=.35,zorder=4)
                else:ax.plot(x,means,color="#CBD5E1",lw=.65,alpha=.65,zorder=1)
            gain_s3=100*(idx[model,bench,"prism_evolve",3]["mean_utility"]["mean"]-
                         idx[model,bench,"ablation_unweighted",3]["mean_utility"]["mean"])
            ax.text(.04,.93,f"Stage 3 vs. Unw.: +{gain_s3:.2f} pts",transform=ax.transAxes,
                    ha="left",va="top",fontsize=6.5,fontweight="bold",color=COLORS["spar"])
            ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}",pad=6.5);ax.set_xlim(-.12,3.12)
            ax.set_ylim(low,high);ax.set_xticks(x);ax.grid(axis="y");ax.set_xlabel("Training stage $k$")
            if col==0:ax.set_ylabel("Dev utility")
        fig.text(.075,.98,BENCH_LABELS[bench],va="top",fontsize=8.4,fontweight="bold");legend(fig,other=True)
        records.append(save_figure(fig,out_dir,f"rq4_01_stage_evolution_{bench}",f"RQ4 | {BENCH_LABELS[bench]} utility evolution",
            "Three-seed means; SPAR and Unweighted bands are sample SD. All methods also appear in complete stage-value tables.",["metrics.stage_results"]))
    return records


def stage_tables(metrics,comparisons,out_dir):
    idx={(r["model"],r["benchmark"],r["method"],r["stage_k"]):r["values"] for r in metrics["stage_results"]};records=[]
    for bench in BENCHMARKS:
        for pair in [(0,1),(2,3)]:
            settings=[(m,k) for m in MODEL_IDS for k in pair];rows=[];scores=[]
            for method in METHODS:
                vals=[idx[m,bench,method,k]["mean_utility"] for m,k in settings]
                rows.append([METHOD_LABELS[method]]+[(v["mean"],v["std"]) for v in vals]);scores.append([v["mean"] for v in vals])
            records.append(white_table(rows,["Method"]+[f"Stage {k}" for m,k in settings],[MODEL_LABELS[m] for m in MODEL_IDS],
                out_dir,f"rq4_01_stage_values_{bench}_{pair[0]}_{pair[1]}",f"RQ4 | {BENCH_LABELS[bench]} complete stages {pair}",
                "Mean +/- sample SD over three seeds. Column maxima are bold where values differ.",["metrics.stage_results"],
                corner=f"{BENCH_LABELS[bench]}\nMean ± SD",scores=scores))
    return records


def feedback_utilization(metrics,comparisons,out_dir):
    idx=main_index(metrics);records=[];y=np.arange(len(METHODS))
    for bench in BENCHMARKS:
        fig,axes=plt.subplots(1,3,figsize=(7.6,3.1),sharey=True,gridspec_kw={"wspace":.33})
        fig.subplots_adjust(left=.15,right=.94,bottom=.18,top=.79)
        for col,(ax,model) in enumerate(zip(axes,MODEL_IDS)):
            for y_div,ls_div in [(.5,":"),(6.5,"--"),(12.5,"-")]:
                ax.axhline(y_div,color="#CBD5E1",lw=.45,ls=ls_div,zorder=0)
            unw_mean=idx[model,bench,"ablation_unweighted"]["final_dev"]["delta_use"]["mean"]
            ax.axvline(unw_mean,color=COLORS["unweighted"],lw=.65,ls="--",alpha=.5,zorder=1)
            for row,method in enumerate(METHODS):
                v=idx[model,bench,method]["final_dev"]["delta_use"]
                color=COLORS["spar"] if method=="prism_evolve" else COLORS["unweighted"] if method=="ablation_unweighted" else "#94A3B8"
                marker="o" if method=="prism_evolve" else "^" if method=="ablation_unweighted" else "s"
                ax.barh(row,v["mean"],height=.42,color=color,alpha=.13 if method in FOCUS[:2] else .06,edgecolor="none",zorder=1)
                ax.errorbar(v["mean"],row,xerr=v["std"],fmt=marker,color=color,ecolor=color if method in FOCUS[:2] else "#CBD5E1",
                            elinewidth=.75,capsize=1.3,capthick=.65,markersize=3.8 if method in FOCUS[:2] else 2.8,zorder=4)
                ax.text(1.025,row,f"{v['mean']:.4f}",transform=ax.get_yaxis_transform(),ha="left",va="center",
                        fontsize=6.0,fontweight="bold" if method=="prism_evolve" else "normal",
                        color=COLORS["spar"] if method=="prism_evolve" else COLORS["ink"])
            ax.axhspan(12.5,13.5,color="#E6F3F6",alpha=.75,zorder=0)
            ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}",pad=6.5)
            ax.set_yticks(y,[METHOD_LABELS[m] for m in METHODS],fontsize=6.8);ax.set_ylim(-.65,13.65)
            ax.set_xlim(0,.85);ax.set_xticks([0,.4,.8]);ax.set_xlabel(r"$\Delta_{\mathrm{use}}$")
            ax.grid(axis="x");ax.tick_params(axis="y",length=2)
            ax.text(1.025,1.015,"Mean",transform=ax.transAxes,fontsize=6.1,fontweight="bold",ha="left",va="bottom")
        for label in axes[0].get_yticklabels():
            if label.get_text()=="SPAR":
                label.set_fontweight("bold");label.set_color(COLORS["spar"])
        fig.text(.15,.98,BENCH_LABELS[bench],va="top",fontsize=8.4,fontweight="bold")
        fig.legend(handles=[Line2D([],[],color=COLORS["spar"],marker="o",ls="",label="SPAR"),
                            Line2D([],[],color=COLORS["unweighted"],marker="^",ls="--",lw=.7,label="Unweighted"),
                            Line2D([],[],color="#94A3B8",marker="s",ls="",label="Other methods")],
                   loc="upper right",bbox_to_anchor=(.995,.999),ncol=3,fontsize=7.1,columnspacing=1.1,handletextpad=.3)
        records.append(save_figure(fig,out_dir,f"rq4_02_feedback_utilization_{bench}",f"RQ4 | {BENCH_LABELS[bench]} feedback",
            "Fixed 24-task diagnostic; points are means and whiskers sample SD over three seeds.",["metrics.main_results.final_dev.delta_use"]))
    return records


def feedback_values(metrics,comparisons,out_dir):
    idx=main_index(metrics);settings=[(b,m) for b in BENCHMARKS for m in MODEL_IDS];rows=[];scores=[]
    for method in METHODS:
        vals=[idx[m,b,method]["final_dev"]["delta_use"] for b,m in settings]
        rows.append([METHOD_LABELS[method]]+[(v["mean"],v["std"]) for v in vals]);scores.append([v["mean"] for v in vals])
    return white_table(rows,["Method"]+[SHORT[m] for b,m in settings],[BENCH_LABELS[b] for b in BENCHMARKS],
        out_dir,"rq4_02_feedback_values","RQ4 | Complete feedback values",
        "Fixed 24-task diagnostic; mean +/- sample SD over three seeds. Highest means per column are bold.",
        ["metrics.main_results.final_dev.delta_use"],corner=r"$\Delta_{\mathrm{use}}$ mean ± SD",scores=scores)


def feedback_stage_change(metrics,comparisons,out_dir):
    idx={(r["model"],r["benchmark"]):r for r in comparisons["stage_changes"]}
    main_idx=main_index(metrics)
    settings=[(b,m) for b in BENCHMARKS for m in MODEL_IDS]
    fig,ax=plt.subplots(figsize=(7.6,2.3));fig.subplots_adjust(left=.22,right=.97,bottom=.24,top=.80)
    for row,(bench,model) in enumerate(settings):
        start,end=idx[model,bench]["delta_use"]["reference"],idx[model,bench]["delta_use"]["ours"]
        unw_end=main_idx[model,bench,"ablation_unweighted"]["final_dev"]["delta_use"]["mean"]
        ax.plot([start,end],[row,row],color=COLORS["spar"],lw=1.7,alpha=.45,zorder=1)
        ax.scatter([start],[row],s=24,facecolors="white",edgecolors="#64748B",linewidths=.9,zorder=3)
        ax.scatter([unw_end],[row],s=22,marker="^",color=COLORS["unweighted"],edgecolors="white",linewidths=.35,zorder=4)
        ax.scatter([end],[row],s=28,color=COLORS["spar"],edgecolors="white",linewidths=.35,zorder=5)
        ax.annotate(f"{start:.4f}",(start,row),xytext=(-6,0),textcoords="offset points",ha="right",va="center",fontsize=6.5,color="#475569")
        ax.annotate(f"{end:.4f} (+{end-start:.3f})",(end,row),xytext=(6,0),textcoords="offset points",
                    ha="left",va="center",fontsize=6.5,fontweight="bold",color=COLORS["spar"])
    ax.set_yticks(range(6),[f"{SHORT[m]} / {'AppWorld' if b=='appworld' else 'PB-XL'}" for b,m in settings],fontsize=7)
    ax.set_ylim(5.6,-.6);ax.set_xlim(.05,.84);ax.set_xticks([.1,.2,.3,.4,.5,.6,.7,.8]);ax.grid(axis="x")
    ax.set_xlabel(r"Masked-feedback utilization $\Delta_{\mathrm{use}}$");ax.axhline(2.5,color="#CBD5E1",lw=.65,ls="--")
    fig.legend(handles=[Line2D([],[],color="#64748B",marker="o",mfc="white",ls="",label="Stage 0 (Initial)"),
                        Line2D([],[],color=COLORS["unweighted"],marker="^",ls="",label="Unweighted (Stage 3)"),
                        Line2D([],[],color=COLORS["spar"],marker="o",ls="",label="SPAR (Stage 3)")],
               loc="upper center",bbox_to_anchor=(.58,1.0),ncol=3,fontsize=7.0,columnspacing=1.2)
    return save_figure(fig,out_dir,"rq4_02_feedback_stage_change","RQ4 | Feedback stage endpoints",
        "Fixed 24-task diagnostic; three-seed mean endpoints.",["comparisons.stage_changes"])


def trajectory_outcomes(metrics,comparisons,out_dir):
    idx={(r["model"],r["benchmark"],r["method"]):r for r in comparisons["trajectory_outcomes"]}
    categories=[("success_without_recorded_failure",COLORS["spar"],"Success / no failure"),
                ("recovered_success",COLORS["probe"],"Recovered success"),
                ("final_failure","#E2E8F0","Final failure")]
    records=[];y=np.arange(len(METHODS))
    for bench in BENCHMARKS:
        fig,axes=plt.subplots(1,3,figsize=(7.6,3.05),sharex=True,sharey=True)
        fig.subplots_adjust(left=.16,right=.985,bottom=.18,top=.78,wspace=.18)
        for col,(ax,model) in enumerate(zip(axes,MODEL_IDS)):
            for y_div,ls_div in [(.5,":"),(6.5,"--"),(12.5,"-")]:
                ax.axhline(y_div,color="#94A3B8",lw=.45,ls=ls_div,zorder=4)
            rows=[idx[model,bench,method] for method in METHODS];left=np.zeros(len(METHODS))
            for key,color,label in categories:
                widths=np.array([r[key]/r["seed_task_evaluations"]*100 for r in rows])
                ax.barh(y,widths,left=left,height=.68,color=color,edgecolor="white",lw=.55,zorder=2)
                for j,width in enumerate(widths):
                    if width>=11.5:
                        ax.text(left[j]+width/2,j,f"{width:.1f}%",ha="center",va="center",
                                fontsize=5.3 if width<18 else 5.7,
                                fontweight="bold" if (j==len(METHODS)-1 and key!="final_failure") else "normal",
                                color="white" if key!="final_failure" else COLORS["ink"],zorder=3)
                left+=widths
            n=rows[0]["seed_task_evaluations"];ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]} ($N={n}$)",fontsize=8.1)
            ax.set_yticks(y,[METHOD_LABELS[m] for m in METHODS],fontsize=6.7);ax.tick_params(axis="y",length=0)
            ax.set_ylim(-.65,13.65);ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_xlabel("Test evaluations (%)")
            ax.spines["left"].set_visible(False)
        for label in axes[0].get_yticklabels():
            if label.get_text()=="SPAR":
                label.set_fontweight("bold");label.set_color(COLORS["spar"])
        fig.text(.16,.982,BENCH_LABELS[bench],ha="left",va="top",fontsize=7.8,fontweight="bold")
        fig.legend(handles=[Patch(fc=c,label=l) for _,c,l in categories],loc="upper right",bbox_to_anchor=(.99,.995),
                   ncol=3,fontsize=6.8,columnspacing=.9,handletextpad=.3)
        records.append(save_figure(fig,out_dir,f"rq4_03_trajectory_outcomes_{bench}",f"RQ4 | {BENCH_LABELS[bench]} trajectory outcomes",
            "Three-seed pooled counts; successful evaluations without a recorded failure, recovered successes, and final failures.",
            ["comparisons.trajectory_outcomes"]))
    return records


def conditional_recovery(metrics,comparisons,out_dir):
    idx={(r["model"],r["benchmark"],r["method"]):r for r in comparisons["trajectory_outcomes"]}
    settings=[(b,m) for b in BENCHMARKS for m in MODEL_IDS];rows=[];scores=[]
    for method in METHODS:
        vals=[idx[m,b,method] for b,m in settings]
        rows.append([METHOD_LABELS[method]]+[f"{r['recovered_success']}/{r['evaluations_with_recorded_failure']} ({100*r['pooled_recovery_rate']:.1f}%)" if r["evaluations_with_recorded_failure"] else "—" for r in vals])
        scores.append([r["pooled_recovery_rate"] if r["pooled_recovery_rate"] is not None else np.nan for r in vals])
    return white_table(rows,["Method"]+[SHORT[m] for b,m in settings],[BENCH_LABELS[b] for b in BENCHMARKS],out_dir,
        "rq4_04_conditional_recovery","RQ4 | Conditional recovery counts",
        "Recovered successes / evaluations with recorded failure; highest pooled conditional rates are bold.",["comparisons.trajectory_outcomes"],
        corner="Recovered / failure (%)",first=.20,scores=scores,height=2.95)


def budget_response(metrics,comparisons,out_dir):
    idx={(r["model"],r["method"],r["budget"]):r["values"] for r in metrics["budgets"]};budgets=sorted({r["budget"] for r in metrics["budgets"]})
    fig,axes=plt.subplots(2,3,figsize=(7.6,3.25),sharex=True,sharey="row")
    fig.subplots_adjust(left=.075,right=.985,bottom=.185,top=.84,wspace=.18,hspace=.21)
    for col,model in enumerate(MODEL_IDS):
        for row,(field,scale,ylabel) in enumerate([("mean_utility",1,"Dev utility"),("success_rate",100,"Dev success (%)")]):
            ax=axes[row,col];ax.axvline(1.0,color="#CBD5E1",lw=.7,ls=(0,(2,3)),zorder=0);ax.grid(axis="y")
            for method in reversed(FOCUS):
                vals=[idx[model,method,b][field]*scale for b in budgets]
                ax.plot(budgets,vals,**STYLES[method],markeredgecolor="white",markeredgewidth=.4,
                        zorder={"prism_evolve":4,"ablation_unweighted":5,"frozen_base":3}[method])
            ax.set_xlim(.55,1.45);ax.set_xticks(budgets,["0.6×","1.0×","1.4×"])
            ax.tick_params(axis="x",pad=3.5)
            if row==0:
                ax.set_ylim(.18,.87);ax.set_yticks([.2,.4,.6,.8]);ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}",pad=6.5)
                gain=100*(idx[model,"prism_evolve",budgets[0]][field]-idx[model,"ablation_unweighted",budgets[0]][field])
                ax.text(.04,.94,f"Tight budget: +{gain:.2f} pts",transform=ax.transAxes,ha="left",va="top",
                        fontsize=6.8,fontweight="bold",color=COLORS["spar"])
            else:ax.set_ylim(20,103);ax.set_yticks([25,50,75,100])
            if col==0:ax.set_ylabel(ylabel)
    fig.text(.075,.98,"AppWorld",va="top",fontsize=8.4,fontweight="bold");legend(fig,budget=True)
    fig.supxlabel(r"Budget multiplier $B/B_0$",y=.015,fontsize=7.8)
    return save_figure(fig,out_dir,"rq5_01_budget_response","RQ5 | Budget response: focused comparison",
        "AppWorld means; the vertical line is default budget. Tight-budget labels are SPAR minus Unweighted utility x100.",["metrics.budgets"])


def budget_tables(metrics,comparisons,out_dir):
    idx={(r["model"],r["method"],r["budget"]):r["values"] for r in metrics["budgets"]};budgets=sorted({r["budget"] for r in metrics["budgets"]})
    settings=[(m,b) for m in MODEL_IDS for b in budgets];records=[]
    for field,scale,suffix,corner in [("mean_utility",1,"utility","AppWorld dev utility"),("success_rate",100,"success","AppWorld dev success (%)")]:
        scores=[[idx[m,method,b][field]*scale for m,b in settings] for method in METHODS]
        rows=[[METHOD_LABELS[method]]+[format(v,".4f" if field=="mean_utility" else ".2f") for v in vals] for method,vals in zip(METHODS,scores)]
        records.append(white_table(rows,["Method"]+[f"{b:.1f}×" for m,b in settings],[MODEL_LABELS[m] for m in MODEL_IDS],out_dir,
            f"rq5_02_budget_values_{suffix}",f"RQ5 | Complete budget {suffix} values",
            "All 14 methods and three budgets; three-seed means. Highest values per column are bold.",["metrics.budgets"],
            corner=corner,first=.18,scores=scores))
    return records


def split_schedule(metrics,comparisons,out_dir):
    idx={(r["model"],r["schedule"]):r["values"] for r in metrics["split_steps"]};schedules=["fixed_T0","uniform_T0_1","uniform_T0_2"]
    labels=["Root\n$T=0$","Uniform\n$T\\in\\{0,1\\}$","Uniform\n$T\\in\\{0,1,2\\}$"]
    fig,axes=plt.subplots(2,3,figsize=(7.6,3.4),sharex=True,sharey="row")
    fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.86,wspace=.18,hspace=.32)
    for col,model in enumerate(MODEL_IDS):
        for row,(split,field) in enumerate([("final_test","mean_utility"),("final_dev","delta_use")]):
            ax=axes[row,col]
            means=[idx[model,s][split][field]["mean"] for s in schedules]
            ax.axhline(means[0],color=COLORS["spar"],lw=.7,ls=":",alpha=.45,zorder=1)
            ax.plot([0,1],means[:2],color="#94A3B8",lw=.9,ls="--",zorder=2)
            ax.plot([1,2],means[1:],color="#94A3B8",lw=.9,ls="--",zorder=2)
            for x,s in enumerate(schedules):
                v=idx[model,s][split][field];color=COLORS["spar"] if x==0 else "#64748B"
                ax.errorbar(x,v["mean"],yerr=v["std"],fmt="o",color=color,ecolor=color,ms=4.3 if x==0 else 3.6,
                            elinewidth=.85,capsize=2,capthick=.8,zorder=4)
                y_txt = v["mean"]+v["std"]+.045 if x==0 else v["mean"]-v["std"]-.095
                va_txt = "bottom" if x==0 else "top"
                ax.text(x,y_txt,f"{v['mean']:.4f}",ha="center",va=va_txt,fontsize=6.2,
                        fontweight="bold" if x==0 else "normal",color=COLORS["spar"] if x==0 else COLORS["ink"],
                        bbox=dict(boxstyle="round,pad=0.08",fc="white",ec="none",alpha=.88),zorder=5)
            ax.set_xticks(range(3),labels,fontsize=6.4,linespacing=1.85);ax.set_xlim(-.4,2.4);ax.set_ylim(0,.98);ax.set_yticks([0,.4,.8]);ax.grid(axis="y")
            ax.tick_params(axis="x",pad=4.0)
            if row==0:ax.set_title(f"({chr(97+col)}) {MODEL_LABELS[model]}")
            if col==0:ax.set_ylabel("Test utility" if row==0 else r"Dev $\Delta_{\mathrm{use}}$")
    return save_figure(fig,out_dir,"rq5_03_split_schedule","RQ5 | Training split location",
        "AppWorld; point = mean, whisker = sample SD over three seeds. Uniform schedules sample T from the displayed set.",["metrics.split_steps"])


def build(metrics,comparisons,out_dir):
    records=stage_evolution(metrics,comparisons,out_dir)
    records.extend(stage_tables(metrics,comparisons,out_dir));records.extend(feedback_utilization(metrics,comparisons,out_dir))
    records.extend([feedback_values(metrics,comparisons,out_dir),feedback_stage_change(metrics,comparisons,out_dir)])
    records.extend(trajectory_outcomes(metrics,comparisons,out_dir))
    records.extend([conditional_recovery(metrics,comparisons,out_dir),budget_response(metrics,comparisons,out_dir)])
    records.extend(budget_tables(metrics,comparisons,out_dir));records.append(split_schedule(metrics,comparisons,out_dir))
    return records
