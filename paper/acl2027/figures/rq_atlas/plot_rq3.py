"""Exact-enumeration and Monte Carlo views from the saved verification results."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from atlas_common import COLORS, save_figure

RLOO_COLOR = "#64748B"

def build(metrics, comparisons, out_dir):
    verification = metrics["exact_verification"]["values"]
    exact, mc = verification["exact_enumeration"], verification["monte_carlo"]
    records = []

    fig = plt.figure(figsize=(8.6, 4.6))
    gs = fig.add_gridspec(1, 2, left=.075, right=.975, bottom=.20, top=.78,
                          width_ratios=[1, 1.35], wspace=.23)
    ax, table_ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    gate = np.linspace(0, .5, 251)
    ax.plot(gate, np.full_like(gate, 100), color=COLORS["spar"], lw=2.2)
    ax.plot(gate, 200 * gate * (1-gate), color=RLOO_COLOR, lw=1.7, ls="--")
    ax.plot(gate, 100 * (1-gate)**2, color=COLORS["unweighted"], lw=1.7, ls=":")
    miss = 100 * exact["p_002_no_info_branch_prob"]
    ax.scatter([.02], [miss], color=COLORS["unweighted"], s=28, zorder=4)
    ax.text(.48, 10, f"$p=0.02$\n{miss:.2f}% zero-I miss", ha="right", fontsize=8,
            linespacing=1.45, color=COLORS["ink"])
    ax.set(xlim=(0, .5), ylim=(0, 108), xlabel="Initial information-mode probability $p$",
           ylabel="Pair coverage / miss probability (%)")
    ax.set_title("(a) Sampling coverage")
    ax.grid(axis="y")
    handles = [Line2D([0], [0], color=COLORS["spar"], lw=2.1, label="Stratified pair"),
               Line2D([0], [0], color=RLOO_COLOR, lw=1.7, ls="--", label="On-policy pair"),
               Line2D([0], [0], color=COLORS["unweighted"], lw=1.7, ls=":", label="Zero-I miss")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065, .885), ncol=3)

    table_ax.axis("off")
    table_ax.set_title("(b) Gradient agreement")
    rows = [
        ("Analytical", exact["analytical_gradient"]),
        ("Stratified exact", exact["enumerated_stratified_gradient"]),
        ("RLOO exact", exact["enumerated_rloo_gradient"]),
        ("Autograd", exact["pytorch_autograd_gradient"]),
        ("Stratified MC", mc["stratified_mean_grad"]),
        ("RLOO MC", mc["rloo_mean_grad"]),
    ]
    cells = [[name] + [f"{v if v != 0 else 0:+.5f}" for v in values] for name, values in rows]
    table = table_ax.table(cellText=cells, colLabels=["Estimator", "$g_1$", "$g_2$", "$g_3$"],
                           colWidths=[.40, .20, .20, .20], cellLoc="right", bbox=[0, .10, 1, .78])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("#F1F3F5" if row == 0 else ("#E9F3F5" if row >= 5 else "white"))
        cell.set_text_props(color=COLORS["ink"], weight="bold" if row == 0 else "normal")
        if col == 0:
            cell.set_text_props(ha="left")
        if row in (0, 6):
            cell.visible_edges = "B"
            cell.set_linewidth(.55)
            cell.set_edgecolor("#A6B1B8")
    table_ax.text(0, .985, f"Exact objective: {exact['J_exact_fraction']} = {exact['J_exact']:.4f}",
                  transform=table_ax.transAxes, va="top", fontsize=8.2)
    table_ax.text(0, -.035, f"Variance identity: LHS = RHS = {exact['theorem_5_2_lhs']:.7f}",
                  transform=table_ax.transAxes, fontsize=8.1)
    records.append(save_figure(fig, out_dir, "rq3_01_coverage_and_gradients",
        "RQ3  |  Sampling coverage and gradient agreement",
        f"Exact two-stage setting. MC: {mc['num_episodes']:,} episodes. Gradient coordinates follow the stored parameter-vector order.",
        ["metrics/exact_verification"]))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.19, top=.78, wspace=.30)
    configs = [
        ("(a) Total gradient variance", "Variance", [exact["var_sum_stratified"], mc["stratified_var_sum"]],
         [exact["var_sum_rloo"], mc["rloo_var_sum"]], ".5f"),
        ("(b) Expected pair cost", "Normalized cost", [exact["cost_stratified"], mc["stratified_mean_pair_cost"]],
         [exact["cost_rloo"], mc["rloo_mean_pair_cost"]], ".4f"),
        ("(c) Variance × cost", "Variance–cost product", [exact["var_cost_product_stratified"], mc["stratified_var_cost_product"]],
         [exact["var_cost_product_rloo"], mc["rloo_var_cost_product"]], ".5f"),
    ]
    x = np.array([0., 1.15])
    width = .32
    for ax, (title, ylabel, strat, rloo, fmt) in zip(axes, configs):
        for offset, values, color in [(-width/2-.02, strat, COLORS["spar"]),
                                      (width/2+.02, rloo, RLOO_COLOR)]:
            bars = ax.bar(x + offset, values, width, color=color, zorder=3)
            for bar, value in zip(bars, values):
                ax.annotate(format(value, fmt), (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.1)
        maximum = max(strat + rloo)
        ax.set(xticks=x, xticklabels=["Exact", "Monte Carlo"], ylabel=ylabel,
               ylim=(0, maximum*1.43), xlim=(-.55, 1.70))
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y")
        changes = [100*(s/r-1) for s,r in zip(strat, rloo)]
        ax.text(.5, .95, f"{changes[0]:+.2f}%  /  {changes[1]:+.2f}%", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, fontweight="bold")
    handles = [plt.Rectangle((0,0),1,1,color=COLORS["spar"],label="Stratified"),
               plt.Rectangle((0,0),1,1,color=RLOO_COLOR,label="Two-sample RLOO")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065,.89), ncol=2)
    records.append(save_figure(fig, out_dir, "rq3_02_variance_cost",
        "RQ3  |  Exact and Monte Carlo variance–cost comparison",
        f"Percentages compare stratified with RLOO (Exact / Monte Carlo); MC uses {mc['num_episodes']:,} episodes.",
        ["metrics/exact_verification"]))
    return records
