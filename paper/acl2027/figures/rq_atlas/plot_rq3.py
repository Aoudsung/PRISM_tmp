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

    fig = plt.figure(figsize=(7.6, 2.85))
    gs = fig.add_gridspec(1, 2, left=.075, right=.985, bottom=.19, top=.81,
                          width_ratios=[1.02, 1.28], wspace=.24)
    ax, table_ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    gate = np.linspace(0, .5, 251)
    ax.axvspan(0, .06, color=COLORS["unweighted"], alpha=.07, zorder=0)
    ax.plot(gate, np.full_like(gate, 100), color=COLORS["spar"], lw=2.2, zorder=3)
    ax.plot(gate, 200 * gate * (1-gate), color=RLOO_COLOR, lw=1.7, ls="--", zorder=2)
    ax.plot(gate, 100 * (1-gate)**2, color=COLORS["unweighted"], lw=1.5, ls=":", zorder=2)
    miss = 100 * exact["p_002_no_info_branch_prob"]
    onpol_p002 = 200 * 0.02 * 0.98
    ax.scatter([.02], [miss], color=COLORS["unweighted"], s=24, marker="^", zorder=4)
    ax.scatter([.02], [onpol_p002], color=RLOO_COLOR, s=20, marker="s", zorder=4)
    ax.annotate(f"Zero-I miss ($p=0.02$): {miss:.2f}%", xy=(.02, miss), xytext=(.15, 86),
                arrowprops=dict(arrowstyle="-", color=COLORS["unweighted"], lw=.65),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.92),
                ha="left", va="center", fontsize=6.6, fontweight="bold", color=COLORS["unweighted"])
    ax.annotate(f"On-policy pair: {onpol_p002:.2f}%", xy=(.02, onpol_p002), xytext=(.13, 8),
                arrowprops=dict(arrowstyle="-", color=RLOO_COLOR, lw=.65),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.92),
                ha="left", va="center", fontsize=6.5, color=COLORS["muted"])
    ax.set(xlim=(0, .5), ylim=(0, 109), xlabel="Initial information-mode probability $p$",
           ylabel="Pair coverage / miss probability (%)")
    ax.set_title("(a) Sampling coverage")
    ax.grid(axis="y")
    handles = [Line2D([0], [0], color=COLORS["spar"], lw=2.1, label="Stratified pair (100%)"),
               Line2D([0], [0], color=RLOO_COLOR, lw=1.7, ls="--", marker="s", ms=3.2, label="On-policy pair"),
               Line2D([0], [0], color=COLORS["unweighted"], lw=1.5, ls=":", marker="^", ms=3.5, label="Zero-I miss")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .995), ncol=3,
               fontsize=6.8, columnspacing=1.1, handlelength=1.5)

    table_ax.axis("off")
    table_ax.set_title(f"(b) Gradient agreement (MC: {mc['num_episodes']:,})")
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
                           colWidths=[.40, .20, .20, .20], cellLoc="right", bbox=[0, 0, 1, .70])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        is_strat = row in (2, 5)
        cell.set_facecolor("#F1F3F5" if row == 0 else ("#E6F3F6" if is_strat else "white"))
        cell.set_text_props(color=COLORS["spar"] if (is_strat and col == 0) else COLORS["ink"],
                            weight="bold" if (row == 0 or is_strat) else "normal")
        if col == 0:
            cell.set_text_props(ha="left")
    for y, lw in [(.70, .7), (.60, .45), (.20, .45), (0, .7)]:
        table_ax.plot([0, 1], [y, y], transform=table_ax.transAxes,
                      color="#94A3B8", lw=lw, clip_on=False)
    for y, name, value in [(.95, r"Objective $J(\theta_0)$", f"{exact['J_exact_fraction']} = {exact['J_exact']:.4f}"),
                           (.82, "Variance identity: LHS = RHS", f"{exact['theorem_5_2_lhs']:.7f}")]:
        table_ax.text(0, y, name, transform=table_ax.transAxes, va="top", fontsize=7)
        table_ax.text(1, y, value, transform=table_ax.transAxes, va="top", ha="right", fontsize=7, fontweight="bold")
    records.append(save_figure(fig, out_dir, "rq3_01_coverage_and_gradients",
        "RQ3  |  Sampling coverage and gradient agreement",
        f"Exact two-stage setting. MC: {mc['num_episodes']:,} episodes. Gradient coordinates follow the stored parameter-vector order.",
        ["metrics/exact_verification"]))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75))
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.81, wspace=.32)
    configs = [
        ("(a) Total gradient variance", "Variance", [exact["var_sum_stratified"], mc["stratified_var_sum"]],
         [exact["var_sum_rloo"], mc["rloo_var_sum"]], ".5f"),
        ("(b) Expected pair cost", "Normalized cost", [exact["cost_stratified"], mc["stratified_mean_pair_cost"]],
         [exact["cost_rloo"], mc["rloo_mean_pair_cost"]], ".4f"),
        ("(c) Variance × cost", "Variance–cost product", [exact["var_cost_product_stratified"], mc["stratified_var_cost_product"]],
         [exact["var_cost_product_rloo"], mc["rloo_var_cost_product"]], ".5f"),
    ]
    x = np.array([0., 1.18])
    width = .33
    for ax, (title, ylabel, strat, rloo, fmt) in zip(axes, configs):
        for offset, values, color in [(-width/2-.02, strat, COLORS["spar"]),
                                      (width/2+.02, rloo, RLOO_COLOR)]:
            bars = ax.bar(x + offset, values, width, color=color, edgecolor="white", lw=.4, zorder=3)
            for bar, value in zip(bars, values):
                ax.annotate(format(value, fmt), (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.2, fontweight="bold" if color == COLORS["spar"] else "normal",
                            color=COLORS["spar"] if color == COLORS["spar"] else COLORS["ink"])
        maximum = max(strat + rloo)
        ax.set(xticks=x, xticklabels=["Exact", "MC (10k)"], ylabel=ylabel,
               ylim=(0, maximum*1.48), xlim=(-.58, 1.76))
        ax.set_title(title, fontsize=8.4)
        ax.grid(axis="y")
        changes = [100*(s/r-1) for s, r in zip(strat, rloo)]
        for xp, s_val, r_val, change in zip(x, strat, rloo, changes):
            y_top = max(s_val, r_val) + maximum * .17
            ax.plot([xp - width/2, xp - width/2, xp + width/2, xp + width/2],
                    [y_top - maximum*.03, y_top, y_top, y_top - maximum*.03],
                    color="#94A3B8", lw=.6, zorder=2)
            ax.text(xp, y_top + maximum*.03, rf"$\Delta$ {change:+.2f}%", ha="center", va="bottom",
                    fontsize=6.6, fontweight="bold",
                    color=COLORS["spar"] if change < 0 else COLORS["muted"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS["spar"], label="Stratified (SPAR)"),
               plt.Rectangle((0, 0), 1, 1, color=RLOO_COLOR, label="Two-sample RLOO")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .995), ncol=2,
               fontsize=6.9, handlelength=1.4)
    records.append(save_figure(fig, out_dir, "rq3_02_variance_cost",
        "RQ3  |  Exact and Monte Carlo variance–cost comparison",
        f"Percentages compare stratified with RLOO (Exact / Monte Carlo); MC uses {mc['num_episodes']:,} episodes.",
        ["metrics/exact_verification"]))
    return records
