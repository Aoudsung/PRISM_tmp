#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-grade experimental figure generator for our ACL 2027 paper:
SPAR: Breaking Information Self-Locking in Long-Horizon Agents via
Stratified Probing with Aligned Reweighting.

Vector PDF and 300-dpi PNG figures with compact, print-oriented annotations.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ============================================================================
# 1. Global Typography, Chrome & Print Palette Configuration
# ============================================================================
ACL_RCPARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.titlesize": 8.4,
    "axes.titleweight": "bold",
    "axes.titlepad": 7.5,
    "axes.labelsize": 7.8,
    "axes.labelpad": 5.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.4,
    "legend.labelspacing": 0.62,
    "figure.titlesize": 9.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#94A3B8",
    "axes.labelcolor": "#0F172A",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": "#E2E8EE",
    "grid.linestyle": "-",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.85,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.dpi": 300,
}
mpl.rcParams.update(ACL_RCPARAMS)

PALETTE = {
    "SPAR":                "#005F73",  # Deep Royal Teal (Ours - Primary Focal)
    "SPAR_Accent":         "#0A9396",  # Bright Teal (Ours - Secondary)
    "Branch_Probe":        "#007796",  # Mode I (Discover / Probe)
    "Branch_Exec":         "#D97706",  # Mode E (Execute / Submit - Warm Amber)
    "Ablation_Unweighted": "#C83E4D",  # Coral Crimson (Over-probing warning)
    "Step_DPO":            "#6D597A",  # Muted Amethyst (Same-prefix preference)
    "PoS_EnvProbe":        "#4D7C6F",  # Sage Olive (Test-time probing)
    "WorldEvolver":        "#3B7097",  # Steel Blue (World model memory)
    "InfoPO":              "#6BA3BE",  # Soft Sky Slate (Intrinsic info gain)
    "GiGPO":               "#64748B",  # Cool Slate (Group on-policy RL)
    "AREW":                "#94A3B8",  # Light Slate (Advantage reweighting)
    "Frozen_Base":         "#CBD5E1",  # Neutral Grey (Untrained baseline)
    "Frozen_Line":         "#64748B",  # Darker grey for reference dashed lines
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)


def save_both(fig, stem):
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path} and {png_path}")


# ============================================================================
# FIGURE 1: Conceptual & Quantitative Framework Overview (Double-Column Teaser)
# ============================================================================
def plot_fig1_overview():
    # Full-bleed axes on a 7.8 x 3.05 inch canvas (561.6 x 219.6 pt)
    fig = plt.figure(figsize=(7.8, 3.05))
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def draw_box(x, y, w, h, fc, ec, lw=1.0, ls="-", radius=1.0, alpha=1.0, zorder=2):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.05,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
            alpha=alpha, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color="#334155", lw=1.2, style="-|>", rad=0.0, ls="-", zorder=4):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style, mutation_scale=9.5,
            color=color, linewidth=lw, linestyle=ls, zorder=zorder
        )
        ax.add_patch(arr)

    # --- THREE MAIN ZONE CONTAINERS ---
    # Zone (a): Left (x=0.5..24.5, width=24.0)
    draw_box(0.5, 1.0, 24.0, 40.0, "#F8FAFC", "#CBD5E1", lw=0.9, radius=1.2, zorder=1)
    ax.text(12.5, 39.2, "(a) Context & Cold-Start Starvation", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#0F172A")

    # Zone (b): Center (x=25.8..72.2, width=46.4)
    draw_box(25.8, 1.0, 46.4, 40.0, "#F0F9FA", "#005F73", lw=1.1, radius=1.2, zorder=1)
    ax.text(49.0, 39.2, "(b) Training: Stratified Two-Branch Rollout & Aligned Reweighting",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#005F73")

    # Zone (c): Right (x=73.5..99.5, width=26.0)
    draw_box(73.5, 1.0, 26.0, 40.0, "#F8FAFC", "#CBD5E1", lw=0.9, radius=1.2, zorder=1)
    ax.text(86.5, 39.2, "(c) Test-Time Single-Stream Policy", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#0F172A")

    # --- ZONE (a) CONTENT ---
    # Public History box
    draw_box(1.6, 25.8, 21.8, 11.2, "#FFFFFF", "#64748B", lw=0.85, radius=0.9)
    ax.text(12.5, 34.8, r"Public History $h_T$ & Budget $B_T$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#1E293B")
    ax.text(12.5, 31.3, "Task: Update 'Alex Smith' in CRM", ha="center", va="center",
            fontsize=7.0, color="#334155")
    ax.text(12.5, 28.0, "State: 3 ambiguous IDs in directory", ha="center", va="center",
            fontsize=6.9, color="#B45309", fontweight="bold")

    # Standard On-Policy Rollout Starvation Box
    draw_box(1.6, 2.2, 21.8, 21.4, "#FFF5F5", "#C83E4D", lw=0.85, ls="--", radius=0.9)
    ax.text(12.5, 21.3, "Standard On-Policy Rollout", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#991B1B")
    ax.text(12.5, 17.9, r"Initial probe gate $p(h_T) = 0.02$", ha="center", va="center",
            fontsize=7.0, color="#334155")
    ax.text(12.5, 14.3, r"$P(N_I = 0 \mid h_T) = (1-p)^2 = \mathbf{96.0\%}$",
            ha="center", va="center", fontsize=7.1, color="#C83E4D", fontweight="bold")
    draw_box(2.7, 3.4, 19.6, 7.8, "#FEE2E2", "#C83E4D", lw=0.7, radius=0.7)
    ax.text(12.5, 8.8, r"Both rollouts choose Execute $(E, E)$", ha="center", va="center",
            fontsize=6.7, color="#7F1D1D")
    ax.text(12.5, 5.4, "Zero Contrastive Probe Signal", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color="#991B1B")

    # --- ZONE (b) CONTENT ---
    # Snapshot node
    draw_box(27.0, 16.0, 8.2, 12.0, "#FFFFFF", "#005F73", lw=1.0, radius=0.9)
    ax.text(31.1, 25.4, "Snapshot", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#005F73")
    ax.text(31.1, 21.8, r"$(s_T, h_T)$", ha="center", va="center", fontsize=7.8, color="#0F172A")
    ax.text(31.1, 18.2, "1:1 Split", ha="center", va="center", fontsize=6.9,
            color="#0A9396", fontweight="bold")

    # Arrow from h_T in Zone (a) to Snapshot in Zone (b)
    draw_arrow(23.4, 30.5, 27.0, 23.5, color="#005F73", lw=1.2, rad=-0.12)

    # Top Branch: Information Mode I (x=37.2..54.4, y=23.4..37.0)
    draw_box(37.2, 23.4, 17.2, 13.6, "#E0F4F8", "#007796", lw=1.0, radius=0.9)
    ax.text(45.8, 34.8, r"Information Branch $I$ ($\tau_I$)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#005F73")
    ax.text(45.8, 31.5, r"1. $q \sim \pi_{\theta, I}$: probe directory",
            ha="center", va="center", fontsize=6.8, color="#0F172A")
    ax.text(45.8, 28.5, r"2. Disambiguates ID $\rightarrow$ Execute",
            ha="center", va="center", fontsize=6.8, color="#0F172A")
    draw_box(38.6, 24.3, 14.4, 2.7, "#005F73", "#005F73", lw=0.5, radius=0.5)
    ax.text(45.8, 25.65, r"Terminal Utility $G_I = \mathbf{0.78}$",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#FFFFFF")

    # Bottom Branch: Execution Mode E (x=37.2..54.4, y=7.2..20.8)
    draw_box(37.2, 7.2, 17.2, 13.6, "#FEF3C7", "#D97706", lw=1.0, radius=0.9)
    ax.text(45.8, 18.6, r"Execution Branch $E$ ($\tau_E$)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#B45309")
    ax.text(45.8, 15.3, r"1. $a \sim \pi_{\theta, E}$: direct modify",
            ha="center", va="center", fontsize=6.8, color="#0F172A")
    ax.text(45.8, 12.3, r"2. Wrong ID modified $\rightarrow$ Fails",
            ha="center", va="center", fontsize=6.8, color="#0F172A")
    draw_box(38.6, 8.1, 14.4, 2.7, "#B45309", "#B45309", lw=0.5, radius=0.5)
    ax.text(45.8, 9.45, r"Terminal Utility $G_E = \mathbf{0.00}$",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#FFFFFF")

    # Arrows from Snapshot to Branches
    draw_arrow(35.2, 24.0, 37.2, 30.0, color="#007796", lw=1.25, rad=0.1)
    draw_arrow(35.2, 19.5, 37.2, 14.0, color="#D97706", lw=1.25, rad=-0.1)

    # Aligned Reweighting Box (x=56.2..71.0, y=7.2..37.0)
    draw_box(56.2, 7.2, 14.8, 29.8, "#FFFFFF", "#005F73", lw=1.1, radius=0.9)
    ax.text(63.6, 35.0, "Aligned Gate", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#005F73")
    ax.text(63.6, 32.3, "Reweighting", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#005F73")

    # Three gradient terms inside updater with generous vertical spacing
    draw_box(57.0, 24.6, 13.2, 6.2, "#F0FDF4", "#15803D", lw=0.7, radius=0.6)
    ax.text(63.6, 29.0, "Root Gate Update:", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#166534")
    ax.text(63.6, 26.2, r"$(G_I - G_E)\nabla_\theta p_\theta$", ha="center", va="center", fontsize=7.0, color="#0F172A")

    draw_box(57.0, 17.4, 13.2, 6.2, "#E0F4F8", "#007796", lw=0.7, radius=0.6)
    ax.text(63.6, 21.8, "Mode-I Calibrated:", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#005F73")
    ax.text(63.6, 19.0, r"$p(G_I - G_E)\nabla_\theta \ell_I$", ha="center", va="center", fontsize=7.0, color="#0F172A")

    draw_box(57.0, 10.2, 13.2, 6.2, "#FEF3C7", "#D97706", lw=0.7, radius=0.6)
    ax.text(63.6, 14.6, "Mode-E Calibrated:", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#B45309")
    ax.text(63.6, 11.8, r"$(1-p)(G_E - G_I)\nabla_\theta \ell_E$", ha="center", va="center", fontsize=6.7, color="#0F172A")

    ax.text(63.6, 8.5, "Unbiased (-44.5% Var)", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="#005F73")

    # Arrows from Branches to Updater
    draw_arrow(54.4, 30.0, 56.2, 27.5, color="#007796", lw=1.2)
    draw_arrow(54.4, 14.0, 56.2, 16.5, color="#D97706", lw=1.2)

    # Bottom banner strictly inside Zone (b) (x=27.0..71.0, y=2.1..5.7)
    draw_box(27.0, 2.1, 44.0, 3.6, "#005F73", "#005F73", lw=0.6, radius=0.7)
    ax.text(49.0, 3.9, r"Unified Utility:  $U_B(\tau) = Y(\tau)\cdot (1 - C(\tau)/B)_+$  (Zero Step Rewards)",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#FFFFFF")

    # --- ZONE (c) CONTENT ---
    # Arrow from Updater to Test-Time Policy
    draw_arrow(71.0, 22.0, 74.6, 30.5, color="#005F73", lw=1.35, rad=0.1)

    # Evolved Policy Box (x=74.6..98.4, y=25.8..37.0)
    draw_box(74.6, 25.8, 23.8, 11.2, "#FFFFFF", "#005F73", lw=1.0, radius=0.9)
    ax.text(86.5, 34.8, r"Evolved Policy $\pi_{\theta^*}$ (No Tree)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#005F73")
    ax.text(86.5, 31.3, r"Ambiguous $h_t \Rightarrow p_{\theta^*}=0.84$ (Probe)",
            ha="center", va="center", fontsize=6.7, color="#007796", fontweight="bold")
    ax.text(86.5, 27.9, r"Resolved $h_{t+1} \Rightarrow p_{\theta^*}=0.08$ (Exec)",
            ha="center", va="center", fontsize=6.7, color="#B45309", fontweight="bold")

    # Mini Bar Comparison inside Zone (c) (x=74.6..98.4, y=2.1..23.8)
    draw_box(74.6, 2.1, 23.8, 21.7, "#FFFFFF", "#CBD5E1", lw=0.85, radius=0.9)
    ax.text(86.5, 21.6, r"PlanBench-XL Test $\mathbb{E}[U_B] \uparrow$",
            ha="center", va="center", fontsize=7.3, fontweight="bold", color="#0F172A")

    # Draw 4 horizontal mini-bars for Llama-3.1-8B PlanBench-XL
    methods_mini = ["Frozen", "GiGPO", "Step-DPO", "SPAR"]
    vals_mini = [0.268, 0.342, 0.561, 0.668]
    cols_mini = [PALETTE["Frozen_Base"], PALETTE["GiGPO"], PALETTE["Step_DPO"], PALETTE["SPAR"]]
    y_positions = [5.0, 9.2, 13.4, 17.6]

    for m_label, val, col, yp in zip(methods_mini, vals_mini, cols_mini, y_positions):
        bar_w = val * 11.2  # max ~7.48 units
        rect = mpatches.Rectangle((82.2, yp - 1.25), bar_w, 2.5, facecolor=col, edgecolor="none", zorder=3)
        ax.add_patch(rect)
        weight = "bold" if "SPAR" in m_label else "normal"
        t_col = "#005F73" if "SPAR" in m_label else "#334155"
        ax.text(81.6, yp, m_label, ha="right", va="center", fontsize=6.7, fontweight=weight, color=t_col)
        val_str = f"{val:.3f}" if "SPAR" not in m_label else f"{val:.3f} (+0.40)"
        ax.text(82.6 + bar_w, yp, val_str, ha="left", va="center", fontsize=6.5, fontweight=weight, color=t_col)

    save_both(fig, "fig1_spar_overview")


# ============================================================================
# FIGURE 2: Sampling Coverage, Variance Dominance & Cross-Stage Evolution (1x3)
# ============================================================================
def plot_fig2_coverage_and_evolution():
    fig, axes = plt.subplots(
        1, 3, figsize=(7.6, 2.92),
        gridspec_kw={"width_ratios": [1.02, 1.0, 1.0], "wspace": 0.34}
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.18, top=0.70)

    # --- Panel (a): Contrastive Pair Coverage vs. p(h) ---
    ax0 = axes[0]
    p_grid = np.linspace(0.01, 0.50, 200)
    cov_strat = np.ones_like(p_grid) * 100.0
    cov_rloo = 2.0 * p_grid * (1.0 - p_grid) * 100.0
    starvation_rloo = ((1.0 - p_grid) ** 2) * 100.0

    ax0.plot(p_grid, cov_strat, color=PALETTE["SPAR"], lw=2.1, label="SPAR Stratified (1:1)")
    ax0.plot(p_grid, cov_rloo, color=PALETTE["GiGPO"], lw=1.7, ls="--", label="On-Policy RLOO ($n=2$)")
    ax0.plot(p_grid, starvation_rloo, color=PALETTE["Ablation_Unweighted"], lw=1.6, ls=":", label=r"Zero-Probe Miss $(1-p)^2$")

    ax0.fill_between(p_grid, cov_rloo, cov_strat, color=PALETTE["SPAR_Accent"], alpha=0.07)

    # State the cold-start coordinate explicitly; no leader crosses the curves.
    ax0.scatter([0.02], [96.04], color=PALETTE["Ablation_Unweighted"], s=24, zorder=5)
    ax0.scatter([0.02], [100.0], color=PALETTE["SPAR"], s=24, zorder=5)
    ax0.text(
        0.96, 0.055, "Cold start ($p=0.02$)\n96.0% zero-probe miss",
        transform=ax0.transAxes, ha="right", va="bottom", linespacing=1.35,
        fontsize=6.6, color="#334155"
    )
    ax0.annotate("100% exact coverage", xy=(0.49, 100), xytext=(0, 4),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.6, fontweight="bold", color="#0F172A")

    ax0.set_title(r"(a) Coverage vs. Gate $p(h)$")
    ax0.set_xlabel(r"Initial Probe Gate $p(h)$")
    ax0.set_ylabel("Episode Proportion (%)")
    ax0.set_xlim(0.0, 0.50)
    ax0.set_ylim(0, 109)
    ax0.grid(True)

    # Panel (a) legend placed cleanly in top-left header band above Panel (a)
    h0, l0 = ax0.get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper left", bbox_to_anchor=(0.04, 0.99),
               ncol=1, fontsize=6.9, labelspacing=0.45, handlelength=1.6)

    # --- Panel (b): AppWorld Cross-Stage Self-Evolution (Qwen2.5-7B Dev E[U_B]) ---
    ax1 = axes[1]
    stages = np.array([0, 1, 2, 3])
    app_curves = {
        "SPAR (Ours)":         ([0.2723, 0.4000, 0.5284, 0.5967], PALETTE["SPAR"], "-", "o", 2.1, 4.8),
        "Ablation-Unweighted": ([0.2723, 0.3326, 0.4394, 0.5895], PALETTE["Ablation_Unweighted"], "--", "^", 1.6, 4.2),
        "Step-DPO":            ([0.2723, 0.4118, 0.4768, 0.5145], PALETTE["Step_DPO"], "-.", "v", 1.5, 4.0),
        "PoS+EnvProbe":        ([0.2723, 0.4800, 0.4800, 0.4800], PALETTE["PoS_EnvProbe"], "-", "s", 1.4, 3.8),
        "InfoPO":              ([0.2723, 0.2995, 0.3518, 0.3882], PALETTE["InfoPO"], "--", "D", 1.4, 3.8),
        "GiGPO":               ([0.2723, 0.2836, 0.3036, 0.3154], PALETTE["GiGPO"], ":", "x", 1.4, 4.0),
    }
    ax1.axhline(0.2723, color=PALETTE["Frozen_Line"], ls="--", lw=1.0, alpha=0.8, label="Frozen Base")
    for label, (vals, col, ls, mk, lw, ms) in app_curves.items():
        vals_arr = np.array(vals)
        ax1.plot(stages, vals_arr, color=col, ls=ls, marker=mk, lw=lw, markersize=ms, label=label)
        if label == "SPAR (Ours)":
            ax1.fill_between(stages, vals_arr - 0.016, vals_arr + 0.016, color=col, alpha=0.12)

    # Offset in points keeps the endpoint label clear at both export resolutions.
    ax1.annotate(
        "0.597 (+0.324)", xy=(3, 0.5967), xytext=(-2, 6),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=6.9, fontweight="bold", color="#0F172A"
    )

    ax1.set_title("(b) AppWorld Self-Evolution")
    ax1.set_xlabel("Self-Evolution Stage $k$")
    ax1.set_ylabel(r"Budget-Aware Utility $\mathbb{E}[U_B] \uparrow$")
    ax1.set_xticks(stages)
    ax1.set_xticklabels(["Stage 0", "Stage 1", "Stage 2", "Stage 3"])
    ax1.set_ylim(0.22, 0.68)
    ax1.grid(True)

    # --- Panel (c): PlanBench-XL Cross-Stage Self-Evolution (Llama-3.1-8B Dev E[U_B]) ---
    ax2 = axes[2]
    pb_curves = {
        "SPAR (Ours)":         ([0.2232, 0.3309, 0.5823, 0.6765], PALETTE["SPAR"], "-", "o", 2.1, 4.8),
        "Ablation-Unweighted": ([0.2232, 0.2751, 0.4430, 0.5911], PALETTE["Ablation_Unweighted"], "--", "^", 1.6, 4.2),
        "PoS+EnvProbe":        ([0.2232, 0.5690, 0.5690, 0.5690], PALETTE["PoS_EnvProbe"], "-", "s", 1.4, 3.8),
        "Step-DPO":            ([0.2232, 0.3554, 0.4513, 0.5591], PALETTE["Step_DPO"], "-.", "v", 1.5, 4.0),
        "InfoPO":              ([0.2232, 0.2554, 0.3191, 0.3668], PALETTE["InfoPO"], "--", "D", 1.4, 3.8),
        "GiGPO":               ([0.2232, 0.2232, 0.2477, 0.2554], PALETTE["GiGPO"], ":", "x", 1.4, 4.0),
    }
    ax2.axhline(0.2232, color=PALETTE["Frozen_Line"], ls="--", lw=1.0, alpha=0.8)
    for label, (vals, col, ls, mk, lw, ms) in pb_curves.items():
        vals_arr = np.array(vals)
        ax2.plot(stages, vals_arr, color=col, ls=ls, marker=mk, lw=lw, markersize=ms, label=label)
        if label == "SPAR (Ours)":
            ax2.fill_between(stages, vals_arr - 0.015, vals_arr + 0.015, color=col, alpha=0.12)

    ax2.annotate(
        "0.677 (+0.453)", xy=(3, 0.6765), xytext=(-2, 6),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=6.9, fontweight="bold", color="#0F172A"
    )

    ax2.set_title("(c) PlanBench-XL Self-Evolution")
    ax2.set_xlabel("Self-Evolution Stage $k$")
    ax2.set_ylabel(r"Budget-Aware Utility $\mathbb{E}[U_B] \uparrow$")
    ax2.set_xticks(stages)
    ax2.set_xticklabels(["Stage 0", "Stage 1", "Stage 2", "Stage 3"])
    ax2.set_ylim(0.18, 0.74)
    ax2.grid(True)

    # Shared top legend for Panels (b) and (c) in top-right header band
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.99),
               ncol=4, fontsize=7.0, columnspacing=0.95, handletextpad=0.35, labelspacing=0.55)

    save_both(fig, "fig2_coverage_and_evolution")


# ============================================================================
# FIGURE 3: Multi-Backbone Main Benchmark Comparison with +Delta Callouts
# ============================================================================
def plot_fig3_multibackbone_main():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 2.85), sharey=True,
        gridspec_kw={"wspace": 0.12}
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.76)

    # Concise, generous x-tick labels with linespacing=1.35 so they never collide
    backbones = [
        "Qwen2.5-7B\n(7.6B)",
        "Llama-3.1-8B\n(8.0B)",
        "Qwen3.8-27B\n(27.0B)",
    ]
    x = np.arange(len(backbones))
    width = 0.105

    methods = [
        ("Frozen Base",         PALETTE["Frozen_Base"]),
        ("GiGPO",               PALETTE["GiGPO"]),
        ("InfoPO",              PALETTE["InfoPO"]),
        ("WorldEvolver",        PALETTE["WorldEvolver"]),
        ("PoS+EnvProbe",        PALETTE["PoS_EnvProbe"]),
        ("Step-DPO",            PALETTE["Step_DPO"]),
        ("Ablation-Unweighted", PALETTE["Ablation_Unweighted"]),
        ("SPAR (Ours)",         PALETTE["SPAR"]),
    ]

    appworld_data = {
        "Frozen Base":         ([0.2583, 0.2720, 0.3341], [0.0109, 0.0121, 0.0101]),
        "GiGPO":               ([0.3094, 0.3295, 0.3952], [0.0299, 0.0268, 0.0071]),
        "InfoPO":              ([0.3634, 0.3955, 0.5268], [0.0289, 0.0035, 0.0676]),
        "WorldEvolver":        ([0.3636, 0.3827, 0.4487], [0.0188, 0.0052, 0.0081]),
        "PoS+EnvProbe":        ([0.5112, 0.5583, 0.6649], [0.0021, 0.0059, 0.0104]),
        "Step-DPO":            ([0.5583, 0.5871, 0.6770], [0.0713, 0.0636, 0.0398]),
        "Ablation-Unweighted": ([0.5842, 0.5279, 0.6484], [0.1161, 0.1105, 0.0432]),
        "SPAR (Ours)":         ([0.6130, 0.6279, 0.6948], [0.1138, 0.1102, 0.0091]),
    }

    planbench_data = {
        "Frozen Base":         ([0.2633, 0.2682, 0.3290], [0.0304, 0.0338, 0.0429]),
        "GiGPO":               ([0.3163, 0.3419, 0.4070], [0.0395, 0.0558, 0.0395]),
        "InfoPO":              ([0.3975, 0.4163, 0.5279], [0.0447, 0.0449, 0.0689]),
        "WorldEvolver":        ([0.4441, 0.4541, 0.5227], [0.0253, 0.0347, 0.0228]),
        "PoS+EnvProbe":        ([0.4921, 0.5627, 0.6630], [0.0234, 0.0224, 0.0081]),
        "Step-DPO":            ([0.5356, 0.5607, 0.6643], [0.0730, 0.0658, 0.0358]),
        "Ablation-Unweighted": ([0.5932, 0.5939, 0.6386], [0.0605, 0.0828, 0.0444]),
        "SPAR (Ours)":         ([0.6243, 0.6677, 0.6847], [0.0470, 0.0197, 0.0130]),
    }

    n_m = len(methods)
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    for ax, data_dict, title_str in [
        (axes[0], appworld_data, "(a) AppWorld Held-Out Test (168 Tasks)"),
        (axes[1], planbench_data, "(b) PlanBench-XL Retail Test (90 Queries)"),
    ]:
        base_means = data_dict["Frozen Base"][0]
        for idx, (m_name, col) in enumerate(methods):
            means, stds = data_dict[m_name]
            bars = ax.bar(
                x + offsets[idx], means, width * 0.90,
                yerr=stds, capsize=1.6,
                color=col, edgecolor="none",
                error_kw=dict(lw=0.55, ecolor="#64748B", capthick=0.55),
                label=m_name, zorder=3
            )
            if m_name == "SPAR (Ours)":
                for b_idx, bar in enumerate(bars):
                    val = means[b_idx]
                    delta = val - base_means[b_idx]
                    adj_top = max(
                        data_dict[m][0][b_idx] + data_dict[m][1][b_idx]
                        for m, _ in methods[-3:]
                    )
                    top_y = adj_top + 0.022
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0, top_y,
                        f"{val:.3f}\n(+{delta:.2f})",
                        ha="center", va="bottom", fontsize=6.5, linespacing=1.25,
                        fontweight="bold", color="#0F172A"
                    )

        ax.set_title(title_str)
        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=7.6, linespacing=1.35)
        ax.set_ylim(0.0, 0.92)
        ax.grid(True, axis="y")

    axes[0].set_ylabel(r"Normalized Success-Over-Time $\mathbb{E}[U_B] \uparrow$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.99),
               ncol=4, fontsize=7.3, columnspacing=1.2, handletextpad=0.4, labelspacing=0.55)

    save_both(fig, "fig3_multibackbone_main_comparison")


# ============================================================================
# FIGURE 4: Aligned Reweighting vs. Over-Probing, Budget & Horizon Selectivity
# ============================================================================
def plot_fig4_reweighting_and_budget():
    # Use a 2-row synchronized stacked panel for (a) so NO twinx() right-axis collides with Panel (b)!
    fig = plt.figure(figsize=(7.6, 2.95))
    outer_gs = GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.12, 1.0, 1.0],
        wspace=0.33, left=0.07, right=0.985, bottom=0.19, top=0.71
    )

    # --- Panel (a): Synchronized Top (Utility E[U_B]) & Bottom (Interaction Cost E[C]/B) ---
    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], height_ratios=[1.35, 1.0], hspace=0.16)
    ax0_top = fig.add_subplot(gs_a[0])
    ax0_bot = fig.add_subplot(gs_a[1], sharex=ax0_top)

    groups = ["Qwen\nAppW", "Llama\nAppW", "Qwen\nPB-XL", "Llama\nPB-XL"]
    gx = np.arange(len(groups))
    bw = 0.25

    u_stepdpo = [0.5583, 0.5871, 0.5356, 0.5607]
    u_unw     = [0.5842, 0.5279, 0.5932, 0.5939]
    u_spar    = [0.6130, 0.6279, 0.6243, 0.6677]

    c_stepdpo = [0.3412, 0.3450, 0.3420, 0.3390]
    c_unw     = [0.3582, 0.3610, 0.3572, 0.3536]
    c_spar    = [0.3295, 0.3235, 0.3309, 0.3208]

    ax0_top.bar(gx - bw, u_stepdpo, bw * 0.90, color=PALETTE["Step_DPO"], label="Step-DPO")
    ax0_top.bar(gx,      u_unw,     bw * 0.90, color=PALETTE["Ablation_Unweighted"], label="w/o Gate Weight")
    bars_spar = ax0_top.bar(gx + bw, u_spar, bw * 0.90, color=PALETTE["SPAR"], label="SPAR (Ours)")

    gains_vs_unw = ["+2.9", "+10.0", "+3.1", "+7.4"]
    for idx, bar in enumerate(bars_spar):
        ax0_top.text(
            bar.get_x() + bar.get_width() / 2.0, u_spar[idx] + 0.015,
            gains_vs_unw[idx], ha="center", va="bottom",
            fontsize=6.4, fontweight="bold", color="#0F172A"
        )

    ax0_top.set_title("(a) Reweighting vs. Over-Probing")
    ax0_top.set_ylabel(r"Utility $\mathbb{E}[U_B] \uparrow$", fontsize=7.6)
    ax0_top.set_ylim(0.42, 0.76)
    ax0_top.text(0.02, 0.97, "Gain vs. w/o Gate Weight (pts)",
                 transform=ax0_top.transAxes, va="top", fontsize=5.9, color="#64748B")
    ax0_top.grid(True, axis="y")
    plt.setp(ax0_top.get_xticklabels(), visible=False)

    # Bottom sub-panel: Interaction Cost E[C]/B (lower is better)
    ax0_bot.bar(gx - bw, c_stepdpo, bw * 0.90, color=PALETTE["Step_DPO"])
    ax0_bot.bar(gx,      c_unw,     bw * 0.90, color=PALETTE["Ablation_Unweighted"])
    bars_c_spar = ax0_bot.bar(gx + bw, c_spar, bw * 0.90, color=PALETTE["SPAR"])

    cost_drops = ["-8.0%", "-10.4%", "-7.4%", "-9.3%"]
    for idx, bar in enumerate(bars_c_spar):
        bracket_y = max(c_unw[idx], c_spar[idx]) + 0.007
        ax0_bot.plot([gx[idx], gx[idx], gx[idx] + bw, gx[idx] + bw],
                     [bracket_y - 0.003, bracket_y, bracket_y, bracket_y - 0.003],
                     color="#64748B", lw=0.6)
        ax0_bot.text(
            gx[idx] + bw / 2.0, bracket_y + 0.003,
            cost_drops[idx], ha="center", va="bottom",
            fontsize=6.2, fontweight="bold", color="#0F172A"
        )

    ax0_bot.set_ylabel(r"Cost $\mathbb{E}[C]/B \downarrow$", fontsize=7.6)
    ax0_bot.set_ylim(0.26, 0.405)
    ax0_bot.set_yticks([0.28, 0.32, 0.36, 0.40])
    ax0_bot.set_xticks(gx)
    ax0_bot.set_xticklabels(groups, fontsize=7.1, linespacing=1.28)
    ax0_bot.grid(True, axis="y")

    # Legend for Panel (a) in top-left header band
    ha, la = ax0_top.get_legend_handles_labels()
    fig.legend(ha, la, loc="upper left", bbox_to_anchor=(0.05, 0.98),
               ncol=1, fontsize=6.9, labelspacing=0.45, handlelength=1.4)

    # --- Panel (b): Budget Sensitivity B in {0.6 B_0, 1.0 B_0, 1.4 B_0} ---
    ax1 = fig.add_subplot(outer_gs[1])
    budgets = np.array([0.6, 1.0, 1.4])
    b_curves = [
        ("SPAR (Ours)",        [0.4290, 0.6130, 0.6918], PALETTE["SPAR"],                "-",  "o", 2.1),
        ("w/o Gate Weight",    [0.3678, 0.5374, 0.6094], PALETTE["Ablation_Unweighted"], "--", "^", 1.6),
        ("Step-DPO",           [0.3891, 0.5422, 0.6079], PALETTE["Step_DPO"],            "-.", "v", 1.5),
        ("PoS+EnvProbe",       [0.3602, 0.5062, 0.5688], PALETTE["PoS_EnvProbe"],        "-",  "s", 1.4),
        ("InfoPO",             [0.3121, 0.4282, 0.4779], PALETTE["InfoPO"],              "--", "D", 1.4),
        ("GiGPO",              [0.2567, 0.3482, 0.3874], PALETTE["GiGPO"],               ":",  "x", 1.4),
        ("Frozen Base",        [0.2144, 0.2877, 0.3191], PALETTE["Frozen_Line"],         "--", ".", 1.2),
    ]
    for label, vals, col, ls, mk, lw in b_curves:
        ax1.plot(budgets, vals, color=col, ls=ls, marker=mk, lw=lw, markersize=4.3, label=label)

    # A local difference bracket replaces the diagonal arrow onto one series.
    ax1.plot([0.565, 0.55, 0.55, 0.565], [0.3678, 0.3678, 0.4290, 0.4290],
             color=PALETTE["SPAR"], lw=0.8)
    ax1.text(
        0.04, 0.93, "Tight budget: +6.1 pts\nvs. w/o Gate Weight",
        transform=ax1.transAxes, ha="left", va="top", linespacing=1.35,
        fontsize=6.5, color="#0F172A"
    )

    ax1.set_title(r"(b) Budget Scaling ($B / B_0$)")
    ax1.set_xlabel(r"Budget Multiplier $B / B_0$")
    ax1.set_ylabel(r"Budget-Aware Utility $\mathbb{E}[U_B] \uparrow$")
    ax1.set_xticks(budgets)
    ax1.set_xticklabels(["0.6x\n(Tight)", "1.0x\n(Default)", "1.4x\n(Generous)"], linespacing=1.30)
    ax1.set_xlim(0.50, 1.50)
    ax1.set_ylim(0.18, 0.75)
    ax1.grid(True)

    # Legend for Panel (b) in top-center header band (3 cols -> 3 compact rows)
    hb, lb = ax1.get_legend_handles_labels()
    fig.legend(hb, lb, loc="upper center", bbox_to_anchor=(0.50, 0.99),
               ncol=3, fontsize=6.6, columnspacing=0.75, handletextpad=0.3, labelspacing=0.45)

    # --- Panel (c): Split-Step Horizon Protocol ---
    ax2 = fig.add_subplot(outer_gs[2])
    split_labels = ["Root Split\n($T=0$)", "2-Step\n($T \\leq 1$)", "3-Step\n($T \\leq 2$)"]
    sx = np.arange(len(split_labels))
    sw = 0.32

    qwen_split_u  = [0.6130, 0.4788, 0.3758]
    llama_split_u = [0.6279, 0.5052, 0.4127]
    qwen_duse     = [0.5033, 0.3612, 0.3306]
    llama_duse    = [0.5579, 0.3892, 0.3513]

    b1 = ax2.bar(sx - sw/2, qwen_split_u,  sw * 0.90, color=PALETTE["SPAR_Accent"], label=r"Qwen $\mathbb{E}[U_B]$")
    b2 = ax2.bar(sx + sw/2, llama_split_u, sw * 0.90, color=PALETTE["SPAR"],        label=r"Llama $\mathbb{E}[U_B]$")

    ax2.plot(sx - sw/2, qwen_duse,  color="#D97706", ls="--", marker="o", lw=1.3, ms=4.0, label=r"Qwen $\Delta_{\mathrm{use}}$")
    ax2.plot(sx + sw/2, llama_duse, color="#B45309", ls="-",  marker="s", lw=1.4, ms=4.0, label=r"Llama $\Delta_{\mathrm{use}}$")

    for bar, val in zip(b2, llama_split_u):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.2f}",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#0F172A"
        )

    ax2.set_title("(c) Split-Step Schedule")
    ax2.set_xticks(sx)
    ax2.set_xticklabels(split_labels, fontsize=7.1, linespacing=1.32)
    ax2.set_ylabel(r"Test $\mathbb{E}[U_B]$ & Dev $\Delta_{\mathrm{use}} \uparrow$")
    ax2.set_ylim(0.20, 0.73)
    ax2.grid(True, axis="y")

    # Legend for Panel (c) in top-right header band with generous labelspacing for math brackets
    hc, lc = ax2.get_legend_handles_labels()
    fig.legend(hc, lc, loc="upper right", bbox_to_anchor=(0.985, 0.98),
               ncol=2, fontsize=6.6, columnspacing=0.75, handletextpad=0.3, labelspacing=0.85)

    save_both(fig, "fig4_reweighting_and_budget_selectivity")


# ============================================================================
# FIGURE 5: Counterfactual Feedback Utilization (Delta_use) & Trajectory Breakdown
# ============================================================================
def plot_fig5_feedback_and_breakdown():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 2.90), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.14}
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.19, top=0.76)

    # --- Panel (a): Counterfactual Masked-Feedback Utilization Delta_use Dumbbell ---
    ax0 = axes[0]
    methods_ord = [
        "Frozen Base",
        "AREW / T3",
        "GiGPO",
        "InfoPO",
        "WorldEvolver",
        "w/o Gate Weight",
        "PoS+EnvProbe",
        "Step-DPO",
        "SPAR (Ours)",
    ]
    y_pos = np.arange(len(methods_ord))

    duse_qwen7b   = np.array([0.2042, 0.2333, 0.2625, 0.3208, 0.3694, 0.3912, 0.4083, 0.4958, 0.5033])
    duse_llama8b  = np.array([0.2333, 0.2333, 0.2917, 0.3500, 0.3694, 0.3897, 0.4158, 0.5250, 0.5579])
    duse_qwen27b  = np.array([0.2819, 0.3500, 0.3500, 0.4569, 0.4083, 0.4737, 0.5362, 0.6611, 0.6708])

    for i in range(len(methods_ord)):
        ax0.plot([duse_qwen7b[i], duse_qwen27b[i]], [y_pos[i], y_pos[i]],
                 color="#CBD5E1", lw=2.2, zorder=2)

    ax0.scatter(duse_qwen7b,  y_pos, color=PALETTE["InfoPO"],      s=30, marker="o", label="Qwen2.5-7B", zorder=4)
    ax0.scatter(duse_llama8b, y_pos, color=PALETTE["Branch_Exec"], s=32, marker="s", label="Llama-3.1-8B", zorder=4)
    ax0.scatter(duse_qwen27b, y_pos, color=PALETTE["SPAR"],        s=38, marker="D", label="Qwen3.8-27B", zorder=5)

    ax0.axhspan(7.55, 8.45, color="#E0F4F8", alpha=0.65, zorder=1)
    ax0.axhspan(4.55, 5.45, color="#FEE2E2", alpha=0.45, zorder=1)

    # The range covers all three backbones; the note names the compared rows.
    gains = 100 * np.array([duse_qwen7b[8] - duse_qwen7b[5],
                            duse_llama8b[8] - duse_llama8b[5],
                            duse_qwen27b[8] - duse_qwen27b[5]])
    ax0.text(
        0.46, 1.5, f"+{gains.min():.1f} to +{gains.max():.1f} pts",
        fontsize=7.0, fontweight="bold", color="#0F172A", va="bottom"
    )
    ax0.text(
        0.46, 1.35, "SPAR vs. w/o Gate Weight\nacross all three backbones",
        fontsize=6.1, linespacing=1.4, color="#475569", va="top"
    )

    ax0.set_title(r"(a) Masked-Feedback Gain $\Delta_{\mathrm{use}} \uparrow$")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(methods_ord, fontsize=7.5)
    for ticklabel, m_name in zip(ax0.get_yticklabels(), methods_ord):
        if m_name == "SPAR (Ours)":
            ticklabel.set_fontweight("bold")

    ax0.set_xlabel(r"$\Delta_{\mathrm{use}}(\pi) = \mathbb{E}[U_{\mathrm{visible}} - U_{\mathrm{masked}}]$", labelpad=7.5)
    ax0.set_xlim(0.15, 0.74)
    ax0.set_ylim(-0.65, 8.65)
    ax0.grid(True, axis="x")

    # Legend for Panel (a) in top-left header band
    h0, l0 = ax0.get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper left", bbox_to_anchor=(0.13, 0.98),
               ncol=3, fontsize=7.1, columnspacing=0.9, handletextpad=0.25)

    # --- Panel (b): 100% Horizontal Stacked Bar of Trajectory Outcome Breakdown ---
    ax1 = axes[1]
    cat_1_direct_succ  = np.array([36.7, 42.2, 47.4, 57.0, 64.1, 68.5, 79.6, 63.0, 74.1])
    cat_2_recov_succ   = np.array([ 0.0,  0.0,  0.0,  1.5,  0.0, 13.4,  0.0, 27.0, 24.0])
    cat_3_overprobe    = np.array([ 8.3,  9.8, 11.6, 12.5, 11.9,  8.1,  9.4,  7.5,  1.2])
    cat_4_blind_drift  = 100.0 - (cat_1_direct_succ + cat_2_recov_succ + cat_3_overprobe)

    segments = [
        (cat_1_direct_succ, "#005F73", "Informed 1st-Pass Success", "#FFFFFF"),
        (cat_2_recov_succ,  "#0A9396", "Post-Error Recovery Success", "#FFFFFF"),
        (cat_3_overprobe,   "#F59E0B", "Cost Overrun / Over-Probe", "#0F172A"),
        (cat_4_blind_drift, "#C83E4D", "Uninformed Execution Drift", "#FFFFFF"),
    ]

    left_acc = np.zeros(len(methods_ord))
    bar_h = 0.66
    for seg_vals, seg_col, seg_label, txt_col in segments:
        ax1.barh(y_pos, seg_vals, height=bar_h, left=left_acc,
                 color=seg_col, edgecolor="#FFFFFF", linewidth=0.5, label=seg_label, zorder=3)
        for idx, val in enumerate(seg_vals):
            # Only print inside bar when segment is >= 11.5% wide (~25 pt) so text never overflows segment
            if val >= 11.5:
                cx = left_acc[idx] + val / 2.0
                ax1.text(cx, y_pos[idx], f"{val:.1f}%", ha="center", va="center",
                         fontsize=6.3, fontweight="bold", color=txt_col, zorder=5)
        left_acc += seg_vals

    ax1.set_title("(b) Trajectory Outcome Breakdown (%)")
    ax1.set_yticks(y_pos)
    ax1.tick_params(axis="y", labelleft=False, left=False)
    ax1.set_xlabel("Proportion of Test Episodes (%)", labelpad=7.5)
    ax1.set_xlim(0, 100)
    ax1.grid(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.99),
               ncol=2, fontsize=6.9, columnspacing=0.9, handletextpad=0.35, labelspacing=0.45)

    save_both(fig, "fig5_feedback_utilization_and_recovery")


if __name__ == "__main__":
    plot_fig1_overview()
    plot_fig2_coverage_and_evolution()
    plot_fig3_multibackbone_main()
    plot_fig4_reweighting_and_budget()
    plot_fig5_feedback_and_breakdown()
    print("All 5 ACL 2027 figures regenerated successfully.")
