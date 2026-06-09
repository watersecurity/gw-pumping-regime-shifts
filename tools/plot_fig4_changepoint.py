#!/usr/bin/env python3
"""
Fig 4: Per-Agent BOCPD Changepoint Heterogeneity (4x2 grid).

Each panel shows one non-stationary agent with dual y-axes:
  - Left: pumping depth time series
  - Right: BOCPD level-shift and slope-change posterior probabilities
  - Vertical dashed line: BOCPD-detected CP year (highest combined posterior with p >= 0.3)

8 agents in 4x2 layout. Agent 3 excluded (no CP >= 0.3 threshold).

Data sources:
  results/changepoint_probabilities_agents_cluster2_t0.3.csv
  results/changepoint_probabilities_agents_cluster1_t0.3.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "results" / "figures"

AGENT_ORDER = [2, 12, 14, 18, 20, 24, 28, 29]
CLUSTER_MAP = {2: 2, 24: 2, 28: 2, 29: 2, 12: 1, 14: 1, 18: 1, 20: 1}
CLUSTER2_AGENTS = {2, 24, 28, 29}

# BOCPD-detected CP years (highest combined posterior with p >= 0.3)
BOCPD_CP = {
    2: 2013, 24: 2003, 28: 2011, 29: 2011,
    12: 2003, 14: 2004, 18: 2003, 20: 2004,
}

THRESHOLD = 0.3

RC = {
    "font.family": "Arial", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    c2 = pd.read_csv(
        BASE / "results" / "changepoint_probabilities_agents_cluster2_t0.3.csv")
    c1 = pd.read_csv(
        BASE / "results" / "changepoint_probabilities_agents_cluster1_t0.3.csv")

    years = c2["Year"].values

    agent_data = {}
    for aid in CLUSTER2_AGENTS:
        agent_data[aid] = {
            "irr": c2[f"Agent{aid}_MeanIrrigation_mm"].values,
            "level": c2[f"Agent{aid}_CP_Prob"].values,
            "slope": c2[f"Agent{aid}_CP_Prob_Slope"].values,
            "combined": c2[f"Agent{aid}_CP_Prob_Combined"].values,
        }
    for aid in [12, 14, 18, 20]:
        agent_data[aid] = {
            "irr": c1[f"Agent{aid}_MeanIrrigation_mm"].values,
            "level": c1[f"Agent{aid}_CP_Prob"].values,
            "slope": c1[f"Agent{aid}_CP_Prob_Slope"].values,
            "combined": c1[f"Agent{aid}_CP_Prob_Combined"].values,
        }

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(8)]

    for idx, aid in enumerate(AGENT_ORDER):
        row, col = divmod(idx, 2)
        ax_left = axes[row, col]
        d = agent_data[aid]
        cid = CLUSTER_MAP[aid]

        # Pad probabilities if needed (BOCPD returns n-1 values)
        level_probs = d["level"]
        slope_probs = d["slope"]
        combined = d["combined"]
        if len(level_probs) == len(years) - 1:
            level_probs = np.insert(level_probs, 0, 0.0)
        if len(slope_probs) == len(years) - 1:
            slope_probs = np.insert(slope_probs, 0, 0.0)
        if len(combined) == len(years) - 1:
            combined = np.insert(combined, 0, 0.0)

        # Left y-axis: pumping depth
        ax_left.plot(
            years, d["irr"], color="#1f77b4", linewidth=2,
            marker="o", markersize=4, zorder=5,
        )
        ax_left.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax_left.tick_params(axis="both", length=4)

        # Right y-axis: CP probability bars
        ax_right = ax_left.twinx()
        bar_width = 0.4
        ax_right.bar(
            years - bar_width / 2, level_probs, width=bar_width,
            color="#ff7f0e", alpha=0.7, zorder=2,
        )
        ax_right.bar(
            years + bar_width / 2, slope_probs, width=bar_width,
            color="#2ca02c", alpha=0.7, zorder=2,
        )
        ax_right.set_ylim(0.0, 1.05)
        ax_right.tick_params(axis="y", length=4)

        # Threshold line
        ax_right.axhline(y=THRESHOLD, color="gray", linestyle="--",
                         linewidth=1.0, zorder=3)

        # BOCPD MAP year per agent → 3-year transition window [cp-1, cp+1]
        cp_year = BOCPD_CP[aid]
        window_lo = cp_year - 1
        window_hi = cp_year + 1

        # Translucent band across the window
        ax_left.axvspan(window_lo, window_hi, color="red", alpha=0.12, zorder=2)
        # Bracketing dashed lines at window endpoints
        ax_left.axvline(x=window_lo, color="red", linestyle="--",
                        linewidth=2.0, zorder=4, alpha=0.8)
        ax_left.axvline(x=window_hi, color="red", linestyle="--",
                        linewidth=2.0, zorder=4, alpha=0.8)
        # Black dashed line at the exact BOCPD CP year (point estimate)
        ax_left.axvline(x=cp_year, color="black", linestyle="--",
                        linewidth=2.0, zorder=5)

        # Annotate the window (centered above cp_year)
        ax_left.annotate(
            f"CP window: [{window_lo}, {window_hi}]",
            xy=(cp_year, ax_left.get_ylim()[1]),
            xytext=(0, -5), textcoords="offset points",
            fontsize=14, fontweight="bold", color="red",
            ha="center", va="top",
        )

        # Ensure line drawn on top of bars
        ax_left.set_zorder(ax_right.get_zorder() + 1)
        ax_left.patch.set_visible(False)

        # Title with cluster label
        ax_left.set_title(f"Agent {aid} (C{cid}, CP={cp_year})",
                          fontsize=14, fontweight="bold")

        # Panel label
        ax_left.text(0.01, 0.99, panel_labels[idx], transform=ax_left.transAxes,
                     fontsize=14, fontweight="bold", va="top")

        # Axis labels (only on edges)
        if row == 3:
            ax_left.set_xlabel("Year")
        if col == 0:
            ax_left.set_ylabel("Pumping depth (mm)")
        if col == 1:
            ax_right.set_ylabel("Posterior probability")

    # ── Figure-level legend ──────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", linewidth=2, marker="o",
               markersize=4, label="Pumping depth"),
        Patch(facecolor="#ff7f0e", alpha=0.7, label="Level-shift prob"),
        Patch(facecolor="#2ca02c", alpha=0.7, label="Slope-change prob"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0,
               label=f"Threshold = {THRESHOLD}"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2.0,
               label="CP window [cp±1]"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0,
               label="Changepoint (CP)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.5, -0.04), ncol=6, frameon=False,
        fontsize=17, handlelength=2.8, handletextpad=0.7,
        columnspacing=1.6,
    )

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig4_changepoint.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


if __name__ == "__main__":
    main()
