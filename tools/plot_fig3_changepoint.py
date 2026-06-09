#!/usr/bin/env python3
"""
Fig 3: Changepoint Detection (per-agent BOCPD posteriors, 3x3 grid).

Each panel shows one non-stationary agent with dual y-axes:
  Left:  Annual irrigation depth (blue line)
  Right: Mean/Var shift probability (orange bars) + Slope change probability (green bars)
  CP year marked with a vertical line and annotation.

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

# 9 non-stationary agents: 5 from Cluster 2, 4 from Cluster 1
CLUSTER2_AGENTS = [2, 3, 24, 28, 29]
CLUSTER1_NS_AGENTS = [12, 14, 18, 20]
ALL_NS_AGENTS = sorted(CLUSTER2_AGENTS + CLUSTER1_NS_AGENTS)

THRESHOLD = 0.3

PER_AGENT_CP = {
    2: 2005, 3: 2005, 12: 2004, 14: 1994,
    18: 2004, 20: 2005, 24: 2005, 28: 2005, 29: 2005,
}

RC = {
    "font.family": "Arial", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-agent CP data (threshold 0.3)
    c2 = pd.read_csv(BASE / "results" / "changepoint_probabilities_agents_cluster2_t0.3.csv")
    c1 = pd.read_csv(BASE / "results" / "changepoint_probabilities_agents_cluster1_t0.3.csv")

    years = c2["Year"].values

    # Build per-agent data dict
    agent_data = {}
    for aid in CLUSTER2_AGENTS:
        agent_data[aid] = {
            "irr": c2[f"Agent{aid}_MeanIrrigation_mm"].values,
            "level": c2[f"Agent{aid}_CP_Prob"].values,
            "slope": c2[f"Agent{aid}_CP_Prob_Slope"].values,
        }
    for aid in CLUSTER1_NS_AGENTS:
        agent_data[aid] = {
            "irr": c1[f"Agent{aid}_MeanIrrigation_mm"].values,
            "level": c1[f"Agent{aid}_CP_Prob"].values,
            "slope": c1[f"Agent{aid}_CP_Prob_Slope"].values,
        }

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)

    for idx, aid in enumerate(ALL_NS_AGENTS):
        row, col = divmod(idx, 3)
        ax_left = axes[row, col]
        d = agent_data[aid]

        # Pad probabilities if needed (BOCPD returns n-1 values)
        level_probs = d["level"]
        slope_probs = d["slope"]
        if len(level_probs) == len(years) - 1:
            level_probs = np.insert(level_probs, 0, 0.0)
        if len(slope_probs) == len(years) - 1:
            slope_probs = np.insert(slope_probs, 0, 0.0)

        # Left y-axis: Irrigation depth (blue line)
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
        ax_right.axhline(y=THRESHOLD, color="red", linestyle="--",
                         linewidth=1.2, zorder=3)

        # Annotate max level probability
        max_idx = np.argmax(level_probs)
        ax_right.annotate(
            f"{level_probs[max_idx]:.2f}",
            xy=(years[max_idx] - bar_width / 2, level_probs[max_idx]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#ff7f0e",
        )

        # CP year vertical line + annotation
        cp_year = PER_AGENT_CP[aid]
        ax_left.axvline(x=cp_year, color="red", linestyle=":", linewidth=1.5,
                        zorder=4)
        ax_left.annotate(
            f"CP={cp_year}", xy=(cp_year, ax_left.get_ylim()[1]),
            xytext=(3, -5), textcoords="offset points",
            fontsize=10, fontweight="bold", color="red",
            ha="left", va="top",
        )

        # Ensure line drawn on top of bars
        ax_left.set_zorder(ax_right.get_zorder() + 1)
        ax_left.patch.set_visible(False)

        # Title
        ax_left.set_title(f"Agent {aid}", fontsize=14, fontweight="bold")

        # Axis labels (only on edges)
        if row == 2:
            ax_left.set_xlabel("Year")
        if col == 0:
            ax_left.set_ylabel("Irrigation Depth (mm)")
        if col == 2:
            ax_right.set_ylabel("Posterior Probability")

    # Figure-level legend
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", linewidth=2, marker="o",
               markersize=4, label="Irrigation Depth"),
        Patch(facecolor="#ff7f0e", alpha=0.7, label="Mean/Var Shift Prob"),
        Patch(facecolor="#2ca02c", alpha=0.7, label="Slope Change Prob"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
               label=f"Threshold = {THRESHOLD}"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=1.5,
               label="Detected CP Year"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center",
        bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False, fontsize=13,
    )

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig3_changepoint.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


if __name__ == "__main__":
    main()
