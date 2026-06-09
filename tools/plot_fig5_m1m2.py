#!/usr/bin/env python3
"""
Fig 5: M1 vs M2 Irrigation Predictions + Prediction Intervals.

3×3 panel grid (8 agents + 1 empty) showing per-agent observed irrigation +
M1/M2 ensemble median and 90% PIs, using per-agent BOCPD-derived CPs.

Data sources:
  results/transition_window/per_agent_detail.csv  (best CP per agent)
  results/cluster_irrigation_summary.csv          (full observed 1993-2020)
  Bootstrap predictions regenerated inline from run_transition_window.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pathlib import Path

# Per-agent CP years (best CP by M2 KGE from transition window analysis)
AGENT_CP = {
    2:  2012,   # best CP from transition window
    24: 2004,
    28: 2012,
    29: 2011,
    12: 2004,
    14: 2004,
    18: 2004,
    20: 2005,
}
AGENT_ORDER = [2, 12, 14, 18, 20, 24, 28, 29]

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "results" / "figures"

RC = {
    "font.family": "Arial", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load summary to get best CP per agent
    summary = pd.read_csv(BASE / "results" / "transition_window" / "summary.csv")

    # Load full observed series (1993-2020) for all agents
    irr = pd.read_csv(BASE / "results" / "cluster_irrigation_summary.csv")

    # Load MODFLOW/RRCA predictions if available, otherwise use transition window
    rrca_path = BASE / "results" / "modflow_propagation" / "rrca_export" / "rrca_summary.csv"
    has_rrca = rrca_path.exists()
    if has_rrca:
        rrca = pd.read_csv(rrca_path)

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(9)]

    for idx, aid in enumerate(AGENT_ORDER):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        cp = AGENT_CP[aid]

        # Full observed series
        obs_col = f"Agent_{aid}"
        obs_years = irr["Year"].values
        obs_vals = irr[obs_col].values

        # Use RRCA predictions if available
        if has_rrca and aid in rrca["agent"].values:
            agent_sum = rrca[rrca["agent"] == aid].sort_values("year")

            m1_mask = agent_sum["m1_median_mm"].notna()
            m1 = agent_sum[m1_mask]
            if len(m1) > 0:
                ax.fill_between(m1["year"], m1["m1_pi_lo_mm"], m1["m1_pi_hi_mm"],
                                color="tab:blue", alpha=0.15, zorder=2)
                ax.plot(m1["year"], m1["m1_median_mm"], "--",
                        color="tab:blue", linewidth=1.5, zorder=3)

            m2_mask = agent_sum["m2_median_mm"].notna()
            m2 = agent_sum[m2_mask]
            if len(m2) > 0:
                ax.fill_between(m2["year"], m2["m2_pi_lo_mm"], m2["m2_pi_hi_mm"],
                                color="tab:red", alpha=0.15, zorder=2)
                ax.plot(m2["year"], m2["m2_median_mm"], "--",
                        color="tab:red", linewidth=1.5, zorder=3)

        # Observed (full 1993-2020)
        ax.plot(obs_years, obs_vals, "ko-", linewidth=1.2, markersize=4, zorder=5)

        # CP vertical line
        ax.axvline(cp, color="gray", linestyle="--", linewidth=0.9, zorder=1)

        ax.set_title(f"Agent {aid} (CP={cp})")
        ax.grid(False)
        ax.tick_params(length=4)
        ax.text(0.01, 0.99, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

        if row == 2:
            ax.set_xlabel("Year")
        if col == 0:
            ax.set_ylabel("Pumping depth (mm yr$^{-1}$)")

    # Empty panel for 9th position (Agent 3 dropped)
    axes[2, 2].axis("off")

    # Shared legend
    legend_handles = [
        plt.Line2D([0], [0], color="k", marker="o", markersize=4,
                   linewidth=1.2, label="Observed"),
        plt.Line2D([0], [0], color="tab:blue", linestyle="--",
                   linewidth=1.5, label="M1 median"),
        Patch(facecolor="tab:blue", alpha=0.15, label="M1 90% PI"),
        plt.Line2D([0], [0], color="tab:red", linestyle="--",
                   linewidth=1.5, label="M2 median"),
        Patch(facecolor="tab:red", alpha=0.15, label="M2 90% PI"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False)

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig5_m1m2_predictions.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


if __name__ == "__main__":
    main()
