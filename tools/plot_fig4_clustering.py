#!/usr/bin/env python3
"""
Fig 4: Temporal Clustering Results (DTC).

Panel (a): Co-association heatmap from 10 DTC runs (k=2).
Panel (b): Annual irrigation time series by cluster (Cluster 1 vs Cluster 2).
Panel (c): GIS map — user will add separately.

Data sources:
  results/cluster_coassociation_k2.csv  (from cluster_robustness.py)
  results/dtc_cluster_assignments_k2.csv
  results/cluster_irrigation_summary.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

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

    # Load data
    coassoc = pd.read_csv(BASE / "results" / "cluster_coassociation_k2.csv", index_col=0)
    assignments = pd.read_csv(BASE / "results" / "dtc_cluster_assignments_k2.csv")
    irrigation = pd.read_csv(BASE / "results" / "cluster_irrigation_summary.csv")

    # Identify cluster members
    cluster2_agents = sorted(assignments[assignments["Cluster"] == 2]["AgentID"].tolist())
    cluster1_agents = sorted(assignments[assignments["Cluster"] == 1]["AgentID"].tolist())

    # Sort co-association matrix: Cluster 2 agents first, then Cluster 1
    sorted_agents = cluster2_agents + cluster1_agents
    # Convert index to int for matching
    coassoc.index = coassoc.index.astype(int)
    coassoc.columns = coassoc.columns.astype(int)
    coassoc_sorted = coassoc.loc[sorted_agents, sorted_agents]

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True,
                              gridspec_kw={"width_ratios": [1, 1.5]})

    # ── Panel (a): Co-association heatmap ─────────────────────────────────
    ax = axes[0]
    im = ax.imshow(coassoc_sorted.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    # Add cluster boundary line
    n_c2 = len(cluster2_agents)
    ax.axhline(n_c2 - 0.5, color="black", linewidth=1.5)
    ax.axvline(n_c2 - 0.5, color="black", linewidth=1.5)

    # Tick labels
    tick_labels = [str(a) for a in sorted_agents]
    ax.set_xticks(range(len(sorted_agents)))
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=90)
    ax.set_yticks(range(len(sorted_agents)))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Agent ID")
    ax.set_ylabel("Agent ID")

    # Cluster labels on sides
    ax.text(-0.15, n_c2 / 2 / len(sorted_agents), "C2",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="center", color="tab:red")
    ax.text(-0.15, (n_c2 + len(cluster1_agents) / 2) / len(sorted_agents), "C1",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="center", color="tab:blue")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Co-association frequency")
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", color="white")

    # ── Panel (b): Time series by cluster ─────────────────────────────────
    ax2 = axes[1]
    years = irrigation["Year"].values

    # Plot individual agent series
    for aid in cluster1_agents:
        col = f"Agent_{aid}"
        if col in irrigation.columns:
            ax2.plot(years, irrigation[col].values, "-",
                     color="tab:blue", alpha=0.15, linewidth=0.8, zorder=1)

    for aid in cluster2_agents:
        col = f"Agent_{aid}"
        if col in irrigation.columns:
            ax2.plot(years, irrigation[col].values, "-",
                     color="tab:red", alpha=0.3, linewidth=0.8, zorder=1)

    # Compute and plot cluster means
    c1_cols = [f"Agent_{a}" for a in cluster1_agents if f"Agent_{a}" in irrigation.columns]
    c2_cols = [f"Agent_{a}" for a in cluster2_agents if f"Agent_{a}" in irrigation.columns]
    c1_mean = irrigation[c1_cols].mean(axis=1).values
    c2_mean = irrigation[c2_cols].mean(axis=1).values

    ax2.plot(years, c1_mean, "o-", color="tab:blue", linewidth=2, markersize=4,
             zorder=5, label=f"Cluster 1 mean (n={len(cluster1_agents)})")
    ax2.plot(years, c2_mean, "s-", color="tab:red", linewidth=2, markersize=4,
             zorder=5, label=f"Cluster 2 mean (n={len(cluster2_agents)})")

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Irrigation depth (mm yr$^{-1}$)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(False)
    ax2.tick_params(length=4)
    ax2.text(0.01, 0.99, "(b)", transform=ax2.transAxes,
             fontsize=14, fontweight="bold", va="top")

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig4_clustering.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


if __name__ == "__main__":
    main()
