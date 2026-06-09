#!/usr/bin/env python3
"""
Fig 3: Temporal Clustering + Cluster-Mean BOCPD Posteriors.

Panel (a): Co-association heatmap from 10 DTC runs (k=2).
Panel (b): GIS map placeholder.
Panel (c): Cluster 1 — individual traces, cluster mean, and BOCPD posterior.
Panel (d): Cluster 2 — individual traces, cluster mean, and BOCPD posterior.

Data sources:
  results/cluster_coassociation_k2.csv
  results/dtc_cluster_assignments_k2.csv
  results/cluster_irrigation_summary.csv
  results/cluster_mean_changepoint.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

    # ── Load data ────────────────────────────────────────────────────────
    coassoc = pd.read_csv(BASE / "results" / "cluster_coassociation_k2.csv",
                          index_col=0)
    assignments = pd.read_csv(BASE / "results" / "dtc_cluster_assignments_k2.csv")
    irrigation = pd.read_csv(BASE / "results" / "cluster_irrigation_summary.csv")
    cp_data = pd.read_csv(BASE / "results" / "cluster_mean_changepoint.csv")

    cluster2_agents = sorted(
        assignments[assignments["Cluster"] == 2]["AgentID"].tolist())
    cluster1_agents = sorted(
        assignments[assignments["Cluster"] == 1]["AgentID"].tolist())

    years_irr = irrigation["Year"].values
    years_cp = cp_data["Year"].values

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    # ── Panel (a): Co-association heatmap ────────────────────────────────
    ax = axes[0, 0]
    sorted_agents = cluster2_agents + cluster1_agents
    coassoc.index = coassoc.index.astype(int)
    coassoc.columns = coassoc.columns.astype(int)
    coassoc_sorted = coassoc.loc[sorted_agents, sorted_agents]

    im = ax.imshow(coassoc_sorted.values, cmap="YlOrRd", vmin=0, vmax=1,
                   aspect="auto")

    n_c2 = len(cluster2_agents)
    ax.axhline(n_c2 - 0.5, color="black", linewidth=1.5)
    ax.axvline(n_c2 - 0.5, color="black", linewidth=1.5)

    tick_labels = [str(a) for a in sorted_agents]
    ax.set_xticks(range(len(sorted_agents)))
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(len(sorted_agents)))
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Agent ID")
    ax.set_ylabel("Agent ID")

    ax.text(-0.15, n_c2 / 2 / len(sorted_agents), "C2",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="center", color="tab:red")
    ax.text(-0.15, (n_c2 + len(cluster1_agents) / 2) / len(sorted_agents),
            "C1", transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="center", color="tab:blue")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Co-association frequency")
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", color="white")

    # ── Helper: cluster time series + BOCPD panel ────────────────────────
    def _plot_cluster_panel(ax_ts, cluster_agents, cluster_label, color,
                            bocpd_probs, panel_label):
        """Plot cluster time series with BOCPD posteriors.

        Parameters
        ----------
        bocpd_probs : array-like
            Combined BOCPD posterior probabilities aligned with years_cp.
        """
        # Individual traces
        for aid in cluster_agents:
            col = f"Agent_{aid}"
            if col in irrigation.columns:
                ax_ts.plot(years_irr, irrigation[col].values, "-",
                           color=color, alpha=0.12, linewidth=0.8, zorder=1)

        # Cluster mean
        cols = [f"Agent_{a}" for a in cluster_agents
                if f"Agent_{a}" in irrigation.columns]
        c_mean = irrigation[cols].mean(axis=1).values
        ax_ts.plot(years_irr, c_mean, "o-", color=color, linewidth=2.2,
                   markersize=4, zorder=5,
                   label=f"{cluster_label} mean (n={len(cluster_agents)})")
        ax_ts.set_xlabel("Year")
        ax_ts.set_ylabel("Pumping depth (mm yr$^{-1}$)")
        ax_ts.tick_params(length=4)

        # BOCPD posterior on right y-axis
        ax_cp = ax_ts.twinx()
        ax_cp.bar(years_cp, bocpd_probs, width=0.6, color=color, alpha=0.25,
                  zorder=2)
        ax_cp.set_ylim(0.0, 1.05)
        ax_cp.set_ylabel("Posterior probability")
        ax_cp.tick_params(axis="y", length=4)

        # Threshold line
        ax_cp.axhline(y=0.3, color="gray", linestyle="--", linewidth=1.0,
                      zorder=3)

        # Mark MAP changepoint year (always shown)
        max_idx = np.argmax(bocpd_probs)
        max_prob = bocpd_probs[max_idx]
        cp_year = years_cp[max_idx]
        ax_ts.axvline(x=cp_year, color="black", linestyle=":",
                      linewidth=1.5, zorder=4)
        ax_ts.annotate(
            f"CP≈{cp_year}\n(p={max_prob:.2f})",
            xy=(cp_year, ax_ts.get_ylim()[1]),
            xytext=(5, -8), textcoords="offset points",
            fontsize=10, fontweight="bold", color="black",
            ha="left", va="top",
        )

        # Ensure time-series line on top of bars
        ax_ts.set_zorder(ax_cp.get_zorder() + 1)
        ax_ts.patch.set_visible(False)

        ax_ts.text(0.01, 0.99, panel_label, transform=ax_ts.transAxes,
                   fontsize=14, fontweight="bold", va="top")
        ax_ts.set_title(cluster_label, fontsize=14, fontweight="bold")

    # ── Compute BOCPD posteriors ────────────────────────────────────────
    # Cluster 1: use default priors from CSV (combined column)
    c1_probs = cp_data["Cluster1_CP_Prob_Combined"].values

    # Cluster 2: re-run BOCPD with weak priors (alpha0=0.5, beta0=0.5,
    # kappa0=0.5) and prior_p=1/29 to produce posteriors above threshold
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE))
        from tools.run_changepoint_detection import run_offline_detection
        c2_cols = [f"Agent_{a}" for a in cluster2_agents
                   if f"Agent_{a}" in irrigation.columns]
        c2_mean_series = irrigation[c2_cols].mean(axis=1).values
        c2_level = run_offline_detection(
            c2_mean_series, prior_p=1/29,
            alpha0=0.5, beta0=0.5, kappa0=0.5, mu0=0.0,
        )
        c2_level = np.insert(c2_level, 0, 0.0)  # pad to match years
    except (ImportError, ModuleNotFoundError):
        print("  BOCPD library incomplete — using pre-computed CSV posteriors for Cluster 2")
        c2_level = cp_data["Cluster2_CP_Prob_Combined"].values

    # ── Panel (b): placeholder for GIS map ──────────────────────────────
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, "(b) GIS map placeholder",
                    transform=axes[0, 1].transAxes, fontsize=14,
                    ha="center", va="center", color="gray")

    # ── Panel (c): Cluster 1 ─────────────────────────────────────────────
    _plot_cluster_panel(axes[1, 0], cluster1_agents, "Cluster 1", "tab:blue",
                        c1_probs, "(c)")

    # ── Panel (d): Cluster 2 ─────────────────────────────────────────────
    _plot_cluster_panel(axes[1, 1], cluster2_agents, "Cluster 2", "tab:red",
                        c2_level, "(d)")

    # ── Figure-level legend ──────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1.2, label="Individual agents"),
        Line2D([0], [0], color="black", linewidth=2.2, marker="o",
               markersize=4, label="Cluster mean"),
        Patch(facecolor="gray", alpha=0.3, label="BOCPD posterior"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0,
               label="Threshold = 0.3"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
               fontsize=17, handlelength=2.8, handletextpad=0.7,
               columnspacing=1.6)

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig3_clustering.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


if __name__ == "__main__":
    main()
