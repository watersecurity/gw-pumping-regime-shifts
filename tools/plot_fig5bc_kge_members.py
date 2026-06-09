#!/usr/bin/env python3
"""
Fig 5b/5c: M1 vs M2 with KGE-selected ensemble members + 90% PI.

Fig 5b: Central line = member whose KGE is closest to the median KGE
        across 100 ensemble members (median-KGE member).
Fig 5c: Central line = member with the highest KGE (best-KGE member).

Both figures keep the same 90% PI bands as Fig 5.

Data sources:
  results/cluster_irrigation_summary.csv          (full observed 1993-2020)
  results/modflow_propagation/rrca_export/rrca_summary.csv  (PI bands)
  results/modflow_propagation/rrca_export/rrca_members_m1.csv
  results/modflow_propagation/rrca_export/rrca_members_m2.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pathlib import Path

AGENT_CP = {
    2: 2012, 12: 2004, 14: 2004, 18: 2004,
    20: 2005, 24: 2004, 28: 2012, 29: 2011,
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


def compute_kge(y_true, y_pred):
    """Kling-Gupta Efficiency (KGE)."""
    obs_mean = np.mean(y_true)
    obs_std = np.std(y_true)
    if obs_std == 0 or obs_mean == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / obs_std
    beta = np.mean(y_pred) / obs_mean
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def select_kge_member(agent_members, obs_years, obs_vals, eval_start, method):
    """Select an ensemble member by KGE criterion.

    Parameters
    ----------
    agent_members : DataFrame with columns [year, depth_mm, member]
    obs_years, obs_vals : full observed series (1993-2020)
    eval_start : first eval year (CP + 1)
    method : "median" or "best"

    Returns
    -------
    selected_member_id, selected_series (DataFrame with year, depth_mm)
    """
    # Observed for eval window
    obs_mask = (obs_years >= eval_start) & (obs_years <= 2020)
    obs_eval_years = obs_years[obs_mask]
    obs_eval_vals = obs_vals[obs_mask]

    # Compute KGE for each member on eval window
    kges = {}
    for m, grp in agent_members.groupby("member"):
        grp_eval = grp[(grp["year"] >= eval_start) & (grp["year"] <= 2020)].sort_values("year")
        if len(grp_eval) != len(obs_eval_vals):
            continue
        kge = compute_kge(obs_eval_vals, grp_eval["depth_mm"].values)
        if not np.isnan(kge):
            kges[m] = kge

    if method == "best":
        sel = max(kges, key=kges.get)
    else:  # median
        median_kge = np.nanmedian(list(kges.values()))
        sel = min(kges, key=lambda m: abs(kges[m] - median_kge))

    sel_kge = kges[sel]
    sel_series = agent_members[agent_members["member"] == sel].sort_values("year")
    return sel, sel_series, sel_kge


def plot_figure(mode, irr, rrca_summary, m1_members, m2_members):
    """Plot a single 3x3 figure with KGE-selected member lines."""
    if mode == "median_kge":
        m1_label = "M1 median-KGE member"
        m2_label = "M2 median-KGE member"
        filename = "Fig5b_m1m2_ens_kge_median"
        method = "median"
    else:
        m1_label = "M1 best-KGE member"
        m2_label = "M2 best-KGE member"
        filename = "Fig5c_m1m2_best_kge"
        method = "best"

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(8)]

    for idx, aid in enumerate(AGENT_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        cp = AGENT_CP[aid]
        eval_start = cp + 1

        # Full observed series
        obs_col = f"Agent_{aid}"
        obs_years = irr["Year"].values
        obs_vals = irr[obs_col].values

        # PI bands from rrca_summary
        agent_sum = rrca_summary[rrca_summary["agent"] == aid].sort_values("year")

        # M1 PI + selected member
        m1_mask = agent_sum["m1_median_mm"].notna()
        m1_pi = agent_sum[m1_mask]
        if len(m1_pi) > 0:
            ax.fill_between(m1_pi["year"], m1_pi["m1_pi_lo_mm"], m1_pi["m1_pi_hi_mm"],
                            color="tab:blue", alpha=0.15, zorder=2)
        m1_agent = m1_members[m1_members["agent"] == aid]
        _, m1_sel, m1_kge = select_kge_member(m1_agent, obs_years, obs_vals, eval_start, method)
        # Plot selected member for years that have PI bands
        m1_plot = m1_sel[m1_sel["year"].isin(m1_pi["year"])]
        ax.plot(m1_plot["year"], m1_plot["depth_mm"], "--",
                color="tab:blue", linewidth=1.5, zorder=3)

        # M2 PI + selected member
        m2_mask = agent_sum["m2_median_mm"].notna()
        m2_pi = agent_sum[m2_mask]
        if len(m2_pi) > 0:
            ax.fill_between(m2_pi["year"], m2_pi["m2_pi_lo_mm"], m2_pi["m2_pi_hi_mm"],
                            color="tab:red", alpha=0.15, zorder=2)
        m2_agent = m2_members[m2_members["agent"] == aid]
        _, m2_sel, m2_kge = select_kge_member(m2_agent, obs_years, obs_vals, eval_start, method)
        m2_plot = m2_sel[m2_sel["year"].isin(m2_pi["year"])]
        ax.plot(m2_plot["year"], m2_plot["depth_mm"], "--",
                color="tab:red", linewidth=1.5, zorder=3)

        # Observed (full 1993-2020)
        ax.plot(obs_years, obs_vals, "ko-", linewidth=1.2, markersize=4, zorder=5)

        # CP vertical line
        ax.axvline(cp, color="gray", linestyle="--", linewidth=0.9, zorder=1)

        # KGE annotations — per-panel corner to avoid overlap with data; M1 stacked above M2
        kge_corner = {
            2: "top-left", 12: "top-right", 14: "top-right", 18: "top-right",
            20: "top-right", 24: "top-right", 28: "top-left", 29: "top-left",
        }.get(aid, "top-right")
        if kge_corner == "top-right":
            x_kge, ha_kge = 0.97, "right"
        else:
            x_kge, ha_kge = 0.03, "left"
        ax.text(x_kge, 0.91, f"M1 KGE={m1_kge:.2f}",
                transform=ax.transAxes, fontsize=14, color="tab:blue",
                ha=ha_kge, va="top")
        ax.text(x_kge, 0.83, f"M2 KGE={m2_kge:.2f}",
                transform=ax.transAxes, fontsize=14, color="tab:red",
                ha=ha_kge, va="top")

        ax.set_title(f"Agent {aid} (CP*={cp})", fontsize=14, fontweight="bold")
        ax.grid(False)
        ax.tick_params(length=4)
        ax.text(0.01, 0.99, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

        if row == 3:
            ax.set_xlabel("Year")
        if col == 0:
            ax.set_ylabel("Pumping depth (mm yr$^{-1}$)")

    # Shared legend
    legend_handles = [
        plt.Line2D([0], [0], color="k", marker="o", markersize=4,
                   linewidth=1.2, label="Observed"),
        plt.Line2D([0], [0], color="tab:blue", linestyle="--",
                   linewidth=1.5, label=m1_label),
        Patch(facecolor="tab:blue", alpha=0.15, label="M1 90% PI"),
        plt.Line2D([0], [0], color="tab:red", linestyle="--",
                   linewidth=1.5, label=m2_label),
        Patch(facecolor="tab:red", alpha=0.15, label="M2 90% PI"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False,
               fontsize=17, handlelength=2.8, handletextpad=0.7,
               columnspacing=1.6)

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"{filename}.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    irr = pd.read_csv(BASE / "results" / "cluster_irrigation_summary.csv")
    rrca_summary = pd.read_csv(BASE / "results" / "modflow_propagation" / "rrca_export" / "rrca_summary.csv")
    m1_members = pd.read_csv(BASE / "results" / "modflow_propagation" / "rrca_export" / "rrca_members_m1.csv")
    m2_members = pd.read_csv(BASE / "results" / "modflow_propagation" / "rrca_export" / "rrca_members_m2.csv")

    plot_figure("median_kge", irr, rrca_summary, m1_members, m2_members)
    plot_figure("best_kge", irr, rrca_summary, m1_members, m2_members)


if __name__ == "__main__":
    main()
