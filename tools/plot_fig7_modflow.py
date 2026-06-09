#!/usr/bin/env python3
"""
Fig 7: MODFLOW Coupled Groundwater Uncertainty Propagation (4x2 grid).

Each panel shows one non-stationary agent with best-KGE ensemble member
trajectories for M1 and M2, 90% prediction intervals, and the observed
(baseline) reference trajectory. Pre-CP training period is shaded gray.

Data sources:
  results/modflow_propagation/coupled_ensemble/coupled_waterhead_summary.csv
  results/modflow_propagation/coupled_ensemble/coupled_waterhead_observed.csv
  results/modflow_propagation/coupled_ensemble/coupled_waterhead_realizations.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

AGENT_ORDER = [2, 12, 14, 18, 20, 24, 28, 29]
PER_AGENT_CP = {
    2: 2012, 12: 2004, 14: 2004, 18: 2004,
    20: 2005, 24: 2004, 28: 2012, 29: 2011,
}

DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "modflow_propagation" / "coupled_ensemble"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

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


def select_best_kge_member(realizations, observed, aid, model, eval_start):
    """Find the ensemble member with the highest KGE for a given agent/model."""
    obs = observed[(observed["agent"] == aid)].sort_values("year")
    obs_eval = obs[(obs["year"] >= eval_start) & (obs["year"] <= 2020)]
    y_true = obs_eval["relative_mm"].values
    eval_years = obs_eval["year"].values

    agent_model = realizations[
        (realizations["agent"] == aid) & (realizations["model"] == model)
    ]

    best_kge = -np.inf
    best_member = None
    for m, grp in agent_model.groupby("member"):
        grp_eval = grp[grp["year"].isin(eval_years)].sort_values("year")
        if len(grp_eval) != len(y_true):
            continue
        kge = compute_kge(y_true, grp_eval["relative_mm"].values)
        if not np.isnan(kge) and kge > best_kge:
            best_kge = kge
            best_member = m

    # Get the full series for the best member
    best_series = agent_model[agent_model["member"] == best_member].sort_values("year")
    return best_member, best_kge, best_series


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(DATA_DIR / "coupled_waterhead_summary.csv")
    observed = pd.read_csv(DATA_DIR / "coupled_waterhead_observed.csv")
    realizations = pd.read_csv(DATA_DIR / "coupled_waterhead_realizations.csv")

    # Filter to 8 non-stationary agents
    observed = observed[observed["agent"].isin(AGENT_ORDER)].copy()

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(8)]

    for idx, aid in enumerate(AGENT_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        cp = PER_AGENT_CP[aid]
        eval_start = cp + 1

        # Observed (reference trajectory)
        obs = observed[observed["agent"] == aid].sort_values("year")

        # M1 and M2 summaries (PI bands)
        m1_sum = summary[(summary["agent"] == aid) & (summary["model"] == "m1")].sort_values("year")
        m2_sum = summary[(summary["agent"] == aid) & (summary["model"] == "m2")].sort_values("year")

        # Pre-CP shading
        ax.axvspan(obs["year"].min(), cp, color="lightgray", alpha=0.2, zorder=0)

        # M1 PI band + best-KGE member
        m1_pi = m1_sum[m1_sum["p5_mm"].notna()]
        if len(m1_pi) > 0:
            ax.fill_between(m1_pi["year"], m1_pi["p5_mm"], m1_pi["p95_mm"],
                            color="tab:blue", alpha=0.15, zorder=2)

        _, m1_kge, m1_best = select_best_kge_member(realizations, observed, aid, "m1", cp)
        m1_plot = m1_best[m1_best["year"].isin(m1_pi["year"])]
        ax.plot(m1_plot["year"], m1_plot["relative_mm"], "--",
                color="tab:blue", linewidth=1.5, zorder=3)

        # M2 PI band + best-KGE member
        m2_pi = m2_sum[m2_sum["p5_mm"].notna()]
        if len(m2_pi) > 0:
            ax.fill_between(m2_pi["year"], m2_pi["p5_mm"], m2_pi["p95_mm"],
                            color="tab:red", alpha=0.15, zorder=2)

        _, m2_kge, m2_best = select_best_kge_member(realizations, observed, aid, "m2", eval_start)
        m2_plot = m2_best[m2_best["year"].isin(m2_pi["year"])]
        ax.plot(m2_plot["year"], m2_plot["relative_mm"], "--",
                color="tab:red", linewidth=1.5, zorder=3)

        # Observed reference
        ax.plot(obs["year"], obs["relative_mm"], "ko-",
                linewidth=1.2, markersize=4, zorder=5)

        # CP vertical line
        ax.axvline(cp, color="gray", linestyle="--", linewidth=0.9, zorder=1)

        # KGE annotations — per-panel corner to avoid overlap with data; M1 stacked above M2
        kge_corner = {
            2: "top-right", 12: "top-right", 14: "top-right", 18: "top-right",
            20: "top-right", 24: "top-right",
            28: "bottom-right", 29: "bottom-right",
        }.get(aid, "top-right")
        if kge_corner == "top-right":
            x_kge, ha_kge, y1, y2, va_kge = 0.97, "right", 0.91, 0.83, "top"
        elif kge_corner == "top-left":
            x_kge, ha_kge, y1, y2, va_kge = 0.03, "left", 0.91, 0.83, "top"
        elif kge_corner == "bottom-right":
            x_kge, ha_kge, y1, y2, va_kge = 0.97, "right", 0.13, 0.05, "bottom"
        else:
            x_kge, ha_kge, y1, y2, va_kge = 0.03, "left", 0.13, 0.05, "bottom"
        ax.text(x_kge, y1, f"M1 KGE={m1_kge:.2f}",
                transform=ax.transAxes, fontsize=14, color="tab:blue",
                ha=ha_kge, va=va_kge)
        ax.text(x_kge, y2, f"M2 KGE={m2_kge:.2f}",
                transform=ax.transAxes, fontsize=14, color="tab:red",
                ha=ha_kge, va=va_kge)

        # Formatting
        ax.set_title(f"Agent {aid} (CP*={cp})", fontsize=14, fontweight="bold")
        ax.grid(False)
        ax.tick_params(length=4)
        ax.text(0.01, 0.99, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

        if row == 3:
            ax.set_xlabel("Year")
        if col == 0:
            ax.set_ylabel("GW level change\nfrom 1993 (mm)")

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color="k", marker="o", markersize=4,
               linewidth=1.2, label="Observed"),
        Line2D([0], [0], color="tab:blue", linestyle="--",
               linewidth=1.5, label="M1 best KGE"),
        Patch(facecolor="tab:blue", alpha=0.15, label="M1 90% PI"),
        Line2D([0], [0], color="tab:red", linestyle="--",
               linewidth=1.5, label="M2 best KGE"),
        Patch(facecolor="tab:red", alpha=0.15, label="M2 90% PI"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False,
               fontsize=17, handlelength=2.8, handletextpad=0.7,
               columnspacing=1.6)

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig7_modflow_coupled.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)
    print("Done.")


if __name__ == "__main__":
    main()
