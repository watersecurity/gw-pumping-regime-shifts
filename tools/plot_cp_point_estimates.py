#!/usr/bin/env python3
"""
Plot CP sensitivity point estimates — 1×2 grid (Agent 4, Agent 11)
with all three CP candidates (2005, 2007, 2009) overlaid.

Retrains the point-estimate model for each CP using saved best_params,
then plots train + eval predictions on a common eval window (2010–2020).

Output: results/cluster2_cp_sensitivity_point_estimates.pdf (.png)
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Add parent so we can import from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_xgboost_abm import (
    AGENT_IDS,
    DATA_DIR,
    RESULTS_DIR,
    load_agent_data,
    aggregate_to_annual,
    prepare_features,
    load_tuned_params,
    train_with_early_stopping,
)

# ── Configuration ────────────────────────────────────────────────────────────
CP_CANDIDATES = [2005, 2007, 2009]
COMMON_EVAL_START = 2010
CP_COLORS = {2005: "tab:blue", 2007: "tab:orange", 2009: "tab:green"}
OUTPUT_STEM = "cluster2_cp_sensitivity_point_estimates"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and aggregate ────────────────────────────────────────────────
    print("Loading data...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    print(f"  {len(annual)} rows, years {annual['Year'].min()}–{annual['Year'].max()}")

    # Common eval window
    df_eval = annual[annual["Year"] >= COMMON_EVAL_START].copy()

    # ── 2. Retrain for each CP and collect predictions ───────────────────────
    # Storage: cp -> agent_id -> {train_years, train_pred, eval_years, eval_pred, eval_r2}
    cp_results = {}

    for cp in CP_CANDIDATES:
        params_path = RESULTS_DIR / f"sensitivity_cp{cp}" / f"best_params_cp{cp}.json"
        if not params_path.exists():
            print(f"  WARNING: {params_path} not found, skipping CP={cp}")
            continue

        best_params = load_tuned_params(params_path)
        train_params = {
            **best_params,
            "objective": "reg:squarederror",
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }

        df_train = annual[annual["Year"] <= cp].copy()
        X_train, y_train = prepare_features(df_train)
        X_eval, y_eval = prepare_features(df_eval)

        print(f"  CP={cp}: training on {df_train['Year'].min()}–{cp} "
              f"({len(df_train)} obs), eval {COMMON_EVAL_START}–2020 ({len(df_eval)} obs)")

        model = train_with_early_stopping(X_train, y_train, train_params, val_years=3)
        pred_train = model.predict(X_train)
        pred_eval = model.predict(X_eval)

        agent_results = {}
        for aid in AGENT_IDS:
            # Training
            tr_mask = df_train["AgentID"] == aid
            tr_years = df_train.loc[tr_mask, "Year"].values
            tr_pred = pred_train[tr_mask.values]

            # Eval
            ev_mask = df_eval["AgentID"] == aid
            ev_years = df_eval.loc[ev_mask, "Year"].values
            ev_obs = y_eval[ev_mask.values]
            ev_pred = pred_eval[ev_mask.values]
            ev_r2 = r2_score(ev_obs, ev_pred)

            agent_results[aid] = {
                "train_years": tr_years,
                "train_pred": tr_pred,
                "eval_years": ev_years,
                "eval_pred": ev_pred,
                "eval_r2": ev_r2,
            }
            print(f"    Agent {aid}: eval R² = {ev_r2:.3f}")

        cp_results[cp] = agent_results

    if not cp_results:
        print("ERROR: No CP results found. Exiting.")
        return

    # ── 3. Plot ──────────────────────────────────────────────────────────────
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
    })

    n_agents = len(AGENT_IDS)
    ncols = min(n_agents, 3)
    nrows = math.ceil(n_agents / ncols)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(n_agents)]
    bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for col_idx, aid in enumerate(AGENT_IDS):
        ax = axes.flat[col_idx]

        # Full observed time series
        obs_sub = annual[annual["AgentID"] == aid].sort_values("Year")
        _, y_obs = prepare_features(obs_sub)
        ax.plot(obs_sub["Year"].values, y_obs, "ko-", label="Observed",
                linewidth=1.2, markersize=4, zorder=5)

        # Overlay each CP candidate
        r2_texts = []
        for cp in CP_CANDIDATES:
            if cp not in cp_results:
                continue
            res = cp_results[cp][aid]
            color = CP_COLORS[cp]

            # Training predictions (solid)
            ax.plot(res["train_years"], res["train_pred"], "-",
                    color=color, linewidth=1.5, zorder=2,
                    label=f"CP={cp}" if col_idx == 0 else None)

            # Eval predictions (dashed)
            ax.plot(res["eval_years"], res["eval_pred"], "--",
                    color=color, linewidth=1.5, zorder=2)

            # Vertical CP line
            ax.axvline(cp, color=color, linestyle=":", linewidth=0.9, alpha=0.6)

            r2_texts.append((cp, res["eval_r2"], color))

        # Stacked R² annotations (bottom-right, with white background)
        for i, (cp, r2, color) in enumerate(r2_texts):
            ax.text(0.98, 0.04 + i * 0.09,
                    f"CP={cp}: R$^2$={r2:.2f}",
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=11, color=color, bbox=bbox_props, zorder=10)

        # Panel label (below title area)
        ax.text(0.02, 0.95, panel_labels[col_idx],
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold", bbox=bbox_props, zorder=10)

        ax.set_title(f"Agent {aid}")
        ax.set_xlabel("Year")
        if col_idx % ncols == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")
        ax.tick_params(axis="both", which="major", length=4)

    # Hide unused subplots
    for i in range(n_agents, nrows * ncols):
        axes.flat[i].set_visible(False)

    # Simplified legend below the panels
    from matplotlib.lines import Line2D
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.4", linestyle="-", linewidth=1.5))
    labels.append("Train")
    handles.append(Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.5))
    labels.append("Eval")
    fig.legend(handles, labels, loc="lower center",
               ncol=len(handles), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Cluster 2 — CP Sensitivity Point Predictions",
                 fontsize=16)

    output_pdf = RESULTS_DIR / f"{OUTPUT_STEM}.pdf"
    output_png = RESULTS_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"\n  Saved {output_pdf}")
    print(f"  Saved {output_png}")


if __name__ == "__main__":
    main()
