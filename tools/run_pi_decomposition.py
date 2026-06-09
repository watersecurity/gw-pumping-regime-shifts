#!/usr/bin/env python3
"""
PI Width Decomposition — Bootstrap vs Hyperparameter Jitter.

For each CP in {2005, 2007, 2009}, runs three 100-member ensembles:
  1. Full        — block-bootstrap years + hyperparameter jitter  (existing)
  2. Bootstrap   — block-bootstrap years, fixed Optuna-best params
  3. Jitter      — fixed training set,    jittered hyperparams

Outputs:
  results/cluster2_pi_decomposition_summary.csv   — table of PI widths & RMSE spreads
  results/cluster2_pi_decomposition.pdf/.png       — grouped bar chart
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_xgboost_abm import (
    AGENT_IDS,
    CP_YEAR,
    BLOCK_SIZE,
    MASTER_SEED,
    RESULTS_DIR,
    load_agent_data,
    aggregate_to_annual,
    prepare_features,
    load_tuned_params,
    train_with_early_stopping,
    constrained_block_bootstrap_years,
    _anchor_tail_for_cp,
    sample_hyperparams,
    compute_full_metrics,
)

# ── Configuration ────────────────────────────────────────────────────────────
CP_CANDIDATES = [2005, 2007, 2009]
COMMON_EVAL_START = 2010
N_BOOT = 100


def run_decomposed_ensemble(df_train, df_eval, best_params, cp, rng,
                            n_boot=N_BOOT, mode="full"):
    """Run ensemble with a specific uncertainty source isolated.

    Args:
        mode: "full"      — bootstrap + jitter (default, replicates run_cp)
              "bootstrap"  — bootstrap years, fixed params
              "jitter"     — fixed training set, jittered params
    """
    X_eval, y_eval = prepare_features(df_eval)
    eval_agents = df_eval["AgentID"].values
    n_eval = len(y_eval)
    preds_matrix = np.zeros((n_boot, n_eval))
    metrics_list = []

    pre_cp_years = np.arange(1993, CP_YEAR)
    anchor_tail = _anchor_tail_for_cp(cp, base_cp=CP_YEAR)

    # Fixed params for bootstrap-only mode
    fixed_params = {
        **best_params,
        "objective": "reg:squarederror",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    for i in range(n_boot):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        # --- Training data ---
        if mode in ("full", "bootstrap"):
            boot_years = constrained_block_bootstrap_years(
                pre_cp_years, anchor_tail, BLOCK_SIZE, member_rng)
            boot_frames = [df_train[df_train["Year"] == yr] for yr in boot_years]
            df_boot = pd.concat(boot_frames, ignore_index=True)
            X_boot, y_boot = prepare_features(df_boot)
        else:  # jitter: fixed training set
            X_boot, y_boot = prepare_features(df_train)

        # --- Hyperparameters ---
        if mode in ("full", "jitter"):
            params = sample_hyperparams(member_rng, base_params=best_params)
        else:  # bootstrap: fixed params
            params = dict(fixed_params)
            params["random_state"] = int(member_rng.integers(0, 2**31))

        model = train_with_early_stopping(X_boot, y_boot, params, val_years=3)
        preds = model.predict(X_eval)
        preds_matrix[i] = preds

        m = compute_full_metrics(y_eval, preds, eval_agents)
        m["ensemble_id"] = i
        metrics_list.append(m)

    # Summarize
    pi_widths = (np.percentile(preds_matrix, 95, axis=0)
                 - np.percentile(preds_matrix, 5, axis=0))
    rmse_arr = np.array([m["overall_RMSE"] for m in metrics_list])

    return {
        "PI_width_mean": np.mean(pi_widths),
        "PI_width_median": np.median(pi_widths),
        "RMSE_median": np.median(rmse_arr),
        "RMSE_IQR_lo": np.percentile(rmse_arr, 25),
        "RMSE_IQR_hi": np.percentile(rmse_arr, 75),
        "RMSE_spread": np.percentile(rmse_arr, 75) - np.percentile(rmse_arr, 25),
    }


def plot_decomposition(df_summary, output_pdf):
    """Grouped bar chart: PI width and RMSE IQR by CP and ensemble mode."""
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

    modes = ["Full", "Bootstrap", "Jitter"]
    mode_colors = {"Full": "0.35", "Bootstrap": "tab:blue", "Jitter": "tab:orange"}
    cps = sorted(df_summary["CP"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    metrics = [("PI_width_mean", "Mean PI Width (mm)"),
               ("RMSE_spread", "RMSE IQR (mm)")]
    panel_labels = ["(a)", "(b)"]

    bar_width = 0.25
    x = np.arange(len(cps))

    for ax, (col, ylabel), plabel in zip(axes, metrics, panel_labels):
        for j, mode in enumerate(modes):
            sub = df_summary[df_summary["Mode"] == mode].sort_values("CP")
            vals = sub[col].values
            ax.bar(x + j * bar_width, vals, bar_width,
                   label=mode, color=mode_colors[mode], edgecolor="white")
            # Value labels on bars
            for xi, v in zip(x + j * bar_width, vals):
                ax.text(xi, v + ax.get_ylim()[1] * 0.01, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=10)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([f"CP={c}" for c in cps])
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.99, plabel, transform=ax.transAxes,
                va="top", ha="left", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=4)

    # Adjust y-limits to make room for value labels
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("PI Width Decomposition — Bootstrap vs Jitter",
                 fontsize=16)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    df_eval = annual[annual["Year"] >= COMMON_EVAL_START].copy()
    print(f"  Eval: {COMMON_EVAL_START}–2020 ({len(df_eval)} obs)")

    rows = []
    for cp in CP_CANDIDATES:
        params_path = RESULTS_DIR / f"sensitivity_cp{cp}" / f"best_params_cp{cp}.json"
        if not params_path.exists():
            print(f"  WARNING: {params_path} not found, skipping CP={cp}")
            continue

        best_params = load_tuned_params(params_path)
        df_train = annual[annual["Year"] <= cp].copy()

        for mode in ("full", "bootstrap", "jitter"):
            print(f"  CP={cp}, mode={mode}...")
            rng = np.random.default_rng(MASTER_SEED + cp)
            result = run_decomposed_ensemble(
                df_train, df_eval, best_params, cp, rng,
                n_boot=N_BOOT, mode=mode)
            row = {"CP": cp, "Mode": mode.capitalize(), **result}
            rows.append(row)
            print(f"    PI width mean={result['PI_width_mean']:.1f}, "
                  f"RMSE IQR={result['RMSE_spread']:.1f}")

    df_summary = pd.DataFrame(rows)

    # Save summary CSV
    csv_path = RESULTS_DIR / "cluster2_pi_decomposition_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # Print table
    print("\n" + "=" * 72)
    print("PI DECOMPOSITION SUMMARY")
    print("=" * 72)
    print(f"{'CP':<6} {'Mode':<12} {'PI Width':>10} {'RMSE med':>10} "
          f"{'RMSE IQR':>10}")
    print("-" * 72)
    for _, r in df_summary.iterrows():
        print(f"{r['CP']:<6} {r['Mode']:<12} {r['PI_width_mean']:>10.1f} "
              f"{r['RMSE_median']:>10.1f} {r['RMSE_spread']:>10.1f}")
    print("=" * 72)

    # Plot
    plot_decomposition(df_summary,
                       RESULTS_DIR / "cluster2_pi_decomposition.pdf")


if __name__ == "__main__":
    main()
