#!/usr/bin/env python3
"""
Year-Feature Leakage Check: M1 vs M2 with and without Year as a feature.

Runs the M1/M2 point-estimate comparison twice:
  1. With Year (default pipeline)
  2. Without Year (drop_year=True)

If M2's advantage holds when Year is dropped, the benefit is real
(not an artifact of Year-as-feature extrapolation).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent so we can import the pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_xgboost_abm import (
    AGENT_IDS, CP_YEAR, MASTER_SEED, RESULTS_DIR,
    load_agent_data, aggregate_to_annual, split_data,
    prepare_features, optuna_tune, compute_full_metrics,
)


def run_comparison(df_m1_train, df_m2_train, df_eval, drop_year):
    """Run M1/M2 point-estimate comparison with or without Year feature."""
    tag = "no_year" if drop_year else "with_year"
    print(f"\n{'='*60}")
    print(f"  M1 vs M2 — {tag}")
    print(f"{'='*60}")

    X_m1, y_m1 = prepare_features(df_m1_train, drop_year=drop_year)
    X_m2, y_m2 = prepare_features(df_m2_train, drop_year=drop_year)
    X_eval, y_eval = prepare_features(df_eval, drop_year=drop_year)
    eval_agents = df_eval["AgentID"].values

    # Need year arrays for time-aware splitting inside optuna_tune
    years_m1 = df_m1_train["Year"].values
    years_m2 = df_m2_train["Year"].values

    print(f"  Features: {list(X_m1.columns)}")
    print(f"  M1 train shape: {X_m1.shape}")
    print(f"  M2 train shape: {X_m2.shape}")
    print(f"  Eval shape:     {X_eval.shape}")

    print("\n  Tuning M1 (Stationary)...")
    model_m1, params_m1 = optuna_tune(
        X_m1, y_m1, seed=MASTER_SEED, years=years_m1)

    print("\n  Tuning M2 (CP-Aware)...")
    model_m2, params_m2 = optuna_tune(
        X_m2, y_m2, seed=MASTER_SEED + 1, years=years_m2)

    pred_m1 = model_m1.predict(X_eval)
    pred_m2 = model_m2.predict(X_eval)

    metrics_m1 = compute_full_metrics(y_eval, pred_m1, eval_agents)
    metrics_m2 = compute_full_metrics(y_eval, pred_m2, eval_agents)

    rmse_m1 = metrics_m1["overall_RMSE"]
    rmse_m2 = metrics_m2["overall_RMSE"]
    r2_m1 = metrics_m1["overall_R2"]
    r2_m2 = metrics_m2["overall_R2"]
    mae_m1 = metrics_m1["overall_MAE"]
    mae_m2 = metrics_m2["overall_MAE"]

    delta_rmse = rmse_m1 - rmse_m2
    pct_improve = delta_rmse / rmse_m1 * 100

    print(f"\n  --- Results ({tag}) ---")
    print(f"  M1  RMSE={rmse_m1:.2f}  MAE={mae_m1:.2f}  R²={r2_m1:.3f}")
    print(f"  M2  RMSE={rmse_m2:.2f}  MAE={mae_m2:.2f}  R²={r2_m2:.3f}")
    print(f"  Delta RMSE (M1-M2): {delta_rmse:.2f}  ({pct_improve:+.1f}%)")
    print(f"  M2 better? {'YES' if delta_rmse > 0 else 'NO'}")

    # Per-agent breakdown
    for aid in AGENT_IDS:
        r_m1 = metrics_m1[f"RMSE_agent{aid}"]
        r_m2 = metrics_m2[f"RMSE_agent{aid}"]
        r2_a1 = metrics_m1[f"R2_agent{aid}"]
        r2_a2 = metrics_m2[f"R2_agent{aid}"]
        print(f"  Agent {aid}:  M1 RMSE={r_m1:.2f} R²={r2_a1:.3f}"
              f"  |  M2 RMSE={r_m2:.2f} R²={r2_a2:.3f}"
              f"  |  Δ={r_m1 - r_m2:.2f}")

    return {
        "variant": tag,
        "M1_RMSE": rmse_m1, "M1_MAE": mae_m1, "M1_R2": r2_m1,
        "M2_RMSE": rmse_m2, "M2_MAE": mae_m2, "M2_R2": r2_m2,
        "delta_RMSE": delta_rmse, "pct_improve": pct_improve,
        "M1_params": params_m1, "M2_params": params_m2,
    }


def main():
    print("Loading and aggregating data...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    print(f"  Annual dataset: {len(annual)} rows "
          f"({annual['Year'].nunique()} years × {len(AGENT_IDS)} agents)")

    # Split: M1 train ≤ 2004, M2 train ≤ 2005, eval ≥ 2006
    df_m1_train, df_eval = split_data(
        annual, train_end_year=CP_YEAR - 1, eval_start_year=CP_YEAR + 1)
    df_m2_train, _ = split_data(
        annual, train_end_year=CP_YEAR, eval_start_year=CP_YEAR + 1)

    print(f"  M1 train: {len(df_m1_train)} obs (≤{CP_YEAR-1})")
    print(f"  M2 train: {len(df_m2_train)} obs (≤{CP_YEAR})")
    print(f"  Eval:     {len(df_eval)} obs ({CP_YEAR+1}–2020)")

    # Run both variants
    result_with = run_comparison(df_m1_train, df_m2_train, df_eval,
                                 drop_year=False)
    result_no = run_comparison(df_m1_train, df_m2_train, df_eval,
                               drop_year=True)

    # Summary comparison
    print("\n" + "=" * 60)
    print("YEAR-FEATURE LEAKAGE CHECK — SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<12} {'M1 RMSE':>8} {'M2 RMSE':>8} "
          f"{'Δ RMSE':>8} {'% Impr':>8} {'M1 R²':>7} {'M2 R²':>7}")
    print("-" * 60)
    for r in [result_with, result_no]:
        print(f"{r['variant']:<12} {r['M1_RMSE']:>8.2f} {r['M2_RMSE']:>8.2f} "
              f"{r['delta_RMSE']:>8.2f} {r['pct_improve']:>+7.1f}% "
              f"{r['M1_R2']:>7.3f} {r['M2_R2']:>7.3f}")
    print("=" * 60)

    if result_no["delta_RMSE"] > 0:
        print("\nCONCLUSION: M2's advantage HOLDS without Year feature.")
        print("  The M1/M2 delta is NOT an artifact of Year extrapolation.")
    else:
        print("\nCONCLUSION: M2's advantage DISAPPEARS without Year feature.")
        print("  The M1/M2 delta may be confounded by Year-as-feature.")

    # Save results
    out_dir = RESULTS_DIR / "leakage_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for r in [result_with, result_no]:
        summary_rows.append({k: v for k, v in r.items()
                             if k not in ("M1_params", "M2_params")})
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "year_leakage_check.csv", index=False)
    print(f"\nSaved results to {out_dir / 'year_leakage_check.csv'}")


if __name__ == "__main__":
    main()
