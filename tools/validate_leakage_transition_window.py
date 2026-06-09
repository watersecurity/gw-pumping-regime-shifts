#!/usr/bin/env python3
"""
Leakage validation and PI decomposition for the 9-agent transition-window analysis.

Runs two analyses for CP=2005:
  A. Year-feature ablation: M1 vs M2 with and without Year feature
  B. PI decomposition: full vs bootstrap-only vs jitter-only ensemble

Produces:
  results/transition_window/leakage_year_ablation.csv
  results/transition_window/pi_decomposition.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_transition_window import (
    AGENT_IDS,
    BLOCK_SIZE,
    EVAL_END,
    EVAL_START,
    N_BOOT,
    OUTPUT_DIR,
    _run_bootstrap_ensemble,
    aggregate_to_annual,
    load_agent_data,
    optuna_tune,
    prepare_features,
    run_cp_candidate,
    train_with_early_stopping,
)
from run_xgboost_abm import MASTER_SEED, sample_hyperparams

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

CP = 2005  # Primary changepoint


def run_year_ablation(annual, df_eval):
    """Run M1 vs M2 with and without Year feature for CP=2005."""
    print("\n" + "=" * 60)
    print("A. Year-Feature Ablation (CP=2005)")
    print("=" * 60)

    results = []
    for drop_year, label in [(False, "with_year"), (True, "no_year")]:
        print(f"\n  Variant: {label}")
        summary, _ = run_cp_candidate(
            cp=CP, annual=annual, df_eval=df_eval,
            master_seed=MASTER_SEED, n_boot=N_BOOT,
            block_size=BLOCK_SIZE, drop_year=drop_year,
        )
        results.append({
            "variant": label,
            "M1_RMSE": summary["M1_RMSE"],
            "M1_R2": summary["M1_R2"],
            "M2_RMSE": summary["M2_RMSE"],
            "M2_R2": summary["M2_R2"],
            "delta_RMSE": summary["M1_RMSE"] - summary["M2_RMSE"],
            "pct_improve": (summary["M1_RMSE"] - summary["M2_RMSE"])
                           / summary["M1_RMSE"] * 100,
            "M1_RMSE_ens_median": summary["M1_RMSE_ens_median"],
            "M2_RMSE_ens_median": summary["M2_RMSE_ens_median"],
            "M1_R2_ens_median": summary["M1_R2_ens_median"],
            "M2_R2_ens_median": summary["M2_R2_ens_median"],
        })

    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "leakage_year_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(df.to_string(index=False))
    return df


def run_pi_decomposition(annual, df_eval):
    """Run PI decomposition: full, bootstrap-only, jitter-only for M2 at CP=2005."""
    print("\n" + "=" * 60)
    print("B. PI Decomposition (CP=2005, M2)")
    print("=" * 60)

    # Prepare M2 training data
    df_m2_train = annual[annual["Year"] <= CP].copy()
    X_eval, y_eval = prepare_features(df_eval)
    eval_agents = df_eval["AgentID"].values

    # Tune M2 once (same seed as transition-window analysis)
    m2_seed = MASTER_SEED + CP + 1
    print(f"  Tuning M2 (seed={m2_seed})...")
    _, m2_best_params = optuna_tune(
        *prepare_features(df_m2_train), seed=m2_seed)

    results = []
    for mode in ["full", "bootstrap_only", "jitter_only"]:
        print(f"\n  Mode: {mode}")
        rmse_arr, r2_arr, _, _ = _run_bootstrap_ensemble(
            cp=CP, model_label="M2", df_train=df_m2_train,
            X_eval=X_eval, y_eval=y_eval, eval_agents=eval_agents,
            best_params=m2_best_params, master_seed=MASTER_SEED,
            n_boot=N_BOOT, block_size=BLOCK_SIZE, annual=annual,
            ensemble_mode=mode,
        )

        # PI width: use ensemble predictions to compute per-observation PI
        # For pooled PI width, we report the spread of pooled RMSE as a proxy
        results.append({
            "mode": mode,
            "RMSE_median": np.median(rmse_arr),
            "RMSE_IQR_lo": np.percentile(rmse_arr, 25),
            "RMSE_IQR_hi": np.percentile(rmse_arr, 75),
            "RMSE_spread": np.percentile(rmse_arr, 75) - np.percentile(rmse_arr, 25),
            "R2_median": np.median(r2_arr),
            "R2_IQR_lo": np.percentile(r2_arr, 25),
            "R2_IQR_hi": np.percentile(r2_arr, 75),
        })

    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "pi_decomposition.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    print("Leakage Validation & PI Decomposition")
    print("=" * 60)
    print(f"Agents: {AGENT_IDS}")
    print(f"CP: {CP}")
    print(f"Eval window: {EVAL_START}-{EVAL_END}")
    print(f"Bootstrap members: {N_BOOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data for all 9 agents...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    df_eval = annual[
        (annual["Year"] >= EVAL_START) & (annual["Year"] <= EVAL_END)
    ].copy()
    print(f"  Annual records: {len(annual)} rows")
    print(f"  Eval records: {len(df_eval)} rows")

    # Run analyses
    run_year_ablation(annual, df_eval)
    run_pi_decomposition(annual, df_eval)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
