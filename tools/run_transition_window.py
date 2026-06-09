#!/usr/bin/env python3
"""
Transition-Window Robustness Analysis (per-agent CP design).

Each of the 8 non-stationary agents has its own BOCPD-derived CP year.
For each agent, tests M1 vs M2 across CP-1, CP, CP+1 (3 candidates).

M1 (Stationary):        trains on years < candidate
M2 (Changepoint-Aware): trains on years <= candidate
Eval:                    candidate+1 through 2020

Produces:
  results/transition_window/summary.csv         — per-agent summary (8 rows)
  results/transition_window/per_agent_detail.csv — agent × candidate (24 rows)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure sibling imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_xgboost_abm import (
    BLOCK_SIZE,
    JITTER_WIDTHS,
    MASTER_SEED,
    N_SENS_BOOT,
    XGB_DEFAULTS,
    aggregate_to_annual,
    compute_kge,
    constrained_block_bootstrap_years,
    load_agent_data,
    optuna_tune,
    sample_hyperparams,
    train_with_early_stopping,
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ────────────────────────────────────────────────────────────────
# Per-agent CP years derived from BOCPD robustness sweep (threshold 0.3).
# Agent 3 dropped (no CP >= 0.3; effectively stationary).
AGENT_CP = {
    2:  2013,   # C2, p=0.41
    24: 2003,   # C2, p=0.33
    28: 2011,   # C2, p=0.33
    29: 2011,   # C2, p=0.32
    12: 2003,   # C1, p=0.33
    14: 2004,   # C1, p=0.31
    18: 2003,   # C1, p=0.88
    20: 2004,   # C1, p=0.32
}
AGENT_IDS = sorted(AGENT_CP.keys())
EVAL_END = 2020
N_BOOT = 100                                        # N_SENS_BOOT convention
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "transition_window"


# ── Local Feature Preparation (8-agent version) ─────────────────────────────
def prepare_features(df, agent_ids=None, add_agent_dummies=True, drop_year=False):
    """Build feature matrix X and target vector y for all 8 non-stationary agents.

    This is a local version that creates agent dummies for all 8
    non-stationary agents, unlike the 5-agent version in run_xgboost_abm.py.

    Args:
        df: DataFrame with feature columns and Irrigation_Depth target.
        agent_ids: List of agent IDs for dummy creation. Defaults to AGENT_IDS.
        add_agent_dummies: If True (default), add one-hot encoded AgentID columns.
        drop_year: If True, exclude Year from the feature matrix.
    """
    if agent_ids is None:
        agent_ids = AGENT_IDS

    feature_cols = [
        "Year", "Precipitation", "Temperature",
        "Corn", "Wheat", "Soybeans", "Sorghum", "Diesel",
    ]
    X = df[feature_cols].copy()
    if drop_year:
        X = X.drop(columns=["Year"])
    if add_agent_dummies:
        for aid in agent_ids:
            X[f"Agent_{aid}"] = (df["AgentID"] == aid).astype(int)
    y = df["Irrigation_Depth"].values
    return X, y


# ── Local Metrics (8-agent version) ─────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R2, Bias, KGE."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
        "KGE": compute_kge(y_true, y_pred),
    }


def compute_per_agent_metrics(y_true, y_pred, agent_ids_arr, agent_ids=None):
    """Compute per-agent metrics for all 8 non-stationary agents.

    Args:
        y_true: array of true values
        y_pred: array of predicted values
        agent_ids_arr: array of agent IDs corresponding to each sample
        agent_ids: list of agent IDs to iterate over (default: AGENT_IDS)

    Returns:
        dict mapping agent_id -> metrics dict
    """
    if agent_ids is None:
        agent_ids = AGENT_IDS
    result = {}
    for aid in agent_ids:
        mask = agent_ids_arr == aid
        if mask.sum() == 0:
            continue
        result[aid] = compute_metrics(y_true[mask], y_pred[mask])
    return result


# ── Per-Agent-CP-Candidate Logic ──────────────────────────────────────────
def run_agent_cp_candidate(aid, cp, annual_agent, master_seed=MASTER_SEED,
                           n_boot=N_BOOT, block_size=BLOCK_SIZE,
                           ensemble_mode="full"):
    """Run M1 vs M2 comparison for a single agent at a single CP candidate.

    Args:
        aid: agent ID
        cp: changepoint year candidate
        annual_agent: annual DataFrame filtered to this agent only
        master_seed: root seed
        n_boot: bootstrap ensemble members
        block_size: block size for moving-block bootstrap
        ensemble_mode: "full" (bootstrap+jitter), "bootstrap_only", or
            "jitter_only".

    Returns:
        detail_row: dict with metrics for this agent × candidate
    """
    eval_start = cp + 1
    df_m1_train = annual_agent[annual_agent["Year"] < cp].copy()
    df_m2_train = annual_agent[annual_agent["Year"] <= cp].copy()
    df_eval = annual_agent[
        (annual_agent["Year"] >= eval_start) & (annual_agent["Year"] <= EVAL_END)
    ].copy()

    m1_years = sorted(df_m1_train["Year"].unique())
    m2_years = sorted(df_m2_train["Year"].unique())
    eval_years = sorted(df_eval["Year"].unique())

    if len(m1_years) < 3 or len(eval_years) < 2:
        print(f"    Skipping CP={cp}: insufficient data "
              f"(M1 train={len(m1_years)}yr, eval={len(eval_years)}yr)")
        return None

    print(f"    CP={cp}: M1 train {m1_years[0]}-{m1_years[-1]} ({len(m1_years)}yr), "
          f"M2 train {m2_years[0]}-{m2_years[-1]} ({len(m2_years)}yr), "
          f"eval {eval_years[0]}-{eval_years[-1]} ({len(eval_years)}yr)")

    # Prepare features (single agent, no agent dummies needed)
    X_m1_tune, y_m1_tune = prepare_features(df_m1_train, agent_ids=[aid],
                                             drop_year=False)
    X_m2_tune, y_m2_tune = prepare_features(df_m2_train, agent_ids=[aid],
                                             drop_year=False)
    X_m1_train, y_m1_train = prepare_features(df_m1_train, agent_ids=[aid])
    X_m2_train, y_m2_train = prepare_features(df_m2_train, agent_ids=[aid])
    X_eval, y_eval = prepare_features(df_eval, agent_ids=[aid])

    m1_years_arr = df_m1_train["Year"].values
    m2_years_arr = df_m2_train["Year"].values

    # Optuna tuning
    m1_seed = master_seed + aid * 100 + cp
    m2_seed = master_seed + aid * 100 + cp + 1

    _, m1_best_params = optuna_tune(X_m1_tune, y_m1_tune, seed=m1_seed,
                                     objective_metric="kge")
    _, m2_best_params = optuna_tune(X_m2_tune, y_m2_tune, seed=m2_seed,
                                     objective_metric="kge")

    # Point-estimate predictions
    m1_params = {**m1_best_params, "objective": "reg:squarederror",
                 "early_stopping_rounds": 50, "verbosity": 0}
    m2_params = {**m2_best_params, "objective": "reg:squarederror",
                 "early_stopping_rounds": 50, "verbosity": 0}

    m1_point = train_with_early_stopping(X_m1_train, y_m1_train, m1_params,
                                         val_years=3, years=m1_years_arr)
    m1_preds = m1_point.predict(X_eval)

    m2_point = train_with_early_stopping(X_m2_train, y_m2_train, m2_params,
                                         val_years=3, years=m2_years_arr)
    m2_preds = m2_point.predict(X_eval)

    m1_metrics = compute_metrics(y_eval, m1_preds)
    m2_metrics = compute_metrics(y_eval, m2_preds)

    print(f"      M1: RMSE={m1_metrics['RMSE']:.1f}, "
          f"KGE={m1_metrics['KGE']:.3f}, R2={m1_metrics['R2']:.3f}")
    print(f"      M2: RMSE={m2_metrics['RMSE']:.1f}, "
          f"KGE={m2_metrics['KGE']:.3f}, R2={m2_metrics['R2']:.3f}")

    # Bootstrap ensemble
    m1_ens = _run_agent_bootstrap(
        df_m1_train, X_eval, y_eval, m1_best_params,
        master_seed + aid * 100 + cp, n_boot, block_size,
        agent_ids=[aid], ensemble_mode=ensemble_mode)
    m2_ens = _run_agent_bootstrap(
        df_m2_train, X_eval, y_eval, m2_best_params,
        master_seed + aid * 100 + cp + 1000, n_boot, block_size,
        agent_ids=[aid], ensemble_mode=ensemble_mode)

    detail_row = {
        "AgentID": aid,
        "CP_Year": cp,
        "CP_Offset": cp - AGENT_CP[aid],  # -1, 0, or +1
        "N_train_M1": len(m1_years),
        "N_train_M2": len(m2_years),
        "N_eval": len(eval_years),
        "M1_RMSE": m1_metrics["RMSE"],
        "M1_MAE": m1_metrics["MAE"],
        "M1_R2": m1_metrics["R2"],
        "M1_KGE": m1_metrics["KGE"],
        "M1_Bias": m1_metrics["Bias"],
        "M2_RMSE": m2_metrics["RMSE"],
        "M2_MAE": m2_metrics["MAE"],
        "M2_R2": m2_metrics["R2"],
        "M2_KGE": m2_metrics["KGE"],
        "M2_Bias": m2_metrics["Bias"],
        "M2_wins_RMSE": int(m2_metrics["RMSE"] < m1_metrics["RMSE"]),
        "M2_wins_KGE": int(m2_metrics["KGE"] > m1_metrics["KGE"]),
        "M2_wins_R2": int(m2_metrics["R2"] > m1_metrics["R2"]),
        "M1_RMSE_ens_median": np.median(m1_ens["rmse"]),
        "M2_RMSE_ens_median": np.median(m2_ens["rmse"]),
        "M1_KGE_ens_median": np.nanmedian(m1_ens["kge"]),
        "M2_KGE_ens_median": np.nanmedian(m2_ens["kge"]),
        "M1_R2_ens_median": np.median(m1_ens["r2"]),
        "M2_R2_ens_median": np.median(m2_ens["r2"]),
        "M2_wins_ens_RMSE": int(np.median(m2_ens["rmse"]) < np.median(m1_ens["rmse"])),
        "M2_wins_ens_KGE": int(np.nanmedian(m2_ens["kge"]) > np.nanmedian(m1_ens["kge"])),
        "M2_wins_ens_R2": int(np.median(m2_ens["r2"]) > np.median(m1_ens["r2"])),
    }
    return detail_row


def _run_agent_bootstrap(df_train, X_eval, y_eval, best_params,
                          seed, n_boot, block_size, agent_ids,
                          ensemble_mode="full"):
    """Run bootstrap ensemble for a single agent's model.

    Returns:
        dict with keys 'rmse', 'r2', 'kge' — each an array of shape (n_boot,).
    """
    train_years_sorted = sorted(df_train["Year"].unique())

    # Fallback for very short training sets
    if len(train_years_sorted) <= 3:
        X_fb, y_fb = prepare_features(df_train, agent_ids=agent_ids)
        preds = train_with_early_stopping(
            X_fb, y_fb, {
                **best_params, "objective": "reg:squarederror",
                "early_stopping_rounds": 50, "verbosity": 0,
            }, val_years=min(2, len(train_years_sorted) - 1),
            years=df_train["Year"].values).predict(X_eval)
        return {
            "rmse": np.array([np.sqrt(mean_squared_error(y_eval, preds))]),
            "r2": np.array([r2_score(y_eval, preds)]),
            "kge": np.array([compute_kge(y_eval, preds)]),
        }

    anchor_tail = train_years_sorted[-3:]
    pre_cp_years = train_years_sorted[:-3]

    rng = np.random.default_rng(seed)
    rmse_arr = np.zeros(n_boot)
    r2_arr = np.zeros(n_boot)
    kge_arr = np.zeros(n_boot)

    for i in range(n_boot):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        if ensemble_mode == "jitter_only":
            df_boot = df_train
        else:
            boot_years = constrained_block_bootstrap_years(
                pre_cp_years, anchor_tail, block_size, member_rng)
            frames = [df_train[df_train["Year"] == yr] for yr in boot_years]
            df_boot = pd.concat(frames, ignore_index=True)

        boot_years_arr = df_boot["Year"].values
        X_boot, y_boot = prepare_features(df_boot, agent_ids=agent_ids)

        if ensemble_mode == "bootstrap_only":
            params = {**best_params, "objective": "reg:squarederror",
                      "early_stopping_rounds": 50, "verbosity": 0}
        else:
            params = sample_hyperparams(member_rng, base_params=best_params)

        model = train_with_early_stopping(
            X_boot, y_boot, params, val_years=3, years=boot_years_arr)
        preds = model.predict(X_eval)
        rmse_arr[i] = np.sqrt(mean_squared_error(y_eval, preds))
        r2_arr[i] = r2_score(y_eval, preds)
        kge_arr[i] = compute_kge(y_eval, preds)

    return {"rmse": rmse_arr, "r2": r2_arr, "kge": kge_arr}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Run per-agent transition-window robustness analysis."""
    print("Transition-Window Robustness Analysis (per-agent CP)")
    print("=" * 60)
    print(f"Agents: {AGENT_IDS}")
    print(f"Per-agent CPs: {AGENT_CP}")
    print(f"Bootstrap members: {N_BOOT}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and aggregate data for all agents
    print("Loading data...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    print(f"  Annual records: {len(annual)} rows, "
          f"years {annual['Year'].min()}-{annual['Year'].max()}")

    all_detail_rows = []

    for aid in AGENT_IDS:
        base_cp = AGENT_CP[aid]
        candidates = [base_cp - 1, base_cp, base_cp + 1]
        annual_agent = annual[annual["AgentID"] == aid].copy()

        print(f"\n{'='*60}")
        print(f"Agent {aid} (base CP={base_cp}, window={candidates})")
        print(f"{'='*60}")

        for cp in candidates:
            row = run_agent_cp_candidate(aid, cp, annual_agent)
            if row is not None:
                all_detail_rows.append(row)

    # Save detail results
    detail_df = pd.DataFrame(all_detail_rows)
    detail_path = OUTPUT_DIR / "per_agent_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    # Build per-agent summary (best candidate per agent by KGE)
    summary_rows = []
    for aid in AGENT_IDS:
        agent_rows = detail_df[detail_df["AgentID"] == aid]
        if agent_rows.empty:
            continue
        # Best candidate = the one where M2 KGE is highest
        best_idx = agent_rows["M2_KGE"].idxmax()
        best = agent_rows.loc[best_idx]
        summary_rows.append({
            "AgentID": aid,
            "Base_CP": AGENT_CP[aid],
            "Best_CP": int(best["CP_Year"]),
            "M1_RMSE": best["M1_RMSE"],
            "M2_RMSE": best["M2_RMSE"],
            "RMSE_change_pct": (best["M2_RMSE"] - best["M1_RMSE"]) / best["M1_RMSE"] * 100,
            "M1_KGE": best["M1_KGE"],
            "M2_KGE": best["M2_KGE"],
            "M1_R2": best["M1_R2"],
            "M2_R2": best["M2_R2"],
            "M2_wins_RMSE": int(best["M2_wins_RMSE"]),
            "M2_wins_KGE": int(best["M2_wins_KGE"]),
            "M2_wins_ens_RMSE": int(best["M2_wins_ens_RMSE"]),
            "M2_wins_ens_KGE": int(best["M2_wins_ens_KGE"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("Results saved:")
    print(f"  {detail_path} ({len(detail_df)} rows)")
    print(f"  {summary_path} ({len(summary_df)} rows)")

    # Print summary
    print(f"\n{'='*60}")
    print("Per-Agent Summary (best CP by M2 KGE)")
    print(f"{'='*60}")
    cols = ["AgentID", "Base_CP", "Best_CP",
            "M1_RMSE", "M2_RMSE", "RMSE_change_pct",
            "M1_KGE", "M2_KGE", "M2_wins_RMSE", "M2_wins_KGE"]
    print(summary_df[cols].to_string(index=False))

    n_wins_rmse = summary_df["M2_wins_RMSE"].sum()
    n_wins_kge = summary_df["M2_wins_KGE"].sum()
    n = len(summary_df)
    print(f"\nM2 wins: RMSE {n_wins_rmse}/{n}, KGE {n_wins_kge}/{n}")
    print("\nDone.")


if __name__ == "__main__":
    main()
