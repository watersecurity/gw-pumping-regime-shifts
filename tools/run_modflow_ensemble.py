#!/usr/bin/env python3
"""
MODFLOW Uncertainty Propagation: Partially Pooled XGBoost M1 vs M2 Ensemble.

Expands the XGBoost M1 vs M2 pipeline from Agent 12 (single agent) and
Phase 1 uniform-CP pooled model to a partially pooled framework covering all
8 non-stationary agents with heterogeneous per-agent BOCPD changepoint years.

Per-agent CP years (best CP from BOCPD transition-window analysis, KGE):
  Agents 12, 14, 18, 24 → CP=2004
  Agent 20 → CP=2005
  Agent 29 → CP=2011
  Agents 2, 28 → CP=2012
  Agent 3 excluded (no CP ≥ 0.3 threshold; effectively stationary)

M1 (Stationary):        trains on years < CP  (pre-changepoint only)
M2 (Changepoint-Aware): trains on years <= CP (includes CP year)

Produces:
  results/modflow_propagation/uncertainty_summary.csv
      Per-agent M1 vs M2 PI widths, medians, and ensemble RMSE
  results/modflow_propagation/rrca_export/rrca_observed.csv
      Observed values through each agent's CP year (8 agents)
  results/modflow_propagation/rrca_export/rrca_members_m1.csv
      Full 100-member M1 ensemble predictions (long-format, mm + inches)
  results/modflow_propagation/rrca_export/rrca_members_m2.csv
      Full 100-member M2 ensemble predictions (long-format, mm + inches)
  results/modflow_propagation/rrca_export/rrca_summary.csv
      Per agent-year M1/M2 medians and [5th, 95th] PIs (mm + inches)

Design decisions (from Phase 2 CONTEXT.md):
  D-03/D-04: Partially pooled framework — 8 agents fit jointly with agent dummies
  D-05/D-06: Training rows defined per agent's detected CP (heterogeneous windows)
  D-07/D-08: 100 bootstrap members, constrained block bootstrap (block_size=3)
  D-09/D-10: Both mm (native) and inches (authoritative) output columns
  D-11/D-12: Observed through CP year; predictions replace CP+1 through 2020
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_xgboost_abm import (
    BLOCK_SIZE,
    JITTER_WIDTHS,
    MASTER_SEED,
    aggregate_to_annual,
    compute_kge,
    constrained_block_bootstrap_years,
    load_agent_data,
    optuna_tune,
    sample_hyperparams,
    train_with_early_stopping,
)
from run_transition_window import (
    compute_metrics,
    compute_per_agent_metrics,
    prepare_features,
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

# Per-agent CP years — best CP from BOCPD transition-window analysis (KGE)
# Agent 3 excluded: no CP >= 0.3 threshold (effectively stationary)
PER_AGENT_CP = {
    2:  2012,
    12: 2004,
    14: 2004,
    18: 2004,
    20: 2005,
    24: 2004,
    28: 2012,
    29: 2011,
}

# Per-agent eval start = CP + 1 (D-12)
PER_AGENT_EVAL_START = {aid: cp + 1 for aid, cp in PER_AGENT_CP.items()}

AGENT_IDS_8 = sorted(PER_AGENT_CP.keys())  # 8 non-stationary agents (Agent 3 dropped)
N_BOOT = 100        # Bootstrap members per model (D-07)
MM_TO_INCHES = 25.4
EVAL_END = 2020

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "modflow_propagation"
RRCA_DIR = RESULTS_DIR / "rrca_export"


# ── Training / Eval Set Construction ──────────────────────────────────────────

def build_pooled_train(annual, model="M1"):
    """Build partially pooled training DataFrame respecting per-agent CP years.

    For M1: each agent contributes rows where Year < CP (pre-changepoint)
    For M2: each agent contributes rows where Year <= CP (through CP year)

    Args:
        annual: Annual DataFrame with columns AgentID, Year, Irrigation_Depth, etc.
        model: "M1" (pre-CP only) or "M2" (through CP year).

    Returns:
        Concatenated DataFrame of per-agent CP-filtered rows.
    """
    frames = []
    for aid, cp in PER_AGENT_CP.items():
        agent_df = annual[annual["AgentID"] == aid]
        if model == "M1":
            frames.append(agent_df[agent_df["Year"] < cp])
        else:  # M2
            frames.append(agent_df[agent_df["Year"] <= cp])
    return pd.concat(frames, ignore_index=True)


def build_pooled_eval(annual):
    """Build pooled evaluation DataFrame with per-agent eval start dates.

    Each agent's eval window starts at CP+1 (per PER_AGENT_EVAL_START)
    and runs through 2020.

    Args:
        annual: Annual DataFrame with all 8 agents.

    Returns:
        Concatenated DataFrame sorted by (AgentID, Year) with 104 rows
        when annual covers 1993-2020 for all 8 agents.
    """
    frames = []
    for aid, eval_start in PER_AGENT_EVAL_START.items():
        agent_df = annual[annual["AgentID"] == aid]
        frames.append(
            agent_df[
                (agent_df["Year"] >= eval_start) &
                (agent_df["Year"] <= EVAL_END)
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["AgentID", "Year"]).reset_index(drop=True)


# ── Optuna Tuning ─────────────────────────────────────────────────────────────

def run_optuna_tuning(df_m1_train, df_m2_train, n_trials=100):
    """Tune hyperparameters for M1 and M2 pooled training sets.

    Tunes once per model on the concatenated 8-agent training set following
    the Phase 1 convention (avoid per-agent tuning: D-03/D-04).

    Args:
        df_m1_train: M1 training DataFrame (pre-CP rows for all agents).
        df_m2_train: M2 training DataFrame (through-CP rows for all agents).
        n_trials: Number of Optuna trials (default 100).

    Returns:
        (m1_best_params, m2_best_params): tuple of hyperparameter dicts.
    """
    print(f"  Tuning M1 ({len(df_m1_train)} rows)...")
    X_m1, y_m1 = prepare_features(df_m1_train, agent_ids=AGENT_IDS_8, drop_year=False)
    _, m1_best_params = optuna_tune(X_m1, y_m1, n_trials=n_trials, seed=MASTER_SEED,
                                     objective_metric="kge")

    print(f"  Tuning M2 ({len(df_m2_train)} rows)...")
    X_m2, y_m2 = prepare_features(df_m2_train, agent_ids=AGENT_IDS_8, drop_year=False)
    _, m2_best_params = optuna_tune(X_m2, y_m2, n_trials=n_trials, seed=MASTER_SEED + 1,
                                     objective_metric="kge")

    return m1_best_params, m2_best_params


# ── Bootstrap Ensemble ────────────────────────────────────────────────────────

def run_pooled_bootstrap(df_train, X_eval, best_params,
                         n_boot=N_BOOT, master_seed=MASTER_SEED, seed_offset=0,
                         return_models=False):
    """Run moving-block bootstrap ensemble on pooled training set.

    Follows the Phase 1 convention from run_transition_window._run_bootstrap_ensemble:
    - Global anchor tail = last 3 unique training years
    - Pre-anchor = remaining years
    - Per-member: block bootstrap years, resample rows, jitter hyperparams, train, predict

    For M1 the global anchor tail will be [2002, 2003, 2004] and pre-anchor
    [1993, ..., 2001] (9 years >= block_size=3). Agent 14's single 1993-year
    rows participate normally in bootstrap draws (Pitfall 1 from RESEARCH.md).

    Args:
        df_train: Training DataFrame with Year and AgentID columns.
        X_eval: Feature matrix for evaluation (prepared with prepare_features).
        best_params: Optuna-tuned hyperparameter dict.
        n_boot: Number of bootstrap members.
        master_seed: Base seed for RNG.
        seed_offset: Offset added to master_seed for M1/M2 separation.
        return_models: If True, also return the list of trained models.

    Returns:
        np.ndarray of shape (n_boot, n_eval_rows).
        If return_models=True: (np.ndarray, list of models).
    """
    train_years_sorted = sorted(df_train["Year"].unique())
    anchor_tail = train_years_sorted[-3:]
    pre_anchor_years = train_years_sorted[:-3]

    # Verify no overlap (Pitfall 2)
    overlap = set(pre_anchor_years) & set(anchor_tail)
    if overlap:
        raise ValueError(
            f"Bootstrap anchor contamination: overlap = {overlap}"
        )

    rng = np.random.default_rng(master_seed + seed_offset)
    member_preds = []
    models = [] if return_models else None

    for i in range(n_boot):
        member_rng = np.random.default_rng(rng.integers(0, 2 ** 31))

        boot_years = constrained_block_bootstrap_years(
            pre_anchor_years, anchor_tail, BLOCK_SIZE, member_rng
        )
        frames = [df_train[df_train["Year"] == yr] for yr in boot_years]
        df_boot = pd.concat(frames, ignore_index=True)

        # Prepare features with all 8 agent dummies (Pitfall 3)
        X_boot, y_boot = prepare_features(
            df_boot, agent_ids=AGENT_IDS_8, drop_year=False
        )

        params = sample_hyperparams(member_rng, base_params=best_params)
        model = train_with_early_stopping(
            X_boot, y_boot, params,
            val_years=3, years=df_boot["Year"].values,
        )
        member_preds.append(model.predict(X_eval))
        if return_models:
            models.append(model)

        if (i + 1) % 10 == 0:
            print(f"    Bootstrap member {i + 1}/{n_boot} complete")

    preds_arr = np.array(member_preds)  # shape: (n_boot, n_eval_rows)
    if return_models:
        return preds_arr, models
    return preds_arr


# ── Per-Agent PI Summary ──────────────────────────────────────────────────────

def compute_agent_pi_summary(member_preds, y_eval, eval_agents, model_label):
    """Compute per-agent prediction interval summary.

    For each of the 8 non-stationary agents: median prediction, 5th/95th PI,
    mean PI width, RMSE of median vs observed, and ensemble RMSE median.

    Uses canonical column name `pi_width_mean_mm` per RESEARCH.md Pattern 5.
    The `m1_` / `m2_` prefix is added when combining M1 and M2 summaries in
    the output CSV.

    Args:
        member_preds: np.ndarray of shape (n_boot, n_eval_rows).
        y_eval: Observed values for eval rows.
        eval_agents: Array of AgentID for each eval row.
        model_label: "M1" or "M2" (stored in each row for reference).

    Returns:
        List of dicts, one per agent, with keys:
          agent, model, cp_year, eval_start, eval_end,
          rmse_point, r2_point,
          pi_width_mean_mm, pi_lo_mean_mm, pi_hi_mean_mm,
          ens_rmse_median
    """
    from sklearn.metrics import r2_score, mean_squared_error

    rows = []
    for aid in AGENT_IDS_8:
        mask = eval_agents == aid
        if mask.sum() == 0:
            continue

        agent_preds = member_preds[:, mask]   # (n_boot, n_agent_years)
        y_true = y_eval[mask]

        median_pred = np.median(agent_preds, axis=0)
        lo = np.percentile(agent_preds, 5, axis=0)
        hi = np.percentile(agent_preds, 95, axis=0)
        pi_width = hi - lo

        # Median RMSE across per-member RMSEs (ensemble spread metric)
        per_member_rmse = np.array([
            np.sqrt(np.mean((agent_preds[m] - y_true) ** 2))
            for m in range(agent_preds.shape[0])
        ])
        per_member_kge = np.array([
            compute_kge(y_true, agent_preds[m])
            for m in range(agent_preds.shape[0])
        ])

        rows.append({
            "agent": int(aid),
            "model": model_label,
            "cp_year": int(PER_AGENT_CP[aid]),
            "eval_start": int(PER_AGENT_EVAL_START[aid]),
            "eval_end": int(EVAL_END),
            "rmse_point": float(np.sqrt(np.mean((median_pred - y_true) ** 2))),
            "r2_point": float(r2_score(y_true, median_pred)),
            "kge_point": float(compute_kge(y_true, median_pred)),
            "pi_width_mean_mm": float(pi_width.mean()),
            "pi_lo_mean_mm": float(lo.mean()),
            "pi_hi_mean_mm": float(hi.mean()),
            "ens_rmse_median": float(np.median(per_member_rmse)),
            "ens_kge_median": float(np.nanmedian(per_member_kge)),
        })

    return rows


# ── RRCA Export Functions ──────────────────────────────────────────────────────

def build_rrca_observed_rows(annual):
    """Build observed RRCA rows for the 8 non-stationary agents.

    Covers the 8 non-stationary agents only (not all 43 RRCA agents) — scope
    matches the Phase 2 ensemble predictions. Observed values are retained
    through each agent's CP year per D-11.

    Args:
        annual: Annual DataFrame with all 8 agents.

    Returns:
        DataFrame with columns: year, agent, depth_mm, depth_inches, source,
        model, member.
        Rows: observed through each agent's CP year.
    """
    frames = []
    for aid, cp in PER_AGENT_CP.items():
        agent_df = annual[
            (annual["AgentID"] == aid) & (annual["Year"] <= cp)
        ].copy()
        if len(agent_df) == 0:
            continue
        obs_df = pd.DataFrame({
            "year": agent_df["Year"].astype(int).values,
            "agent": agent_df["AgentID"].astype(int).values,
            "depth_mm": agent_df["Irrigation_Depth"].round(2).values,
            "depth_inches": (agent_df["Irrigation_Depth"] / MM_TO_INCHES).round(1).values,
            "source": "observed",
            "model": pd.array([pd.NA] * len(agent_df), dtype="object"),
            "member": pd.array([pd.NA] * len(agent_df), dtype="Int64"),
        })
        frames.append(obs_df)

    return pd.concat(frames, ignore_index=True)


def build_rrca_member_rows(eval_df, member_preds, model_label):
    """Build full-member RRCA rows for ensemble predictions.

    Args:
        eval_df: Eval DataFrame with Year and AgentID columns.
        member_preds: np.ndarray of shape (n_boot, n_eval_rows).
        model_label: "M1" or "M2".

    Returns:
        DataFrame with columns: year, agent, depth_mm, depth_inches,
        source, model, member (one row per bootstrap member per eval row).
    """
    years = eval_df["Year"].values
    agents = eval_df["AgentID"].values
    n_boot, n_eval = member_preds.shape

    rows = []
    for member_idx in range(n_boot):
        for row_idx in range(n_eval):
            pred_mm = float(member_preds[member_idx, row_idx])
            rows.append({
                "year": int(years[row_idx]),
                "agent": int(agents[row_idx]),
                "depth_mm": round(pred_mm, 2),
                "depth_inches": round(pred_mm / MM_TO_INCHES, 1),
                "source": "ensemble",
                "model": model_label,
                "member": member_idx,
            })

    return pd.DataFrame(rows)


def build_rrca_summary_rows(eval_df, m1_member_preds, m2_member_preds):
    """Build per agent-year summary rows with M1 and M2 medians and PIs.

    Computes M1/M2 median, 5th and 95th percentile predictions for each
    agent-year in the eval set.

    Args:
        eval_df: Eval DataFrame with Year, AgentID, and Irrigation_Depth.
        m1_member_preds: np.ndarray of shape (n_boot, n_eval_rows) for M1.
        m2_member_preds: np.ndarray of shape (n_boot, n_eval_rows) for M2.

    Returns:
        DataFrame with columns: year, agent, observed_mm, observed_inches,
        m1_median_mm, m1_median_inches, m2_median_mm, m2_median_inches,
        m1_pi_lo_mm, m1_pi_hi_mm, m2_pi_lo_mm, m2_pi_hi_mm,
        m1_pi_lo_inches, m1_pi_hi_inches, m2_pi_lo_inches, m2_pi_hi_inches.
    """
    years = eval_df["Year"].values
    agents = eval_df["AgentID"].values
    observed_mm = eval_df["Irrigation_Depth"].values

    m1_median = np.median(m1_member_preds, axis=0)
    m1_lo = np.percentile(m1_member_preds, 5, axis=0)
    m1_hi = np.percentile(m1_member_preds, 95, axis=0)

    m2_median = np.median(m2_member_preds, axis=0)
    m2_lo = np.percentile(m2_member_preds, 5, axis=0)
    m2_hi = np.percentile(m2_member_preds, 95, axis=0)

    rows = []
    for i in range(len(eval_df)):
        obs_mm = float(observed_mm[i])
        rows.append({
            "year": int(years[i]),
            "agent": int(agents[i]),
            "observed_mm": round(obs_mm, 2),
            "observed_inches": round(obs_mm / MM_TO_INCHES, 1),
            "m1_median_mm": round(float(m1_median[i]), 2),
            "m1_median_inches": round(float(m1_median[i]) / MM_TO_INCHES, 1),
            "m2_median_mm": round(float(m2_median[i]), 2),
            "m2_median_inches": round(float(m2_median[i]) / MM_TO_INCHES, 1),
            "m1_pi_lo_mm": round(float(m1_lo[i]), 2),
            "m1_pi_hi_mm": round(float(m1_hi[i]), 2),
            "m2_pi_lo_mm": round(float(m2_lo[i]), 2),
            "m2_pi_hi_mm": round(float(m2_hi[i]), 2),
            "m1_pi_lo_inches": round(float(m1_lo[i]) / MM_TO_INCHES, 1),
            "m1_pi_hi_inches": round(float(m1_hi[i]) / MM_TO_INCHES, 1),
            "m2_pi_lo_inches": round(float(m2_lo[i]) / MM_TO_INCHES, 1),
            "m2_pi_hi_inches": round(float(m2_hi[i]) / MM_TO_INCHES, 1),
        })

    return pd.DataFrame(rows)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    """Run the full partially pooled MODFLOW uncertainty propagation pipeline.

    Steps:
      a. Load and aggregate 8-agent annual data
      b. Build M1 and M2 pooled training sets
      c. Build pooled eval set
      d. Optuna tuning (one pass per model on concatenated set)
      e. Bootstrap M1 (100 members, seed_offset=0)
      f. Bootstrap M2 (100 members, seed_offset=1000)
      g. Compute per-agent PI summary for both models
      h. Save uncertainty summary CSV
      i. Save RRCA observed, M1 member, M2 member, and summary CSVs
    """
    print("=" * 60)
    print("MODFLOW Uncertainty Propagation: Partially Pooled M1 vs M2")
    print("=" * 60)

    # ── a. Load data ─────────────────────────────────────────────────────────
    print("\n[1/9] Loading 8-agent data...")
    raw = load_agent_data(AGENT_IDS_8)
    annual = aggregate_to_annual(raw)
    print(f"  Loaded {len(annual)} annual rows for {annual['AgentID'].nunique()} agents "
          f"({annual['Year'].min()}-{annual['Year'].max()})")

    # ── b. Build M1/M2 training sets ─────────────────────────────────────────
    print("\n[2/9] Building pooled training sets...")
    df_m1_train = build_pooled_train(annual, model="M1")
    df_m2_train = build_pooled_train(annual, model="M2")
    print(f"  M1 training: {len(df_m1_train)} rows")
    print(f"  M2 training: {len(df_m2_train)} rows")
    print(f"  NOTE: Agent 14 contributes {len(df_m1_train[df_m1_train['AgentID'] == 14])} "
          f"M1 row(s) — sparse contribution borrows from pooled model")

    # ── c. Build eval set ────────────────────────────────────────────────────
    print("\n[3/9] Building eval set...")
    df_eval = build_pooled_eval(annual)
    print(f"  Eval: {len(df_eval)} rows (per-agent eval start from PER_AGENT_EVAL_START)")
    X_eval, y_eval = prepare_features(df_eval, agent_ids=AGENT_IDS_8, drop_year=False)
    eval_agents = df_eval["AgentID"].values

    # ── d. Optuna tuning ─────────────────────────────────────────────────────
    print("\n[4/9] Running Optuna tuning (100 trials per model)...")
    m1_best_params, m2_best_params = run_optuna_tuning(df_m1_train, df_m2_train)
    print(f"  M1 best: {m1_best_params}")
    print(f"  M2 best: {m2_best_params}")

    # ── e. Bootstrap M1 ──────────────────────────────────────────────────────
    print(f"\n[5/9] Running M1 bootstrap ({N_BOOT} members)...")
    m1_member_preds, m1_models = run_pooled_bootstrap(
        df_m1_train, X_eval, m1_best_params,
        n_boot=N_BOOT, master_seed=MASTER_SEED, seed_offset=0,
        return_models=True,
    )
    print(f"  M1 bootstrap complete: shape {m1_member_preds.shape}")

    # ── f. Bootstrap M2 ──────────────────────────────────────────────────────
    print(f"\n[6/9] Running M2 bootstrap ({N_BOOT} members, seed_offset=1000)...")
    m2_member_preds = run_pooled_bootstrap(
        df_m2_train, X_eval, m2_best_params,
        n_boot=N_BOOT, master_seed=MASTER_SEED, seed_offset=1000,
    )
    print(f"  M2 bootstrap complete: shape {m2_member_preds.shape}")

    # ── g. Per-agent PI summary ───────────────────────────────────────────────
    print("\n[7/9] Computing per-agent PI summaries...")
    m1_summary = compute_agent_pi_summary(
        m1_member_preds, y_eval, eval_agents, model_label="M1"
    )
    m2_summary = compute_agent_pi_summary(
        m2_member_preds, y_eval, eval_agents, model_label="M2"
    )

    # Combine M1 and M2 into a wide-format summary
    m1_df = pd.DataFrame(m1_summary).set_index("agent")
    m2_df = pd.DataFrame(m2_summary).set_index("agent")

    summary_rows = []
    for aid in AGENT_IDS_8:
        row = {
            "agent": aid,
            "cp_year": PER_AGENT_CP[aid],
            "eval_start": PER_AGENT_EVAL_START[aid],
            "eval_end": EVAL_END,
            "m1_rmse_point": m1_df.loc[aid, "rmse_point"],
            "m2_rmse_point": m2_df.loc[aid, "rmse_point"],
            "m1_r2_point": m1_df.loc[aid, "r2_point"],
            "m2_r2_point": m2_df.loc[aid, "r2_point"],
            "m1_kge_point": m1_df.loc[aid, "kge_point"],
            "m2_kge_point": m2_df.loc[aid, "kge_point"],
            "m1_pi_width_mean_mm": m1_df.loc[aid, "pi_width_mean_mm"],
            "m2_pi_width_mean_mm": m2_df.loc[aid, "pi_width_mean_mm"],
            "m1_pi_lo_mean_mm": m1_df.loc[aid, "pi_lo_mean_mm"],
            "m1_pi_hi_mean_mm": m1_df.loc[aid, "pi_hi_mean_mm"],
            "m2_pi_lo_mean_mm": m2_df.loc[aid, "pi_lo_mean_mm"],
            "m2_pi_hi_mean_mm": m2_df.loc[aid, "pi_hi_mean_mm"],
            "m1_ens_rmse_median": m1_df.loc[aid, "ens_rmse_median"],
            "m2_ens_rmse_median": m2_df.loc[aid, "ens_rmse_median"],
            "m1_ens_kge_median": m1_df.loc[aid, "ens_kge_median"],
            "m2_ens_kge_median": m2_df.loc[aid, "ens_kge_median"],
            "m2_wins_rmse": int(m2_df.loc[aid, "rmse_point"] < m1_df.loc[aid, "rmse_point"]),
            "m2_wins_r2": int(m2_df.loc[aid, "r2_point"] > m1_df.loc[aid, "r2_point"]),
            "m2_wins_kge": int(m2_df.loc[aid, "kge_point"] > m1_df.loc[aid, "kge_point"]),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    for row in summary_rows:
        print(f"  Agent {row['agent']:2d}: M1 RMSE={row['m1_rmse_point']:.1f}  "
              f"M2 RMSE={row['m2_rmse_point']:.1f}  "
              f"M1 KGE={row['m1_kge_point']:.3f}  M2 KGE={row['m2_kge_point']:.3f}  "
              f"PI_width M1={row['m1_pi_width_mean_mm']:.1f} M2={row['m2_pi_width_mean_mm']:.1f} mm  "
              f"M2 wins: RMSE={bool(row['m2_wins_rmse'])} KGE={bool(row['m2_wins_kge'])}")

    n_m2_wins = sum(r["m2_wins_rmse"] for r in summary_rows)
    n_m2_wins_kge = sum(r["m2_wins_kge"] for r in summary_rows)
    print(f"  M2 wins RMSE: {n_m2_wins}/8 agents")
    print(f"  M2 wins KGE:  {n_m2_wins_kge}/8 agents")

    # ── h. Save uncertainty summary ──────────────────────────────────────────
    print("\n[8/9] Saving output CSVs...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RRCA_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "uncertainty_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # ── i. Save RRCA files ────────────────────────────────────────────────────
    # Observed rows (8 non-stationary agents only, through CP year)
    obs_df = build_rrca_observed_rows(annual)
    obs_path = RRCA_DIR / "rrca_observed.csv"
    obs_df.to_csv(obs_path, index=False)
    print(f"  Saved: {obs_path} ({len(obs_df)} rows)")

    # Build CP-year eval rows for M1 (M1 can predict at CP since it trains on < CP)
    cp_frames = []
    for aid, cp in PER_AGENT_CP.items():
        cp_rows = annual[(annual["AgentID"] == aid) & (annual["Year"] == cp)]
        cp_frames.append(cp_rows)
    df_cp_eval = pd.concat(cp_frames, ignore_index=True).sort_values(
        ["AgentID", "Year"]
    ).reset_index(drop=True)
    X_cp_eval, _ = prepare_features(df_cp_eval, agent_ids=AGENT_IDS_8, drop_year=False)

    # Predict CP year with each M1 bootstrap model
    m1_cp_preds = np.array([m.predict(X_cp_eval) for m in m1_models])
    print(f"  M1 CP-year predictions: shape {m1_cp_preds.shape}")

    # M1 full-member rows (eval window + CP year)
    m1_members_eval_df = build_rrca_member_rows(df_eval, m1_member_preds, model_label="M1")
    m1_members_cp_df = build_rrca_member_rows(df_cp_eval, m1_cp_preds, model_label="M1")
    m1_members_df = pd.concat(
        [m1_members_cp_df, m1_members_eval_df], ignore_index=True
    ).sort_values(["agent", "year", "member"]).reset_index(drop=True)
    m1_members_path = RRCA_DIR / "rrca_members_m1.csv"
    m1_members_df.to_csv(m1_members_path, index=False)
    print(f"  Saved: {m1_members_path} ({len(m1_members_df)} rows, "
          f"incl. {len(m1_members_cp_df)} CP-year rows)")

    # M2 full-member rows (eval window only — CP year is in M2 training)
    m2_members_df = build_rrca_member_rows(df_eval, m2_member_preds, model_label="M2")
    m2_members_path = RRCA_DIR / "rrca_members_m2.csv"
    m2_members_df.to_csv(m2_members_path, index=False)
    print(f"  Saved: {m2_members_path} ({len(m2_members_df)} rows)")

    # RRCA summary (per agent-year medians and PIs)
    # Include CP-year rows for M1 with M2 columns as NaN
    rrca_summary_eval_df = build_rrca_summary_rows(df_eval, m1_member_preds, m2_member_preds)

    # Build CP-year summary rows (M1 only, M2 = NaN)
    m1_cp_median = np.median(m1_cp_preds, axis=0)
    m1_cp_lo = np.percentile(m1_cp_preds, 5, axis=0)
    m1_cp_hi = np.percentile(m1_cp_preds, 95, axis=0)
    cp_summary_rows = []
    for i in range(len(df_cp_eval)):
        obs_mm = float(df_cp_eval.iloc[i]["Irrigation_Depth"])
        cp_summary_rows.append({
            "year": int(df_cp_eval.iloc[i]["Year"]),
            "agent": int(df_cp_eval.iloc[i]["AgentID"]),
            "observed_mm": round(obs_mm, 2),
            "observed_inches": round(obs_mm / MM_TO_INCHES, 1),
            "m1_median_mm": round(float(m1_cp_median[i]), 2),
            "m1_median_inches": round(float(m1_cp_median[i]) / MM_TO_INCHES, 1),
            "m2_median_mm": np.nan,
            "m2_median_inches": np.nan,
            "m1_pi_lo_mm": round(float(m1_cp_lo[i]), 2),
            "m1_pi_hi_mm": round(float(m1_cp_hi[i]), 2),
            "m2_pi_lo_mm": np.nan,
            "m2_pi_hi_mm": np.nan,
            "m1_pi_lo_inches": round(float(m1_cp_lo[i]) / MM_TO_INCHES, 1),
            "m1_pi_hi_inches": round(float(m1_cp_hi[i]) / MM_TO_INCHES, 1),
            "m2_pi_lo_inches": np.nan,
            "m2_pi_hi_inches": np.nan,
        })
    cp_summary_df = pd.DataFrame(cp_summary_rows)
    rrca_summary_df = pd.concat(
        [cp_summary_df, rrca_summary_eval_df], ignore_index=True
    ).sort_values(["agent", "year"]).reset_index(drop=True)

    rrca_summary_path = RRCA_DIR / "rrca_summary.csv"
    rrca_summary_df.to_csv(rrca_summary_path, index=False)
    print(f"  Saved: {rrca_summary_path} ({len(rrca_summary_df)} rows, "
          f"incl. {len(cp_summary_df)} CP-year rows with M2=NaN)")

    # Free M1 models to reclaim memory
    del m1_models

    print("\n[9/9] Done.")
    print("=" * 60)
    print(f"  Outputs in: {RESULTS_DIR}")
    print(f"  M2 wins RMSE: {n_m2_wins}/8 agents  "
          f"({100 * n_m2_wins // 8}%)")
    print(f"  M2 wins KGE:  {n_m2_wins_kge}/8 agents  "
          f"({100 * n_m2_wins_kge // 8}%)")


if __name__ == "__main__":
    main()
