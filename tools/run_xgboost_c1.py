#!/usr/bin/env python3
"""
Pooled-by-Regime XGBoost for Cluster 1 — Agents 12, 18, 20.

Pools all 3 agents per regime with relative-time encoding (t_rel),
benchmarks linear models (Ridge, ElasticNet) against XGBoost, and uses
rolling-origin CV with agent-balanced macro-RMSE scoring.

Point estimation: pooled-by-regime with model selection.
Ensemble: per-agent bootstrap prediction intervals (unchanged).

Changepoint signals:
  Agent 12: level CP at 2004 (p=0.32–0.41)
  Agent 18: slope CP at 2004 (p=0.64–0.91)
  Agent 20: level CP at 2005 (p=0.31–0.40)
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Import reusable functions from Cluster 2 pipeline
from run_xgboost_abm import (
    load_agent_data,
    aggregate_to_annual,
    prepare_features,
    compute_metrics,
    make_constrained_moving_block_year_sequence,
    bootstrap_panel_by_year_sequence,
    train_with_early_stopping,
)

# ── Constants ────────────────────────────────────────────────────────────────
AGENT_CONFIGS = {
    12: {"cp_year": 2004},
    18: {"cp_year": 2004},
    20: {"cp_year": 2005},
}
AGENT_IDS = list(AGENT_CONFIGS.keys())

MASTER_SEED = 142          # well separated from Cluster 2's seed space (42)
N_TWO_REGIME_BOOT = 100
BLOCK_SIZE = 3

N_OPTUNA_TRIALS_POINT = 100

# Feature column constants for pooled-by-regime pipeline
BASE_FEATURES = ["Precipitation", "Temperature", "Corn", "Wheat",
                 "Soybeans", "Sorghum", "Diesel"]
AGENT_DUMMIES = ["Agent_12", "Agent_18", "Agent_20"]

# PRE regime feature sets (unbounded t_rel OK — only 2 steps of extrapolation)
PRE_FEATURE_SETS = {
    "A_base": BASE_FEATURES + AGENT_DUMMIES,
    "B_trel": ["t_rel"] + BASE_FEATURES + AGENT_DUMMIES,
    "C_trel_interact": ["t_rel"] + BASE_FEATURES + AGENT_DUMMIES +
                       ["t_rel_x_Agent_12", "t_rel_x_Agent_18", "t_rel_x_Agent_20"],
}

# POST regime feature sets (NO raw unbounded t_rel)
POST_FEATURE_SETS = {
    "A_base": BASE_FEATURES + AGENT_DUMMIES,
    "B_cap6": ["t_rel_cap"] + BASE_FEATURES + AGENT_DUMMIES,
    "B_cap8": ["t_rel_cap"] + BASE_FEATURES + AGENT_DUMMIES,
    "B_cap10": ["t_rel_cap"] + BASE_FEATURES + AGENT_DUMMIES,
    "B_capmax": ["t_rel_cap"] + BASE_FEATURES + AGENT_DUMMIES,
    "C_bins": ["bin_0_3", "bin_4_7", "bin_8plus"] + BASE_FEATURES + AGENT_DUMMIES,
    "D_cap8_interact": ["t_rel_cap"] + BASE_FEATURES + AGENT_DUMMIES +
                       ["t_rel_cap_x_Agent_12", "t_rel_cap_x_Agent_18",
                        "t_rel_cap_x_Agent_20"],
}

# Cap values for POST B_cap* feature sets
_POST_CAP_MAP = {"B_cap6": 6, "B_cap8": 8, "B_cap10": 10}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TWO_REGIME_DIR = RESULTS_DIR / "two_regime_c1"

# Pooled CP sensitivity constants
CP_CANDIDATES_SENS = [2004, 2005, 2006]
COMMON_EVAL_START = 2007
N_OPTUNA_TRIALS_SENS = 100

# POST-CP split-point sweep constants
POST_SPLIT_CANDIDATES = [2013, 2014, 2015, 2016, 2017]
SPLIT_SWEEP_DIR = TWO_REGIME_DIR / "split_sweep"

# Conservative, fixed search space for CP sensitivity
CONSERVATIVE_SEARCH_SPACE = {
    "max_depth": (2, 3),
    "min_child_weight": (5, 20),
    "gamma": (0.5, 5.0),
    "subsample": (0.7, 0.9),
    "colsample_bytree": (0.7, 0.9),
    "reg_lambda": (5.0, 30.0),
    "reg_alpha": (0.0, 10.0),
    "learning_rate": (0.03, 0.1),
    "n_estimators": (50, 500),
}


# ── Feature Helpers ──────────────────────────────────────────────────────────
def prepare_features_c1(df, drop_year=False):
    """Build feature matrix with Cluster 1 agent dummies.

    Returns X with 11 features (Year + 7 base + 3 agent dummies) or
    10 features if drop_year=True.
    """
    feature_cols = ["Year", "Precipitation", "Temperature",
                    "Corn", "Wheat", "Soybeans", "Sorghum", "Diesel"]
    X = df[feature_cols].copy()
    if drop_year:
        X = X.drop(columns=["Year"])
    for aid in AGENT_IDS:
        X[f"Agent_{aid}"] = (df["AgentID"] == aid).astype(int)
    y = df["Irrigation_Depth"].values
    return X, y


# ── Pooled-by-Regime Helpers ────────────────────────────────────────────────
def build_regime_datasets(annual):
    """Build pooled regime DataFrames with all derived columns.

    Returns (pre_tv, pre_te, post_tv, post_te) DataFrames with:
    - t_rel (Year - agent's cp_year)
    - 7 base predictors
    - 3 agent dummies
    - t_rel interaction columns (t_rel × agent dummies)
    - t_rel_cap_6, t_rel_cap_8, t_rel_cap_10, t_rel_cap_trainmax
    - t_rel_cap interaction columns for each cap variant
    - bin_0_3, bin_4_7, bin_8plus (time bins)
    - metadata: Year, AgentID, cp_year
    """
    pieces = []
    for aid, config in AGENT_CONFIGS.items():
        cp_year = config["cp_year"]
        df_agent = annual[annual["AgentID"] == aid].copy()
        df_agent["cp_year"] = cp_year
        df_agent["t_rel"] = df_agent["Year"] - cp_year
        pieces.append(df_agent)
    df_all = pd.concat(pieces, ignore_index=True)

    # Add agent dummies
    for aid in AGENT_IDS:
        df_all[f"Agent_{aid}"] = (df_all["AgentID"] == aid).astype(int)

    # Add t_rel interaction features
    for aid in AGENT_IDS:
        df_all[f"t_rel_x_Agent_{aid}"] = df_all["t_rel"] * df_all[f"Agent_{aid}"]

    # Add capped t_rel columns
    for cap_val in [6, 8, 10]:
        col = f"t_rel_cap_{cap_val}"
        df_all[col] = df_all["t_rel"].clip(upper=cap_val)
        # Interactions with capped t_rel
        for aid in AGENT_IDS:
            df_all[f"{col}_x_Agent_{aid}"] = (
                df_all[col] * df_all[f"Agent_{aid}"])

    # Time bins
    df_all["bin_0_3"] = ((df_all["t_rel"] >= 0) &
                         (df_all["t_rel"] <= 3)).astype(int)
    df_all["bin_4_7"] = ((df_all["t_rel"] >= 4) &
                         (df_all["t_rel"] <= 7)).astype(int)
    df_all["bin_8plus"] = (df_all["t_rel"] >= 8).astype(int)

    # PRE regime: rows where Year < cp_year
    pre = df_all[df_all["Year"] < df_all["cp_year"]].copy()
    pre_tv = pre[pre["t_rel"] <= -3].copy().reset_index(drop=True)
    pre_te = pre[pre["t_rel"].isin([-2, -1])].copy().reset_index(drop=True)

    # POST regime: rows where Year >= cp_year
    post = df_all[df_all["Year"] >= df_all["cp_year"]].copy()
    post_tv = post[post["Year"] <= 2016].copy().reset_index(drop=True)
    post_te = post[post["Year"] >= 2017].copy().reset_index(drop=True)

    # Add trainmax-capped t_rel for POST (cap at max t_rel in trainval)
    post_trainmax = post_tv["t_rel"].max()
    for df in [post_tv, post_te]:
        df["t_rel_cap_trainmax"] = df["t_rel"].clip(upper=post_trainmax)
        for aid in AGENT_IDS:
            df[f"t_rel_cap_trainmax_x_Agent_{aid}"] = (
                df["t_rel_cap_trainmax"] * df[f"Agent_{aid}"])

    print(f"  PRE  trainval: {len(pre_tv)} rows, test: {len(pre_te)} rows")
    print(f"  POST trainval: {len(post_tv)} rows, test: {len(post_te)} rows")
    print(f"  POST trainmax t_rel cap: {post_trainmax}")

    return pre_tv, pre_te, post_tv, post_te


def build_post_datasets_for_split(annual, test_start_year):
    """Build POST-regime datasets with a parameterized train/test split.

    Same feature engineering as build_regime_datasets() POST section,
    but splits at test_start_year instead of hardcoded 2017.

    Returns (post_tv, post_te) DataFrames.
    """
    pieces = []
    for aid, config in AGENT_CONFIGS.items():
        cp_year = config["cp_year"]
        df_agent = annual[annual["AgentID"] == aid].copy()
        df_agent["cp_year"] = cp_year
        df_agent["t_rel"] = df_agent["Year"] - cp_year
        pieces.append(df_agent)
    df_all = pd.concat(pieces, ignore_index=True)

    # Add agent dummies
    for aid in AGENT_IDS:
        df_all[f"Agent_{aid}"] = (df_all["AgentID"] == aid).astype(int)

    # Add t_rel interaction features
    for aid in AGENT_IDS:
        df_all[f"t_rel_x_Agent_{aid}"] = df_all["t_rel"] * df_all[f"Agent_{aid}"]

    # Add capped t_rel columns
    for cap_val in [6, 8, 10]:
        col = f"t_rel_cap_{cap_val}"
        df_all[col] = df_all["t_rel"].clip(upper=cap_val)
        for aid in AGENT_IDS:
            df_all[f"{col}_x_Agent_{aid}"] = (
                df_all[col] * df_all[f"Agent_{aid}"])

    # Time bins
    df_all["bin_0_3"] = ((df_all["t_rel"] >= 0) &
                         (df_all["t_rel"] <= 3)).astype(int)
    df_all["bin_4_7"] = ((df_all["t_rel"] >= 4) &
                         (df_all["t_rel"] <= 7)).astype(int)
    df_all["bin_8plus"] = (df_all["t_rel"] >= 8).astype(int)

    # POST regime: rows where Year >= cp_year
    post = df_all[df_all["Year"] >= df_all["cp_year"]].copy()
    post_tv = post[post["Year"] < test_start_year].copy().reset_index(drop=True)
    post_te = post[post["Year"] >= test_start_year].copy().reset_index(drop=True)

    # Add trainmax-capped t_rel for POST (cap at max t_rel in trainval)
    post_trainmax = post_tv["t_rel"].max()
    for df in [post_tv, post_te]:
        df["t_rel_cap_trainmax"] = df["t_rel"].clip(upper=post_trainmax)
        for aid in AGENT_IDS:
            df[f"t_rel_cap_trainmax_x_Agent_{aid}"] = (
                df["t_rel_cap_trainmax"] * df[f"Agent_{aid}"])

    print(f"  POST trainval: {len(post_tv)} rows "
          f"(years {post_tv['Year'].min()}-{post_tv['Year'].max()}), "
          f"test: {len(post_te)} rows "
          f"(years {post_te['Year'].min()}-{post_te['Year'].max()})")
    print(f"  POST trainmax t_rel cap: {post_trainmax}")

    return post_tv, post_te


def get_feature_columns(feature_set_name, regime="pre"):
    """Return list of actual DataFrame column names for a feature set.

    For POST B_cap* variants, maps the abstract 't_rel_cap' to the
    appropriate 't_rel_cap_N' column in the DataFrame. Interactions
    are similarly remapped.
    """
    if regime == "pre":
        cols = list(PRE_FEATURE_SETS[feature_set_name])
        return cols

    # POST regime
    template = list(POST_FEATURE_SETS[feature_set_name])

    if feature_set_name.startswith("B_cap") or feature_set_name.startswith("D_cap"):
        # Determine which capped column to use
        if feature_set_name == "B_capmax":
            src_col = "t_rel_cap_trainmax"
        elif feature_set_name == "D_cap8_interact":
            src_col = "t_rel_cap_8"
        else:
            cap_val = _POST_CAP_MAP[feature_set_name]
            src_col = f"t_rel_cap_{cap_val}"

        # Replace abstract names with actual column names
        result = []
        for c in template:
            if c == "t_rel_cap":
                result.append(src_col)
            elif c.startswith("t_rel_cap_x_Agent_"):
                aid_str = c.replace("t_rel_cap_x_Agent_", "")
                result.append(f"{src_col}_x_Agent_{aid_str}")
            else:
                result.append(c)
        return result

    # A_base, C_bins — columns are literal
    return template


def make_pre_rolling_folds(df_tv):
    """Rolling-origin folds on t_rel for PRE regime (2-year val blocks).

    Returns list of (train_idx, val_idx) numpy arrays into df_tv.
    """
    t_rel_vals = sorted(df_tv["t_rel"].unique())
    min_t = t_rel_vals[0]
    max_t = t_rel_vals[-1]

    folds = []
    val_len = 2
    # Need at least 3 training t_rel values per agent
    cutoff = min_t + 2
    while True:
        val_start = cutoff + 1
        val_end = val_start + val_len - 1
        if val_end > max_t:
            break
        train_mask = df_tv["t_rel"] <= cutoff
        val_mask = (df_tv["t_rel"] >= val_start) & (df_tv["t_rel"] <= val_end)
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            folds.append((
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            ))
        cutoff += 1
    return folds


def make_post_rolling_folds(df_tv):
    """Rolling-origin folds on calendar Year for POST regime (4-year val blocks).

    Returns list of (train_idx, val_idx) numpy arrays into df_tv.
    """
    folds = []
    val_len = 4
    for train_end in range(2008, 2013):
        val_years = list(range(train_end + 1, train_end + 1 + val_len))
        max_val = max(val_years)
        if max_val > df_tv["Year"].max():
            break
        train_mask = df_tv["Year"] <= train_end
        val_mask = df_tv["Year"].isin(val_years)
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            folds.append((
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            ))
    return folds


def make_post_rolling_folds_adaptive(df_tv):
    """Adaptive rolling-origin folds for POST regime with variable trainval size.

    Scales val_len and fold count to trainval size:
    - Determines the common trainval year range across agents
    - trainval ≤ 10 common years → val_len=2
    - trainval > 10 common years → val_len=3
    - Ensures at least 4 years of training data in every fold
    - Uses calendar Year for fold boundaries

    Returns list of (train_idx, val_idx) numpy arrays into df_tv.
    """
    # Get common years (years that all agents share in trainval)
    all_years = sorted(df_tv["Year"].unique())
    n_years = len(all_years)

    if n_years <= 10:
        val_len = 2
    else:
        val_len = 3

    min_train_years = 4  # minimum training years per fold

    folds = []
    # The earliest possible train_end must leave at least min_train_years
    # Each agent contributes 1 row per year, so we count unique years
    # train_end marks the last year of the training partition
    earliest_train_end = all_years[min_train_years - 1]
    latest_train_end = all_years[-1] - val_len  # leave room for val block

    for train_end in range(earliest_train_end, latest_train_end + 1):
        val_years = list(range(train_end + 1, train_end + 1 + val_len))
        max_val = max(val_years)
        if max_val > df_tv["Year"].max():
            break
        train_mask = df_tv["Year"] <= train_end
        val_mask = df_tv["Year"].isin(val_years)
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            folds.append((
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            ))
    return folds


def compute_macro_rmse(y_true, y_pred, agent_ids):
    """Per-agent RMSE averaged across agents."""
    agent_rmses = []
    for aid in sorted(set(agent_ids)):
        mask = agent_ids == aid
        if mask.sum() > 0:
            agent_rmses.append(
                np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    return np.mean(agent_rmses)


def compute_macro_mae(y_true, y_pred, agent_ids):
    """Per-agent MAE averaged across agents."""
    agent_maes = []
    for aid in sorted(set(agent_ids)):
        mask = agent_ids == aid
        if mask.sum() > 0:
            agent_maes.append(
                mean_absolute_error(y_true[mask], y_pred[mask]))
    return np.mean(agent_maes)


def compute_max_agent_rmse(y_true, y_pred, agent_ids):
    """Maximum per-agent RMSE (worst-agent performance)."""
    agent_rmses = []
    for aid in sorted(set(agent_ids)):
        mask = agent_ids == aid
        if mask.sum() > 0:
            agent_rmses.append(
                np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    return max(agent_rmses) if agent_rmses else float("inf")


def build_model_candidates(regime_name="pre"):
    """Build list of model candidate dicts for benchmarking.

    Each dict: {"name": str, "type": str, "model_factory": callable,
                "param_grid": list_of_dicts or None}.

    All POST feature sets are safe (no raw unbounded t_rel), so gblinear
    is allowed for both regimes.
    """
    candidates = []

    # Ridge
    candidates.append({
        "name": "Ridge",
        "type": "sklearn",
        "model_factory": lambda **kw: Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(**kw)),
        ]),
        "param_grid": list(ParameterGrid({
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        })),
    })

    # ElasticNet
    candidates.append({
        "name": "ElasticNet",
        "type": "sklearn",
        "model_factory": lambda **kw: Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(max_iter=10000, **kw)),
        ]),
        "param_grid": list(ParameterGrid({
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
            "l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
        })),
    })

    # XGBoost gbtree (tuned via Optuna)
    candidates.append({
        "name": "XGBoost_gbtree",
        "type": "xgb_optuna",
        "model_factory": None,
        "param_grid": None,
    })

    # XGBoost gblinear
    candidates.append({
        "name": "XGBoost_gblinear",
        "type": "sklearn",
        "model_factory": lambda **kw: XGBRegressor(
            booster="gblinear", objective="reg:squarederror",
            verbosity=0, n_estimators=200, **kw),
        "param_grid": list(ParameterGrid({
            "reg_lambda": [0.1, 1.0, 10.0],
            "reg_alpha": [0.0, 1.0, 10.0],
        })),
    })

    return candidates


def cv_score_model(candidate, X_tv, y_tv, agent_ids_tv, folds, seed=42,
                   transform="identity"):
    """Run rolling-origin CV for a model candidate.

    Returns dict with name, best_params, mean_macro_rmse, mean_macro_mae,
    mean_max_agent_rmse, and for XGBoost gbtree also n_estimators.

    If transform="log1p", fits on log1p(y) and scores on original scale.
    """
    name = candidate["name"]
    ctype = candidate["type"]

    def _apply_transform(y):
        return np.log1p(y) if transform == "log1p" else y

    def _inverse_transform(y):
        if transform == "log1p":
            y = np.clip(y, None, 20)  # cap at exp(20) ≈ 485M to avoid overflow
            return np.expm1(y)
        return y

    if ctype == "sklearn":
        best_rmse = float("inf")
        best_mae = float("inf")
        best_max_agent = float("inf")
        best_params = {}

        for params in candidate["param_grid"]:
            fold_rmses = []
            fold_maes = []
            fold_max_agent = []
            for train_idx, val_idx in folds:
                X_tr = X_tv.iloc[train_idx].values
                y_tr = _apply_transform(y_tv[train_idx])
                X_va = X_tv.iloc[val_idx].values
                y_va = y_tv[val_idx]
                aids_va = agent_ids_tv[val_idx]

                model = candidate["model_factory"](**params)
                model.fit(X_tr, y_tr)
                pred = _inverse_transform(model.predict(X_va))

                fold_rmses.append(compute_macro_rmse(y_va, pred, aids_va))
                fold_maes.append(compute_macro_mae(y_va, pred, aids_va))
                fold_max_agent.append(
                    compute_max_agent_rmse(y_va, pred, aids_va))

            mean_rmse = np.mean(fold_rmses)
            mean_mae = np.mean(fold_maes)
            mean_max = np.mean(fold_max_agent)
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_mae = mean_mae
                best_max_agent = mean_max
                best_params = params

        return {
            "name": name,
            "best_params": best_params,
            "mean_macro_rmse": best_rmse,
            "mean_macro_mae": best_mae,
            "mean_max_agent_rmse": best_max_agent,
        }

    elif ctype == "xgb_optuna":
        sampler = optuna.samplers.TPESampler(seed=seed)
        ss = CONSERVATIVE_SEARCH_SPACE

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", *ss["min_child_weight"]),
                "gamma": trial.suggest_float("gamma", *ss["gamma"], log=True),
                "subsample": trial.suggest_float("subsample", *ss["subsample"]),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *ss["colsample_bytree"]),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *ss["reg_lambda"], log=True),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *ss["reg_alpha"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *ss["learning_rate"], log=True),
                "n_estimators": trial.suggest_int(
                    "n_estimators", *ss["n_estimators"]),
                "objective": "reg:squarederror",
                "verbosity": 0,
                "random_state": seed,
            }

            fold_scores = []
            fold_max_scores = []
            fold_best_iters = []
            for train_idx, val_idx in folds:
                X_tr = X_tv.iloc[train_idx].values
                y_tr = _apply_transform(y_tv[train_idx])
                X_va = X_tv.iloc[val_idx].values
                y_va = y_tv[val_idx]
                aids_va = agent_ids_tv[val_idx]

                n_val = max(1, int(0.2 * len(y_tr)))
                xgb_params = {k: v for k, v in params.items()}
                es_rounds = min(50, max(5, len(y_tr) // 3))
                xgb_params["early_stopping_rounds"] = es_rounds

                model = XGBRegressor(**xgb_params)
                model.fit(
                    X_tr[:-n_val], y_tr[:-n_val],
                    eval_set=[(X_tr[-n_val:], y_tr[-n_val:])],
                    verbose=False,
                )
                fold_best_iters.append(
                    getattr(model, "best_iteration",
                            params["n_estimators"]) + 1)
                pred = _inverse_transform(model.predict(X_va))
                fold_scores.append(compute_macro_rmse(y_va, pred, aids_va))
                fold_max_scores.append(
                    compute_max_agent_rmse(y_va, pred, aids_va))

            trial.set_user_attr("fold_n_estimators", fold_best_iters)
            trial.set_user_attr("mean_max_agent_rmse",
                                float(np.mean(fold_max_scores)))
            return np.mean(fold_scores)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS_POINT,
                       show_progress_bar=False)

        best_params = study.best_trial.params
        best_params["objective"] = "reg:squarederror"
        best_params["verbosity"] = 0
        best_params["random_state"] = seed

        fold_n_est = study.best_trial.user_attrs["fold_n_estimators"]
        final_n_est = int(np.median(fold_n_est))
        best_params["n_estimators"] = final_n_est
        best_params.pop("early_stopping_rounds", None)

        return {
            "name": name,
            "best_params": best_params,
            "mean_macro_rmse": study.best_value,
            "mean_macro_mae": float("nan"),
            "mean_max_agent_rmse": study.best_trial.user_attrs[
                "mean_max_agent_rmse"],
            "fold_n_estimators": fold_n_est,
            "final_n_estimators": final_n_est,
        }


def select_best_model(cv_results):
    """Select winner from CV results.

    Primary: lowest mean_macro_rmse.
    Secondary: lowest mean_max_agent_rmse.
    Tertiary: simpler model (Ridge > ElasticNet > XGBoost).
    """
    simplicity_order = {"Ridge": 0, "ElasticNet": 1,
                        "XGBoost_gblinear": 2, "XGBoost_gbtree": 3}

    sorted_results = sorted(cv_results, key=lambda r: (
        r["mean_macro_rmse"],
        r.get("mean_max_agent_rmse", float("inf")),
        simplicity_order.get(r["name"], 99),
    ))
    return sorted_results[0]


# ── Rolling-Origin CV for CP Sensitivity ────────────────────────────────────
def _make_rolling_folds(cp, val_len=3):
    """Generate rolling-origin fold specs for CP sensitivity tuning.

    Example for CP=2006, val_len=3:
        Fold 0: train 1993-2001, val 2002-2004
        Fold 1: train 1993-2002, val 2003-2005
        Fold 2: train 1993-2003, val 2004-2006

    Returns list of dicts with 'train_years' and 'val_years' lists.
    """
    folds = []
    for i in range(val_len):
        val_end = cp - (val_len - 1 - i)
        val_start = val_end - val_len + 1
        train_end = val_start - 1
        folds.append({
            "train_years": list(range(1993, train_end + 1)),
            "val_years": list(range(val_start, val_end + 1)),
        })
    return folds


def _make_rolling_folds_regime(trainval_years, val_len=2, n_folds=3):
    """Generate rolling-origin fold specs for an arbitrary trainval period.

    Example for trainval 1993-2001, val_len=2, n_folds=3:
        Fold 0: train 1993-1997, val 1998-1999
        Fold 1: train 1993-1998, val 1999-2000
        Fold 2: train 1993-1999, val 2000-2001

    Returns list of dicts with 'train_years' and 'val_years' lists.
    """
    tv = sorted(trainval_years)
    folds = []
    for i in range(n_folds):
        # Work backward from the end of trainval
        val_end_idx = len(tv) - 1 - (n_folds - 1 - i) * 1
        val_start_idx = val_end_idx - val_len + 1
        train_end_idx = val_start_idx - 1
        if train_end_idx < 0:
            continue  # not enough data for this fold
        folds.append({
            "train_years": tv[: train_end_idx + 1],
            "val_years": tv[val_start_idx: val_end_idx + 1],
        })
    return folds


def _optuna_tune_macro(X, y, agent_ids, folds, seed, years=None):
    """Optuna tuning with macro-averaged per-agent RMSE across rolling folds.

    For each trial:
      - For each fold: fit on pooled train rows with early stopping on val rows,
        then compute per-agent RMSE on val, average across agents = fold score.
      - Trial score = mean(fold scores).

    After tuning: final_n_estimators = median(best_iterations across folds of
    best trial).

    Args:
        X: Feature DataFrame (pooled across agents).
        y: Target array.
        agent_ids: Array of AgentID per row.
        folds: List of fold dicts from _make_rolling_folds().
        seed: Random seed for TPE sampler.
        years: Year array per row (required if X lacks a 'Year' column).

    Returns:
        (best_params_with_n_estimators, fold_details_dict)
    """
    from sklearn.metrics import mean_squared_error as mse

    agent_ids = np.asarray(agent_ids)
    unique_agents = np.sort(np.unique(agent_ids))
    X_arr = X.values if hasattr(X, "values") else X

    # Determine years array
    if years is not None:
        years_arr = np.asarray(years)
    elif hasattr(X, "columns") and "Year" in X.columns:
        years_arr = X["Year"].values
    else:
        raise ValueError("Must provide years array or X with 'Year' column")

    # Pre-compute fold masks
    fold_masks = []
    for fold in folds:
        train_yrs = set(fold["train_years"])
        val_yrs = set(fold["val_years"])
        tmask = np.array([yr in train_yrs for yr in years_arr])
        vmask = np.array([yr in val_yrs for yr in years_arr])
        fold_masks.append((tmask, vmask))

    ss = CONSERVATIVE_SEARCH_SPACE
    sampler = optuna.samplers.TPESampler(seed=seed)

    # Storage for best trial's fold details
    best_fold_n_estimators = []

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
            "gamma": trial.suggest_float("gamma", *ss["gamma"]),
            "subsample": trial.suggest_float("subsample", *ss["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"]),
            "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
            "n_estimators": trial.suggest_int("n_estimators", *ss["n_estimators"]),
            "objective": "reg:squarederror",
            "early_stopping_rounds": 30,
            "verbosity": 0,
        }

        fold_scores = []
        fold_best_iters = []
        for tmask, vmask in fold_masks:
            model = XGBRegressor(**params)
            model.fit(
                X_arr[tmask], y[tmask],
                eval_set=[(X_arr[vmask], y[vmask])],
                verbose=False,
            )
            best_iter = getattr(model, "best_iteration", params["n_estimators"])
            fold_best_iters.append(best_iter + 1)

            # Per-agent RMSE on val
            pred_val = model.predict(X_arr[vmask])
            agent_rmses = []
            for aid in unique_agents:
                agent_val_mask = agent_ids[vmask] == aid
                if agent_val_mask.sum() > 0:
                    agent_rmse = np.sqrt(mse(
                        y[vmask][agent_val_mask],
                        pred_val[agent_val_mask],
                    ))
                    agent_rmses.append(agent_rmse)
            fold_scores.append(np.mean(agent_rmses))

        # Store fold iterations for later retrieval
        trial.set_user_attr("fold_n_estimators", fold_best_iters)
        return np.mean(fold_scores)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS_SENS, show_progress_bar=False)

    best_params = study.best_trial.params
    best_fold_n_estimators = study.best_trial.user_attrs["fold_n_estimators"]
    final_n_est = int(np.median(best_fold_n_estimators))

    # Remove early_stopping_rounds from params, add final n_estimators
    best_params.pop("early_stopping_rounds", None)
    best_params["n_estimators"] = final_n_est

    fold_details = {
        "best_value": study.best_value,
        "fold_n_estimators": best_fold_n_estimators,
        "final_n_estimators": final_n_est,
    }

    print(f"  Optuna best macro RMSE: {study.best_value:.2f}")
    print(f"  Fold n_estimators: {best_fold_n_estimators} → median={final_n_est}")
    print(f"  Best params: { {k: v for k, v in best_params.items() if k != 'n_estimators'} }")

    return best_params, fold_details


# ── Regime Builder ───────────────────────────────────────────────────────────
def make_regimes(cp_year):
    """Build pre-CP and post-CP regime dicts from a given changepoint year.

    Pre-CP:  full=1993–(CP-1), trainval=1993–(CP-3), test=(CP-2)–(CP-1),
             anchor=last 3 of trainval
    Post-CP: full=CP–2020, trainval=CP–2016, test=2017–2020,
             anchor=last 3 of trainval
    """
    # Pre-CP regime
    pre_full = list(range(1993, cp_year))           # 1993 to CP-1
    pre_trainval = list(range(1993, cp_year - 2))   # 1993 to CP-3
    pre_test = list(range(cp_year - 2, cp_year))    # CP-2 to CP-1
    pre_anchor = pre_trainval[-3:]                   # last 3 of trainval

    regime_a = {
        "name": "pre",
        "full_years": pre_full,
        "trainval_years": pre_trainval,
        "test_years": pre_test,
        "anchor_years": pre_anchor,
    }

    # Post-CP regime
    post_full = list(range(cp_year, 2021))          # CP to 2020
    post_trainval = list(range(cp_year, 2017))      # CP to 2016
    post_test = [2017, 2018, 2019, 2020]
    post_anchor = post_trainval[-3:]                 # last 3 of trainval

    regime_b = {
        "name": "post",
        "full_years": post_full,
        "trainval_years": post_trainval,
        "test_years": post_test,
        "anchor_years": post_anchor,
    }

    return regime_a, regime_b


# ── Point Estimation (Pooled-by-Regime v2) ───────────────────────────────────
def _refit_and_predict(best, candidates, X_tv, y_tv, X_te, transform):
    """Refit winning model on full trainval and predict.

    Returns (final_model, pred_tv, pred_te).
    """
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv

    if best["name"] in ("Ridge", "ElasticNet", "XGBoost_gblinear"):
        for cand in candidates:
            if cand["name"] == best["name"]:
                final_model = cand["model_factory"](**best["best_params"])
                break
        final_model.fit(X_tv.values, y_fit)
    else:  # XGBoost gbtree
        refit_params = {
            k: v for k, v in best["best_params"].items()
            if k not in ("early_stopping_rounds",)
        }
        refit_params["objective"] = "reg:squarederror"
        refit_params["verbosity"] = 0
        final_model = XGBRegressor(**refit_params)
        final_model.fit(X_tv.values, y_fit)

    pred_tv = final_model.predict(X_tv.values)
    pred_te = final_model.predict(X_te.values)

    if transform == "log1p":
        pred_tv = np.expm1(np.clip(pred_tv, None, 20))
        pred_te = np.expm1(np.clip(pred_te, None, 20))

    return final_model, pred_tv, pred_te


def _save_model_diagnostics(final_model, best, feature_cols, regime_name):
    """Save coefficient or feature importance summaries."""
    if best["name"] in ("Ridge", "ElasticNet"):
        # Extract coefficients from pipeline
        if hasattr(final_model, "named_steps"):
            estimator = final_model.named_steps.get(
                "ridge", final_model.named_steps.get("enet"))
        else:
            estimator = final_model
        if hasattr(estimator, "coef_"):
            coef_df = pd.DataFrame({
                "feature": feature_cols,
                "coefficient": estimator.coef_,
            })
            if hasattr(estimator, "intercept_"):
                coef_df = pd.concat([coef_df, pd.DataFrame({
                    "feature": ["intercept"],
                    "coefficient": [estimator.intercept_],
                })], ignore_index=True)
            coef_path = TWO_REGIME_DIR / f"pooled_coefficients_{regime_name}.csv"
            coef_df.to_csv(coef_path, index=False)
            print(f"  Saved {coef_path}")
    elif best["name"] in ("XGBoost_gbtree", "XGBoost_gblinear"):
        if hasattr(final_model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": final_model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_path = (TWO_REGIME_DIR /
                        f"pooled_feature_importance_{regime_name}.csv")
            imp_df.to_csv(imp_path, index=False)
            print(f"  Saved {imp_path}")


def run_point(annual):
    """Pooled-by-regime point estimation v2 with regime-specific feature sets.

    Iterates over (feature_set, model, transform) combos per regime,
    selects best by macro-RMSE with max-agent-RMSE tie-breaker.
    """
    print("=" * 60)
    print("CLUSTER 1 — POOLED-BY-REGIME POINT ESTIMATION v2")
    print("=" * 60)

    TWO_REGIME_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build regime datasets
    print("\nBuilding pooled regime datasets...")
    pre_tv, pre_te, post_tv, post_te = build_regime_datasets(annual)

    regime_specs = [
        ("pre", pre_tv, pre_te, make_pre_rolling_folds, PRE_FEATURE_SETS),
        ("post", post_tv, post_te, make_post_rolling_folds, POST_FEATURE_SETS),
    ]

    all_summaries = []
    all_predictions = []
    all_leaderboard = []

    for regime_name, df_tv, df_te, fold_fn, feat_sets in regime_specs:
        print(f"\n{'─' * 60}")
        print(f"REGIME: {regime_name.upper()}")
        print(f"{'─' * 60}")

        y_tv = df_tv["Irrigation_Depth"].values
        agent_ids_tv = df_tv["AgentID"].values
        y_te = df_te["Irrigation_Depth"].values
        agent_ids_te = df_te["AgentID"].values

        print(f"  Trainval: {len(df_tv)} rows, "
              f"{df_tv['AgentID'].nunique()} agents")
        print(f"  Test:     {len(df_te)} rows")

        # Generate folds once (shared across feature sets)
        folds = fold_fn(df_tv)
        print(f"  CV folds: {len(folds)}")
        for i, (tr_idx, va_idx) in enumerate(folds):
            tr_years = sorted(df_tv.iloc[tr_idx]["Year"].unique())
            va_years = sorted(df_tv.iloc[va_idx]["Year"].unique())
            print(f"    Fold {i}: train {len(tr_idx)} rows "
                  f"({tr_years[0]}-{tr_years[-1]}), "
                  f"val {len(va_idx)} rows "
                  f"({va_years[0]}-{va_years[-1]})")

        # 2. Score all (feature_set, model, transform) combos
        regime_cv_results = []
        regime_seed = MASTER_SEED + (0 if regime_name == "pre" else 100)

        for fs_name in feat_sets:
            feature_cols = get_feature_columns(fs_name, regime=regime_name)
            print(f"\n  Feature set: {fs_name} ({len(feature_cols)} cols)")
            print(f"    Columns: {feature_cols}")

            X_tv = df_tv[feature_cols].copy()
            X_te = df_te[feature_cols].copy()

            candidates = build_model_candidates(regime_name)

            for transform in ["identity", "log1p"]:
                for cand in candidates:
                    combo_label = (f"{fs_name}/{cand['name']}"
                                   f"/{transform}")
                    print(f"    Scoring {combo_label}...", end="")
                    result = cv_score_model(
                        cand, X_tv, y_tv, agent_ids_tv, folds,
                        seed=regime_seed, transform=transform,
                    )
                    result["feature_set"] = fs_name
                    result["transform"] = transform
                    result["feature_cols"] = feature_cols
                    regime_cv_results.append(result)
                    print(f" macro_RMSE={result['mean_macro_rmse']:.2f}"
                          f"  max_agent={result.get('mean_max_agent_rmse', float('nan')):.2f}")

                    all_leaderboard.append({
                        "regime": regime_name,
                        "feature_set": fs_name,
                        "model": result["name"],
                        "transform": transform,
                        "mean_macro_rmse": result["mean_macro_rmse"],
                        "mean_macro_mae": result.get("mean_macro_mae",
                                                     float("nan")),
                        "mean_max_agent_rmse": result.get(
                            "mean_max_agent_rmse", float("nan")),
                        "best_params": json.dumps(
                            result.get("best_params", {})),
                    })

        # 3. Select best model with sanity-check fallback
        ranked = sorted(regime_cv_results, key=lambda r: (
            r["mean_macro_rmse"],
            r.get("mean_max_agent_rmse", float("inf")),
            {"Ridge": 0, "ElasticNet": 1,
             "XGBoost_gblinear": 2, "XGBoost_gbtree": 3}.get(
                r["name"], 99),
        ))

        tv_max = y_tv.max()
        pred_bound = 2 * tv_max
        print(f"\n  Sanity check: pred bound = ±{pred_bound:.0f} "
              f"(2× trainval max {tv_max:.0f})")
        print(f"  Checking {len(ranked)} candidates...")
        best = None
        n_skipped = 0
        for candidate_best in ranked:
            cand_fs = candidate_best["feature_set"]
            cand_transform = candidate_best["transform"]
            cand_cols = candidate_best["feature_cols"]

            X_tv_cand = df_tv[cand_cols].copy()
            X_te_cand = df_te[cand_cols].copy()
            cands = build_model_candidates(regime_name)

            _, pred_tv_cand, pred_te_cand = _refit_and_predict(
                candidate_best, cands, X_tv_cand, y_tv, X_te_cand,
                cand_transform)

            max_pred = max(pred_tv_cand.max(), pred_te_cand.max())
            min_pred = min(pred_tv_cand.min(), pred_te_cand.min())
            if max_pred > pred_bound or min_pred < -pred_bound:
                n_skipped += 1
                print(f"\n  SKIP ({n_skipped}): {candidate_best['name']}/{cand_fs}/"
                      f"{cand_transform} — pred range "
                      f"[{min_pred:.0f}, {max_pred:.0f}] exceeds "
                      f"±{pred_bound:.0f}")
                continue

            best = candidate_best
            print(f"  Passed sanity check ({n_skipped} skipped, "
                  f"{len(ranked) - n_skipped - 1} untested)")
            break

        if best is None:
            print(f"\n  ERROR: All {len(ranked)} models exceed prediction "
                  f"bounds (±{pred_bound:.0f})!")
            print("  Falling back to top-ranked with lowest macro RMSE...")
            best = ranked[0]

        best_fs = best["feature_set"]
        best_transform = best["transform"]
        best_cols = best["feature_cols"]

        print(f"\n  WINNER: {best['name']} / {best_fs} / {best_transform}")
        print(f"    Time encoding: {best_fs}")
        print(f"    macro RMSE = {best['mean_macro_rmse']:.2f}")
        print(f"    max_agent RMSE = {best.get('mean_max_agent_rmse', float('nan')):.2f}")
        print(f"    Params: {best['best_params']}")

        # Save best params
        params_path = (TWO_REGIME_DIR /
                       f"pooled_best_params_{regime_name}_v2.json")
        with open(params_path, "w") as f:
            json.dump({
                "model": best["name"],
                "feature_set": best_fs,
                "transform": best_transform,
                "feature_cols": best_cols,
                **best,
            }, f, indent=2, default=str)
        print(f"  Saved {params_path}")

        # 4. Refit winner on full trainval
        X_tv_best = df_tv[best_cols].copy()
        X_te_best = df_te[best_cols].copy()
        candidates = build_model_candidates(regime_name)

        final_model, pred_tv, pred_te = _refit_and_predict(
            best, candidates, X_tv_best, y_tv, X_te_best, best_transform)

        # 6. Compute metrics
        train_metrics = compute_metrics(y_tv, pred_tv)
        test_metrics = compute_metrics(y_te, pred_te)

        print(f"\n  POOLED Train — RMSE={train_metrics['RMSE']:.2f}, "
              f"R²={train_metrics['R2']:.3f}")
        print(f"  POOLED Test  — RMSE={test_metrics['RMSE']:.2f}, "
              f"R²={test_metrics['R2']:.3f}")

        all_summaries.append({
            "regime": regime_name,
            "scope": "pooled",
            "AgentID": "all",
            "model": best["name"],
            "feature_set": best_fs,
            "transform": best_transform,
            "train_RMSE": train_metrics["RMSE"],
            "train_MAE": train_metrics["MAE"],
            "train_R2": train_metrics["R2"],
            "train_Bias": train_metrics["Bias"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_R2": test_metrics["R2"],
            "test_Bias": test_metrics["Bias"],
        })

        # Per-agent metrics
        for aid in AGENT_IDS:
            mask_tv = agent_ids_tv == aid
            mask_te = agent_ids_te == aid
            if mask_tv.sum() > 0 and mask_te.sum() > 0:
                a_train_m = compute_metrics(
                    y_tv[mask_tv], pred_tv[mask_tv])
                a_test_m = compute_metrics(
                    y_te[mask_te], pred_te[mask_te])
                print(f"  Agent {aid} — "
                      f"train R²={a_train_m['R2']:.3f}, "
                      f"test R²={a_test_m['R2']:.3f}")

                all_summaries.append({
                    "regime": regime_name,
                    "scope": "per_agent",
                    "AgentID": aid,
                    "model": best["name"],
                    "feature_set": best_fs,
                    "transform": best_transform,
                    "train_RMSE": a_train_m["RMSE"],
                    "train_MAE": a_train_m["MAE"],
                    "train_R2": a_train_m["R2"],
                    "train_Bias": a_train_m["Bias"],
                    "test_RMSE": a_test_m["RMSE"],
                    "test_MAE": a_test_m["MAE"],
                    "test_R2": a_test_m["R2"],
                    "test_Bias": a_test_m["Bias"],
                })

        # Save predictions
        pred_df = pd.DataFrame({
            "regime": regime_name,
            "Year": list(df_tv["Year"].values) +
                    list(df_te["Year"].values),
            "AgentID": list(df_tv["AgentID"].values) +
                       list(df_te["AgentID"].values),
            "cp_year": list(df_tv["cp_year"].values) +
                       list(df_te["cp_year"].values),
            "t_rel": list(df_tv["t_rel"].values) +
                     list(df_te["t_rel"].values),
            "Obs": np.concatenate([y_tv, y_te]),
            "Pred": np.concatenate([pred_tv, pred_te]),
            "Split": (["train"] * len(y_tv) +
                      ["test"] * len(y_te)),
            "model": best["name"],
            "feature_set": best_fs,
        })
        pred_path = (TWO_REGIME_DIR /
                     f"pooled_predictions_{regime_name}_v2.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"  Saved {pred_path}")
        all_predictions.append(pred_df)

        # Save metrics
        metrics_path = (TWO_REGIME_DIR /
                        f"pooled_metrics_{regime_name}_v2.csv")
        metrics_rows = [
            {"Split": "train", "scope": "pooled", **train_metrics},
            {"Split": "test", "scope": "pooled", **test_metrics},
        ]
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"  Saved {metrics_path}")

        # Save model diagnostics
        _save_model_diagnostics(final_model, best, best_cols, regime_name)

    # Save leaderboard
    lb_path = TWO_REGIME_DIR / "pooled_regime_leaderboard.csv"
    pd.DataFrame(all_leaderboard).to_csv(lb_path, index=False)
    print(f"\nSaved {lb_path}")

    # Combined summary
    summary_path = TWO_REGIME_DIR / "pooled_point_summary_v2.csv"
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    # 3-way comparison
    print("\nBuilding comparison...")
    comp_parts = []

    # Old per-agent results
    old_path = TWO_REGIME_DIR / "two_regime_c1_point_summary.csv"
    if old_path.exists():
        old_df = pd.read_csv(old_path)
        old_df["approach"] = "per_agent_old"
        comp_parts.append(old_df)

    # v1 pooled results
    v1_path = TWO_REGIME_DIR / "pooled_point_summary.csv"
    if v1_path.exists():
        v1_df = pd.read_csv(v1_path)
        v1_df["approach"] = "pooled_v1"
        comp_parts.append(v1_df)

    # v2 results
    v2_df = pd.DataFrame(all_summaries)
    v2_df["approach"] = "pooled_v2"
    comp_parts.append(v2_df)

    if comp_parts:
        comparison = pd.concat(comp_parts, ignore_index=True)
        comp_path = TWO_REGIME_DIR / "pooled_vs_previous_comparison_v2.csv"
        comparison.to_csv(comp_path, index=False)
        print(f"Saved {comp_path}")

    # Plot
    print("\nGenerating pooled point-estimate plots...")
    for regime_name in ["pre", "post"]:
        plot_pooled_point(
            annual, regime_name,
            TWO_REGIME_DIR / f"pooled_point_{regime_name}_v2.pdf")

    # Per-agent two-regime plots from pooled v2 predictions
    print("\nGenerating per-agent pooled point-estimate plots...")
    for aid in AGENT_IDS:
        out_pdf = TWO_REGIME_DIR / f"pooled_two_regime_agent{aid}_point_v2.pdf"
        plot_pooled_agent_point(aid, annual, out_pdf)

    # Final report
    print("\n" + "=" * 60)
    print("CLUSTER 1 — POOLED POINT ESTIMATION v2 SUMMARY")
    print("=" * 60)
    for s in all_summaries:
        agent_label = (f"Agent {s['AgentID']}"
                       if s['scope'] == 'per_agent' else "POOLED")
        print(f"  {s['regime']:4s} {agent_label:10s} "
              f"[{s['model']}/{s['feature_set']}/{s['transform']}]  "
              f"train R²={s['train_R2']:.3f}  "
              f"test RMSE={s['test_RMSE']:.2f} "
              f"R²={s['test_R2']:.3f}")
    print("=" * 60)


# ── Ensemble ─────────────────────────────────────────────────────────────────
def run_ensemble(annual):
    """Ensemble step: load saved params + bootstrap PI for each agent."""
    print("=" * 60)
    print("CLUSTER 1 — TWO-REGIME ENSEMBLE (PER-AGENT)")
    print("=" * 60)

    TWO_REGIME_DIR.mkdir(parents=True, exist_ok=True)

    combined_summaries = []

    for aid, config in AGENT_CONFIGS.items():
        cp_year = config["cp_year"]
        regime_a, regime_b = make_regimes(cp_year)

        print(f"\n{'─' * 50}")
        print(f"Agent {aid}  (CP year = {cp_year})")
        print(f"{'─' * 50}")

        df_agent = annual[annual["AgentID"] == aid].copy()

        for regime in [regime_a, regime_b]:
            name = regime["name"]
            seed_offset = aid * 10 + (0 if name == "pre" else 100)

            print(f"\n  --- Regime: {name} ({regime['full_years'][0]}–"
                  f"{regime['full_years'][-1]}) ---")
            print(f"    Trainval: {regime['trainval_years'][0]}–"
                  f"{regime['trainval_years'][-1]} "
                  f"({len(regime['trainval_years'])} years)")
            print(f"    Test:     {regime['test_years']}")
            print(f"    Anchor:   {regime['anchor_years']}")

            # Load saved params
            params_path = TWO_REGIME_DIR / f"best_params_{aid}_{name}.json"
            if not params_path.exists():
                raise FileNotFoundError(
                    f"Params not found: {params_path}\n"
                    "Run --two-regime-point first.")
            with open(params_path) as f:
                best_params = json.load(f)
            print(f"    Loaded {params_path}")

            # Filter data
            df_trainval = df_agent[
                df_agent["Year"].isin(regime["trainval_years"])].copy()
            df_test = df_agent[
                df_agent["Year"].isin(regime["test_years"])].copy()

            X_test, y_test = prepare_features(
                df_test, add_agent_dummies=False, drop_year=True)

            # Bootstrap ensemble
            rng = np.random.default_rng(MASTER_SEED + seed_offset + 1)
            n_test = len(y_test)
            preds_matrix = np.zeros((N_TWO_REGIME_BOOT, n_test))

            print(f"    Running {N_TWO_REGIME_BOOT}-member bootstrap...")
            for i in range(N_TWO_REGIME_BOOT):
                member_rng = np.random.default_rng(rng.integers(0, 2**31))

                # Bootstrap years
                boot_years = make_constrained_moving_block_year_sequence(
                    regime["trainval_years"], regime["anchor_years"],
                    BLOCK_SIZE, member_rng)

                df_boot = bootstrap_panel_by_year_sequence(
                    df_trainval, boot_years)
                boot_years_arr = df_boot["Year"].values.copy()
                X_boot, y_boot = prepare_features(
                    df_boot, add_agent_dummies=False, drop_year=True)

                # Fixed params — only random_state varies
                params = {
                    **best_params,
                    "objective": "reg:squarederror",
                    "early_stopping_rounds": 50,
                    "verbosity": 0,
                    "random_state": int(member_rng.integers(0, 2**31)),
                }
                model = train_with_early_stopping(
                    X_boot, y_boot, params, val_years=3,
                    years=boot_years_arr)

                preds_matrix[i] = model.predict(X_test)

            # Compute prediction intervals
            p05 = np.percentile(preds_matrix, 5, axis=0)
            p50 = np.percentile(preds_matrix, 50, axis=0)
            p95 = np.percentile(preds_matrix, 95, axis=0)

            pi_df = pd.DataFrame({
                "Year": df_test["Year"].values,
                "AgentID": aid,
                "Obs": y_test,
                "p05": p05,
                "p50": p50,
                "p95": p95,
            })

            pi_path = TWO_REGIME_DIR / f"prediction_intervals_{aid}_{name}.csv"
            pi_df.to_csv(pi_path, index=False)
            print(f"    Saved {pi_path}")

            # Summary statistics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            med_rmse = np.sqrt(mean_squared_error(y_test, p50))
            med_mae = mean_absolute_error(y_test, p50)
            med_r2 = r2_score(y_test, p50) if len(y_test) > 1 else float("nan")

            pi_widths = p95 - p05
            coverage = np.mean((y_test >= p05) & (y_test <= p95))

            member_rmses = np.array([
                np.sqrt(mean_squared_error(y_test, preds_matrix[i]))
                for i in range(N_TWO_REGIME_BOOT)
            ])

            summary = {
                "AgentID": aid,
                "cp_year": cp_year,
                "regime": name,
                "RMSE_median_pred": med_rmse,
                "MAE_median_pred": med_mae,
                "R2_median_pred": med_r2,
                "RMSE_ens_median": np.median(member_rmses),
                "RMSE_ens_IQR_lo": np.percentile(member_rmses, 25),
                "RMSE_ens_IQR_hi": np.percentile(member_rmses, 75),
                "PI_width_mean": np.mean(pi_widths),
                "PI_width_median": np.median(pi_widths),
                "PI_coverage_90": coverage,
            }

            # Save per-agent-regime summary
            summary_path = TWO_REGIME_DIR / f"summary_{aid}_{name}.csv"
            pd.DataFrame([summary]).to_csv(summary_path, index=False)
            print(f"    Saved {summary_path}")

            combined_summaries.append(summary)

            print(f"    RMSE (median pred): {med_rmse:.2f}")
            print(f"    R² (median pred):   {med_r2:.3f}")
            print(f"    PI width (mean):    {np.mean(pi_widths):.2f}")
            print(f"    PI coverage (90%):  {coverage:.3f}")

    # Combined summary
    combined_path = TWO_REGIME_DIR / "two_regime_c1_ensemble_summary.csv"
    pd.DataFrame(combined_summaries).to_csv(combined_path, index=False)
    print(f"\nSaved {combined_path}")

    # Plot per-agent PI figures
    print("\nGenerating per-agent PI plots...")
    for aid in AGENT_IDS:
        plot_agent_pi(aid, annual,
                      TWO_REGIME_DIR / f"two_regime_agent{aid}_pi.pdf")

    # Final report
    print("\n" + "=" * 60)
    print("CLUSTER 1 — ENSEMBLE SUMMARY")
    print("=" * 60)
    for s in combined_summaries:
        print(f"  Agent {s['AgentID']} {s['regime']:4s}  "
              f"RMSE={s['RMSE_median_pred']:.2f}  "
              f"R²={s['R2_median_pred']:.3f}  "
              f"PI_width={s['PI_width_mean']:.1f}  "
              f"coverage={s['PI_coverage_90']:.3f}")
    print("=" * 60)


# ── Plotting ─────────────────────────────────────────────────────────────────
_RC_PARAMS = {
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.0,
}


def plot_pooled_point(annual, regime_name, output_pdf):
    """Plot pooled-by-regime point estimates (1x3 panel grid, one per agent).

    Reads saved pooled_predictions_v2 CSV from TWO_REGIME_DIR.
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    pred_path = TWO_REGIME_DIR / f"pooled_predictions_{regime_name}_v2.csv"
    pred_df = pd.read_csv(pred_path)

    # Extract model/feature_set info for title
    model_name = pred_df["model"].iloc[0] if "model" in pred_df.columns else ""
    fs_name = (pred_df["feature_set"].iloc[0]
               if "feature_set" in pred_df.columns else "")

    colors = {"pre": "tab:blue", "post": "tab:red"}
    color = colors[regime_name]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             constrained_layout=True)

    for col_idx, aid in enumerate(AGENT_IDS):
        ax = axes[col_idx]
        cp_year = AGENT_CONFIGS[aid]["cp_year"]

        # Agent's predictions
        agent_pred = pred_df[pred_df["AgentID"] == aid].sort_values("Year")
        train_pred = agent_pred[agent_pred["Split"] == "train"]
        test_pred = agent_pred[agent_pred["Split"] == "test"]

        # Full observed time series for context
        obs = annual[annual["AgentID"] == aid].sort_values("Year")
        ax.plot(obs["Year"].values,
                obs["Irrigation_Depth"].values,
                "ko-", label="Observed" if col_idx == 0 else None,
                linewidth=1.2, markersize=4, zorder=5)

        # Training period shading
        tv_years = train_pred["Year"].values
        if len(tv_years) > 0:
            shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
            if col_idx == 0:
                shade_kw["label"] = "Training period"
            ax.axvspan(tv_years.min() - 0.3, tv_years.max() + 0.3,
                       **shade_kw)

        # Train predictions (solid)
        ax.plot(train_pred["Year"].values, train_pred["Pred"].values,
                "-", color=color, linewidth=1.5,
                label="Train pred" if col_idx == 0 else None,
                zorder=3)

        # Train R²
        if len(train_pred) > 1:
            train_r2 = r2_score(train_pred["Obs"].values,
                                train_pred["Pred"].values)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(tv_years.min() + 0.3, 0.88,
                    f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

        # Test predictions (dashed)
        ax.plot(test_pred["Year"].values, test_pred["Pred"].values,
                "--", color=color, linewidth=1.5,
                label="Test pred" if col_idx == 0 else None,
                zorder=4)

        # Test R²
        if len(test_pred) > 1:
            test_r2 = r2_score(test_pred["Obs"].values,
                               test_pred["Pred"].values)
            ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=12, color="black")

        # CP line
        ax.axvline(cp_year, color="0.6", linestyle="--", linewidth=1,
                   zorder=1)

        # Panel label
        panel_label = f"({chr(ord('a') + col_idx)})"
        ax.text(0.01, 0.99, panel_label,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        ax.set_title(f"Agent {aid} (CP={cp_year})")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)
        ax.set_xlabel("Year")
        if col_idx == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.05))

    regime_label = "Pre-CP" if regime_name == "pre" else "Post-CP"
    title_suffix = f" [{model_name}, {fs_name}]" if model_name else ""
    fig.suptitle(f"Cluster 1 — Pooled {regime_label} Point Estimates"
                 f"{title_suffix}",
                 fontsize=16, y=1.09)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_pooled_agent_point(aid, annual, output_pdf):
    """Plot per-agent two-regime point estimates from pooled v2 predictions.

    Reads pooled_predictions_{pre,post}_v2.csv filtered by AgentID.
    Same visual style as plot_agent_point().
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    config = AGENT_CONFIGS[aid]
    cp_year = config["cp_year"]
    regime_a, regime_b = make_regimes(cp_year)
    regimes = [regime_a, regime_b]
    colors = {"pre": "tab:blue", "post": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for col_idx, regime in enumerate(regimes):
        name = regime["name"]
        color = colors[name]
        ax = axes[col_idx]

        # Load pooled predictions, filter to this agent
        pred_path = TWO_REGIME_DIR / f"pooled_predictions_{name}_v2.csv"
        pred_all = pd.read_csv(pred_path)
        pred_df = pred_all[pred_all["AgentID"] == aid].copy()
        train_pred = pred_df[pred_df["Split"] == "train"].sort_values("Year")
        test_pred = pred_df[pred_df["Split"] == "test"].sort_values("Year")

        # Full regime observed data
        df_full = annual[(annual["AgentID"] == aid) &
                         (annual["Year"].isin(regime["full_years"]))].copy()
        df_full = df_full.sort_values("Year")

        # Observed
        ax.plot(df_full["Year"].values,
                df_full["Irrigation_Depth"].values,
                "ko-", label="Observed", linewidth=1.2, markersize=5,
                zorder=5)

        # Training period shading
        tv_min = regime["trainval_years"][0]
        tv_max = regime["trainval_years"][-1]
        shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
        if col_idx == 0:
            shade_kw["label"] = "Training period"
        ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

        # Train predictions (solid)
        ax.plot(train_pred["Year"].values, train_pred["Pred"].values,
                "-", color=color, linewidth=1.5, label="Train pred",
                zorder=3)

        # Train R²
        if len(train_pred) > 1:
            train_r2 = r2_score(train_pred["Obs"].values,
                                train_pred["Pred"].values)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

        # Test predictions (dashed)
        ax.plot(test_pred["Year"].values, test_pred["Pred"].values,
                "--", color=color, linewidth=1.5, label="Test pred",
                zorder=4)

        # Test R²
        if len(test_pred) > 1:
            test_r2 = r2_score(test_pred["Obs"].values,
                               test_pred["Pred"].values)
            ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=12, color="black")

        # Changepoint line
        ax.axvline(cp_year, color="0.6", linestyle="--", linewidth=1,
                   zorder=1)

        # Panel label
        panel_label = f"({chr(ord('a') + col_idx)})"
        ax.text(0.01, 0.99, panel_label,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        regime_label = "Pre-CP" if name == "pre" else "Post-CP"
        ax.set_title(f"{regime_label} ({regime['full_years'][0]}"
                     f"\u2013{regime['full_years'][-1]})")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)
        ax.set_xlabel("Year")
        if col_idx == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(f"Agent {aid} — Pooled Two-Regime Point Estimates "
                 f"(CP = {cp_year})",
                 fontsize=16, y=1.09)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_agent_point(aid, annual, output_pdf):
    """Plot per-agent two-regime point estimates (2 panels: pre, post).

    Reads saved point_predictions CSVs from TWO_REGIME_DIR.
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    config = AGENT_CONFIGS[aid]
    cp_year = config["cp_year"]
    regime_a, regime_b = make_regimes(cp_year)
    regimes = [regime_a, regime_b]
    colors = {"pre": "tab:blue", "post": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for col_idx, regime in enumerate(regimes):
        name = regime["name"]
        color = colors[name]
        ax = axes[col_idx]

        # Load saved predictions
        pred_path = TWO_REGIME_DIR / f"point_predictions_{aid}_{name}.csv"
        pred_df = pd.read_csv(pred_path)
        train_pred = pred_df[pred_df["Split"] == "train"].sort_values("Year")
        test_pred = pred_df[pred_df["Split"] == "test"].sort_values("Year")

        # Full regime observed data
        df_full = annual[(annual["AgentID"] == aid) &
                         (annual["Year"].isin(regime["full_years"]))].copy()
        df_full = df_full.sort_values("Year")

        # Observed
        ax.plot(df_full["Year"].values,
                df_full["Irrigation_Depth"].values,
                "ko-", label="Observed", linewidth=1.2, markersize=5,
                zorder=5)

        # Training period shading
        tv_min = regime["trainval_years"][0]
        tv_max = regime["trainval_years"][-1]
        shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
        if col_idx == 0:
            shade_kw["label"] = "Training period"
        ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

        # Train predictions (solid)
        ax.plot(train_pred["Year"].values, train_pred["Pred"].values,
                "-", color=color, linewidth=1.5, label="Train pred",
                zorder=3)

        # Train R²
        if len(train_pred) > 1:
            train_r2 = r2_score(train_pred["Obs"].values,
                                train_pred["Pred"].values)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

        # Test predictions (dashed)
        ax.plot(test_pred["Year"].values, test_pred["Pred"].values,
                "--", color=color, linewidth=1.5, label="Test pred",
                zorder=4)

        # Test R²
        if len(test_pred) > 1:
            test_r2 = r2_score(test_pred["Obs"].values,
                               test_pred["Pred"].values)
            ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=12, color="black")

        # Changepoint line
        ax.axvline(cp_year, color="0.6", linestyle="--", linewidth=1,
                   zorder=1)

        # Panel label
        panel_label = f"({chr(ord('a') + col_idx)})"
        ax.text(0.01, 0.99, panel_label,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        regime_label = "Pre-CP" if name == "pre" else "Post-CP"
        ax.set_title(f"{regime_label} ({regime['full_years'][0]}"
                     f"\u2013{regime['full_years'][-1]})")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)
        ax.set_xlabel("Year")
        if col_idx == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(f"Agent {aid} — Two-Regime Point Estimates (CP = {cp_year})",
                 fontsize=16, y=1.09)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_agent_pi(aid, annual, output_pdf):
    """Plot per-agent two-regime prediction intervals (2 panels: pre, post).

    Reads saved point_predictions and prediction_intervals CSVs.
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    config = AGENT_CONFIGS[aid]
    cp_year = config["cp_year"]
    regime_a, regime_b = make_regimes(cp_year)
    regimes = [regime_a, regime_b]
    colors = {"pre": "tab:blue", "post": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for col_idx, regime in enumerate(regimes):
        name = regime["name"]
        color = colors[name]
        ax = axes[col_idx]

        # Load saved point predictions (for train portion)
        pred_path = TWO_REGIME_DIR / f"point_predictions_{aid}_{name}.csv"
        pred_df = pd.read_csv(pred_path)
        train_pred = pred_df[pred_df["Split"] == "train"].sort_values("Year")

        # Load PI data
        pi_path = TWO_REGIME_DIR / f"prediction_intervals_{aid}_{name}.csv"
        pi_df = pd.read_csv(pi_path).sort_values("Year")

        # Full regime observed data
        df_full = annual[(annual["AgentID"] == aid) &
                         (annual["Year"].isin(regime["full_years"]))].copy()
        df_full = df_full.sort_values("Year")

        # Observed
        ax.plot(df_full["Year"].values,
                df_full["Irrigation_Depth"].values,
                "ko-", label="Observed", linewidth=1.2, markersize=5,
                zorder=5)

        # Training period shading
        tv_min = regime["trainval_years"][0]
        tv_max = regime["trainval_years"][-1]
        shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
        if col_idx == 0:
            shade_kw["label"] = "Training period"
        ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

        # Train predictions (solid, from point estimates)
        ax.plot(train_pred["Year"].values, train_pred["Pred"].values,
                "-", color=color, linewidth=1.5, label="Train pred",
                zorder=3)

        # Train R²
        if len(train_pred) > 1:
            train_r2 = r2_score(train_pred["Obs"].values,
                                train_pred["Pred"].values)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

        # Test: PI band + median
        test_yrs = pi_df["Year"].values
        ax.fill_between(test_yrs, pi_df["p05"], pi_df["p95"],
                        alpha=0.25, color=color, label="90% PI",
                        zorder=2)
        ax.plot(test_yrs, pi_df["p50"], "--", color=color,
                linewidth=1.5, label="Test median", zorder=4)

        # Test R² (from median)
        if len(pi_df) > 1:
            test_r2 = r2_score(pi_df["Obs"].values, pi_df["p50"].values)
            ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=12, color="black")

        # Changepoint line
        ax.axvline(cp_year, color="0.6", linestyle="--", linewidth=1,
                   zorder=1)

        # Panel label
        panel_label = f"({chr(ord('a') + col_idx)})"
        ax.text(0.01, 0.99, panel_label,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        regime_label = "Pre-CP" if name == "pre" else "Post-CP"
        ax.set_title(f"{regime_label} ({regime['full_years'][0]}"
                     f"\u2013{regime['full_years'][-1]})")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)
        ax.set_xlabel("Year")
        if col_idx == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(
        f"Agent {aid} — Two-Regime Prediction Intervals (CP = {cp_year})",
        fontsize=16, y=1.09)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


# ── CP Sensitivity Analysis ──────────────────────────────────────────────────
CP_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def _run_cp_sensitivity_pass(annual, drop_year=False):
    """Single CP sensitivity pass (with or without Year feature).

    Returns (all_results, summary_rows, predictions_rows) for downstream
    consumption by run_cp_sensitivity().
    """
    label = "no-Year" if drop_year else "with-Year"
    print(f"\n{'=' * 60}")
    print(f"CLUSTER 1 — CP SENSITIVITY ({label.upper()})")
    print(f"{'=' * 60}")
    print(f"  CP candidates:    {CP_CANDIDATES_SENS}")
    print(f"  Common eval start: {COMMON_EVAL_START}")
    print(f"  Agents pooled:    {AGENT_IDS}")
    print(f"  Drop Year feature: {drop_year}")

    # Eval set: all agents × COMMON_EVAL_START–2020
    df_eval = annual[annual["Year"] >= COMMON_EVAL_START].copy()
    X_eval, y_eval = prepare_features_c1(df_eval, drop_year=drop_year)
    print(f"\n  Eval set: {len(df_eval)} rows ({df_eval['Year'].nunique()} years "
          f"× {df_eval['AgentID'].nunique()} agents)")
    print(f"  Features: {X_eval.shape[1]} — {list(X_eval.columns)}")

    all_results = {}
    summary_rows = []
    predictions_rows = []

    for cp in CP_CANDIDATES_SENS:
        print(f"\n{'─' * 50}")
        print(f"CP = {cp}")
        print(f"{'─' * 50}")

        df_train = annual[annual["Year"] <= cp].copy()
        X_train, y_train = prepare_features_c1(df_train, drop_year=drop_year)

        print(f"  Train: 1993–{cp} ({len(df_train)} rows, "
              f"{df_train['AgentID'].nunique()} agents)")
        print(f"  Features: {X_train.shape[1]}")

        # Rolling-origin folds
        folds = _make_rolling_folds(cp)
        for i, fold in enumerate(folds):
            print(f"    Fold {i}: train {fold['train_years'][0]}-"
                  f"{fold['train_years'][-1]}, "
                  f"val {fold['val_years'][0]}-{fold['val_years'][-1]}")

        # Macro-averaged Optuna tuning
        pool_seed = MASTER_SEED + cp
        print(f"  Tuning (seed={pool_seed})...")
        best_params, fold_details = _optuna_tune_macro(
            X_train, y_train,
            agent_ids=df_train["AgentID"].values,
            folds=folds,
            seed=pool_seed,
            years=df_train["Year"].values,
        )

        # Save params
        suffix = "_noyear" if drop_year else ""
        params_path = TWO_REGIME_DIR / f"best_params_pooled_cp{cp}{suffix}.json"
        with open(params_path, "w") as f:
            json.dump({**best_params, **fold_details}, f, indent=2)
        print(f"  Saved {params_path}")

        # Final retrain on full 1993–CP (no early stopping)
        retrain_params = {
            k: v for k, v in best_params.items()
            if k != "early_stopping_rounds"
        }
        retrain_params["objective"] = "reg:squarederror"
        retrain_params["verbosity"] = 0
        model = XGBRegressor(**retrain_params)
        X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
        X_eval_arr = X_eval.values if hasattr(X_eval, "values") else X_eval
        model.fit(X_train_arr, y_train)

        pred_train = model.predict(X_train_arr)
        pred_eval = model.predict(X_eval_arr)

        # Overall metrics
        train_metrics = compute_metrics(y_train, pred_train)
        eval_metrics = compute_metrics(y_eval, pred_eval)

        print(f"  Train (pooled) — RMSE={train_metrics['RMSE']:.2f}, "
              f"R²={train_metrics['R2']:.3f}")
        print(f"  Eval  (pooled) — RMSE={eval_metrics['RMSE']:.2f}, "
              f"R²={eval_metrics['R2']:.3f}")

        # Per-agent eval metrics
        per_agent_r2 = {}
        for aid in AGENT_IDS:
            mask_eval = df_eval["AgentID"] == aid
            if mask_eval.sum() > 1:
                agent_r2 = r2_score(y_eval[mask_eval.values],
                                    pred_eval[mask_eval.values])
            else:
                agent_r2 = float("nan")
            per_agent_r2[aid] = agent_r2
            print(f"    Agent {aid} eval R² = {agent_r2:.3f}")

        # Store results for plotting
        all_results[cp] = {
            "train_df": df_train,
            "pred_train": pred_train,
            "eval_df": df_eval,
            "pred_eval": pred_eval,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "per_agent_r2": per_agent_r2,
        }

        # Store predictions
        for split_name, split_df, split_pred in [
            ("train", df_train, pred_train),
            ("eval", df_eval, pred_eval),
        ]:
            for idx_row in range(len(split_df)):
                predictions_rows.append({
                    "year": split_df.iloc[idx_row]["Year"],
                    "agent_id": int(split_df.iloc[idx_row]["AgentID"]),
                    "obs": split_df.iloc[idx_row]["Irrigation_Depth"],
                    "pred": split_pred[idx_row],
                    "split": split_name,
                    "cp": cp,
                })

        # Summary rows — one overall + per-agent
        summary_rows.append({
            "scope": "pooled",
            "AgentID": "all",
            "cp_candidate": cp,
            "train_n": len(df_train),
            "eval_n": len(df_eval),
            "n_estimators": best_params["n_estimators"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "eval_RMSE": eval_metrics["RMSE"],
            "eval_MAE": eval_metrics["MAE"],
            "eval_R2": eval_metrics["R2"],
            "eval_Bias": eval_metrics["Bias"],
        })
        for aid in AGENT_IDS:
            mask_tr = df_train["AgentID"] == aid
            mask_ev = df_eval["AgentID"] == aid
            agent_train_m = compute_metrics(
                y_train[mask_tr.values], pred_train[mask_tr.values])
            agent_eval_m = compute_metrics(
                y_eval[mask_ev.values], pred_eval[mask_ev.values])
            summary_rows.append({
                "scope": "per_agent",
                "AgentID": aid,
                "cp_candidate": cp,
                "train_n": int(mask_tr.sum()),
                "eval_n": int(mask_ev.sum()),
                "n_estimators": best_params["n_estimators"],
                "train_RMSE": agent_train_m["RMSE"],
                "train_R2": agent_train_m["R2"],
                "eval_RMSE": agent_eval_m["RMSE"],
                "eval_MAE": agent_eval_m["MAE"],
                "eval_R2": agent_eval_m["R2"],
                "eval_Bias": agent_eval_m["Bias"],
            })

    return all_results, summary_rows, predictions_rows


def _add_delta_rows(summary_rows):
    """Append delta rows (CP2005-CP2004, CP2006-CP2005, CP2006-CP2004)."""
    delta_pairs = [
        (2005, 2004, "Δ(2005−2004)"),
        (2006, 2005, "Δ(2006−2005)"),
        (2006, 2004, "Δ(2006−2004)"),
    ]
    delta_cols = ["train_RMSE", "train_R2", "eval_RMSE", "eval_MAE",
                  "eval_R2", "eval_Bias"]

    df_sum = pd.DataFrame(summary_rows)
    delta_rows = []
    for cp_hi, cp_lo, label in delta_pairs:
        for scope in df_sum["scope"].unique():
            for agent_id in df_sum.loc[df_sum["scope"] == scope, "AgentID"].unique():
                row_hi = df_sum[(df_sum["cp_candidate"] == cp_hi) &
                                (df_sum["scope"] == scope) &
                                (df_sum["AgentID"] == agent_id)]
                row_lo = df_sum[(df_sum["cp_candidate"] == cp_lo) &
                                (df_sum["scope"] == scope) &
                                (df_sum["AgentID"] == agent_id)]
                if len(row_hi) == 1 and len(row_lo) == 1:
                    delta = {"scope": scope, "AgentID": agent_id,
                             "cp_candidate": label,
                             "train_n": "", "eval_n": "",
                             "n_estimators": ""}
                    for col in delta_cols:
                        delta[col] = (row_hi[col].values[0] -
                                      row_lo[col].values[0])
                    delta_rows.append(delta)
    return summary_rows + delta_rows


def run_cp_sensitivity(annual):
    """CP sensitivity: pooled agents 12/18/20 with rolling-origin CV.

    Runs two passes:
      1. With Year feature (primary)
      2. Without Year feature (Year-ablation diagnostic)

    Uses macro-averaged per-agent RMSE as Optuna objective with
    conservative fixed search space and early-stopping within folds.
    """
    TWO_REGIME_DIR.mkdir(parents=True, exist_ok=True)

    # ── Primary pass (with Year) ──
    all_results, summary_rows, pred_rows = _run_cp_sensitivity_pass(
        annual, drop_year=False)

    # Add delta rows
    summary_with_deltas = _add_delta_rows(summary_rows)

    # Save summary CSV
    summary_path = TWO_REGIME_DIR / "c1_cp_sensitivity_summary.csv"
    pd.DataFrame(summary_with_deltas).to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # Save predictions CSV
    pred_path = TWO_REGIME_DIR / "c1_cp_sensitivity_predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"Saved {pred_path}")

    # Plot
    print("\nGenerating sensitivity plot...")
    plot_cp_sensitivity(annual, all_results,
                        TWO_REGIME_DIR / "c1_cp_sensitivity_point_estimates.pdf")

    # Save per-agent metrics CSV
    df_sum = pd.DataFrame(summary_rows)
    per_agent_df = df_sum[df_sum["scope"] == "per_agent"][
        ["AgentID", "cp_candidate", "train_RMSE", "train_R2",
         "eval_RMSE", "eval_R2"]].copy()
    per_agent_path = TWO_REGIME_DIR / "c1_cp_sensitivity_per_agent_metrics.csv"
    per_agent_df.to_csv(per_agent_path, index=False)
    print(f"Saved {per_agent_path}")

    # Plot per-agent metrics
    print("\nGenerating per-agent metrics plot...")
    plot_cp_sensitivity_per_agent_metrics(
        summary_rows,
        TWO_REGIME_DIR / "c1_cp_sensitivity_per_agent_metrics.pdf")

    # Print primary summary
    print("\n" + "=" * 60)
    print("CLUSTER 1 — CP SENSITIVITY SUMMARY (WITH YEAR)")
    print("=" * 60)
    for row in summary_with_deltas:
        label = (f"  {row['scope']:10s} Agent={str(row['AgentID']):3s} "
                 f"CP={str(row['cp_candidate']):14s}")
        print(f"{label}  train R²={row['train_R2']:+.3f}  "
              f"eval RMSE={row['eval_RMSE']:+.2f} R²={row['eval_R2']:+.3f}")
    print("=" * 60)

    # ── Year-ablation pass ──
    all_results_ny, summary_rows_ny, pred_rows_ny = _run_cp_sensitivity_pass(
        annual, drop_year=True)

    summary_with_deltas_ny = _add_delta_rows(summary_rows_ny)

    summary_ny_path = TWO_REGIME_DIR / "c1_cp_sensitivity_summary_noyear.csv"
    pd.DataFrame(summary_with_deltas_ny).to_csv(summary_ny_path, index=False)
    print(f"\nSaved {summary_ny_path}")

    pred_ny_path = TWO_REGIME_DIR / "c1_cp_sensitivity_predictions_noyear.csv"
    pd.DataFrame(pred_rows_ny).to_csv(pred_ny_path, index=False)
    print(f"Saved {pred_ny_path}")

    # Print Year-ablation summary
    print("\n" + "=" * 60)
    print("CLUSTER 1 — CP SENSITIVITY SUMMARY (NO YEAR)")
    print("=" * 60)
    for row in summary_with_deltas_ny:
        label = (f"  {row['scope']:10s} Agent={str(row['AgentID']):3s} "
                 f"CP={str(row['cp_candidate']):14s}")
        print(f"{label}  train R²={row['train_R2']:+.3f}  "
              f"eval RMSE={row['eval_RMSE']:+.2f} R²={row['eval_R2']:+.3f}")
    print("=" * 60)

    # ── Interpretation ──
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    df_primary = pd.DataFrame(summary_rows)
    pooled = df_primary[df_primary["scope"] == "pooled"].set_index("cp_candidate")
    for cp_a, cp_b in [(2004, 2005), (2005, 2006), (2004, 2006)]:
        rmse_a = pooled.loc[cp_a, "eval_RMSE"]
        rmse_b = pooled.loc[cp_b, "eval_RMSE"]
        r2_a = pooled.loc[cp_a, "eval_R2"]
        r2_b = pooled.loc[cp_b, "eval_R2"]
        direction = "improves" if rmse_b < rmse_a else "worsens"
        print(f"  CP {cp_a}→{cp_b}: eval RMSE {rmse_a:.2f}→{rmse_b:.2f} "
              f"({direction}), R² {r2_a:.3f}→{r2_b:.3f}")
    print("=" * 60)


def plot_cp_sensitivity_per_agent_metrics(summary_rows, output_pdf):
    """Plot per-agent RMSE and R² across CP candidates (1×2 panels).

    Args:
        summary_rows: list of dicts from _run_cp_sensitivity_pass()
            (non-delta rows only).
        output_pdf: output path for the figure.
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    df = pd.DataFrame(summary_rows)
    df_agents = df[df["scope"] == "per_agent"].copy()
    df_agents["AgentID"] = df_agents["AgentID"].astype(int)

    agent_colors = {12: "tab:blue", 18: "tab:red", 20: "tab:green"}

    fig, (ax_rmse, ax_r2) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True)

    for aid in AGENT_IDS:
        sub = df_agents[df_agents["AgentID"] == aid].sort_values("cp_candidate")
        cps = sub["cp_candidate"].values
        color = agent_colors[aid]

        # RMSE panel
        rmses = sub["eval_RMSE"].values
        ax_rmse.plot(cps, rmses, "o-", color=color, linewidth=1.5,
                     markersize=8, label=f"Agent {aid}", zorder=3)
        for x, v in zip(cps, rmses):
            ax_rmse.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=11,
                             color=color)

        # R² panel
        r2s = sub["eval_R2"].values
        ax_r2.plot(cps, r2s, "o-", color=color, linewidth=1.5,
                   markersize=8, label=f"Agent {aid}", zorder=3)
        for x, v in zip(cps, r2s):
            ax_r2.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=11,
                           color=color)

    # RMSE panel formatting
    ax_rmse.text(0.01, 0.99, "(a)", transform=ax_rmse.transAxes,
                 va="top", ha="left", fontsize=14, fontweight="bold")
    ax_rmse.set_xlabel("CP candidate")
    ax_rmse.set_ylabel("Eval RMSE (mm)")
    ax_rmse.set_xticks(CP_CANDIDATES_SENS)
    ax_rmse.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax_rmse.tick_params(axis="both", which="major", length=4)

    # R² panel formatting
    ax_r2.text(0.01, 0.99, "(b)", transform=ax_r2.transAxes,
               va="top", ha="left", fontsize=14, fontweight="bold")
    ax_r2.axhline(0, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax_r2.set_xlabel("CP candidate")
    ax_r2.set_ylabel("Eval R²")
    ax_r2.set_xticks(CP_CANDIDATES_SENS)
    ax_r2.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax_r2.tick_params(axis="both", which="major", length=4)

    # Legend
    fig.legend(*ax_rmse.get_legend_handles_labels(),
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Cluster 1 — Per-Agent CP Sensitivity Metrics",
                 fontsize=16, y=1.09)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_cp_sensitivity(annual, all_results, output_pdf):
    """Plot CP sensitivity: 1×3 per-agent grid with pooled predictions.

    Args:
        annual: full annual DataFrame (for observed time series).
        all_results: dict {cp: {train_df, pred_train, eval_df, pred_eval,
                                per_agent_r2, ...}}.
        output_pdf: output path.
    """
    from matplotlib.lines import Line2D

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    n_agents = len(AGENT_IDS)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(n_agents)]
    bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8)

    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5.5),
                             constrained_layout=True)
    if n_agents == 1:
        axes = [axes]

    for col_idx, aid in enumerate(AGENT_IDS):
        ax = axes[col_idx]

        # Full observed time series
        obs_sub = annual[annual["AgentID"] == aid].sort_values("Year")
        ax.plot(obs_sub["Year"].values,
                obs_sub["Irrigation_Depth"].values,
                "ko-", label="Observed", linewidth=1.2, markersize=4,
                zorder=5)

        # Overlay each CP candidate
        r2_texts = []
        for i, cp in enumerate(CP_CANDIDATES_SENS):
            res = all_results[cp]
            color = CP_COLORS[i]

            # Extract this agent's predictions from pooled results
            train_mask = res["train_df"]["AgentID"] == aid
            eval_mask = res["eval_df"]["AgentID"] == aid

            train_years = res["train_df"].loc[train_mask, "Year"].values
            train_pred = res["pred_train"][train_mask.values]
            eval_years = res["eval_df"].loc[eval_mask, "Year"].values
            eval_pred = res["pred_eval"][eval_mask.values]

            # Training predictions (solid)
            ax.plot(train_years, train_pred, "-",
                    color=color, linewidth=1.5, zorder=2,
                    label=f"CP={cp}" if col_idx == 0 else None)

            # Eval predictions (dashed)
            ax.plot(eval_years, eval_pred, "--",
                    color=color, linewidth=1.5, zorder=2)

            # Vertical CP line
            ax.axvline(cp, color=color, linestyle=":", linewidth=0.9,
                       alpha=0.6)

            agent_r2 = res["per_agent_r2"][aid]
            r2_texts.append((cp, agent_r2, color))

        # Stacked R² annotations (bottom-right)
        for i, (cp, r2, color) in enumerate(r2_texts):
            ax.text(0.98, 0.04 + i * 0.09,
                    f"CP={cp}: R$^2$={r2:.2f}",
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=11, color=color, bbox=bbox_props, zorder=10)

        # Panel label
        ax.text(0.02, 0.95, panel_labels[col_idx],
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold", bbox=bbox_props, zorder=10)

        ax.set_title(f"Agent {aid}")
        ax.set_xlabel("Year")
        if col_idx == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")
        ax.tick_params(axis="both", which="major", length=4)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)

    # Figure legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.4", linestyle="-", linewidth=1.5))
    labels.append("Train")
    handles.append(Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.5))
    labels.append("Eval")
    fig.legend(handles, labels, loc="lower center",
               ncol=len(handles), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Cluster 1 \u2014 CP Sensitivity (Pooled Agents)",
                 fontsize=16)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


# ── POST-CP Split-Point Sweep ─────────────────────────────────────────────────
def run_post_split_sweep(annual):
    """Sweep POST-CP train/test split points to find best test performance.

    For each test_start_year in POST_SPLIT_CANDIDATES, builds POST datasets,
    runs full model selection (feature_set × model × transform), and records
    metrics. Generates per-agent plots for each split and a comparison plot.
    """
    print("=" * 60)
    print("CLUSTER 1 — POST-CP SPLIT-POINT SWEEP")
    print("=" * 60)

    SPLIT_SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    sweep_summaries = []

    for test_start in POST_SPLIT_CANDIDATES:
        print(f"\n{'═' * 60}")
        print(f"  TEST START YEAR: {test_start}")
        print(f"{'═' * 60}")

        # 1. Build datasets
        post_tv, post_te = build_post_datasets_for_split(annual, test_start)

        # 2. Generate adaptive CV folds
        folds = make_post_rolling_folds_adaptive(post_tv)
        print(f"  CV folds: {len(folds)}")
        for i, (tr_idx, va_idx) in enumerate(folds):
            tr_years = sorted(post_tv.iloc[tr_idx]["Year"].unique())
            va_years = sorted(post_tv.iloc[va_idx]["Year"].unique())
            print(f"    Fold {i}: train {len(tr_idx)} rows "
                  f"({tr_years[0]}-{tr_years[-1]}), "
                  f"val {len(va_idx)} rows "
                  f"({va_years[0]}-{va_years[-1]})")

        y_tv = post_tv["Irrigation_Depth"].values
        agent_ids_tv = post_tv["AgentID"].values
        y_te = post_te["Irrigation_Depth"].values
        agent_ids_te = post_te["AgentID"].values

        # 3. Score all (feature_set, model, transform) combos
        regime_cv_results = []
        all_leaderboard = []
        regime_seed = MASTER_SEED + 100  # POST regime seed

        for fs_name in POST_FEATURE_SETS:
            feature_cols = get_feature_columns(fs_name, regime="post")
            print(f"\n  Feature set: {fs_name} ({len(feature_cols)} cols)")

            X_tv = post_tv[feature_cols].copy()
            X_te = post_te[feature_cols].copy()

            candidates = build_model_candidates("post")

            for transform in ["identity", "log1p"]:
                for cand in candidates:
                    combo_label = f"{fs_name}/{cand['name']}/{transform}"
                    print(f"    Scoring {combo_label}...", end="")
                    result = cv_score_model(
                        cand, X_tv, y_tv, agent_ids_tv, folds,
                        seed=regime_seed, transform=transform,
                    )
                    result["feature_set"] = fs_name
                    result["transform"] = transform
                    result["feature_cols"] = feature_cols
                    regime_cv_results.append(result)
                    print(f" macro_RMSE={result['mean_macro_rmse']:.2f}"
                          f"  max_agent="
                          f"{result.get('mean_max_agent_rmse', float('nan')):.2f}")

                    all_leaderboard.append({
                        "test_start": test_start,
                        "feature_set": fs_name,
                        "model": result["name"],
                        "transform": transform,
                        "mean_macro_rmse": result["mean_macro_rmse"],
                        "mean_macro_mae": result.get("mean_macro_mae",
                                                     float("nan")),
                        "mean_max_agent_rmse": result.get(
                            "mean_max_agent_rmse", float("nan")),
                        "best_params": json.dumps(
                            result.get("best_params", {})),
                    })

        # 4. Select best model with sanity check
        ranked = sorted(regime_cv_results, key=lambda r: (
            r["mean_macro_rmse"],
            r.get("mean_max_agent_rmse", float("inf")),
            {"Ridge": 0, "ElasticNet": 1,
             "XGBoost_gblinear": 2, "XGBoost_gbtree": 3}.get(
                r["name"], 99),
        ))

        tv_max = y_tv.max()
        pred_bound = 2 * tv_max
        best = None
        n_skipped = 0
        for candidate_best in ranked:
            cand_cols = candidate_best["feature_cols"]
            cand_transform = candidate_best["transform"]
            X_tv_cand = post_tv[cand_cols].copy()
            X_te_cand = post_te[cand_cols].copy()
            cands = build_model_candidates("post")

            _, pred_tv_cand, pred_te_cand = _refit_and_predict(
                candidate_best, cands, X_tv_cand, y_tv, X_te_cand,
                cand_transform)

            max_pred = max(pred_tv_cand.max(), pred_te_cand.max())
            min_pred = min(pred_tv_cand.min(), pred_te_cand.min())
            if max_pred > pred_bound or min_pred < -pred_bound:
                n_skipped += 1
                continue

            best = candidate_best
            print(f"\n  Passed sanity check ({n_skipped} skipped)")
            break

        if best is None:
            print(f"\n  WARNING: All models exceed bounds, using top-ranked")
            best = ranked[0]

        best_fs = best["feature_set"]
        best_transform = best["transform"]
        best_cols = best["feature_cols"]

        print(f"\n  WINNER: {best['name']} / {best_fs} / {best_transform}")
        print(f"    macro RMSE = {best['mean_macro_rmse']:.2f}")
        print(f"    Params: {best['best_params']}")

        # 5. Refit on full trainval
        X_tv_best = post_tv[best_cols].copy()
        X_te_best = post_te[best_cols].copy()
        candidates = build_model_candidates("post")

        final_model, pred_tv, pred_te = _refit_and_predict(
            best, candidates, X_tv_best, y_tv, X_te_best, best_transform)

        # 6. Compute metrics — pooled + per-agent
        train_metrics = compute_metrics(y_tv, pred_tv)
        test_metrics = compute_metrics(y_te, pred_te)

        print(f"\n  POOLED Train — RMSE={train_metrics['RMSE']:.2f}, "
              f"R²={train_metrics['R2']:.3f}")
        print(f"  POOLED Test  — RMSE={test_metrics['RMSE']:.2f}, "
              f"R²={test_metrics['R2']:.3f}")

        sweep_summaries.append({
            "test_start": test_start,
            "AgentID": "pooled",
            "model": best["name"],
            "feature_set": best_fs,
            "transform": best_transform,
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_Bias": test_metrics["Bias"],
            "n_train": len(y_tv),
            "n_test": len(y_te),
        })

        for aid in AGENT_IDS:
            mask_tv = agent_ids_tv == aid
            mask_te = agent_ids_te == aid
            if mask_tv.sum() > 0 and mask_te.sum() > 0:
                a_train_m = compute_metrics(
                    y_tv[mask_tv], pred_tv[mask_tv])
                a_test_m = compute_metrics(
                    y_te[mask_te], pred_te[mask_te])
                print(f"  Agent {aid} — "
                      f"train R²={a_train_m['R2']:.3f}, "
                      f"test R²={a_test_m['R2']:.3f}")

                sweep_summaries.append({
                    "test_start": test_start,
                    "AgentID": aid,
                    "model": best["name"],
                    "feature_set": best_fs,
                    "transform": best_transform,
                    "train_RMSE": a_train_m["RMSE"],
                    "train_R2": a_train_m["R2"],
                    "test_RMSE": a_test_m["RMSE"],
                    "test_R2": a_test_m["R2"],
                    "test_MAE": a_test_m["MAE"],
                    "test_Bias": a_test_m["Bias"],
                    "n_train": int(mask_tv.sum()),
                    "n_test": int(mask_te.sum()),
                })

        # Save per-split outputs
        pred_df = pd.DataFrame({
            "Year": list(post_tv["Year"].values) +
                    list(post_te["Year"].values),
            "AgentID": list(post_tv["AgentID"].values) +
                       list(post_te["AgentID"].values),
            "cp_year": list(post_tv["cp_year"].values) +
                       list(post_te["cp_year"].values),
            "t_rel": list(post_tv["t_rel"].values) +
                     list(post_te["t_rel"].values),
            "Obs": np.concatenate([y_tv, y_te]),
            "Pred": np.concatenate([pred_tv, pred_te]),
            "Split": (["train"] * len(y_tv) +
                      ["test"] * len(y_te)),
            "model": best["name"],
            "feature_set": best_fs,
        })
        pred_path = SPLIT_SWEEP_DIR / f"predictions_test{test_start}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Saved {pred_path}")

        params_path = SPLIT_SWEEP_DIR / f"params_test{test_start}.json"
        with open(params_path, "w") as f:
            json.dump({
                "model": best["name"],
                "feature_set": best_fs,
                "transform": best_transform,
                "feature_cols": best_cols,
                "test_start_year": test_start,
                **best,
            }, f, indent=2, default=str)

        lb_path = SPLIT_SWEEP_DIR / f"leaderboard_test{test_start}.csv"
        pd.DataFrame(all_leaderboard).to_csv(lb_path, index=False)

        # Per-agent time-series plots for this split
        for aid in AGENT_IDS:
            out_pdf = (SPLIT_SWEEP_DIR /
                       f"split_agent{aid}_test{test_start}_point.pdf")
            plot_split_sweep_agent_point(
                aid, annual, pred_df, test_start, out_pdf)

    # Save sweep summary
    sweep_df = pd.DataFrame(sweep_summaries)
    sweep_path = SPLIT_SWEEP_DIR / "sweep_summary.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSaved {sweep_path}")

    # Comparison plot
    comp_pdf = SPLIT_SWEEP_DIR / "split_sweep_comparison.pdf"
    plot_split_sweep_comparison(sweep_df, comp_pdf)

    # Final report
    print("\n" + "=" * 60)
    print("SPLIT-POINT SWEEP SUMMARY")
    print("=" * 60)
    for ts in POST_SPLIT_CANDIDATES:
        pooled_row = sweep_df[
            (sweep_df["test_start"] == ts) &
            (sweep_df["AgentID"] == "pooled")]
        if len(pooled_row) > 0:
            r = pooled_row.iloc[0]
            print(f"  test_start={ts}: "
                  f"[{r['model']}/{r['feature_set']}/{r['transform']}]  "
                  f"train R²={r['train_R2']:.3f}  "
                  f"test RMSE={r['test_RMSE']:.2f}  "
                  f"test R²={r['test_R2']:.3f}")
            for aid in AGENT_IDS:
                a_row = sweep_df[
                    (sweep_df["test_start"] == ts) &
                    (sweep_df["AgentID"] == aid)]
                if len(a_row) > 0:
                    a = a_row.iloc[0]
                    print(f"    Agent {aid}: "
                          f"test R²={a['test_R2']:.3f}  "
                          f"test RMSE={a['test_RMSE']:.2f}")

    # Identify best split
    pooled_rows = sweep_df[sweep_df["AgentID"] == "pooled"]
    best_split = pooled_rows.loc[pooled_rows["test_R2"].idxmax()]
    print(f"\n  BEST SPLIT: test_start={int(best_split['test_start'])} "
          f"(pooled test R²={best_split['test_R2']:.3f})")
    print("=" * 60)


def plot_split_sweep_agent_point(aid, annual, pred_df, test_start, output_pdf):
    """Plot per-agent POST-CP point estimates for a given split point.

    Similar to plot_pooled_agent_point() but only shows POST regime
    with the parameterized split.
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    cp_year = AGENT_CONFIGS[aid]["cp_year"]

    agent_pred = pred_df[pred_df["AgentID"] == aid].sort_values("Year")
    train_pred = agent_pred[agent_pred["Split"] == "train"]
    test_pred = agent_pred[agent_pred["Split"] == "test"]

    # Full observed POST data
    obs = annual[(annual["AgentID"] == aid) &
                 (annual["Year"] >= cp_year)].sort_values("Year")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    # Observed
    ax.plot(obs["Year"].values, obs["Irrigation_Depth"].values,
            "ko-", label="Observed", linewidth=1.2, markersize=5, zorder=5)

    # Training period shading
    tv_years = train_pred["Year"].values
    if len(tv_years) > 0:
        ax.axvspan(tv_years.min() - 0.3, tv_years.max() + 0.3,
                   color="lightgray", alpha=0.3, zorder=0,
                   label="Training period")

    # Train predictions (solid)
    ax.plot(train_pred["Year"].values, train_pred["Pred"].values,
            "-", color="tab:red", linewidth=1.5, label="Train pred",
            zorder=3)

    # Test predictions (dashed)
    ax.plot(test_pred["Year"].values, test_pred["Pred"].values,
            "--", color="tab:red", linewidth=1.5, label="Test pred",
            zorder=4)

    # Train R²
    if len(train_pred) > 1:
        train_r2 = r2_score(train_pred["Obs"].values,
                            train_pred["Pred"].values)
        blend = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        ax.text(tv_years.min() + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                transform=blend, va="top", ha="left", fontsize=12,
                color="0.2")

    # Test R²
    if len(test_pred) > 1:
        test_r2 = r2_score(test_pred["Obs"].values,
                           test_pred["Pred"].values)
        ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=12, color="black")

    # CP line
    ax.axvline(cp_year, color="0.6", linestyle="--", linewidth=1, zorder=1)

    ax.set_title(f"Agent {aid} — POST-CP (test≥{test_start}, CP={cp_year})")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax.tick_params(axis="both", which="major", length=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual irrigation depth (mm)")
    ax.legend(loc="lower left", frameon=False, fontsize=11)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")


def plot_split_sweep_comparison(sweep_df, output_pdf):
    """Plot test R² and RMSE vs split point for each agent + pooled.

    1×2 panels: (a) test R² vs test_start, (b) test RMSE vs test_start.
    """
    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_RC_PARAMS)

    agent_colors = {12: "tab:blue", 18: "tab:orange", 20: "tab:green"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for col_idx, (metric, ylabel) in enumerate([
        ("test_R2", "Test R²"),
        ("test_RMSE", "Test RMSE (mm)"),
    ]):
        ax = axes[col_idx]

        # Per-agent lines
        for aid in AGENT_IDS:
            agent_data = sweep_df[sweep_df["AgentID"] == aid].sort_values(
                "test_start")
            ax.plot(agent_data["test_start"].values,
                    agent_data[metric].values,
                    "o-", color=agent_colors[aid], linewidth=1.5,
                    markersize=6, label=f"Agent {aid}", zorder=3)

        # Pooled line (black dashed)
        pooled = sweep_df[sweep_df["AgentID"] == "pooled"].sort_values(
            "test_start")
        ax.plot(pooled["test_start"].values,
                pooled[metric].values,
                "s--", color="black", linewidth=1.5, markersize=6,
                label="Pooled", zorder=4)

        # Highlight best split
        if metric == "test_R2":
            best_idx = pooled[metric].idxmax()
        else:
            best_idx = pooled[metric].idxmin()
        best_row = pooled.loc[best_idx]
        ax.axvline(best_row["test_start"], color="0.7", linestyle=":",
                   linewidth=1, zorder=1)
        ax.scatter([best_row["test_start"]], [best_row[metric]],
                   marker="*", s=200, color="red", zorder=5,
                   label=f"Best ({int(best_row['test_start'])})")

        # Panel label
        panel_label = f"({chr(ord('a') + col_idx)})"
        ax.text(0.01, 0.99, panel_label,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        ax.set_xlabel("Test start year")
        ax.set_ylabel(ylabel)
        ax.set_xticks(POST_SPLIT_CANDIDATES)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Cluster 1 — POST-CP Split-Point Sweep", fontsize=16,
                 y=1.09)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pooled-by-regime XGBoost for Cluster 1 "
                    "(agents 12, 18, 20)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--two-regime-point", action="store_true",
        help="Point estimates: pooled-by-regime with model selection "
             "(Ridge/ElasticNet/XGBoost)")
    mode_group.add_argument(
        "--two-regime-ensemble", action="store_true",
        help="Ensemble: load saved params, bootstrap PI per agent per regime")
    mode_group.add_argument(
        "--cp-sensitivity", action="store_true",
        help="CP sensitivity: pooled agents with CP 2004/2005/2006 cutoffs "
             "(rolling-origin CV, macro-averaged RMSE, Year-ablation)")
    mode_group.add_argument(
        "--post-split-sweep", action="store_true",
        help="Sweep POST-CP train/test split points (test_start 2013-2017)")
    args = parser.parse_args()

    print("Loading and aggregating data...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    print(f"  Annual dataset: {len(annual)} rows "
          f"({annual['Year'].nunique()} years × {len(AGENT_IDS)} agents)")

    if args.two_regime_point:
        run_point(annual)
    elif args.two_regime_ensemble:
        run_ensemble(annual)
    elif args.post_split_sweep:
        run_post_split_sweep(annual)
    else:
        run_cp_sensitivity(annual)


if __name__ == "__main__":
    main()
