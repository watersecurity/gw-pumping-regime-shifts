#!/usr/bin/env python3
"""
Pooled XGBoost ABM for Cluster 2 — Changepoint Benefit Analysis.

Compares two models on a common evaluation window (2006–2020):
  M1 (Stationary):         train 1993–2004
  M2 (Changepoint-Aware):  train 1993–2005 (includes CP year)

Uncertainty quantified via 200-member block-bootstrap ensembles with
hyperparameter jitter. Includes changepoint-year sensitivity analysis.
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_kge(y_true, y_pred):
    """Kling-Gupta Efficiency (KGE).

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where r = Pearson correlation, alpha = std(pred)/std(obs),
    beta = mean(pred)/mean(obs).

    Returns NaN if std(obs)==0 or mean(obs)==0 (degenerate cases).
    """
    obs_mean = np.mean(y_true)
    obs_std = np.std(y_true)
    if obs_std == 0 or obs_mean == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / obs_std
    beta = np.mean(y_pred) / obs_mean
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


# ── Constants ────────────────────────────────────────────────────────────────
AGENT_IDS = [2, 3, 24, 28, 29]
CP_YEAR = 2005
N_BOOT = 200
N_SENS_BOOT = 100
MASTER_SEED = 42
BLOCK_SIZE = 3

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
MODEL_M1_PATH = MODEL_DIR / "cluster2_optuna_M1.json"
MODEL_M2_PATH = MODEL_DIR / "cluster2_optuna_M2.json"
PARAMS_M1_PATH = MODEL_DIR / "cluster2_optuna_params_M1.json"
PARAMS_M2_PATH = MODEL_DIR / "cluster2_optuna_params_M2.json"

XGB_DEFAULTS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "early_stopping_rounds": 50,
    "verbosity": 0,
}

JITTER_RANGES = {
    "max_depth": [2, 3, 4],
    "learning_rate": (0.03, 0.1),
    "subsample": (0.7, 1.0),
    "colsample_bytree": (0.7, 1.0),
    "reg_lambda": (1.0, 10.0),
    "min_child_weight": [1, 3, 5],
}

# Relative jitter widths: centered on tuned (Optuna) hyperparameters
JITTER_WIDTHS = {
    "max_depth": {"additive": 1, "clamp": (2, 8)},           # base ± 1
    "learning_rate": {"mult_range": (0.5, 2.0)},              # multiplicative, log-safe
    "subsample": {"additive": 0.15, "clamp": (0.5, 1.0)},
    "colsample_bytree": {"additive": 0.15, "clamp": (0.5, 1.0)},
    "reg_lambda": {"mult_range": (0.3, 3.0)},                 # multiplicative
    "reg_alpha": {"mult_range": (0.3, 3.0)},                  # multiplicative
    "min_child_weight": {"additive": 2, "clamp": (1, 10)},    # base ± 2
}

# ── Two-Regime Experiment ────────────────────────────────────────────────────
# Split the full record at the 2005 changepoint into two independent regimes.
# Each regime gets its own Optuna tuning, its own train/test split, and a
# bootstrap ensemble with FIXED hyperparameters (no jitter).  Uncertainty
# comes solely from data resampling via constrained moving-block bootstrap
# with an anchor tail (the last 3 trainval years are held fixed to preserve
# the boundary signal).  PI is the [5th, 95th] percentile across members.
REGIME_A = {
    "name": "pre",
    "full_years": list(range(1993, 2005)),          # 1993–2004
    "trainval_years": list(range(1993, 2002)),       # 1993–2001 (9 years)
    "test_years": [2002, 2003, 2004],
    "anchor_years": [1999, 2000, 2001],
}
REGIME_B = {
    "name": "post",
    "full_years": list(range(2005, 2021)),           # 2005–2020
    "trainval_years": list(range(2005, 2017)),        # 2005–2016
    "test_years": [2017, 2018, 2019, 2020],
    "anchor_years": [2014, 2015, 2016],
}
N_TWO_REGIME_BOOT = 100
TWO_REGIME_DIR = RESULTS_DIR / "two_regime"


# ── Data Pipeline ────────────────────────────────────────────────────────────
def load_agent_data(agent_ids):
    """Load and concatenate CSV files for the given agent IDs."""
    frames = []
    for aid in agent_ids:
        fp = DATA_DIR / f"agentdata_{aid}.csv"
        df = pd.read_csv(fp)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["AgentID", "Year", "Month"])


def aggregate_to_annual(df):
    """Aggregate monthly data (May–Oct) to annual per (AgentID, Year)."""
    agg_dict = {
        "Irrigation_Depth": "sum",
        "Precipitation": "sum",
        "Temperature": "mean",
        "Corn": "mean",
        "Wheat": "mean",
        "Soybeans": "mean",
        "Sorghum": "mean",
        "Diesel": "mean",
    }
    annual = df.groupby(["AgentID", "Year"]).agg(agg_dict).reset_index()
    return annual.sort_values(["AgentID", "Year"]).reset_index(drop=True)


def prepare_features(df, add_agent_dummies=True, drop_year=False):
    """Build feature matrix X and target vector y.

    Args:
        df: DataFrame with feature columns and Irrigation_Depth target.
        add_agent_dummies: If True (default), add one-hot encoded AgentID
            columns (10 features). If False, skip agent dummies (8 features),
            suitable for per-agent models.
        drop_year: If True, exclude Year from the feature matrix.
    """
    feature_cols = ["Year", "Precipitation", "Temperature",
                    "Corn", "Wheat", "Soybeans", "Sorghum", "Diesel"]
    X = df[feature_cols].copy()
    if drop_year:
        X = X.drop(columns=["Year"])
    if add_agent_dummies:
        for aid in AGENT_IDS:
            X[f"Agent_{aid}"] = (df["AgentID"] == aid).astype(int)
    y = df["Irrigation_Depth"].values
    return X, y


def fit_linear_time_trend(X_with_year, y):
    """Fit Ridge on Year column. Returns (ridge_model, residuals)."""
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_with_year[["Year"]].values, y)
    return ridge, y - ridge.predict(X_with_year[["Year"]].values)


def predict_with_linear_trend(ridge, xgb_model, X_with_year, X_no_year):
    """Combine linear trend + XGBoost residual prediction."""
    return ridge.predict(X_with_year[["Year"]].values) + xgb_model.predict(X_no_year)


def compute_sample_weights(df, downweight_zero_irrig=None):
    """Compute sample weights, optionally downweighting zero-irrigation rows."""
    if downweight_zero_irrig is None:
        return None
    weights = np.ones(len(df))
    weights[df["Irrigation_Depth"].values == 0] = downweight_zero_irrig
    return weights


def split_data(df, train_end_year, eval_start_year=None):
    """Split into train and eval DataFrames based on year boundaries."""
    if eval_start_year is None:
        eval_start_year = train_end_year + 1
    df_train = df[df["Year"] <= train_end_year].copy()
    df_eval = df[df["Year"] >= eval_start_year].copy()
    return df_train, df_eval


# ── XGBoost Training ────────────────────────────────────────────────────────
def train_with_early_stopping(X_train, y_train, params, val_years=3,
                              refit=False, sample_weight=None, years=None):
    """Train XGBRegressor with time-aware early stopping.

    Uses the last `val_years` years of the training window as validation set.

    Args:
        refit: If True, after finding best_iteration via early stopping,
            retrain on ALL X_train/y_train with n_estimators=best_iteration+1.
        sample_weight: Optional array of sample weights for .fit().
        years: Optional year array for splitting. If None, uses X_train["Year"].
    """
    if years is None:
        years = X_train["Year"].values
    unique_years = np.sort(np.unique(years))
    if len(unique_years) <= val_years:
        # Not enough years for a proper split; train without early stopping
        fallback = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        fallback["n_estimators"] = min(fallback.get("n_estimators", 500), 500)
        model = XGBRegressor(**fallback)
        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **sw_kw)
        return model

    val_cutoff = unique_years[-(val_years)]
    train_mask = years < val_cutoff
    val_mask = years >= val_cutoff

    sw_train = sample_weight[train_mask] if sample_weight is not None else None
    sw_kw = {"sample_weight": sw_train} if sw_train is not None else {}

    model = XGBRegressor(**params)
    model.fit(
        X_train[train_mask], y_train[train_mask],
        eval_set=[(X_train[val_mask], y_train[val_mask])],
        verbose=False,
        **sw_kw,
    )

    if refit:
        best_iteration = getattr(model, "best_iteration", None)
        n_est = (best_iteration + 1) if best_iteration is not None else \
            min(params.get("n_estimators", 500), 500)
        refit_params = {k: v for k, v in params.items()
                        if k != "early_stopping_rounds"}
        refit_params["n_estimators"] = n_est
        model = XGBRegressor(**refit_params)
        sw_kw_full = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **sw_kw_full)

    return model


N_OPTUNA_TRIALS = 100


def optuna_tune(X_train, y_train, n_trials=N_OPTUNA_TRIALS, val_years=3,
                seed=42, search_space_overrides=None, rolling_val_folds=None,
                sample_weight=None, years=None, objective_metric="rmse"):
    """Tune XGBoost hyperparameters with Optuna TPE, then retrain with best params.

    Uses time-aware validation: a single tail split when the validation set is
    large enough (≥ 8 rows), or expanding-window CV when it's smaller (to
    average out the noise from tiny validation sets).

    Args:
        search_space_overrides: Dict mapping param name to (lo, hi) bounds.
            When provided, these bounds replace the adaptive defaults.
        rolling_val_folds: List of {"val_years": [...]} dicts for explicit
            rolling-window CV folds.
        sample_weight: Optional sample weights for .fit() calls.
        years: Optional year array for splitting. If None, uses X_train["Year"].
        objective_metric: Metric to optimize. "rmse" (default, minimized) or
            "kge" (Kling-Gupta Efficiency, maximized).

    Returns (model, best_params) tuple.
    """
    if years is None:
        years_arr = X_train["Year"].values
    else:
        years_arr = np.asarray(years)
    unique_years = np.sort(np.unique(years_arr))

    # Rolling validation folds from config take priority
    if rolling_val_folds is not None:
        cv_folds = []
        for fold_spec in rolling_val_folds:
            val_yrs = fold_spec["val_years"]
            tmask = years_arr < min(val_yrs)
            vmask = np.isin(years_arr, val_yrs)
            cv_folds.append((tmask, vmask))
        use_cv = True
        n_fit = len(X_train)
        print(f"  Using rolling-window CV ({len(cv_folds)} folds) for Optuna")
    else:
        # Adapt val_years to dataset size: never consume more than ~25% of
        # available years for validation (prevents over-thinning on small regimes)
        val_years = min(val_years, max(1, len(unique_years) // 4))

        # Fallback: not enough years to split
        if len(unique_years) <= val_years:
            model = train_with_early_stopping(
                X_train, y_train, XGB_DEFAULTS, val_years=val_years,
                sample_weight=sample_weight, years=years_arr)
            return model, dict(XGB_DEFAULTS)

        val_cutoff = unique_years[-val_years]
        train_mask = years_arr < val_cutoff
        val_mask = years_arr >= val_cutoff

        # When the validation split has fewer than 3 years, the single-split RMSE
        # is too noisy for reliable hyperparameter selection.  Switch to
        # expanding-window CV so Optuna averages over multiple folds.
        use_cv = (val_years < 3)

        if use_cv:
            min_cv_years = max(3, len(unique_years) // 2)
            cv_folds = []
            for vi in range(min_cv_years, len(unique_years)):
                tmask = np.isin(years_arr, unique_years[:vi])
                vmask = years_arr == unique_years[vi]
                cv_folds.append((tmask, vmask))
            # Use full dataset size for search-space bounds (model sees all data
            # across folds, so bounds should reflect the full regime)
            n_fit = len(X_train)
            print(f"  Using expanding-window CV ({len(cv_folds)} folds) for Optuna")
        else:
            X_fit, y_fit = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_train[val_mask], y_train[val_mask]
            n_fit = len(X_fit)

    # Adapt search space to sample size so that small datasets don't get
    # over-regularised (e.g. min_child_weight=8 on 12 rows → constant pred)
    mcw_upper = min(10, max(1, n_fit // 4))
    md_upper = min(6, max(2, int(np.log2(n_fit))))

    # Adaptive early-stopping patience: 50 rounds is too generous when the
    # validation set has only a handful of points (noisy loss never triggers).
    es_rounds = min(50, max(5, n_fit // 3))
    # Cap boosting rounds to avoid memorising tiny datasets
    n_est_upper = min(2000, max(100, n_fit * 10))

    # Regularisation floor: prevent near-zero lambda on tiny datasets
    reg_lambda_lo = max(0.1, 50.0 / n_fit)   # 18 rows → 2.78, 100 rows → 0.5

    sso = search_space_overrides or {}

    sampler = optuna.samplers.TPESampler(seed=seed)

    def objective(trial):
        md_lo, md_hi = sso.get("max_depth", (2, md_upper))
        lr_lo, lr_hi = sso.get("learning_rate",
                                (0.01, min(0.3, max(0.05, 3.0 / n_fit))))
        ss_lo, ss_hi = sso.get("subsample", (0.5, 1.0))
        cs_lo, cs_hi = sso.get("colsample_bytree", (0.5, 1.0))
        rl_lo, rl_hi = sso.get("reg_lambda", (reg_lambda_lo, 20.0))
        ra_lo, ra_hi = sso.get("reg_alpha", (0.0, 10.0))
        mcw_lo, mcw_hi = sso.get("min_child_weight", (1, mcw_upper))

        params = {
            "max_depth": trial.suggest_int("max_depth", int(md_lo), int(md_hi)),
            "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, n_est_upper),
            "subsample": trial.suggest_float("subsample", ss_lo, ss_hi),
            "colsample_bytree": trial.suggest_float("colsample_bytree", cs_lo, cs_hi),
            "reg_lambda": trial.suggest_float("reg_lambda", rl_lo, rl_hi, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", ra_lo, ra_hi),
            "min_child_weight": trial.suggest_int("min_child_weight", int(mcw_lo), int(mcw_hi)),
            "objective": "reg:squarederror",
            "early_stopping_rounds": es_rounds,
            "verbosity": 0,
        }
        # gamma: only searched when present in overrides
        if "gamma" in sso:
            g_lo, g_hi = sso["gamma"]
            params["gamma"] = trial.suggest_float("gamma", g_lo, g_hi)

        if use_cv:
            scores = []
            for tmask, vmask in cv_folds:
                sw_fold = sample_weight[tmask] if sample_weight is not None else None
                sw_kw = {"sample_weight": sw_fold} if sw_fold is not None else {}
                model = XGBRegressor(**params)
                model.fit(X_train[tmask], y_train[tmask],
                          eval_set=[(X_train[vmask], y_train[vmask])],
                          verbose=False, **sw_kw)
                pred = model.predict(X_train[vmask])
                if objective_metric == "kge":
                    scores.append(compute_kge(y_train[vmask], pred))
                else:
                    scores.append(np.sqrt(mean_squared_error(y_train[vmask], pred)))
            if objective_metric == "kge":
                valid = [s for s in scores if not np.isnan(s)]
                return np.mean(valid) if valid else -999.0
            return np.mean(scores)
        else:
            sw_fit = sample_weight[train_mask] if sample_weight is not None else None
            sw_kw = {"sample_weight": sw_fit} if sw_fit is not None else {}
            model = XGBRegressor(**params)
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                      verbose=False, **sw_kw)
            pred_val = model.predict(X_val)
            if objective_metric == "kge":
                kge_val = compute_kge(y_val, pred_val)
                return kge_val if not np.isnan(kge_val) else -999.0
            return np.sqrt(mean_squared_error(y_val, pred_val))

    direction = "maximize" if objective_metric == "kge" else "minimize"
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    metric_label = "KGE" if objective_metric == "kge" else "RMSE"
    fmt = ".4f" if objective_metric == "kge" else ".2f"
    print(f"  Optuna best val {metric_label}: {study.best_value:{fmt}}")
    print(f"  Best params: {best_params}")

    if use_cv:
        # In CV mode, retrain on all data; n_estimators will be capped
        # by _cv_best_n_estimators in the caller
        retrain_params = {k: v for k, v in best_params.items()
                          if k != "early_stopping_rounds"}
        retrain_params["objective"] = "reg:squarederror"
        retrain_params["verbosity"] = 0
        final_model = XGBRegressor(**retrain_params)
        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        final_model.fit(X_train, y_train, **sw_kw)
    else:
        # Retrain with best params + early stopping to find optimal n_estimators
        retrain_params = {
            **best_params,
            "objective": "reg:squarederror",
            "early_stopping_rounds": es_rounds,
            "verbosity": 0,
        }
        sw_fit = sample_weight[train_mask] if sample_weight is not None else None
        sw_kw = {"sample_weight": sw_fit} if sw_fit is not None else {}
        final_model = XGBRegressor(**retrain_params)
        final_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                        verbose=False, **sw_kw)

        # Replace search-space n_estimators with actual early-stopped round count
        # so downstream callers get the regularised tree count
        best_iteration = getattr(final_model, "best_iteration", None)
        if best_iteration is not None:
            best_params["n_estimators"] = best_iteration + 1  # 0-indexed → count
            print(f"  Early-stopped at {best_params['n_estimators']} rounds "
                  f"(search space had {retrain_params['n_estimators']}, "
                  f"patience={es_rounds})")

    return final_model, best_params


def _cv_best_n_estimators(X, y, best_params, min_train_years=5,
                          sample_weight=None, years=None):
    """Determine regularised n_estimators via expanding-window time-series CV.

    Trains one fold per validation year using the Optuna-selected
    hyperparameters + early stopping.  Returns the median best_iteration
    across folds (or the minimum n_estimators if early stopping never fires).

    Args:
        sample_weight: Optional sample weights for .fit() calls.
        years: Optional year array. If None, uses X["Year"].
    """
    if years is None:
        years_arr = X["Year"].values
    else:
        years_arr = np.asarray(years)
    unique_years = np.sort(np.unique(years_arr))
    n_years = len(unique_years)

    if n_years <= min_train_years:
        # Not enough years for even one fold; return params as-is
        return best_params.get("n_estimators", 100)

    # Build expanding-window folds
    best_iters = []
    for val_idx in range(min_train_years, n_years):
        train_years = unique_years[:val_idx]
        val_year = unique_years[val_idx]
        train_mask = np.isin(years_arr, train_years)
        val_mask = years_arr == val_year

        fold_params = {
            k: v for k, v in best_params.items()
            if k != "early_stopping_rounds"
        }
        fold_params["objective"] = "reg:squarederror"
        fold_params["verbosity"] = 0
        # Patience: aggressive for small folds
        n_train = int(train_mask.sum())
        fold_params["early_stopping_rounds"] = min(50, max(5, n_train // 3))

        sw_fold = sample_weight[train_mask] if sample_weight is not None else None
        sw_kw = {"sample_weight": sw_fold} if sw_fold is not None else {}

        model = XGBRegressor(**fold_params)
        model.fit(
            X[train_mask], y[train_mask],
            eval_set=[(X[val_mask], y[val_mask])],
            verbose=False,
            **sw_kw,
        )

        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            best_iters.append(bi + 1)  # 0-indexed → count

    if best_iters:
        cv_n_est = int(np.median(best_iters))
        print(f"  Expanding-window CV n_estimators: "
              f"per-fold={best_iters}, median={cv_n_est}")
    else:
        # Early stopping never fired; fall back to search-space value
        cv_n_est = best_params.get("n_estimators", 100)
        print(f"  Expanding-window CV: early stopping never fired, "
              f"using n_estimators={cv_n_est}")

    return cv_n_est


# ── Model persistence ────────────────────────────────────────────────────────
def save_tuned_model(model, params, model_path, params_path):
    """Save XGBoost model (native JSON) and its tuned hyperparams dict."""
    model_path = Path(model_path)
    params_path = Path(params_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved model  → {model_path}")
    print(f"  Saved params → {params_path}")


def load_tuned_params(params_path):
    """Load tuned hyperparams dict from JSON."""
    with open(params_path) as f:
        return json.load(f)


def load_tuned_model(model_path):
    """Load a saved XGBoost model from native JSON format."""
    model = XGBRegressor()
    model.load_model(str(model_path))
    return model


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R2, Bias."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true),
    }


def compute_full_metrics(y_true, y_pred, agent_ids_arr):
    """Compute overall and per-agent metrics."""
    overall = compute_metrics(y_true, y_pred)
    result = {"overall_" + k: v for k, v in overall.items()}
    for aid in AGENT_IDS:
        mask = agent_ids_arr == aid
        m = compute_metrics(y_true[mask], y_pred[mask])
        for k, v in m.items():
            result[f"{k}_agent{aid}"] = v
    return result


# ── Bootstrap Utilities ─────────────────────────────────────────────────────
def block_bootstrap_years(years, block_size, rng):
    """Block-bootstrap resample year indices (preserving contiguous blocks)."""
    unique_years = np.sort(np.unique(years))
    n = len(unique_years)
    n_blocks = int(np.ceil(n / block_size))
    sampled = []
    for _ in range(n_blocks):
        start = rng.integers(0, n - block_size + 1)
        sampled.extend(unique_years[start:start + block_size])
    return sampled[:n]


def constrained_block_bootstrap_years(pre_cp_years, anchor_tail_years,
                                       block_size, rng):
    """Moving-block bootstrap on pre-CP years with a deterministic anchor tail.

    Args:
        pre_cp_years: array-like of years before the changepoint (e.g. 1993-2004).
        anchor_tail_years: list/array of years to append deterministically
                           (e.g. [2005,2006,2007] for CP=2007).
        block_size: size of contiguous blocks for the moving-block bootstrap.
        rng: numpy random Generator.

    Returns:
        List of ints, length = len(pre_cp_years) + len(anchor_tail_years).
        The last len(anchor_tail_years) entries are exactly anchor_tail_years.
    """
    pre = np.sort(np.asarray(pre_cp_years))
    anchor = list(anchor_tail_years)
    n = len(pre)

    if n < block_size:
        raise ValueError(
            f"pre_cp_years length ({n}) must be >= block_size ({block_size})")

    overlap = set(pre) & set(anchor)
    if overlap:
        raise ValueError(
            f"anchor_tail_years must not overlap pre_cp_years; overlap: {overlap}")

    # Moving-block bootstrap on pre-CP portion only
    n_blocks = int(np.ceil(n / block_size))
    sampled = []
    for _ in range(n_blocks):
        start = rng.integers(0, n - block_size + 1)
        sampled.extend(pre[start:start + block_size].tolist())
    bootstrapped_pre = sampled[:n]

    result = bootstrapped_pre + anchor
    assert result[-len(anchor):] == anchor, (
        f"Tail constraint violated: expected {anchor}, got {result[-len(anchor):]}")
    return result


def _anchor_tail_for_cp(cp_candidate, base_cp=2005):
    """Return the anchor tail for a given CP candidate.

    CP=2005 -> [2005], CP=2007 -> [2005, 2006, 2007], CP=2009 -> [2005, ..., 2009].
    """
    if cp_candidate < base_cp:
        raise ValueError(
            f"cp_candidate ({cp_candidate}) must be >= base_cp ({base_cp})")
    return list(range(base_cp, cp_candidate + 1))


def make_constrained_moving_block_year_sequence(trainval_years, anchor_years,
                                                 block_size, rng,
                                                 target_prefix_len=None):
    """Build a bootstrap year sequence for a regime with a fixed anchor tail.

    Splits trainval_years into a resample-able prefix pool (trainval minus
    anchor) and a deterministic anchor tail.  Delegates the actual bootstrap
    to ``constrained_block_bootstrap_years()``.

    Args:
        trainval_years: all years in the training/validation pool.
        anchor_years: years pinned at the tail (must be a subset/tail of
                      trainval_years).
        block_size: moving-block size.
        rng: numpy random Generator.
        target_prefix_len: number of prefix years to draw.  Defaults to the
                           full prefix pool size so that
                           len(output) == len(trainval_years).

    Returns:
        List[int] of length ``target_prefix_len + len(anchor_years)``.
    """
    trainval = sorted(trainval_years)
    anchor = list(anchor_years)

    prefix_pool = sorted(set(trainval) - set(anchor))
    if target_prefix_len is None:
        target_prefix_len = len(prefix_pool)

    # Validate anchor is tail of trainval
    if trainval[-len(anchor):] != anchor:
        raise ValueError(
            f"anchor_years {anchor} must be the tail of "
            f"trainval_years {trainval}")

    # Delegate to existing constrained bootstrap (prefix_pool acts as pre-CP
    # years, anchor acts as the deterministic tail)
    result = constrained_block_bootstrap_years(
        np.array(prefix_pool), anchor, block_size, rng)

    # Trim prefix if target_prefix_len < len(prefix_pool)
    if target_prefix_len < len(prefix_pool):
        result = result[:target_prefix_len] + anchor

    assert len(result) == target_prefix_len + len(anchor), (
        f"Expected {target_prefix_len + len(anchor)} years, got {len(result)}")
    assert result[-len(anchor):] == anchor, (
        f"Tail constraint violated: expected {anchor}, got {result[-len(anchor):]}")
    return result


def bootstrap_panel_by_year_sequence(df, year_sequence):
    """Build a panel DataFrame by selecting rows for each year in sequence.

    Handles repeated years (from bootstrap resampling) by including all rows
    for each occurrence.

    Args:
        df: DataFrame with a 'Year' column.
        year_sequence: list of years (may contain duplicates).

    Returns:
        Concatenated DataFrame with rows for each year in sequence.

    Raises:
        ValueError: if any year in the sequence has no rows in df.
    """
    frames = []
    for yr in year_sequence:
        sub = df[df["Year"] == yr]
        if len(sub) == 0:
            raise ValueError(f"Year {yr} not found in DataFrame")
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def sample_hyperparams(rng, base_params=None):
    """Sample jittered hyperparameters.

    If base_params is None, uses legacy path (XGB_DEFAULTS + JITTER_RANGES).
    If base_params is provided, applies relative jitter via JITTER_WIDTHS
    centered on the tuned values.
    """
    if base_params is None:
        # Legacy path: absolute jitter around defaults
        params = dict(XGB_DEFAULTS)
        params["max_depth"] = int(rng.choice(JITTER_RANGES["max_depth"]))
        params["learning_rate"] = float(rng.uniform(*JITTER_RANGES["learning_rate"]))
        params["subsample"] = float(rng.uniform(*JITTER_RANGES["subsample"]))
        params["colsample_bytree"] = float(rng.uniform(*JITTER_RANGES["colsample_bytree"]))
        params["reg_lambda"] = float(rng.uniform(*JITTER_RANGES["reg_lambda"]))
        params["min_child_weight"] = int(rng.choice(JITTER_RANGES["min_child_weight"]))
        params["random_state"] = int(rng.integers(0, 2**31))
        return params

    # Tuned-param path: relative jitter centered on Optuna values
    params = {
        "n_estimators": base_params.get("n_estimators", 2000),
        "objective": "reg:squarederror",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }
    for key, spec in JITTER_WIDTHS.items():
        base_val = base_params.get(key)
        if base_val is None:
            continue
        if "mult_range" in spec:
            lo, hi = spec["mult_range"]
            params[key] = float(base_val * rng.uniform(lo, hi))
        elif "additive" in spec:
            delta = spec["additive"]
            lo_clamp, hi_clamp = spec["clamp"]
            jittered = base_val + rng.uniform(-delta, delta)
            if isinstance(base_val, int) or key in ("max_depth", "min_child_weight"):
                params[key] = int(np.clip(round(jittered), lo_clamp, hi_clamp))
            else:
                params[key] = float(np.clip(jittered, lo_clamp, hi_clamp))
    params["random_state"] = int(rng.integers(0, 2**31))
    return params


def run_ensemble(df_train, df_eval, n_boot, rng, force_include_year=None,
                 base_params=None):
    """Run n_boot ensemble members with block bootstrap + hyperparameter jitter.

    Args:
        base_params: If provided, jitter is centered on these tuned values
                     via JITTER_WIDTHS instead of XGB_DEFAULTS + JITTER_RANGES.

    Returns:
        preds_matrix: (n_boot, n_eval) array of predictions
        metrics_list: list of dicts with per-member metrics
    """
    X_eval, y_eval = prepare_features(df_eval)
    eval_agents = df_eval["AgentID"].values
    n_eval = len(y_eval)
    preds_matrix = np.zeros((n_boot, n_eval))
    metrics_list = []

    train_years = np.sort(df_train["Year"].unique())

    for i in range(n_boot):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        # Block bootstrap: resample years
        boot_years = block_bootstrap_years(train_years, BLOCK_SIZE, member_rng)

        # Force include specific year if requested (for M2: include CP year)
        if force_include_year is not None and force_include_year not in boot_years:
            replace_idx = member_rng.integers(0, len(boot_years))
            # Replace the chosen block starting position with one that includes the forced year
            boot_years[replace_idx] = force_include_year

        # Build bootstrapped training set (both agents for each sampled year)
        boot_frames = []
        for yr in boot_years:
            boot_frames.append(df_train[df_train["Year"] == yr])
        df_boot = pd.concat(boot_frames, ignore_index=True)

        X_boot, y_boot = prepare_features(df_boot)
        params = sample_hyperparams(member_rng, base_params=base_params)

        model = train_with_early_stopping(X_boot, y_boot, params, val_years=3)
        preds = model.predict(X_eval)
        preds_matrix[i] = preds

        m = compute_full_metrics(y_eval, preds, eval_agents)
        m["ensemble_id"] = i
        metrics_list.append(m)

    return preds_matrix, metrics_list


def run_cp(cp, annual, df_common_eval, rng, n_boot=N_SENS_BOOT,
           base_cp=CP_YEAR, block_size=BLOCK_SIZE, save_artifacts=True):
    """Run full sensitivity analysis for a single CP candidate.

    Uses constrained_block_bootstrap_years() to guarantee the anchor tail.

    Returns:
        dict with keys: best_params, preds_matrix, metrics_list, pi_df, summary
    """
    # 1. Split: train on all years <= cp
    df_train = annual[annual["Year"] <= cp].copy()
    X_train, y_train = prepare_features(df_train)

    # 2. Optuna tune
    print(f"    Tuning M2 (CP={cp})...")
    _, best_params = optuna_tune(X_train, y_train, seed=MASTER_SEED + cp + 1)

    # 3. Point-estimate prediction on common eval
    X_eval, y_eval = prepare_features(df_common_eval)
    eval_agents = df_common_eval["AgentID"].values
    eval_years = df_common_eval["Year"].values

    # Retrain point-estimate model on full training set
    point_params = {
        **best_params,
        "objective": "reg:squarederror",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }
    point_model = train_with_early_stopping(X_train, y_train, point_params,
                                            val_years=3)
    point_preds = point_model.predict(X_eval)

    # Point-estimate metrics
    point_rmse = np.sqrt(mean_squared_error(y_eval, point_preds))
    point_mae = mean_absolute_error(y_eval, point_preds)
    point_r2 = r2_score(y_eval, point_preds)
    print(f"    M2 RMSE={point_rmse:.2f}, MAE={point_mae:.2f}, "
          f"R²={point_r2:.3f}")

    # 4. Ensemble with constrained block bootstrap
    pre_cp_years = np.arange(1993, base_cp)
    anchor_tail = _anchor_tail_for_cp(cp, base_cp=base_cp)

    n_eval = len(y_eval)
    preds_matrix = np.zeros((n_boot, n_eval))
    metrics_list = []

    print(f"    Running ensemble (N={n_boot})...")
    for i in range(n_boot):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        # Constrained block bootstrap
        boot_years = constrained_block_bootstrap_years(
            pre_cp_years, anchor_tail, block_size, member_rng)

        # Build bootstrapped training set
        boot_frames = []
        for yr in boot_years:
            boot_frames.append(df_train[df_train["Year"] == yr])
        df_boot = pd.concat(boot_frames, ignore_index=True)

        X_boot, y_boot = prepare_features(df_boot)
        params = sample_hyperparams(member_rng, base_params=best_params)

        model = train_with_early_stopping(X_boot, y_boot, params, val_years=3)
        preds = model.predict(X_eval)
        preds_matrix[i] = preds

        m = compute_full_metrics(y_eval, preds, eval_agents)
        m["ensemble_id"] = i
        metrics_list.append(m)

    # 5. Build PI DataFrame
    pi_rows = []
    for j in range(n_eval):
        pi_rows.append({
            "Year": eval_years[j],
            "AgentID": eval_agents[j],
            "Obs": y_eval[j],
            "M2_med": np.median(preds_matrix[:, j]),
            "M2_p05": np.percentile(preds_matrix[:, j], 5),
            "M2_p95": np.percentile(preds_matrix[:, j], 95),
        })
    pi_df = pd.DataFrame(pi_rows)

    # Ensemble metric arrays
    rmse_arr = np.array([m["overall_RMSE"] for m in metrics_list])
    mae_arr = np.array([m["overall_MAE"] for m in metrics_list])
    r2_arr = np.array([m["overall_R2"] for m in metrics_list])

    # PI widths
    pi_widths = (np.percentile(preds_matrix, 95, axis=0)
                 - np.percentile(preds_matrix, 5, axis=0))
    print(f"    PI width — M2: mean={np.mean(pi_widths):.2f}, "
          f"median={np.median(pi_widths):.2f}")

    # 6. Summary dict (backward-compatible with existing CSV columns)
    summary = {
        "CP_Year": cp,
        "M2_RMSE": point_rmse,
        "M2_MAE": point_mae,
        "M2_R2": point_r2,
        "M2_RMSE_ens_median": np.median(rmse_arr),
        "M2_RMSE_ens_IQR_lo": np.percentile(rmse_arr, 25),
        "M2_RMSE_ens_IQR_hi": np.percentile(rmse_arr, 75),
        "M2_MAE_ens_median": np.median(mae_arr),
        "M2_MAE_ens_IQR_lo": np.percentile(mae_arr, 25),
        "M2_MAE_ens_IQR_hi": np.percentile(mae_arr, 75),
        "M2_R2_ens_median": np.median(r2_arr),
        "M2_R2_ens_IQR_lo": np.percentile(r2_arr, 25),
        "M2_R2_ens_IQR_hi": np.percentile(r2_arr, 75),
        "M2_PI_width_mean": np.mean(pi_widths),
        "M2_PI_width_median": np.median(pi_widths),
        "M2_learning_rate": best_params.get("learning_rate"),
        "M2_max_depth": best_params.get("max_depth"),
    }

    # 7. Save artifacts
    if save_artifacts:
        cp_dir = RESULTS_DIR / f"sensitivity_cp{cp}"
        cp_dir.mkdir(parents=True, exist_ok=True)

        with open(cp_dir / f"best_params_cp{cp}.json", "w") as f:
            json.dump(best_params, f, indent=2)

        pd.DataFrame(metrics_list).to_csv(
            cp_dir / f"ensemble_metrics_cp{cp}.csv", index=False)

        pd.DataFrame([summary]).to_csv(
            cp_dir / f"summary_cp{cp}.csv", index=False)

        print(f"    Saved artifacts → {cp_dir}/")

    return {
        "best_params": best_params,
        "preds_matrix": preds_matrix,
        "metrics_list": metrics_list,
        "pi_df": pi_df,
        "summary": summary,
    }


# ── Two-Regime Ensemble ──────────────────────────────────────────────────────
def run_regime_ensemble(regime, annual, best_params, n_members, rng):
    """Run bootstrap ensemble for a single regime with FIXED hyperparameters.

    Unlike run_ensemble(), no hyperparameter jitter is applied.  Uncertainty
    comes solely from data resampling (constrained moving-block bootstrap)
    and XGBoost internal stochasticity via ``random_state``.

    Supports regime improvement flags: use_linear_time, downweight_zero_irrig.

    Args:
        regime: dict with keys name, trainval_years, test_years, anchor_years.
        annual: full annual DataFrame.
        best_params: Optuna-tuned hyperparameters (used as-is for every member).
        n_members: number of ensemble members.
        rng: numpy random Generator.

    Returns:
        preds_matrix: (n_members, n_test) array of predictions.
        metrics_list: list of per-member metric dicts.
    """
    use_linear = regime.get("use_linear_time", False)
    df_trainval = annual[annual["Year"].isin(regime["trainval_years"])].copy()
    df_test = annual[annual["Year"].isin(regime["test_years"])].copy()

    # Prepare test features (with or without Year)
    if use_linear:
        X_test_year, y_test = prepare_features(df_test)
        X_test_noyear = prepare_features(df_test, drop_year=True)[0]
    else:
        X_test, y_test = prepare_features(df_test, drop_year=True)

    test_agents = df_test["AgentID"].values
    n_test = len(y_test)

    preds_matrix = np.zeros((n_members, n_test))
    metrics_list = []

    for i in range(n_members):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        # Constrained bootstrap on trainval years
        boot_years = make_constrained_moving_block_year_sequence(
            regime["trainval_years"], regime["anchor_years"],
            BLOCK_SIZE, member_rng)

        df_boot = bootstrap_panel_by_year_sequence(df_trainval, boot_years)
        boot_years_arr = df_boot["Year"].values.copy()

        # Sample weights for this bootstrap member
        sw = compute_sample_weights(
            df_boot, regime.get("downweight_zero_irrig"))

        if use_linear:
            # Re-fit Ridge per member (captures trend uncertainty)
            X_boot_year, y_boot_raw = prepare_features(df_boot)
            ridge_boot, y_boot = fit_linear_time_trend(
                X_boot_year, y_boot_raw)
            X_boot = prepare_features(df_boot, drop_year=True)[0]
        else:
            X_boot, y_boot = prepare_features(df_boot, drop_year=True)
            ridge_boot = None

        # Fixed params — only random_state varies across members
        params = {
            **best_params,
            "objective": "reg:squarederror",
            "early_stopping_rounds": 50,
            "verbosity": 0,
            "random_state": int(member_rng.integers(0, 2**31)),
        }
        model = train_with_early_stopping(
            X_boot, y_boot, params, val_years=3,
            refit=True, sample_weight=sw, years=boot_years_arr)

        if use_linear:
            preds = predict_with_linear_trend(
                ridge_boot, model, X_test_year, X_test_noyear)
        else:
            preds = model.predict(X_test)
        preds_matrix[i] = preds

        m = compute_full_metrics(y_test, preds, test_agents)
        m["ensemble_id"] = i
        metrics_list.append(m)

    return preds_matrix, metrics_list


def _run_regime_ensemble_per_agent(regime, annual, n_members, rng):
    """Run per-agent bootstrap ensemble for a regime.

    Each agent gets its own saved params and is bootstrapped independently.
    Predictions are stacked across agents to match the pooled output shape.

    Returns:
        preds_matrix: (n_members, n_test) array of predictions
            where n_test = len(test_years) * len(AGENT_IDS).
        metrics_list: list of per-member metric dicts.
    """
    name = regime["name"]
    df_test_all = annual[annual["Year"].isin(regime["test_years"])].copy()
    df_test_all = df_test_all.sort_values(["AgentID", "Year"]).reset_index(
        drop=True)
    _, y_test_all = prepare_features(df_test_all, drop_year=True)
    test_agents_all = df_test_all["AgentID"].values
    n_test = len(y_test_all)

    preds_matrix = np.zeros((n_members, n_test))

    # Build per-agent index mapping into the stacked test array
    agent_slices = {}
    for aid in AGENT_IDS:
        mask = df_test_all["AgentID"] == aid
        agent_slices[aid] = np.where(mask.values)[0]

    for i in range(n_members):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        for aid in AGENT_IDS:
            # Load per-agent params
            params_path = TWO_REGIME_DIR / f"best_params_{name}_agent{aid}.json"
            with open(params_path) as f:
                best_params = json.load(f)

            # Filter to single agent
            df_agent = annual[annual["AgentID"] == aid].copy()
            df_trainval = df_agent[df_agent["Year"].isin(
                regime["trainval_years"])].copy()
            df_test_agent = df_agent[df_agent["Year"].isin(
                regime["test_years"])].copy()

            # Bootstrap years (per-agent RNG fork)
            agent_rng = np.random.default_rng(
                member_rng.integers(0, 2**31))
            boot_years = make_constrained_moving_block_year_sequence(
                regime["trainval_years"], regime["anchor_years"],
                BLOCK_SIZE, agent_rng)

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
                "random_state": int(agent_rng.integers(0, 2**31)),
            }
            model = train_with_early_stopping(
                X_boot, y_boot, params, val_years=3,
                years=boot_years_arr)

            X_test_agent, _ = prepare_features(
                df_test_agent, add_agent_dummies=False, drop_year=True)
            preds = model.predict(X_test_agent)
            preds_matrix[i, agent_slices[aid]] = preds

    # Compute per-member metrics (pooled across agents)
    metrics_list = []
    for i in range(n_members):
        m = compute_full_metrics(
            y_test_all, preds_matrix[i], test_agents_all)
        m["ensemble_id"] = i
        metrics_list.append(m)

    return preds_matrix, metrics_list


def evaluate_regime(preds_matrix, df_test):
    """Compute prediction intervals and summary statistics for a regime.

    Args:
        preds_matrix: (n_members, n_test) predictions.
        df_test: test DataFrame (must contain Year, AgentID, Irrigation_Depth).

    Returns:
        pi_df: DataFrame with Year, AgentID, Obs, p05, p50, p95.
        summary: dict with ensemble metrics and PI statistics.
    """
    _, y_test = prepare_features(df_test)
    test_agents = df_test["AgentID"].values
    test_years = df_test["Year"].values
    n_test = len(y_test)

    # Per-point prediction intervals
    p05 = np.percentile(preds_matrix, 5, axis=0)
    p50 = np.percentile(preds_matrix, 50, axis=0)
    p95 = np.percentile(preds_matrix, 95, axis=0)

    pi_rows = []
    for j in range(n_test):
        pi_rows.append({
            "Year": test_years[j],
            "AgentID": test_agents[j],
            "Obs": y_test[j],
            "p05": p05[j],
            "p50": p50[j],
            "p95": p95[j],
        })
    pi_df = pd.DataFrame(pi_rows)

    # Point-estimate metrics from median
    med_rmse = np.sqrt(mean_squared_error(y_test, p50))
    med_mae = mean_absolute_error(y_test, p50)
    med_r2 = r2_score(y_test, p50)

    # Per-member RMSE spread
    member_rmses = np.array([
        np.sqrt(mean_squared_error(y_test, preds_matrix[i]))
        for i in range(preds_matrix.shape[0])
    ])

    # PI width and coverage
    pi_widths = p95 - p05
    coverage = np.mean((y_test >= p05) & (y_test <= p95))

    summary = {
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
    return pi_df, summary


def _run_regime_per_agent(regime, annual, seed_offset):
    """Train separate per-agent models for a regime (no agent dummies).

    Args:
        regime: dict with trainval_years, test_years, etc.
        annual: full annual DataFrame.
        seed_offset: seed offset for Optuna reproducibility.

    Returns:
        df_pred: DataFrame with Year, AgentID, Split, Obs, Pred columns.
        train_metrics: dict of overall training metrics.
        test_metrics: dict of overall testing metrics.
    """
    name = regime["name"]
    pred_frames = []
    all_train_obs, all_train_pred = [], []
    all_test_obs, all_test_pred = [], []

    for aid in AGENT_IDS:
        print(f"\n  --- Agent {aid} ---")
        df_agent = annual[annual["AgentID"] == aid].copy()
        df_trainval = df_agent[df_agent["Year"].isin(
            regime["trainval_years"])].copy()
        df_test = df_agent[df_agent["Year"].isin(
            regime["test_years"])].copy()

        X_trainval, y_trainval = prepare_features(
            df_trainval, add_agent_dummies=False, drop_year=True)
        X_test, y_test = prepare_features(
            df_test, add_agent_dummies=False, drop_year=True)
        trainval_years_arr = df_trainval["Year"].values

        print(f"    Trainval: {len(df_trainval)} rows, "
              f"features: {X_trainval.shape[1]}")
        print(f"    Test:     {len(df_test)} rows")

        # Optuna tuning (per-agent seed)
        agent_seed = MASTER_SEED + seed_offset + aid
        print(f"    Tuning (seed={agent_seed})...")
        _, best_params = optuna_tune(
            X_trainval, y_trainval, seed=agent_seed,
            years=trainval_years_arr)

        # Determine regularised n_estimators via expanding-window CV
        # Use min_train_years=3 since each agent only has 8 years
        cv_n_est = _cv_best_n_estimators(
            X_trainval, y_trainval, best_params, min_train_years=3,
            years=trainval_years_arr)
        best_params["n_estimators"] = cv_n_est

        # Save per-agent params
        params_path = TWO_REGIME_DIR / f"best_params_{name}_agent{aid}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"    Saved {params_path}")

        # Final retrain on all trainval (no early stopping, CV-capped n_estimators)
        retrain_params = {
            k: v for k, v in best_params.items()
            if k != "early_stopping_rounds"
        }
        retrain_params["objective"] = "reg:squarederror"
        retrain_params["verbosity"] = 0
        model = XGBRegressor(**retrain_params)
        model.fit(X_trainval, y_trainval)

        # Predict
        pred_trainval = model.predict(X_trainval)
        pred_test = model.predict(X_test)

        # Build per-agent prediction DataFrames
        df_pred_tv = df_trainval[["Year", "AgentID"]].copy()
        df_pred_tv["Split"] = "train"
        df_pred_tv["Obs"] = y_trainval
        df_pred_tv["Pred"] = pred_trainval

        df_pred_te = df_test[["Year", "AgentID"]].copy()
        df_pred_te["Split"] = "test"
        df_pred_te["Obs"] = y_test
        df_pred_te["Pred"] = pred_test

        pred_frames.append(df_pred_tv)
        pred_frames.append(df_pred_te)

        all_train_obs.extend(y_trainval)
        all_train_pred.extend(pred_trainval)
        all_test_obs.extend(y_test)
        all_test_pred.extend(pred_test)

        # Per-agent metrics
        agent_train_m = compute_metrics(y_trainval, pred_trainval)
        agent_test_m = compute_metrics(y_test, pred_test)
        print(f"    Train — RMSE={agent_train_m['RMSE']:.2f}, "
              f"R²={agent_train_m['R2']:.3f}")
        print(f"    Test  — RMSE={agent_test_m['RMSE']:.2f}, "
              f"R²={agent_test_m['R2']:.3f}")

    # Merge predictions across agents
    df_pred = pd.concat(pred_frames, ignore_index=True)

    # Overall metrics (pooled across agents)
    train_metrics = compute_metrics(
        np.array(all_train_obs), np.array(all_train_pred))
    test_metrics = compute_metrics(
        np.array(all_test_obs), np.array(all_test_pred))

    return df_pred, train_metrics, test_metrics


def run_two_regime_point(annual):
    """Point estimation for the two-regime experiment: Optuna tune + predict."""
    print("=" * 60)
    print("TWO-REGIME POINT ESTIMATION")
    print("=" * 60)

    TWO_REGIME_DIR.mkdir(parents=True, exist_ok=True)

    combined_summaries = []

    for regime in [REGIME_A, REGIME_B]:
        name = regime["name"]
        print(f"\n--- Regime: {name} ({regime['full_years'][0]}–"
              f"{regime['full_years'][-1]}) ---")
        print(f"  Trainval: {regime['trainval_years'][0]}–"
              f"{regime['trainval_years'][-1]} "
              f"({len(regime['trainval_years'])} years)")
        print(f"  Test:     {regime['test_years']}")

        seed_offset = 10 if name == "pre" else 20

        if regime.get("per_agent", False):
            # Per-agent path: train separate models per agent
            print("  Mode: per-agent (separate models)")
            df_pred, train_metrics, test_metrics = _run_regime_per_agent(
                regime, annual, seed_offset)
        else:
            # Pooled path — supports linear time detrending, sample weighting,
            # search space overrides, and rolling validation folds
            use_linear = regime.get("use_linear_time", False)
            df_trainval = annual[annual["Year"].isin(
                regime["trainval_years"])].copy()
            df_test = annual[annual["Year"].isin(
                regime["test_years"])].copy()

            # Save year arrays before potential Year column removal
            years_trainval = df_trainval["Year"].values.copy()

            # Sample weights
            sw = compute_sample_weights(
                df_trainval, regime.get("downweight_zero_irrig"))

            if use_linear:
                # Linear time detrending: Ridge on Year, XGBoost on residuals
                X_trainval_year, y_trainval_raw = prepare_features(df_trainval)
                ridge, y_trainval = fit_linear_time_trend(
                    X_trainval_year, y_trainval_raw)
                X_trainval = prepare_features(
                    df_trainval, drop_year=True)[0]
                print(f"  Linear time detrending enabled (Ridge on Year)")
            else:
                X_trainval, y_trainval = prepare_features(df_trainval, drop_year=True)
                ridge = None

            print(f"  Tuning (seed={MASTER_SEED + seed_offset})...")
            _, best_params = optuna_tune(
                X_trainval, y_trainval,
                seed=MASTER_SEED + seed_offset,
                search_space_overrides=regime.get("search_space_overrides"),
                rolling_val_folds=regime.get("rolling_val_folds"),
                sample_weight=sw,
                years=years_trainval)

            # Determine regularised n_estimators via expanding-window CV
            cv_n_est = _cv_best_n_estimators(
                X_trainval, y_trainval, best_params,
                sample_weight=sw, years=years_trainval)
            best_params["n_estimators"] = cv_n_est

            # Save best params (with CV-determined n_estimators)
            params_path = TWO_REGIME_DIR / f"best_params_{name}.json"
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"  Saved {params_path}")

            # Final retrain on all trainval (no early stopping)
            retrain_params = {
                k: v for k, v in best_params.items()
                if k != "early_stopping_rounds"
            }
            retrain_params["objective"] = "reg:squarederror"
            retrain_params["verbosity"] = 0
            model = XGBRegressor(**retrain_params)
            sw_kw = {"sample_weight": sw} if sw is not None else {}
            model.fit(X_trainval, y_trainval, **sw_kw)

            # Predict on trainval (in-sample) and test (out-of-sample)
            if use_linear:
                pred_trainval = predict_with_linear_trend(
                    ridge, model, X_trainval_year, X_trainval)
                X_test_year, y_test = prepare_features(df_test)
                X_test_noyear = prepare_features(
                    df_test, drop_year=True)[0]
                pred_test = predict_with_linear_trend(
                    ridge, model, X_test_year, X_test_noyear)
                # Use raw y for metrics (not residuals)
                y_trainval = y_trainval_raw
            else:
                pred_trainval = model.predict(X_trainval)
                X_test, y_test = prepare_features(df_test, drop_year=True)
                pred_test = model.predict(X_test)

            # Build point predictions CSV
            df_pred_tv = df_trainval[["Year", "AgentID"]].copy()
            df_pred_tv["Split"] = "train"
            df_pred_tv["Obs"] = y_trainval
            df_pred_tv["Pred"] = pred_trainval

            df_pred_te = df_test[["Year", "AgentID"]].copy()
            df_pred_te["Split"] = "test"
            df_pred_te["Obs"] = y_test
            df_pred_te["Pred"] = pred_test

            df_pred = pd.concat([df_pred_tv, df_pred_te], ignore_index=True)

            # Compute metrics for train and test
            train_metrics = compute_metrics(y_trainval, pred_trainval)
            test_metrics = compute_metrics(y_test, pred_test)

        # Save predictions (common to both paths)
        pred_path = TWO_REGIME_DIR / f"point_predictions_{name}.csv"
        df_pred.to_csv(pred_path, index=False)
        print(f"  Saved {pred_path}")

        metrics_rows = [
            {"Split": "train", **train_metrics},
            {"Split": "test", **test_metrics},
        ]
        metrics_path = TWO_REGIME_DIR / f"point_metrics_{name}.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"  Saved {metrics_path}")

        print(f"  Train — RMSE={train_metrics['RMSE']:.2f}, "
              f"R²={train_metrics['R2']:.3f}")
        print(f"  Test  — RMSE={test_metrics['RMSE']:.2f}, "
              f"R²={test_metrics['R2']:.3f}")

        summary = {
            "regime": name,
            "train_RMSE": train_metrics["RMSE"],
            "train_MAE": train_metrics["MAE"],
            "train_R2": train_metrics["R2"],
            "train_Bias": train_metrics["Bias"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_R2": test_metrics["R2"],
            "test_Bias": test_metrics["Bias"],
        }
        combined_summaries.append(summary)

    # Combined summary
    combined_path = TWO_REGIME_DIR / "two_regime_point_summary.csv"
    pd.DataFrame(combined_summaries).to_csv(combined_path, index=False)
    print(f"\nSaved {combined_path}")

    # Plot
    print("\nGenerating point-estimate plot...")
    plot_two_regime_point(
        annual, TWO_REGIME_DIR / "two_regime_point_estimates.pdf")
    print("\nGenerating post-CP point-estimate plot...")
    plot_post_cp_point(
        annual, TWO_REGIME_DIR / "post_cp_point_estimates.pdf")

    # Final report
    print("\n" + "=" * 60)
    print("TWO-REGIME POINT ESTIMATION SUMMARY")
    print("=" * 60)
    for s in combined_summaries:
        print(f"  {s['regime']:4s}  train RMSE={s['train_RMSE']:.2f} "
              f"R²={s['train_R2']:.3f}  "
              f"test RMSE={s['test_RMSE']:.2f} R²={s['test_R2']:.3f}")
    print("=" * 60)


def run_two_regime_ensemble(annual):
    """Ensemble step for the two-regime experiment: load params + bootstrap PI."""
    print("=" * 60)
    print("TWO-REGIME ENSEMBLE")
    print("=" * 60)

    TWO_REGIME_DIR.mkdir(parents=True, exist_ok=True)

    combined_summaries = []

    for regime in [REGIME_A, REGIME_B]:
        name = regime["name"]
        print(f"\n--- Regime: {name} ({regime['full_years'][0]}–"
              f"{regime['full_years'][-1]}) ---")
        print(f"  Trainval: {regime['trainval_years'][0]}–"
              f"{regime['trainval_years'][-1]} "
              f"({len(regime['trainval_years'])} years)")
        print(f"  Test:     {regime['test_years']}")
        print(f"  Anchor:   {regime['anchor_years']}")

        seed_offset = 10 if name == "pre" else 20
        rng = np.random.default_rng(MASTER_SEED + seed_offset + 1)

        if regime.get("per_agent", False):
            # Per-agent ensemble: load per-agent params internally
            for aid in AGENT_IDS:
                pa_path = TWO_REGIME_DIR / f"best_params_{name}_agent{aid}.json"
                if not pa_path.exists():
                    raise FileNotFoundError(
                        f"Per-agent params not found: {pa_path}\n"
                        "Run --two-regime-point first to generate them.")
                print(f"  Loaded {pa_path}")

            print(f"  Running per-agent ensemble (N={N_TWO_REGIME_BOOT})...")
            preds_matrix, metrics_list = _run_regime_ensemble_per_agent(
                regime, annual, N_TWO_REGIME_BOOT, rng)
        else:
            # Pooled ensemble: load single params file
            params_path = TWO_REGIME_DIR / f"best_params_{name}.json"
            if not params_path.exists():
                raise FileNotFoundError(
                    f"Best params not found: {params_path}\n"
                    "Run --two-regime-point first to generate them.")
            with open(params_path) as f:
                best_params = json.load(f)
            print(f"  Loaded {params_path}")

            print(f"  Running ensemble (N={N_TWO_REGIME_BOOT})...")
            preds_matrix, metrics_list = run_regime_ensemble(
                regime, annual, best_params, N_TWO_REGIME_BOOT, rng)

        # Save ensemble metrics
        ens_path = TWO_REGIME_DIR / f"ensemble_metrics_{name}.csv"
        pd.DataFrame(metrics_list).to_csv(ens_path, index=False)
        print(f"  Saved {ens_path}")

        # Evaluate
        df_test = annual[annual["Year"].isin(regime["test_years"])].copy()
        pi_df, summary = evaluate_regime(preds_matrix, df_test)

        # Save prediction intervals
        pi_path = TWO_REGIME_DIR / f"prediction_intervals_{name}.csv"
        pi_df.to_csv(pi_path, index=False)
        print(f"  Saved {pi_path}")

        # Save summary
        summary["regime"] = name
        summary_path = TWO_REGIME_DIR / f"summary_{name}.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f"  Saved {summary_path}")

        combined_summaries.append(summary)

        print(f"  RMSE (median pred): {summary['RMSE_median_pred']:.2f}")
        print(f"  R² (median pred):   {summary['R2_median_pred']:.3f}")
        print(f"  PI width (mean):    {summary['PI_width_mean']:.2f}")
        print(f"  PI coverage (90%):  {summary['PI_coverage_90']:.3f}")

    # Combined summary
    combined_path = TWO_REGIME_DIR / "two_regime_summary.csv"
    pd.DataFrame(combined_summaries).to_csv(combined_path, index=False)
    print(f"\nSaved {combined_path}")

    # Plot
    print("\nGenerating PI plot...")
    plot_two_regime_results(
        annual, TWO_REGIME_DIR / "two_regime_point_estimates_pi.pdf")

    # Final report
    print("\n" + "=" * 60)
    print("TWO-REGIME ENSEMBLE SUMMARY")
    print("=" * 60)
    for s in combined_summaries:
        print(f"  {s['regime']:4s}  RMSE={s['RMSE_median_pred']:.2f}  "
              f"R²={s['R2_median_pred']:.3f}  "
              f"PI_width={s['PI_width_mean']:.1f}  "
              f"coverage={s['PI_coverage_90']:.3f}")
    print("=" * 60)


# ── Post-CP Jitter Ensemble ──────────────────────────────────────────────────
def run_post_cp_jitter_ensemble(annual):
    """Generate prediction intervals for post-CP test period (2017–2020).

    Uncertainty comes from hyperparameter jitter only (no data bootstrapping).
    Every ensemble member trains on the same 2005–2016 data with params
    jittered around the Optuna-tuned values via JITTER_WIDTHS.
    """
    print("=" * 60)
    print("POST-CP JITTER ENSEMBLE")
    print("=" * 60)

    TWO_REGIME_DIR.mkdir(parents=True, exist_ok=True)

    regime = REGIME_B
    name = regime["name"]
    print(f"\nRegime: {name} ({regime['full_years'][0]}–"
          f"{regime['full_years'][-1]})")
    print(f"  Trainval: {regime['trainval_years'][0]}–"
          f"{regime['trainval_years'][-1]} "
          f"({len(regime['trainval_years'])} years)")
    print(f"  Test:     {regime['test_years']}")

    # Load Optuna-tuned params
    params_path = TWO_REGIME_DIR / f"best_params_{name}.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Best params not found: {params_path}\n"
            "Run --two-regime-point first to generate them.")
    with open(params_path) as f:
        best_params = json.load(f)
    print(f"  Loaded {params_path}")

    # Prepare data (same for every member)
    df_trainval = annual[annual["Year"].isin(regime["trainval_years"])].copy()
    df_test = annual[annual["Year"].isin(regime["test_years"])].copy()
    X_trainval, y_trainval = prepare_features(df_trainval, drop_year=True)
    X_test, y_test = prepare_features(df_test, drop_year=True)
    trainval_years_arr = df_trainval["Year"].values.copy()
    test_agents = df_test["AgentID"].values
    n_test = len(y_test)

    n_members = N_TWO_REGIME_BOOT
    preds_matrix = np.zeros((n_members, n_test))
    metrics_list = []

    seed_offset = 20
    rng = np.random.default_rng(MASTER_SEED + seed_offset + 1)

    print(f"  Running jitter ensemble (N={n_members}, no data resampling)...")
    for i in range(n_members):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))
        params = sample_hyperparams(member_rng, base_params=best_params)
        model = train_with_early_stopping(
            X_trainval, y_trainval, params, val_years=3,
            refit=True, years=trainval_years_arr)
        preds = model.predict(X_test)
        preds_matrix[i] = preds

        m = compute_full_metrics(y_test, preds, test_agents)
        m["ensemble_id"] = i
        metrics_list.append(m)

    # Save per-member raw predictions (long format)
    test_years_arr = df_test["Year"].values
    member_rows = []
    for i in range(n_members):
        for j in range(n_test):
            member_rows.append({
                "Year": int(test_years_arr[j]),
                "AgentID": int(test_agents[j]),
                "Member": i,
                "Pred_mm": float(preds_matrix[i, j]),
            })
    member_df = pd.DataFrame(member_rows)
    member_path = TWO_REGIME_DIR / f"ensemble_member_predictions_{name}.csv"
    member_df.to_csv(member_path, index=False)
    print(f"  Saved {member_path}")

    # Save ensemble metrics
    ens_path = TWO_REGIME_DIR / f"ensemble_metrics_{name}.csv"
    pd.DataFrame(metrics_list).to_csv(ens_path, index=False)
    print(f"  Saved {ens_path}")

    # Evaluate (PIs + summary)
    pi_df, summary = evaluate_regime(preds_matrix, df_test)

    pi_path = TWO_REGIME_DIR / f"prediction_intervals_{name}.csv"
    pi_df.to_csv(pi_path, index=False)
    print(f"  Saved {pi_path}")

    summary["regime"] = name
    summary_path = TWO_REGIME_DIR / f"summary_{name}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Saved {summary_path}")

    print(f"\n  RMSE (median pred): {summary['RMSE_median_pred']:.2f}")
    print(f"  R² (median pred):   {summary['R2_median_pred']:.3f}")
    print(f"  PI width (mean):    {summary['PI_width_mean']:.2f}")
    print(f"  PI coverage (90%):  {summary['PI_coverage_90']:.3f}")

    # Plot
    print("\nGenerating post-CP PI plot...")
    plot_post_cp_pi(
        annual, TWO_REGIME_DIR / "post_cp_point_estimates_pi.pdf")

    print("\n" + "=" * 60)
    print("POST-CP JITTER ENSEMBLE COMPLETE")
    print("=" * 60)


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_two_regime_point(annual, output_pdf):
    """Plot two-regime point estimates (no PI bands).

    Layout: 2×2 grid (rows = regime [pre, post], cols = agent).
    Each panel shows: full observed series, in-sample training predictions
    (solid line), test predictions (dashed line), training period shading,
    and R² annotations for train and test.

    Reads saved point_predictions CSVs from TWO_REGIME_DIR.
    """
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

    regimes = [REGIME_A, REGIME_B]
    colors = {"pre": "tab:blue", "post": "tab:red"}
    n_agents = len(AGENT_IDS)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(2 * n_agents)]

    fig, axes = plt.subplots(2, n_agents, figsize=(5 * n_agents, 9),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_idx, regime in enumerate(regimes):
        name = regime["name"]
        color = colors[name]

        # Load saved point predictions
        pred_df = pd.read_csv(
            TWO_REGIME_DIR / f"point_predictions_{name}.csv")
        train_pred = pred_df[pred_df["Split"] == "train"]
        test_pred = pred_df[pred_df["Split"] == "test"]

        # Full regime observed data
        df_full = annual[annual["Year"].isin(regime["full_years"])].copy()

        for col_idx, aid in enumerate(AGENT_IDS):
            ax = axes[row_idx, col_idx]

            # --- Observed: full regime series ---
            obs_sub = df_full[df_full["AgentID"] == aid].sort_values("Year")
            ax.plot(obs_sub["Year"].values,
                    obs_sub["Irrigation_Depth"].values,
                    "ko-", label="Observed", linewidth=1.2, markersize=5,
                    zorder=5)

            # --- Training period shading ---
            tv_min = regime["trainval_years"][0]
            tv_max = regime["trainval_years"][-1]
            shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
            if row_idx == 0 and col_idx == 0:
                shade_kw["label"] = "Training period"
            ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

            # --- In-sample training predictions (solid) ---
            tv_sub = train_pred[train_pred["AgentID"] == aid].sort_values(
                "Year")
            ax.plot(tv_sub["Year"].values, tv_sub["Pred"].values,
                    "-", color=color, linewidth=1.5, label="Train pred",
                    zorder=3)

            # Train R²
            if len(tv_sub) > 1:
                train_r2 = r2_score(tv_sub["Obs"].values,
                                    tv_sub["Pred"].values)
                blend = matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)
                ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                        transform=blend, va="top", ha="left", fontsize=12,
                        color="0.2")

            # --- Test predictions (dashed) ---
            te_sub = test_pred[test_pred["AgentID"] == aid].sort_values(
                "Year")
            ax.plot(te_sub["Year"].values, te_sub["Pred"].values,
                    "--", color=color, linewidth=1.5, label="Test pred",
                    zorder=4)

            # Test R²
            if len(te_sub) > 1:
                test_r2 = r2_score(te_sub["Obs"].values,
                                   te_sub["Pred"].values)
                ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                        transform=ax.transAxes, va="top", ha="right",
                        fontsize=12, color="black")

            # --- Panel label ---
            panel_idx = row_idx * n_agents + col_idx
            ax.text(0.01, 0.99, panel_labels[panel_idx],
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=14, fontweight="bold")

            regime_label = ("Pre-CP" if name == "pre" else "Post-CP")
            ax.set_title(f"{regime_label} ({regime['full_years'][0]}"
                         f"\u2013{regime['full_years'][-1]}) \u2014 Agent {aid}")
            ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
            ax.tick_params(axis="both", which="major", length=4)

            if row_idx == 1:
                ax.set_xlabel("Year")
            if col_idx == 0:
                ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure-level legend from first subplot
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Two-Regime XGBoost \u2014 Point Estimates",
                 fontsize=16, y=1.09)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_post_cp_point(annual, output_pdf):
    """Plot post-CP point estimates (2×1: one row per agent).

    Each panel shows the full observed series (1993–2020) for pre-CP context,
    training predictions (solid), test predictions (dashed), training period
    shading, a vertical changepoint line at 2005, and R² annotations.

    Reads saved point_predictions_post.csv from TWO_REGIME_DIR.
    """
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

    regime = REGIME_B
    color = "tab:red"
    n_agents = len(AGENT_IDS)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(n_agents)]

    # Load saved point predictions
    pred_df = pd.read_csv(TWO_REGIME_DIR / "point_predictions_post.csv")
    train_pred = pred_df[pred_df["Split"] == "train"]
    test_pred = pred_df[pred_df["Split"] == "test"]

    fig, axes = plt.subplots(n_agents, 1, figsize=(10, 4 * n_agents),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for row_idx, aid in enumerate(AGENT_IDS):
        ax = axes[row_idx]

        # --- Observed: full record for pre-CP context ---
        obs_sub = annual[annual["AgentID"] == aid].sort_values("Year")
        ax.plot(obs_sub["Year"].values,
                obs_sub["Irrigation_Depth"].values,
                "ko-", label="Observed", linewidth=1.2, markersize=5,
                zorder=5)

        # --- Vertical changepoint line at 2005 ---
        cp_kw = dict(color="gray", linestyle="--", linewidth=1.0, zorder=1)
        if row_idx == 0:
            cp_kw["label"] = "Changepoint (2005)"
        ax.axvline(2005, **cp_kw)

        # --- Training period shading ---
        tv_min = regime["trainval_years"][0]
        tv_max = regime["trainval_years"][-1]
        shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
        if row_idx == 0:
            shade_kw["label"] = "Training period"
        ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

        # --- In-sample training predictions (solid) ---
        tv_sub = train_pred[train_pred["AgentID"] == aid].sort_values("Year")
        ax.plot(tv_sub["Year"].values, tv_sub["Pred"].values,
                "-", color=color, linewidth=1.5, label="Train pred",
                zorder=3)

        # Train R²
        if len(tv_sub) > 1:
            train_r2 = r2_score(tv_sub["Obs"].values,
                                tv_sub["Pred"].values)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

        # --- Test predictions (dashed) ---
        te_sub = test_pred[test_pred["AgentID"] == aid].sort_values("Year")
        ax.plot(te_sub["Year"].values, te_sub["Pred"].values,
                "--", color=color, linewidth=1.5, label="Test pred",
                zorder=4)

        # Test R²
        if len(te_sub) > 1:
            test_r2 = r2_score(te_sub["Obs"].values,
                               te_sub["Pred"].values)
            ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=12, color="black")

        # --- Panel label ---
        ax.text(0.01, 0.99, panel_labels[row_idx],
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        ax.set_title(f"Agent {aid}")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure-level legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Post-CP XGBoost \u2014 Point Estimates",
                 fontsize=16, y=1.09)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_post_cp_pi(annual, output_pdf):
    """Plot post-CP point estimates with prediction interval bands (2×1).

    Mirrors plot_post_cp_point() but overlays the bootstrap PI band on the
    test years (2017–2020).  Reads point_predictions_post.csv for train preds
    and prediction_intervals_post.csv for test PI.
    """
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

    regime = REGIME_B
    color = "tab:red"
    n_agents = len(AGENT_IDS)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(n_agents)]

    # Load saved point predictions (train) and PI (test)
    pred_df = pd.read_csv(TWO_REGIME_DIR / "point_predictions_post.csv")
    train_pred = pred_df[pred_df["Split"] == "train"]
    pi_df = pd.read_csv(
        TWO_REGIME_DIR / "prediction_intervals_post.csv")

    fig, axes = plt.subplots(n_agents, 1, figsize=(10, 4 * n_agents),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for row_idx, aid in enumerate(AGENT_IDS):
        ax = axes[row_idx]

        # --- Observed: full record for pre-CP context ---
        obs_sub = annual[annual["AgentID"] == aid].sort_values("Year")
        ax.plot(obs_sub["Year"].values,
                obs_sub["Irrigation_Depth"].values,
                "ko-", label="Observed", linewidth=1.2, markersize=5,
                zorder=5)

        # --- Vertical changepoint line at 2005 ---
        cp_kw = dict(color="gray", linestyle="--", linewidth=1.0, zorder=1)
        if row_idx == 0:
            cp_kw["label"] = "Changepoint (2005)"
        ax.axvline(2005, **cp_kw)

        # --- Training period shading ---
        tv_min = regime["trainval_years"][0]
        tv_max = regime["trainval_years"][-1]
        shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
        if row_idx == 0:
            shade_kw["label"] = "Training period"
        ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

        # --- In-sample training predictions (solid) ---
        tv_sub = train_pred[train_pred["AgentID"] == aid].sort_values("Year")
        ax.plot(tv_sub["Year"].values, tv_sub["Pred"].values,
                "-", color=color, linewidth=1.5, label="Train pred",
                zorder=3)

        # Train R²
        if len(tv_sub) > 1:
            train_r2 = r2_score(tv_sub["Obs"].values,
                                tv_sub["Pred"].values)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

        # --- Test: PI band + median ---
        pi_sub = pi_df[pi_df["AgentID"] == aid].sort_values("Year")
        test_yrs = pi_sub["Year"].values

        pi_kw = dict(alpha=0.25, color=color, zorder=2)
        if row_idx == 0:
            pi_kw["label"] = "90% PI"
        ax.fill_between(test_yrs, pi_sub["p05"], pi_sub["p95"], **pi_kw)
        ax.plot(test_yrs, pi_sub["p50"], "--", color=color,
                linewidth=1.5, label="Test median", zorder=4)

        # Test R² (from median prediction)
        test_r2 = r2_score(pi_sub["Obs"].values, pi_sub["p50"].values)
        ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=12, color="black")

        # --- Panel label ---
        ax.text(0.01, 0.99, panel_labels[row_idx],
                transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

        ax.set_title(f"Agent {aid}")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure-level legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Post-CP XGBoost \u2014 Point Estimates & Prediction Intervals",
                 fontsize=16, y=1.09)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_two_regime_results(annual, output_pdf):
    """Plot two-regime point estimates + prediction intervals.

    Layout: 2×2 grid (rows = regime [pre, post], cols = agent).
    Each panel shows: full observed series for that regime, in-sample
    training predictions (solid line from saved point predictions),
    test PI band (shaded), and ensemble median on test (dashed line).

    Reads saved point_predictions and PI CSVs from TWO_REGIME_DIR.
    """
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

    regimes = [REGIME_A, REGIME_B]
    colors = {"pre": "tab:blue", "post": "tab:red"}
    n_agents = len(AGENT_IDS)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(2 * n_agents)]

    fig, axes = plt.subplots(2, n_agents, figsize=(5 * n_agents, 9),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_idx, regime in enumerate(regimes):
        name = regime["name"]
        color = colors[name]

        # Load saved point predictions for train portion
        pred_df = pd.read_csv(
            TWO_REGIME_DIR / f"point_predictions_{name}.csv")
        train_pred = pred_df[pred_df["Split"] == "train"]

        # Load PI data for test set
        pi_df = pd.read_csv(TWO_REGIME_DIR / f"prediction_intervals_{name}.csv")

        # Full regime observed data
        df_full = annual[annual["Year"].isin(regime["full_years"])].copy()

        for col_idx, aid in enumerate(AGENT_IDS):
            ax = axes[row_idx, col_idx]

            # --- Observed: full regime series ---
            obs_sub = df_full[df_full["AgentID"] == aid].sort_values("Year")
            ax.plot(obs_sub["Year"].values,
                    obs_sub["Irrigation_Depth"].values,
                    "ko-", label="Observed", linewidth=1.2, markersize=5,
                    zorder=5)

            # --- Training period shading ---
            tv_min = regime["trainval_years"][0]
            tv_max = regime["trainval_years"][-1]
            shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
            if row_idx == 0 and col_idx == 0:
                shade_kw["label"] = "Training period"
            ax.axvspan(tv_min - 0.3, tv_max + 0.3, **shade_kw)

            # --- In-sample training predictions (from saved point predictions) ---
            tv_sub = train_pred[train_pred["AgentID"] == aid].sort_values(
                "Year")
            ax.plot(tv_sub["Year"].values, tv_sub["Pred"].values,
                    "-", color=color, linewidth=1.5, label="Train pred",
                    zorder=3)

            # Train R²
            if len(tv_sub) > 1:
                train_r2 = r2_score(tv_sub["Obs"].values,
                                    tv_sub["Pred"].values)
                blend = matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)
                ax.text(tv_min + 0.3, 0.88, f"R$^2$ = {train_r2:.2f}",
                        transform=blend, va="top", ha="left", fontsize=12,
                        color="0.2")

            # --- Test: PI band + median ---
            pi_sub = pi_df[pi_df["AgentID"] == aid].sort_values("Year")
            test_yrs = pi_sub["Year"].values

            ax.fill_between(test_yrs, pi_sub["p05"], pi_sub["p95"],
                            alpha=0.25, color=color, label="90% PI",
                            zorder=2)
            ax.plot(test_yrs, pi_sub["p50"], "--", color=color,
                    linewidth=1.5, label="Test median", zorder=4)

            # Test R² (from median prediction)
            test_r2 = r2_score(pi_sub["Obs"].values, pi_sub["p50"].values)
            ax.text(0.98, 0.88, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=12, color="black")

            # --- Panel label ---
            panel_idx = row_idx * n_agents + col_idx
            ax.text(0.01, 0.99, panel_labels[panel_idx],
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=14, fontweight="bold")

            regime_label = ("Pre-CP" if name == "pre" else "Post-CP")
            ax.set_title(f"{regime_label} ({regime['full_years'][0]}"
                         f"\u2013{regime['full_years'][-1]}) \u2014 Agent {aid}")
            ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
            ax.tick_params(axis="both", which="major", length=4)

            if row_idx == 1:
                ax.set_xlabel("Year")
            if col_idx == 0:
                ax.set_ylabel("Annual irrigation depth (mm)")

    # Figure-level legend from first subplot
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Two-Regime XGBoost \u2014 Point Estimates & Prediction Intervals",
                 fontsize=16, y=1.09)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_point_estimates(
    df_m1_train, y_m1_train, pred_m1_train,
    df_m2_train, y_m2_train, pred_m2_train,
    df_eval, y_eval, pred_m1_eval, pred_m2_eval,
    cp_year, output_pdf,
):
    """2×2 grid: rows = model (M1, M2), cols = agent, with train+test and inset metrics."""
    # Save original rcParams and apply journal style
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

    # Build full observed series from train + eval
    all_df = pd.concat([df_m2_train, df_eval], ignore_index=True).drop_duplicates(
        subset=["Year", "AgentID"]
    ).sort_values(["AgentID", "Year"])

    models = [
        ("M1 (Stationary)", df_m1_train, y_m1_train, pred_m1_train, pred_m1_eval, "tab:blue"),
        ("M2 (CP-Aware)", df_m2_train, y_m2_train, pred_m2_train, pred_m2_eval, "tab:red"),
    ]

    n_agents = len(AGENT_IDS)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(2 * n_agents)]

    fig, axes = plt.subplots(2, n_agents, figsize=(5 * n_agents, 8),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_idx, (model_name, df_tr, y_tr, pred_tr, pred_ev, color) in enumerate(models):
        for col_idx, aid in enumerate(AGENT_IDS):
            ax = axes[row_idx, col_idx]

            # Observed: full series
            obs_sub = all_df[all_df["AgentID"] == aid].sort_values("Year")
            X_obs, y_obs = prepare_features(obs_sub)
            ax.plot(obs_sub["Year"].values, y_obs, "ko-", label="Observed",
                    linewidth=1.2, markersize=4, zorder=3)

            # Training predictions
            tr_mask = df_tr["AgentID"] == aid
            tr_sub = df_tr[tr_mask].sort_values("Year")
            ax.plot(tr_sub["Year"].values, pred_tr[tr_mask.values], "-",
                    color=color, linewidth=1.5, label="Train pred", zorder=2)

            # Test predictions
            ev_mask = df_eval["AgentID"] == aid
            ev_sub = df_eval[ev_mask].sort_values("Year")
            ax.plot(ev_sub["Year"].values, pred_ev[ev_mask.values], "--",
                    color=color, linewidth=1.5, label="Test pred", zorder=2)

            # Vertical CP line
            ax.axvline(cp_year, color="0.6", linestyle="--", linewidth=1.0)

            # Training period shading
            train_min = df_tr[df_tr["AgentID"] == aid]["Year"].min()
            train_max = df_tr[df_tr["AgentID"] == aid]["Year"].max()
            shade_kw = dict(color="lightgray", alpha=0.3, zorder=0)
            if row_idx == 0 and col_idx == 0:
                shade_kw["label"] = "Training period"
            ax.axvspan(train_min, train_max, **shade_kw)

            # Compute train metrics
            y_tr_agent = y_tr[tr_mask.values]
            pred_tr_agent = pred_tr[tr_mask.values]
            train_r2 = r2_score(y_tr_agent, pred_tr_agent)

            # Compute test metrics
            y_ev_agent = y_eval[ev_mask.values]
            pred_ev_agent = pred_ev[ev_mask.values]
            test_r2 = r2_score(y_ev_agent, pred_ev_agent)

            # Train R² (left, inside shaded training area — use data x so it
            # stays within the shading regardless of axis limits)
            blend = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(train_min + 0.5, 0.85, f"R$^2$ = {train_r2:.2f}",
                    transform=blend, va="top", ha="left", fontsize=12,
                    color="0.2")

            # Test R² (right, no box)
            ax.text(0.98, 0.85, f"R$^2$ = {test_r2:.2f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=12,
                    color="black")

            # Panel label
            panel_idx = row_idx * n_agents + col_idx
            ax.text(0.01, 0.99, panel_labels[panel_idx],
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=14, fontweight="bold")

            ax.set_title(f"{model_name} — Agent {aid}")
            ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
            ax.tick_params(axis="both", which="major", length=4)

            # Axis labels only where needed
            if row_idx == 1:
                ax.set_xlabel("Year")
            if col_idx == 0:
                ax.set_ylabel("Annual irrigation depth (mm)")

    # Single figure-level legend from first subplot
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))

    fig.suptitle("Cluster 2 — XGBoost Point Predictions", fontsize=16, y=1.10)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Restore original rcParams
    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")


def plot_prediction_intervals(df_pi, output_pdf, title=None):
    """Fan plot: observed + M1/M2 median with 5–95% bands, one subplot per agent."""
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx, aid in enumerate(AGENT_IDS):
        ax = axes.flat[idx]
        sub = df_pi[df_pi["AgentID"] == aid].sort_values("Year")
        yrs = sub["Year"].values

        ax.plot(yrs, sub["Obs"], "ko-", label="Observed", linewidth=1.5, zorder=5)

        # M1 band
        ax.fill_between(yrs, sub["M1_p05"], sub["M1_p95"],
                         alpha=0.2, color="blue", label="M1 90% PI")
        ax.plot(yrs, sub["M1_med"], "b--", alpha=0.7, label="M1 median")

        # M2 band
        ax.fill_between(yrs, sub["M2_p05"], sub["M2_p95"],
                         alpha=0.2, color="red", label="M2 90% PI")
        ax.plot(yrs, sub["M2_med"], "r--", alpha=0.7, label="M2 median")

        # Panel label
        ax.text(0.01, 0.99, panel_labels[idx], transform=ax.transAxes,
                va="top", ha="left", fontsize=14, fontweight="bold")

        ax.set_title(f"Agent {aid}")
        ax.set_xlabel("Year")
        if idx % ncols == 0:
            ax.set_ylabel("Annual irrigation depth (mm)")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", which="major", length=4)

    # Hide unused subplots
    for i in range(n_agents, nrows * ncols):
        axes.flat[i].set_visible(False)

    # Single figure-level legend above panels (no overlap with data)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    default_title = "Cluster 2 — Prediction Intervals (200 Ensemble Members)"
    fig.suptitle(title if title else default_title, fontsize=16, y=1.12)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


def plot_sensitivity_pi_grid(pi_data, cp_candidates, output_pdf):
    """2×3 fan-plot grid: rows=agents, cols=CP candidates (M2 only)."""
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
    n_cps = len(cp_candidates)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(n_agents * n_cps)]

    fig, axes = plt.subplots(n_agents, n_cps,
                             figsize=(6 * n_cps, 3.5 * n_agents),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_idx, aid in enumerate(AGENT_IDS):
        for col_idx, cp in enumerate(cp_candidates):
            ax = axes[row_idx, col_idx]
            panel_idx = row_idx * n_cps + col_idx
            df = pi_data[cp]
            sub = df[df["AgentID"] == aid].sort_values("Year")
            yrs = sub["Year"].values

            ax.plot(yrs, sub["Obs"], "ko-", label="Observed",
                    linewidth=1.5, zorder=5, markersize=4)
            ax.fill_between(yrs, sub["M2_p05"], sub["M2_p95"],
                            alpha=0.2, color="red", label="M2 90% PI")
            ax.plot(yrs, sub["M2_med"], "r--", alpha=0.7, label="M2 median")

            ax.text(0.01, 0.99, panel_labels[panel_idx],
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=14, fontweight="bold")

            if row_idx == 0:
                ax.set_title(f"CP = {cp}")
            if row_idx == n_agents - 1:
                ax.set_xlabel("Year")
            if col_idx == 0:
                ax.set_ylabel(f"Agent {aid}\nAnnual irrigation depth (mm)")

            ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
            ax.tick_params(axis="both", which="major", length=4)

    # Single figure-level legend at top (de-duplicate labels)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Cluster 2 — M2 Prediction Intervals by Changepoint Year",
                 fontsize=16, y=1.10)

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    output_png = str(output_pdf).replace(".pdf", ".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    matplotlib.rcParams.update(orig_rc)
    print(f"  Saved {output_pdf}")
    print(f"  Saved {output_png}")


# ── Ensemble output helper ────────────────────────────────────────────────────
def save_ensemble_results(preds_m1_ens, metrics_m1_ens, preds_m2_ens,
                          metrics_m2_ens, df_eval, y_eval):
    """Save ensemble CSVs, fan plot, and summary statistics."""
    # Save ensemble metrics
    df_ens_m1 = pd.DataFrame(metrics_m1_ens)
    df_ens_m2 = pd.DataFrame(metrics_m2_ens)
    ens_m1_path = RESULTS_DIR / "cluster2_ensemble_metrics_M1.csv"
    ens_m2_path = RESULTS_DIR / "cluster2_ensemble_metrics_M2.csv"
    df_ens_m1.to_csv(ens_m1_path, index=False)
    df_ens_m2.to_csv(ens_m2_path, index=False)
    print(f"  Saved {ens_m1_path}")
    print(f"  Saved {ens_m2_path}")

    # Raw ensemble predictions (one column per member)
    eval_years = df_eval["Year"].values
    eval_agent_ids = df_eval["AgentID"].values
    base_cols = {"Year": eval_years, "AgentID": eval_agent_ids, "Obs": y_eval}

    meta = pd.DataFrame(base_cols)
    members_m1 = pd.DataFrame(preds_m1_ens.T,
                               columns=[f"member_{i}" for i in range(preds_m1_ens.shape[0])])
    df_raw_m1 = pd.concat([meta, members_m1], axis=1)
    raw_m1_path = RESULTS_DIR / "cluster2_ensemble_predictions_M1.csv"
    df_raw_m1.to_csv(raw_m1_path, index=False)
    print(f"  Saved {raw_m1_path}")

    members_m2 = pd.DataFrame(preds_m2_ens.T,
                               columns=[f"member_{i}" for i in range(preds_m2_ens.shape[0])])
    df_raw_m2 = pd.concat([meta, members_m2], axis=1)
    raw_m2_path = RESULTS_DIR / "cluster2_ensemble_predictions_M2.csv"
    df_raw_m2.to_csv(raw_m2_path, index=False)
    print(f"  Saved {raw_m2_path}")

    # Prediction intervals

    pi_rows = []
    for j in range(len(y_eval)):
        pi_rows.append({
            "Year": eval_years[j],
            "AgentID": eval_agent_ids[j],
            "Obs": y_eval[j],
            "M1_med": np.median(preds_m1_ens[:, j]),
            "M1_p05": np.percentile(preds_m1_ens[:, j], 5),
            "M1_p95": np.percentile(preds_m1_ens[:, j], 95),
            "M2_med": np.median(preds_m2_ens[:, j]),
            "M2_p05": np.percentile(preds_m2_ens[:, j], 5),
            "M2_p95": np.percentile(preds_m2_ens[:, j], 95),
        })
    df_pi = pd.DataFrame(pi_rows)
    pi_path = RESULTS_DIR / "cluster2_prediction_intervals.csv"
    df_pi.to_csv(pi_path, index=False)
    print(f"  Saved {pi_path}")

    # Fan plot
    plot_prediction_intervals(df_pi, RESULTS_DIR / "cluster2_prediction_intervals.pdf")

    # Ensemble summary: P(M2 better), delta_rmse distribution
    rmse_m1_arr = df_ens_m1["overall_RMSE"].values
    rmse_m2_arr = df_ens_m2["overall_RMSE"].values
    delta_arr = rmse_m1_arr - rmse_m2_arr  # positive = M2 better

    p_m2_better = np.mean(delta_arr > 0)
    summary = {
        "P_M2_better": p_m2_better,
        "delta_rmse_mean": np.mean(delta_arr),
        "delta_rmse_median": np.median(delta_arr),
        "delta_rmse_IQR_lo": np.percentile(delta_arr, 25),
        "delta_rmse_IQR_hi": np.percentile(delta_arr, 75),
        "delta_rmse_p05": np.percentile(delta_arr, 5),
        "delta_rmse_p95": np.percentile(delta_arr, 95),
        "M1_RMSE_median": np.median(rmse_m1_arr),
        "M2_RMSE_median": np.median(rmse_m2_arr),
    }
    summary_path = RESULTS_DIR / "cluster2_ensemble_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Saved {summary_path}")
    print(f"\n  P(M2 better than M1) = {p_m2_better:.3f}")
    print(f"  Δ RMSE (M1−M2): median={np.median(delta_arr):.2f}, "
          f"IQR=[{np.percentile(delta_arr, 25):.2f}, {np.percentile(delta_arr, 75):.2f}]")

    return p_m2_better


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pooled XGBoost ABM for Cluster 2")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--point-only", action="store_true",
                            help="Stop after point estimates (skip bootstrap ensemble)")
    mode_group.add_argument("--ensemble-only", action="store_true",
                            help="Load saved params and run ensemble only (skip Optuna)")
    mode_group.add_argument("--two-regime-point", action="store_true",
                            help="Two-regime point estimates (Optuna tune + predict, no ensemble)")
    mode_group.add_argument("--two-regime-ensemble", action="store_true",
                            help="Two-regime ensemble (load saved params, bootstrap PI)")
    mode_group.add_argument("--post-cp-ensemble", action="store_true",
                            help="Post-CP jitter ensemble (hyperparameter jitter only, no data bootstrap)")
    args = parser.parse_args()

    if args.two_regime_point or args.two_regime_ensemble or args.post_cp_ensemble:
        print("Loading and aggregating data...")
        raw = load_agent_data(AGENT_IDS)
        annual = aggregate_to_annual(raw)
        print(f"  Annual dataset: {len(annual)} rows "
              f"({annual['Year'].nunique()} years × {len(AGENT_IDS)} agents)")
        if args.two_regime_point:
            run_two_regime_point(annual)
        elif args.two_regime_ensemble:
            run_two_regime_ensemble(annual)
        else:
            run_post_cp_jitter_ensemble(annual)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED)

    # ── 1. Load + aggregate ──────────────────────────────────────────────────
    print("Loading and aggregating data...")
    raw = load_agent_data(AGENT_IDS)
    annual = aggregate_to_annual(raw)
    print(f"  Annual dataset: {len(annual)} rows "
          f"({annual['Year'].nunique()} years × {len(AGENT_IDS)} agents)")
    annual_path = RESULTS_DIR / "cluster2_annual_data.csv"
    annual.to_csv(annual_path, index=False)
    print(f"  Saved {annual_path}")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    df_m1_train, df_eval = split_data(annual, train_end_year=CP_YEAR - 1, eval_start_year=CP_YEAR + 1)
    df_m2_train, _ = split_data(annual, train_end_year=CP_YEAR, eval_start_year=CP_YEAR + 1)
    df_cp_year = annual[annual["Year"] == CP_YEAR].copy()

    print(f"  M1 train: {len(df_m1_train)} obs (≤{CP_YEAR-1})")
    print(f"  M2 train: {len(df_m2_train)} obs (≤{CP_YEAR})")
    print(f"  Eval:     {len(df_eval)} obs ({CP_YEAR+1}–2020)")

    # ── Ensemble-only shortcut ──────────────────────────────────────────────
    if args.ensemble_only:
        print("\n--ensemble-only: loading saved params, skipping Optuna tuning.")
        best_params_m1 = load_tuned_params(PARAMS_M1_PATH)
        best_params_m2 = load_tuned_params(PARAMS_M2_PATH)
        print(f"  M1 params: {best_params_m1}")
        print(f"  M2 params: {best_params_m2}")

        X_eval, y_eval = prepare_features(df_eval)

        print(f"\n--- Ensemble Uncertainty (N={N_BOOT}) ---")
        print("  Running M1 ensemble...")
        preds_m1_ens, metrics_m1_ens = run_ensemble(
            df_m1_train, df_eval, N_BOOT,
            np.random.default_rng(rng.integers(0, 2**31)),
            base_params=best_params_m1
        )
        print("  Running M2 ensemble...")
        preds_m2_ens, metrics_m2_ens = run_ensemble(
            df_m2_train, df_eval, N_BOOT,
            np.random.default_rng(rng.integers(0, 2**31)),
            force_include_year=CP_YEAR, base_params=best_params_m2
        )

        p_m2_better = save_ensemble_results(
            preds_m1_ens, metrics_m1_ens, preds_m2_ens, metrics_m2_ens,
            df_eval, y_eval
        )

        print("\n" + "=" * 60)
        print("ENSEMBLE-ONLY SUMMARY")
        print("=" * 60)
        print(f"  P(M2 better than M1) = {p_m2_better:.3f} ({N_BOOT} members)")
        print("=" * 60)
        return

    # ── 3. Point estimates ───────────────────────────────────────────────────
    print("\n--- Point Estimates ---")
    X_m1_train, y_m1_train = prepare_features(df_m1_train)
    X_m2_train, y_m2_train = prepare_features(df_m2_train)
    X_eval, y_eval = prepare_features(df_eval)
    eval_agents = df_eval["AgentID"].values

    print("  Tuning M1 (Stationary)...")
    model_m1, best_params_m1 = optuna_tune(X_m1_train, y_m1_train, seed=MASTER_SEED)
    print("  Tuning M2 (CP-Aware)...")
    model_m2, best_params_m2 = optuna_tune(X_m2_train, y_m2_train, seed=MASTER_SEED + 1)

    # Save tuned models + params for later --ensemble-only runs
    save_tuned_model(model_m1, best_params_m1, MODEL_M1_PATH, PARAMS_M1_PATH)
    save_tuned_model(model_m2, best_params_m2, MODEL_M2_PATH, PARAMS_M2_PATH)

    pred_m1 = model_m1.predict(X_eval)
    pred_m2 = model_m2.predict(X_eval)

    # Training predictions (in-sample fit)
    pred_m1_train = model_m1.predict(X_m1_train)
    pred_m2_train = model_m2.predict(X_m2_train)

    # Save training predictions
    df_m1_tp = df_m1_train[["Year", "AgentID"]].copy()
    df_m1_tp["Observed"] = y_m1_train
    df_m1_tp["Pred"] = pred_m1_train
    df_m1_tp.to_csv(RESULTS_DIR / "cluster2_train_predictions_M1.csv", index=False)
    print(f"  Saved {RESULTS_DIR / 'cluster2_train_predictions_M1.csv'}")

    df_m2_tp = df_m2_train[["Year", "AgentID"]].copy()
    df_m2_tp["Observed"] = y_m2_train
    df_m2_tp["Pred"] = pred_m2_train
    df_m2_tp.to_csv(RESULTS_DIR / "cluster2_train_predictions_M2.csv", index=False)
    print(f"  Saved {RESULTS_DIR / 'cluster2_train_predictions_M2.csv'}")

    metrics_m1 = compute_full_metrics(y_eval, pred_m1, eval_agents)
    metrics_m2 = compute_full_metrics(y_eval, pred_m2, eval_agents)

    # Diagnostic: M1 prediction on CP year
    X_cp, y_cp = prepare_features(df_cp_year)
    pred_cp = model_m1.predict(X_cp)
    cp_rmse = np.sqrt(mean_squared_error(y_cp, pred_cp))
    print(f"  Diagnostic: M1 RMSE on {CP_YEAR} = {cp_rmse:.2f}")

    # Metrics summary
    print(f"\n  M1 overall RMSE: {metrics_m1['overall_RMSE']:.2f}")
    print(f"  M2 overall RMSE: {metrics_m2['overall_RMSE']:.2f}")
    delta_rmse = metrics_m1["overall_RMSE"] - metrics_m2["overall_RMSE"]
    pct_improve = delta_rmse / metrics_m1["overall_RMSE"] * 100
    print(f"  Delta RMSE (M1−M2): {delta_rmse:.2f}  ({pct_improve:+.1f}%)")

    # Save predictions
    df_pred = df_eval[["Year", "AgentID"]].copy()
    df_pred["Observed"] = y_eval
    df_pred["Pred_M1"] = pred_m1
    df_pred["Pred_M2"] = pred_m2
    pred_path = RESULTS_DIR / "cluster2_predictions_point_estimates.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"  Saved {pred_path}")

    # Save metrics
    metrics_rows = []
    for label, m in [("M1_Stationary", metrics_m1), ("M2_CP_Aware", metrics_m2)]:
        row = {"Model": label}
        row.update(m)
        metrics_rows.append(row)
    # Add improvement row
    metrics_rows.append({
        "Model": "M2_improvement",
        "overall_RMSE": delta_rmse,
        "overall_MAE": metrics_m1["overall_MAE"] - metrics_m2["overall_MAE"],
        "overall_R2": metrics_m2["overall_R2"] - metrics_m1["overall_R2"],
        "overall_Bias": None,
        "pct_improve_RMSE": pct_improve,
    })
    metrics_path = RESULTS_DIR / "cluster2_metrics_point_estimates.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"  Saved {metrics_path}")

    # Save Optuna best params
    params_rows = []
    for label, bp in [("M1_Stationary", best_params_m1), ("M2_CP_Aware", best_params_m2)]:
        row = {"Model": label}
        row.update(bp)
        params_rows.append(row)
    params_path = RESULTS_DIR / "cluster2_optuna_best_params.csv"
    pd.DataFrame(params_rows).to_csv(params_path, index=False)
    print(f"  Saved {params_path}")

    # Plot
    plot_point_estimates(
        df_m1_train, y_m1_train, pred_m1_train,
        df_m2_train, y_m2_train, pred_m2_train,
        df_eval, y_eval, pred_m1, pred_m2,
        CP_YEAR, RESULTS_DIR / "cluster2_predictions_point_estimates.pdf",
    )

    if args.point_only:
        print("\n--point-only: skipping ensemble. Done.")
        return

    # ── 4. Ensemble uncertainty ──────────────────────────────────────────────
    print(f"\n--- Ensemble Uncertainty (N={N_BOOT}) ---")
    print("  Running M1 ensemble...")
    preds_m1_ens, metrics_m1_ens = run_ensemble(
        df_m1_train, df_eval, N_BOOT, np.random.default_rng(rng.integers(0, 2**31)),
        base_params=best_params_m1
    )
    print("  Running M2 ensemble...")
    preds_m2_ens, metrics_m2_ens = run_ensemble(
        df_m2_train, df_eval, N_BOOT, np.random.default_rng(rng.integers(0, 2**31)),
        force_include_year=CP_YEAR, base_params=best_params_m2
    )

    p_m2_better = save_ensemble_results(
        preds_m1_ens, metrics_m1_ens, preds_m2_ens, metrics_m2_ens,
        df_eval, y_eval
    )

    # ── 5. Changepoint-year sensitivity (constrained block bootstrap) ────────
    print("\n--- Changepoint-Year Sensitivity (Constrained Bootstrap) ---")
    cp_candidates = [2005, 2007, 2009]
    # Common eval: start from max(candidates) + 1 = 2010
    common_eval_start = max(cp_candidates) + 1
    df_common_eval = annual[annual["Year"] >= common_eval_start].copy()
    print(f"  Common eval window: {common_eval_start}–2020 "
          f"({len(df_common_eval)} obs)")

    sensitivity_rows = []
    pi_data = {}  # CP year → DataFrame for combined fan plot
    for cp in cp_candidates:
        print(f"\n  CP={cp}:")
        result = run_cp(
            cp, annual, df_common_eval,
            np.random.default_rng(rng.integers(0, 2**31)),
            n_boot=N_SENS_BOOT, base_cp=CP_YEAR, block_size=BLOCK_SIZE,
        )
        sensitivity_rows.append(result["summary"])
        pi_data[cp] = result["pi_df"]

    # Combined 2×3 fan plot
    plot_sensitivity_pi_grid(
        pi_data, cp_candidates,
        RESULTS_DIR / "cluster2_sensitivity_pi_grid.pdf"
    )

    sens_path = RESULTS_DIR / "cluster2_cp_year_sensitivity_summary.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sens_path, index=False)
    print(f"\n  Saved {sens_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Cluster 2 XGBoost Changepoint Benefit Analysis")
    print("=" * 60)
    print(f"  Changepoint year:   {CP_YEAR}")
    print(f"  M1 RMSE (point):    {metrics_m1['overall_RMSE']:.2f}")
    print(f"  M2 RMSE (point):    {metrics_m2['overall_RMSE']:.2f}")
    print(f"  Improvement:        {delta_rmse:.2f} ({pct_improve:+.1f}%)")
    print(f"  P(M2 better):       {p_m2_better:.3f} ({N_BOOT} bootstrap members)")
    print(f"  Sensitivity (CP ∈ {cp_candidates}):")
    for row in sensitivity_rows:
        print(f"    CP={row['CP_Year']}: M2 RMSE={row['M2_RMSE']:.2f}, "
              f"PI width={row['M2_PI_width_mean']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
