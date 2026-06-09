#!/usr/bin/env python3
"""
Agent 12 Pre-CP Extrapolation — Non-Stationarity Test.

Trains a "stationary" model on Agent 12 pre-changepoint data (1993–2003,
11 annual rows) and predicts the entire post-CP period (2004–2020, 17 years)
with uncertainty quantification via 200-member bootstrap ensemble.

Tests whether a model trained before the structural break can predict
irrigation behavior afterward — a direct test of the non-stationarity
hypothesis.
"""

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
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ────────────────────────────────────────────────────────────────
AGENT_ID = 12
CP_YEAR = 2004
TRAIN_YEARS = list(range(1993, 2004))   # 1993–2003 (11 rows)
PRED_YEARS = list(range(2004, 2021))    # 2004–2020 (17 rows, all OOS)
MASTER_SEED = 42
N_OPTUNA_TRIALS = 100
N_BOOT = 200
BLOCK_SIZE = 3
ANCHOR_YEARS = [2001, 2002, 2003]       # last 3 of training

BASE_FEATURES = ["Precipitation", "Temperature", "Corn", "Wheat",
                 "Soybeans", "Sorghum", "Diesel"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "two_regime_c1" / "agent12_precp"

# Aggressive regularization search space for n=11
AGENT12_SEARCH_SPACE = {
    "max_depth": (1, 2),
    "min_child_weight": (3, 10),
    "gamma": (0.5, 5.0),
    "subsample": (0.7, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (5.0, 50.0),
    "reg_alpha": (0.0, 10.0),
    "learning_rate": (0.01, 0.1),
    "n_estimators": (20, 200),
}

# Relative jitter widths for bootstrap ensemble (from run_xgboost_abm.py)
JITTER_WIDTHS = {
    "max_depth": {"additive": 1, "clamp": (2, 8)},
    "learning_rate": {"mult_range": (0.5, 2.0)},
    "subsample": {"additive": 0.15, "clamp": (0.5, 1.0)},
    "colsample_bytree": {"additive": 0.15, "clamp": (0.5, 1.0)},
    "reg_lambda": {"mult_range": (0.3, 3.0)},
    "reg_alpha": {"mult_range": (0.3, 3.0)},
    "min_child_weight": {"additive": 2, "clamp": (1, 10)},
}


# ── Data Pipeline ────────────────────────────────────────────────────────────
def load_and_prepare():
    """Load Agent 12 data, aggregate to annual, split at CP."""
    fp = DATA_DIR / f"agentdata_{AGENT_ID}.csv"
    df = pd.read_csv(fp)

    # Aggregate monthly (May-Oct) to annual
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
    annual = annual.sort_values("Year").reset_index(drop=True)

    df_train = annual[annual["Year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    df_pred = annual[annual["Year"].isin(PRED_YEARS)].copy().reset_index(drop=True)

    print(f"Agent {AGENT_ID} Pre-CP Extrapolation:")
    print(f"  Train: {len(df_train)} rows (years {df_train['Year'].min()}-"
          f"{df_train['Year'].max()})")
    print(f"  Pred:  {len(df_pred)} rows (years {df_pred['Year'].min()}-"
          f"{df_pred['Year'].max()})")
    print(f"  Train y range: [{df_train['Irrigation_Depth'].min():.1f}, "
          f"{df_train['Irrigation_Depth'].max():.1f}] mm")

    return df_train, df_pred


# ── LOOCV Infrastructure ────────────────────────────────────────────────────
def loocv_r2(model_factory, X, y, transform="identity", **fit_kwargs):
    """Leave-one-out CV returning R² (computed from all LOO predictions)."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, val_idx in loo.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if transform == "log1p":
            y_tr_t = np.log1p(y_tr)
        else:
            y_tr_t = y_tr

        model = model_factory()
        model.fit(X_tr, y_tr_t, **fit_kwargs)
        pred = model.predict(X_va)

        if transform == "log1p":
            pred = np.clip(pred, None, 20)
            pred = np.expm1(pred)

        preds[val_idx[0]] = pred[0]

    return r2_score(y, preds)


# ── Model Training Functions ────────────────────────────────────────────────
def train_ridge(X_tv, y_tv, transform="identity"):
    """Train Ridge with LOOCV alpha selection."""
    alphas = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]
    best_alpha = None
    best_r2 = -np.inf

    for alpha in alphas:
        def factory(a=alpha):
            return Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=a))])
        r2 = loocv_r2(factory, X_tv, y_tv, transform)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    model = Pipeline([("scaler", StandardScaler()),
                      ("ridge", Ridge(alpha=best_alpha))])
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model.fit(X_tv, y_fit)

    return model, {"alpha": best_alpha}, best_r2


def train_elasticnet(X_tv, y_tv, transform="identity"):
    """Train ElasticNet with LOOCV alpha × l1_ratio selection."""
    alphas = [1e-4, 1e-3, 0.01, 0.1, 1, 10]
    l1_ratios = [0.05, 0.2, 0.5, 0.8, 0.95]
    best_params = {}
    best_r2 = -np.inf

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            def factory(a=alpha, l=l1_ratio):
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("enet", ElasticNet(alpha=a, l1_ratio=l, max_iter=10000)),
                ])
            r2 = loocv_r2(factory, X_tv, y_tv, transform)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(max_iter=10000, **best_params)),
    ])
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model.fit(X_tv, y_fit)

    return model, best_params, best_r2


def train_xgb_gbtree(X_tv, y_tv, transform="identity", seed=MASTER_SEED):
    """Train XGBoost gbtree with Optuna LOOCV tuning."""
    ss = AGENT12_SEARCH_SPACE
    sampler = optuna.samplers.TPESampler(seed=seed)
    loo = LeaveOneOut()

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
            "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *ss["learning_rate"], log=True),
            "n_estimators": trial.suggest_int(
                "n_estimators", *ss["n_estimators"]),
            "objective": "reg:squarederror",
            "verbosity": 0,
        }

        loo_preds = np.zeros(len(y_tv))
        best_iters = []
        for train_idx, val_idx in loo.split(X_tv):
            X_tr, X_va = X_tv[train_idx], X_tv[val_idx]
            y_tr, y_va = y_tv[train_idx], y_tv[val_idx]

            if transform == "log1p":
                y_tr_t = np.log1p(y_tr)
                y_va_t = np.log1p(y_va)
            else:
                y_tr_t = y_tr
                y_va_t = y_va

            xgb_params = dict(params)
            xgb_params["early_stopping_rounds"] = 5
            model = XGBRegressor(**xgb_params)
            model.fit(X_tr, y_tr_t,
                      eval_set=[(X_va, y_va_t)],
                      verbose=False)

            best_iter = getattr(model, "best_iteration",
                                params["n_estimators"]) + 1
            best_iters.append(best_iter)

            pred = model.predict(X_va)
            if transform == "log1p":
                pred = np.clip(pred, None, 20)
                pred = np.expm1(pred)

            loo_preds[val_idx[0]] = pred[0]

        trial.set_user_attr("fold_n_estimators", best_iters)
        return -r2_score(y_tv, loo_preds)  # minimize negative R²

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_trial.params
    best_params["objective"] = "reg:squarederror"
    best_params["verbosity"] = 0

    # Final n_estimators from median of LOOCV best iterations
    fold_n_est = study.best_trial.user_attrs["fold_n_estimators"]
    best_params["n_estimators"] = max(10, int(np.median(fold_n_est)))
    best_params.pop("early_stopping_rounds", None)

    cv_r2 = -study.best_value
    print(f"    Optuna best LOOCV R²: {cv_r2:.3f}")
    print(f"    Best params: {best_params}")

    # Refit on full trainval
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model = XGBRegressor(**best_params)
    model.fit(X_tv, y_fit)

    return model, best_params, cv_r2


def train_xgb_gblinear(X_tv, y_tv, transform="identity"):
    """Train XGBoost gblinear with LOOCV grid search."""
    lambdas = [0.1, 1.0, 10.0, 50.0]
    alphas = [0.0, 1.0, 10.0]
    best_params = {}
    best_r2 = -np.inf

    for reg_lambda in lambdas:
        for reg_alpha in alphas:
            def factory(rl=reg_lambda, ra=reg_alpha):
                return XGBRegressor(
                    booster="gblinear", objective="reg:squarederror",
                    verbosity=0, n_estimators=200,
                    reg_lambda=rl, reg_alpha=ra)

            r2 = loocv_r2(factory, X_tv, y_tv, transform)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"reg_lambda": reg_lambda, "reg_alpha": reg_alpha}

    model = XGBRegressor(
        booster="gblinear", objective="reg:squarederror",
        verbosity=0, n_estimators=200, **best_params)
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model.fit(X_tv, y_fit)

    return model, best_params, best_r2


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R2, Bias."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


# ── Model Benchmarking ──────────────────────────────────────────────────────
def run_all_models(df_train, df_pred):
    """Train all model × transform combos (A_base only), return leaderboard."""
    model_trainers = {
        "Ridge": train_ridge,
        "ElasticNet": train_elasticnet,
        "XGBoost_gbtree": train_xgb_gbtree,
        "XGBoost_gblinear": train_xgb_gblinear,
    }
    transforms = ["identity", "log1p"]

    X_tv = df_train[BASE_FEATURES].values
    y_tv = df_train["Irrigation_Depth"].values.copy()
    X_pred = df_pred[BASE_FEATURES].values
    y_pred = df_pred["Irrigation_Depth"].values.copy()

    results = []
    best_cv_r2 = -np.inf
    best_entry = None

    for transform in transforms:
        for model_name, trainer in model_trainers.items():
            label = f"{model_name}_A_base_{transform}"
            print(f"\n--- {label} ---")

            model, params, cv_r2 = trainer(X_tv, y_tv, transform)

            # Predict train + OOS
            pred_tv = model.predict(X_tv)
            pred_oos = model.predict(X_pred)

            if transform == "log1p":
                pred_tv = np.clip(pred_tv, None, 20)
                pred_tv = np.expm1(pred_tv)
                pred_oos = np.clip(pred_oos, None, 20)
                pred_oos = np.expm1(pred_oos)

            pred_tv = np.clip(pred_tv, 0, None)
            pred_oos = np.clip(pred_oos, 0, None)

            train_metrics = compute_metrics(y_tv, pred_tv)
            pred_metrics = compute_metrics(y_pred, pred_oos)

            # Sanity check
            max_obs = max(y_tv.max(), y_pred.max())
            sane = pred_oos.max() <= 2 * max_obs

            entry = {
                "model": model_name,
                "feature_set": "A_base",
                "transform": transform,
                "cv_r2": cv_r2,
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "pred_RMSE": pred_metrics["RMSE"],
                "pred_MAE": pred_metrics["MAE"],
                "pred_R2": pred_metrics["R2"],
                "pred_Bias": pred_metrics["Bias"],
                "sane": sane,
                "params": json.dumps(params),
            }
            results.append(entry)

            print(f"  CV R²={cv_r2:.3f}")
            print(f"  Train R²={train_metrics['R2']:.3f}  "
                  f"RMSE={train_metrics['RMSE']:.1f}")
            print(f"  Pred  R²={pred_metrics['R2']:.3f}  "
                  f"RMSE={pred_metrics['RMSE']:.1f}  "
                  f"MAE={pred_metrics['MAE']:.1f}")

            if sane and cv_r2 > best_cv_r2:
                best_cv_r2 = cv_r2
                best_entry = {
                    "label": label,
                    "model": model,
                    "params": params,
                    "model_name": model_name,
                    "feature_set": "A_base",
                    "transform": transform,
                    "cols": BASE_FEATURES,
                    "pred_tv": pred_tv,
                    "pred_oos": pred_oos,
                    "train_metrics": train_metrics,
                    "pred_metrics": pred_metrics,
                    "cv_r2": cv_r2,
                }

    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values("cv_r2", ascending=False).reset_index(drop=True)

    return leaderboard, best_entry


# ── Bootstrap Ensemble ───────────────────────────────────────────────────────
def constrained_block_bootstrap_years(prefix_years, anchor_years, block_size, rng):
    """Moving-block bootstrap on prefix years with a deterministic anchor tail.

    Args:
        prefix_years: years to resample (e.g. 1993–2000).
        anchor_years: years pinned at the tail (e.g. [2001, 2002, 2003]).
        block_size: contiguous block size.
        rng: numpy random Generator.

    Returns:
        List of ints: resampled prefix + anchor tail.
    """
    pre = np.sort(np.asarray(prefix_years))
    anchor = list(anchor_years)
    n = len(pre)

    if n < block_size:
        raise ValueError(
            f"prefix_years length ({n}) must be >= block_size ({block_size})")

    n_blocks = int(np.ceil(n / block_size))
    sampled = []
    for _ in range(n_blocks):
        start = rng.integers(0, n - block_size + 1)
        sampled.extend(pre[start:start + block_size].tolist())
    bootstrapped_pre = sampled[:n]

    return bootstrapped_pre + anchor


def bootstrap_panel_by_year_sequence(df, year_sequence):
    """Build a panel DataFrame by selecting rows for each year in sequence."""
    frames = []
    for yr in year_sequence:
        sub = df[df["Year"] == yr]
        if len(sub) == 0:
            raise ValueError(f"Year {yr} not found in DataFrame")
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def sample_hyperparams_gbtree(rng, base_params):
    """Jitter XGBoost gbtree hyperparams centered on tuned values."""
    params = {
        "n_estimators": base_params.get("n_estimators", 200),
        "objective": "reg:squarederror",
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


def sample_hyperparams_gblinear(rng, base_params):
    """Jitter XGBoost gblinear hyperparams (reg_lambda, reg_alpha only)."""
    params = {
        "booster": "gblinear",
        "n_estimators": 200,
        "objective": "reg:squarederror",
        "verbosity": 0,
        "reg_lambda": float(base_params["reg_lambda"] * rng.uniform(0.3, 3.0)),
        "reg_alpha": float(base_params["reg_alpha"] * rng.uniform(0.3, 3.0)
                           if base_params.get("reg_alpha", 0) > 0
                           else rng.uniform(0, 1.0)),
        "random_state": int(rng.integers(0, 2**31)),
    }
    return params


def sample_hyperparams_ridge(rng, base_params):
    """Jitter Ridge alpha multiplicatively."""
    return {"alpha": float(base_params["alpha"] * rng.uniform(0.3, 3.0))}


def sample_hyperparams_elasticnet(rng, base_params):
    """Jitter ElasticNet alpha and l1_ratio."""
    return {
        "alpha": float(base_params["alpha"] * rng.uniform(0.3, 3.0)),
        "l1_ratio": float(np.clip(
            base_params["l1_ratio"] + rng.uniform(-0.15, 0.15), 0.01, 0.99)),
    }


def _build_model(model_name, params, transform):
    """Instantiate a fresh model from name + params."""
    if model_name == "Ridge":
        return Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=params["alpha"]))])
    elif model_name == "ElasticNet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=params["alpha"],
                                l1_ratio=params["l1_ratio"],
                                max_iter=10000)),
        ])
    elif model_name == "XGBoost_gbtree":
        p = {k: v for k, v in params.items()
             if k not in ("early_stopping_rounds",)}
        return XGBRegressor(**p)
    elif model_name == "XGBoost_gblinear":
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _jitter_params(rng, model_name, base_params):
    """Dispatch to model-specific jitter function."""
    if model_name == "XGBoost_gbtree":
        return sample_hyperparams_gbtree(rng, base_params)
    elif model_name == "XGBoost_gblinear":
        return sample_hyperparams_gblinear(rng, base_params)
    elif model_name == "Ridge":
        return sample_hyperparams_ridge(rng, base_params)
    elif model_name == "ElasticNet":
        return sample_hyperparams_elasticnet(rng, base_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_bootstrap_ensemble(df_train, df_pred, best_entry):
    """200-member bootstrap ensemble with data resampling + hyperparam jitter."""
    model_name = best_entry["model_name"]
    base_params = best_entry["params"]
    transform = best_entry["transform"]

    prefix_years = sorted(set(TRAIN_YEARS) - set(ANCHOR_YEARS))
    train_years_all = sorted(TRAIN_YEARS)

    X_pred = df_pred[BASE_FEATURES].values
    y_pred = df_pred["Irrigation_Depth"].values
    max_obs = max(df_train["Irrigation_Depth"].max(),
                  df_pred["Irrigation_Depth"].max())
    clip_hi = 2 * max_obs

    rng = np.random.default_rng(MASTER_SEED + 200)
    n_pred = len(y_pred)
    preds_matrix = np.zeros((N_BOOT, n_pred))

    print(f"\nRunning {N_BOOT}-member bootstrap ensemble ({model_name})...")
    for i in range(N_BOOT):
        member_rng = np.random.default_rng(rng.integers(0, 2**31))

        # Block bootstrap: resample prefix, pin anchor
        boot_years = constrained_block_bootstrap_years(
            prefix_years, ANCHOR_YEARS, BLOCK_SIZE, member_rng)

        # Build bootstrap training set
        df_boot = bootstrap_panel_by_year_sequence(df_train, boot_years)
        X_boot = df_boot[BASE_FEATURES].values
        y_boot = df_boot["Irrigation_Depth"].values.copy()

        # Jitter hyperparams
        jittered = _jitter_params(member_rng, model_name, base_params)
        model = _build_model(model_name, jittered, transform)

        # Train
        y_fit = np.log1p(y_boot) if transform == "log1p" else y_boot
        model.fit(X_boot, y_fit)

        # Predict
        pred = model.predict(X_pred)
        if transform == "log1p":
            pred = np.clip(pred, None, 20)
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, clip_hi)

        preds_matrix[i] = pred

        if (i + 1) % 50 == 0:
            print(f"  Member {i + 1}/{N_BOOT} done")

    return preds_matrix


# ── Evaluate Ensemble ────────────────────────────────────────────────────────
def evaluate_ensemble(preds_matrix, y_obs):
    """Compute ensemble summary statistics."""
    p05 = np.percentile(preds_matrix, 5, axis=0)
    p50 = np.percentile(preds_matrix, 50, axis=0)
    p95 = np.percentile(preds_matrix, 95, axis=0)

    # Metrics on median prediction
    median_metrics = compute_metrics(y_obs, p50)

    # PI stats
    pi_width = p95 - p05
    coverage = np.mean((y_obs >= p05) & (y_obs <= p95))

    # Member RMSE distribution
    member_rmses = [np.sqrt(mean_squared_error(y_obs, preds_matrix[i]))
                    for i in range(preds_matrix.shape[0])]
    rmse_iqr = np.percentile(member_rmses, 75) - np.percentile(member_rmses, 25)

    summary = {
        "median_RMSE": median_metrics["RMSE"],
        "median_MAE": median_metrics["MAE"],
        "median_R2": median_metrics["R2"],
        "median_Bias": median_metrics["Bias"],
        "mean_PI_width": float(np.mean(pi_width)),
        "coverage_90": float(coverage),
        "member_RMSE_IQR": float(rmse_iqr),
        "member_RMSE_median": float(np.median(member_rmses)),
        "n_members": int(preds_matrix.shape[0]),
    }

    return p05, p50, p95, summary


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_precp_extrapolation(df_train, df_pred, pred_tv, p05, p50, p95,
                             train_metrics, pred_metrics, output_dir):
    """Single-panel plot: pre-CP train fit + post-CP extrapolation with PI."""
    saved_rc = matplotlib.rcParams.copy()
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
    })

    years_tv = df_train["Year"].values
    years_pred = df_pred["Year"].values
    y_tv = df_train["Irrigation_Depth"].values
    y_pred = df_pred["Irrigation_Depth"].values

    all_years = np.concatenate([years_tv, years_pred])
    all_obs = np.concatenate([y_tv, y_pred])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.subplots_adjust(top=0.85)

    # Training period shading
    ax.axvspan(years_tv.min() - 0.5, years_tv.max() + 0.5,
               color="lightgray", alpha=0.3, zorder=0, label="Training period")

    # CP line
    ax.axvline(CP_YEAR, color="gray", linestyle="--", linewidth=1,
               label="CP year", zorder=1)

    # PI band
    ax.fill_between(years_pred, p05, p95, color="tab:red", alpha=0.2,
                    zorder=2, label="90% PI")

    # Train predictions
    ax.plot(years_tv, pred_tv, "-", color="tab:blue", linewidth=1.5,
            label="Train pred", zorder=3)

    # Median prediction (OOS)
    ax.plot(years_pred, p50, "--", color="tab:red", linewidth=1.5,
            label="Median pred (OOS)", zorder=4)

    # Observed
    ax.plot(all_years, all_obs, "ko-", linewidth=1.2, markersize=4,
            label="Observed", zorder=5)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Irrigation Depth (mm)")
    # Metrics annotations
    ax.text(years_tv.min() + 0.3, 0.88,
            f"Train R\u00b2={train_metrics['R2']:.2f}, "
            f"RMSE={train_metrics['RMSE']:.2f}",
            transform=ax.get_xaxis_transform(),
            fontsize=12, color="0.2", ha="left")
    ax.text(0.98, 0.88,
            f"Pred R\u00b2={pred_metrics['R2']:.2f}, "
            f"RMSE={pred_metrics['RMSE']:.2f}",
            transform=ax.transAxes,
            fontsize=12, color="black", ha="right")

    ax.tick_params(length=4)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=False, fontsize=12)

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        fig.savefig(output_dir / f"agent12_precp_extrapolation.{fmt}",
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {output_dir}/agent12_precp_extrapolation.pdf/.png")

    matplotlib.rcParams.update(saved_rc)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    df_train, df_pred = load_and_prepare()

    # Step 2: Model benchmarking
    leaderboard, best_entry = run_all_models(df_train, df_pred)

    # Save leaderboard
    lb_path = OUTPUT_DIR / "leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)
    print(f"\nLeaderboard saved to {lb_path}")
    print(f"\n{'='*70}")
    print("TOP MODELS BY CV R² (descending):")
    print("="*70)
    top_cols = ["model", "transform", "cv_r2",
                "train_R2", "pred_RMSE", "pred_R2", "pred_Bias", "sane"]
    print(leaderboard[top_cols].to_string(index=False))

    if best_entry is None:
        print("\nWARNING: No sane model found!")
        return

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_entry['label']} (CV R²={best_entry['cv_r2']:.3f})")
    print(f"  Train: R²={best_entry['train_metrics']['R2']:.3f}  "
          f"RMSE={best_entry['train_metrics']['RMSE']:.1f}")
    print(f"  Pred:  R²={best_entry['pred_metrics']['R2']:.3f}  "
          f"RMSE={best_entry['pred_metrics']['RMSE']:.1f}  "
          f"MAE={best_entry['pred_metrics']['MAE']:.1f}")

    # Step 3: Save point predictions
    pred_df = pd.DataFrame({
        "Year": np.concatenate([df_train["Year"].values,
                                df_pred["Year"].values]),
        "Observed": np.concatenate([df_train["Irrigation_Depth"].values,
                                    df_pred["Irrigation_Depth"].values]),
        "Predicted": np.concatenate([best_entry["pred_tv"],
                                     best_entry["pred_oos"]]),
        "Split": ["train"] * len(df_train) + ["pred"] * len(df_pred),
    })
    pred_path = OUTPUT_DIR / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # Save params
    params_out = {
        "model": best_entry["model_name"],
        "feature_set": best_entry["feature_set"],
        "transform": best_entry["transform"],
        "features": best_entry["cols"],
        "params": {k: v for k, v in best_entry["params"].items()
                   if not isinstance(v, (np.integer, np.floating))
                   or True},
        "cv_r2": best_entry["cv_r2"],
        "train_metrics": best_entry["train_metrics"],
        "pred_metrics": best_entry["pred_metrics"],
    }
    # Ensure JSON serializable
    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    params_path = OUTPUT_DIR / "params.json"
    with open(params_path, "w") as f:
        json.dump(params_out, f, indent=2, default=_jsonify)
    print(f"Params saved to {params_path}")

    # Step 4: Bootstrap ensemble
    preds_matrix = run_bootstrap_ensemble(df_train, df_pred, best_entry)

    # Save full ensemble member predictions
    member_cols = {f"member_{i}": preds_matrix[i] for i in range(N_BOOT)}
    full_ens_df = pd.DataFrame({"Year": df_pred["Year"].values, **member_cols})
    full_ens_path = OUTPUT_DIR / "ensemble_members.csv"
    full_ens_df.to_csv(full_ens_path, index=False)
    print(f"Full ensemble members saved to {full_ens_path}")

    # Step 5: Evaluate ensemble
    y_pred = df_pred["Irrigation_Depth"].values
    p05, p50, p95, summary = evaluate_ensemble(preds_matrix, y_pred)

    # Save ensemble predictions
    ens_df = pd.DataFrame({
        "Year": df_pred["Year"].values,
        "Observed": y_pred,
        "p05": p05,
        "p50": p50,
        "p95": p95,
    })
    ens_path = OUTPUT_DIR / "ensemble_predictions.csv"
    ens_df.to_csv(ens_path, index=False)
    print(f"\nEnsemble predictions saved to {ens_path}")

    # Save ensemble summary
    summary_path = OUTPUT_DIR / "ensemble_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Ensemble summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print("ENSEMBLE SUMMARY:")
    print("="*70)
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Step 6: Plot
    # Use p50 as pred metrics for the plot
    pred_metrics_p50 = compute_metrics(y_pred, p50)
    plot_precp_extrapolation(
        df_train, df_pred,
        best_entry["pred_tv"],
        p05, p50, p95,
        best_entry["train_metrics"],
        pred_metrics_p50,
        OUTPUT_DIR,
    )

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
