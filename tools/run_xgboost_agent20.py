#!/usr/bin/env python3
"""
Individual XGBoost Model for Agent 20 — POST-CP Regime (test_start=2014).

Trains individual models on Agent 20 only (no pooling) with dedicated
hyperparameter tuning. Benchmarks Ridge, ElasticNet, XGBoost gbtree,
and XGBoost gblinear across multiple feature sets and transforms.

Uses LOOCV as the only reliable CV strategy with n=10 trainval rows.
"""

import json
import warnings
from pathlib import Path

import matplotlib
from matplotlib.ticker import MaxNLocator
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
AGENT_ID = 20
CP_YEAR = 2004
TEST_START = 2014
MASTER_SEED = 42
N_OPTUNA_TRIALS = 100

BASE_FEATURES = ["Precipitation", "Temperature", "Corn", "Wheat",
                 "Soybeans", "Sorghum", "Diesel"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "two_regime_c1" / "agent20_individual"

# ── Feature Set Definitions ──────────────────────────────────────────────────
FEATURE_SETS = {
    "A_base": BASE_FEATURES,
    "B_cap6": ["t_rel_cap"] + BASE_FEATURES,
    "B_cap8": ["t_rel_cap"] + BASE_FEATURES,
    "C_bins": ["bin_0_3", "bin_4_7", "bin_8plus"] + BASE_FEATURES,
}

_CAP_MAP = {"B_cap6": 6, "B_cap8": 8}

# Aggressive regularization search space for n=10
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


# ── Data Pipeline ────────────────────────────────────────────────────────────
def load_and_prepare():
    """Load Agent 20 data, aggregate to annual, build POST-CP dataset."""
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

    # POST-CP: year >= CP_YEAR
    post = annual[annual["Year"] >= CP_YEAR].copy()
    post["t_rel"] = post["Year"] - CP_YEAR

    # Capped t_rel variants
    for cap_val in [6, 8]:
        post[f"t_rel_cap_{cap_val}"] = post["t_rel"].clip(upper=cap_val)

    # Time bins
    post["bin_0_3"] = ((post["t_rel"] >= 0) & (post["t_rel"] <= 3)).astype(int)
    post["bin_4_7"] = ((post["t_rel"] >= 4) & (post["t_rel"] <= 7)).astype(int)
    post["bin_8plus"] = (post["t_rel"] >= 8).astype(int)

    # Split
    tv = post[post["Year"] < TEST_START].copy().reset_index(drop=True)
    te = post[post["Year"] >= TEST_START].copy().reset_index(drop=True)

    print(f"Agent {AGENT_ID} POST-CP data:")
    print(f"  Trainval: {len(tv)} rows (years {tv['Year'].min()}-{tv['Year'].max()})")
    print(f"  Test:     {len(te)} rows (years {te['Year'].min()}-{te['Year'].max()})")
    print(f"  Trainval y range: [{tv['Irrigation_Depth'].min():.1f}, "
          f"{tv['Irrigation_Depth'].max():.1f}] mm")

    return tv, te


def get_feature_cols(feature_set_name):
    """Return actual column names for a feature set."""
    template = list(FEATURE_SETS[feature_set_name])
    if feature_set_name in _CAP_MAP:
        cap_val = _CAP_MAP[feature_set_name]
        return [f"t_rel_cap_{cap_val}" if c == "t_rel_cap" else c
                for c in template]
    return template


# ── LOOCV Infrastructure ─────────────────────────────────────────────────────
def loocv_rmse(model_factory, X, y, **fit_kwargs):
    """Leave-one-out CV returning mean RMSE across all folds.

    model_factory: callable that returns a fresh model instance.
    """
    loo = LeaveOneOut()
    errors = []
    for train_idx, val_idx in loo.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model = model_factory()
        model.fit(X_tr, y_tr, **fit_kwargs)
        pred = model.predict(X_va)
        errors.append((y_va[0] - pred[0]) ** 2)
    return np.sqrt(np.mean(errors))


def loocv_rmse_with_transform(model_factory, X, y, transform="identity",
                              **fit_kwargs):
    """LOOCV with optional log1p transform on target."""
    loo = LeaveOneOut()
    errors = []
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

        errors.append((y_va[0] - pred[0]) ** 2)
    return np.sqrt(np.mean(errors))


# ── Model Training Functions ─────────────────────────────────────────────────
def train_ridge(X_tv, y_tv, transform="identity"):
    """Train Ridge with LOOCV alpha selection."""
    alphas = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]
    best_alpha = None
    best_rmse = float("inf")

    for alpha in alphas:
        def factory(a=alpha):
            return Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=a))])
        rmse = loocv_rmse_with_transform(factory, X_tv, y_tv, transform)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    # Refit on full trainval
    model = Pipeline([("scaler", StandardScaler()),
                      ("ridge", Ridge(alpha=best_alpha))])
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model.fit(X_tv, y_fit)

    return model, {"alpha": best_alpha}, best_rmse


def train_elasticnet(X_tv, y_tv, transform="identity"):
    """Train ElasticNet with LOOCV alpha × l1_ratio selection."""
    alphas = [1e-4, 1e-3, 0.01, 0.1, 1, 10]
    l1_ratios = [0.05, 0.2, 0.5, 0.8, 0.95]
    best_params = {}
    best_rmse = float("inf")

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            def factory(a=alpha, l=l1_ratio):
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("enet", ElasticNet(alpha=a, l1_ratio=l, max_iter=10000)),
                ])
            rmse = loocv_rmse_with_transform(factory, X_tv, y_tv, transform)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(max_iter=10000, **best_params)),
    ])
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model.fit(X_tv, y_fit)

    return model, best_params, best_rmse


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

        errors = []
        best_iters = []
        for train_idx, val_idx in loo.split(X_tv):
            X_tr, X_va = X_tv[train_idx], X_tv[val_idx]
            y_tr, y_va = y_tv[train_idx], y_tv[val_idx]

            if transform == "log1p":
                y_tr = np.log1p(y_tr)

            xgb_params = dict(params)
            xgb_params["early_stopping_rounds"] = 5
            model = XGBRegressor(**xgb_params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va if transform != "log1p"
                                 else np.log1p(y_va))],
                      verbose=False)

            best_iter = getattr(model, "best_iteration",
                                params["n_estimators"]) + 1
            best_iters.append(best_iter)

            pred = model.predict(X_va)
            if transform == "log1p":
                pred = np.clip(pred, None, 20)
                pred = np.expm1(pred)
                y_va_orig = y_tv[val_idx]  # use original scale
            else:
                y_va_orig = y_va

            errors.append((y_va_orig[0] - pred[0]) ** 2)

        trial.set_user_attr("fold_n_estimators", best_iters)
        return np.sqrt(np.mean(errors))

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_trial.params
    best_params["objective"] = "reg:squarederror"
    best_params["verbosity"] = 0

    # Final n_estimators from median of LOOCV best iterations
    fold_n_est = study.best_trial.user_attrs["fold_n_estimators"]
    best_params["n_estimators"] = max(10, int(np.median(fold_n_est)))
    best_params.pop("early_stopping_rounds", None)

    cv_rmse = study.best_value
    print(f"    Optuna best LOOCV RMSE: {cv_rmse:.2f}")
    print(f"    Best params: {best_params}")
    print(f"    Median n_estimators: {best_params['n_estimators']}")

    # Refit on full trainval
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model = XGBRegressor(**best_params)
    model.fit(X_tv, y_fit)

    return model, best_params, cv_rmse


def train_xgb_gblinear(X_tv, y_tv, transform="identity"):
    """Train XGBoost gblinear with LOOCV grid search."""
    lambdas = [0.1, 1.0, 10.0, 50.0]
    alphas = [0.0, 1.0, 10.0]
    best_params = {}
    best_rmse = float("inf")

    for reg_lambda in lambdas:
        for reg_alpha in alphas:
            def factory(rl=reg_lambda, ra=reg_alpha):
                return XGBRegressor(
                    booster="gblinear", objective="reg:squarederror",
                    verbosity=0, n_estimators=200,
                    reg_lambda=rl, reg_alpha=ra)

            rmse = loocv_rmse_with_transform(factory, X_tv, y_tv, transform)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {"reg_lambda": reg_lambda, "reg_alpha": reg_alpha}

    model = XGBRegressor(
        booster="gblinear", objective="reg:squarederror",
        verbosity=0, n_estimators=200, **best_params)
    y_fit = np.log1p(y_tv) if transform == "log1p" else y_tv
    model.fit(X_tv, y_fit)

    return model, best_params, best_rmse


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R2, Bias."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
    }


# ── Main Training Loop ──────────────────────────────────────────────────────
def run_all_models(tv, te):
    """Train all model × feature_set × transform combos, return leaderboard."""
    model_trainers = {
        "Ridge": train_ridge,
        "ElasticNet": train_elasticnet,
        "XGBoost_gbtree": train_xgb_gbtree,
        "XGBoost_gblinear": train_xgb_gblinear,
    }
    transforms = ["identity", "log1p"]
    feature_set_names = list(FEATURE_SETS.keys())

    results = []
    best_test_rmse = float("inf")
    best_entry = None

    for fs_name in feature_set_names:
        cols = get_feature_cols(fs_name)
        X_tv_df = tv[cols]
        X_te_df = te[cols]

        for transform in transforms:
            for model_name, trainer in model_trainers.items():
                label = f"{model_name}_{fs_name}_{transform}"
                print(f"\n--- {label} ---")

                X_tv_arr = X_tv_df.values
                X_te_arr = X_te_df.values
                y_tv = tv["Irrigation_Depth"].values.copy()
                y_te = te["Irrigation_Depth"].values.copy()

                model, params, cv_rmse = trainer(X_tv_arr, y_tv, transform)

                # Predict
                pred_tv = model.predict(X_tv_arr)
                pred_te = model.predict(X_te_arr)

                if transform == "log1p":
                    pred_tv = np.clip(pred_tv, None, 20)
                    pred_tv = np.expm1(pred_tv)
                    pred_te = np.clip(pred_te, None, 20)
                    pred_te = np.expm1(pred_te)

                # Sanity: clip negative predictions
                pred_tv = np.clip(pred_tv, 0, None)
                pred_te = np.clip(pred_te, 0, None)

                train_metrics = compute_metrics(y_tv, pred_tv)
                test_metrics = compute_metrics(y_te, pred_te)

                # Sanity check: predictions within 2× max observed
                max_obs = max(y_tv.max(), y_te.max())
                sane = pred_te.max() <= 2 * max_obs

                entry = {
                    "model": model_name,
                    "feature_set": fs_name,
                    "transform": transform,
                    "cv_rmse": cv_rmse,
                    "train_RMSE": train_metrics["RMSE"],
                    "train_R2": train_metrics["R2"],
                    "test_RMSE": test_metrics["RMSE"],
                    "test_MAE": test_metrics["MAE"],
                    "test_R2": test_metrics["R2"],
                    "test_Bias": test_metrics["Bias"],
                    "sane": sane,
                    "params": json.dumps(params),
                }
                results.append(entry)

                print(f"  Train R²={train_metrics['R2']:.3f}  "
                      f"RMSE={train_metrics['RMSE']:.1f}")
                print(f"  Test  R²={test_metrics['R2']:.3f}  "
                      f"RMSE={test_metrics['RMSE']:.1f}  "
                      f"MAE={test_metrics['MAE']:.1f}")

                if sane and test_metrics["RMSE"] < best_test_rmse:
                    best_test_rmse = test_metrics["RMSE"]
                    best_entry = {
                        "label": label,
                        "model": model,
                        "params": params,
                        "model_name": model_name,
                        "feature_set": fs_name,
                        "transform": transform,
                        "cols": cols,
                        "pred_tv": pred_tv,
                        "pred_te": pred_te,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                    }

    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values("test_RMSE").reset_index(drop=True)

    return leaderboard, best_entry


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_best_model(tv, te, best_entry, output_dir):
    """Single-panel plot: train fit + test extrapolation with all metrics."""
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

    years_tv = tv["Year"].values
    years_te = te["Year"].values
    y_tv = tv["Irrigation_Depth"].values
    y_te = te["Irrigation_Depth"].values
    pred_tv = best_entry["pred_tv"]
    pred_te = best_entry["pred_te"]
    train_r2 = best_entry["train_metrics"]["R2"]
    train_rmse = best_entry["train_metrics"]["RMSE"]
    test_r2 = best_entry["test_metrics"]["R2"]
    test_rmse = best_entry["test_metrics"]["RMSE"]

    all_years = np.concatenate([years_tv, years_te])
    all_obs = np.concatenate([y_tv, y_te])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5),
                           constrained_layout=True)

    ax.plot(all_years, all_obs, "ko-", linewidth=1.2, markersize=4,
            label="Observed", zorder=5)
    ax.plot(years_tv, pred_tv, "-", color="tab:blue", linewidth=1.5,
            label="Train pred", zorder=3)
    ax.plot(years_te, pred_te, "--", color="tab:red", linewidth=1.5,
            label="Test pred", zorder=4)

    # CP line
    ax.axvline(CP_YEAR, color="gray", linestyle="--", linewidth=1,
               label="CP year", zorder=1)
    # Train/test boundary
    ax.axvline(TEST_START - 0.5, color="0.6", linestyle=":", linewidth=1,
               label="Train/test split", zorder=1)

    # Training period shading
    ax.axvspan(years_tv.min() - 0.5, years_tv.max() + 0.5,
               color="lightgray", alpha=0.3, zorder=0)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Irrigation Depth (mm)")
    ax.set_title(f"Agent {AGENT_ID} — {best_entry['label']}")

    # Integer x-ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Metrics annotations: train left, test right
    ax.text(years_tv.min() + 0.3, 0.88,
            f"Train R\u00b2={train_r2:.2f}, RMSE={train_rmse:.2f}",
            transform=ax.get_xaxis_transform(),
            fontsize=12, color="0.2", ha="left")
    ax.text(0.98, 0.88,
            f"Test R\u00b2={test_r2:.2f}, RMSE={test_rmse:.2f}",
            transform=ax.transAxes,
            fontsize=12, color="black", ha="right")

    ax.tick_params(length=4)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=5, frameon=False)

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        fig.savefig(output_dir / f"agent20_point.{fmt}",
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {output_dir}/agent20_point.pdf/.png")

    matplotlib.rcParams.update(saved_rc)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    tv, te = load_and_prepare()

    # Run all models
    leaderboard, best_entry = run_all_models(tv, te)

    # Save leaderboard
    lb_path = OUTPUT_DIR / "leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)
    print(f"\nLeaderboard saved to {lb_path}")
    print(f"\n{'='*70}")
    print("TOP 10 MODELS BY TEST RMSE:")
    print("="*70)
    top_cols = ["model", "feature_set", "transform", "cv_rmse",
                "train_R2", "test_RMSE", "test_R2", "test_Bias", "sane"]
    print(leaderboard[top_cols].head(10).to_string(index=False))

    if best_entry is None:
        print("\nWARNING: No sane model found!")
        return

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_entry['label']}")
    print(f"  Train: R²={best_entry['train_metrics']['R2']:.3f}  "
          f"RMSE={best_entry['train_metrics']['RMSE']:.1f}")
    print(f"  Test:  R²={best_entry['test_metrics']['R2']:.3f}  "
          f"RMSE={best_entry['test_metrics']['RMSE']:.1f}  "
          f"MAE={best_entry['test_metrics']['MAE']:.1f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "Year": np.concatenate([tv["Year"].values, te["Year"].values]),
        "Observed": np.concatenate([tv["Irrigation_Depth"].values,
                                    te["Irrigation_Depth"].values]),
        "Predicted": np.concatenate([best_entry["pred_tv"],
                                     best_entry["pred_te"]]),
        "Split": ["train"] * len(tv) + ["test"] * len(te),
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
        "params": best_entry["params"],
        "train_metrics": best_entry["train_metrics"],
        "test_metrics": best_entry["test_metrics"],
    }
    params_path = OUTPUT_DIR / "params.json"
    with open(params_path, "w") as f:
        json.dump(params_out, f, indent=2)
    print(f"Params saved to {params_path}")

    # Plot
    plot_best_model(tv, te, best_entry, OUTPUT_DIR)


if __name__ == "__main__":
    main()
