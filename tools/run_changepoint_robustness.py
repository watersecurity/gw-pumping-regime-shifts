"""Evaluate robustness of BOCPD changepoint detection.

Two modes:
  1. **Cluster mode** (default): sensitivity sweep + jackknife + bootstrap
     on cluster-mean time series.
  2. **Per-agent mode** (``--agent-ids``): sensitivity sweep only on individual
     agent time series (both level and slope detection).

Usage:
    # Cluster mode
    python tools/run_changepoint_robustness.py              # default k=2
    python tools/run_changepoint_robustness.py --k 3
    python tools/run_changepoint_robustness.py --n-boot 500
    python tools/run_changepoint_robustness.py --skip-bootstrap

    # Per-agent mode
    python tools/run_changepoint_robustness.py --agent-ids 12 14 18 20

Outputs:
    Cluster mode:
        results/changepoint_sensitivity_k{K}.csv
        results/changepoint_sensitivity_k{K}.pdf
        results/changepoint_jackknife_k{K}.csv
        results/changepoint_bootstrap_k{K}.csv
        results/changepoint_resampling_k{K}.pdf
    Per-agent mode:
        results/changepoint_sensitivity_agents_{IDS}.csv
        results/changepoint_sensitivity_agents_{IDS}.pdf
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.run_dtc_clustering import load_data
from tools.run_changepoint_detection import (
    load_cluster_assignments,
    run_offline_detection,
    run_offline_slope_detection,
    compute_agent_series,
)

RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Sensitivity grid defaults
# ---------------------------------------------------------------------------
PRIOR_P_GRID = [1 / 50, 1 / 29, 1 / 15]
THRESHOLD_GRID = [0.3, 0.5, 0.7]
PRIOR_SETS = [
    {"label": "default", "alpha0": 1.0, "beta0": 1.0, "kappa0": 1.0, "mu0": 0.0},
    {"label": "weak",    "alpha0": 0.5, "beta0": 0.5, "kappa0": 0.5, "mu0": 0.0},
    {"label": "strong",  "alpha0": 2.0, "beta0": 2.0, "kappa0": 2.0, "mu0": 0.0},
]

# Default BOCPD hyperparameters (for jackknife/bootstrap baseline)
DEFAULT_PRIOR_P = 1 / 29
DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0
DEFAULT_KAPPA0 = 1.0
DEFAULT_MU0 = 0.0
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_cluster_mean(values, years, agent_ids, assignments):
    """Compute mean annual irrigation per cluster.

    Returns DataFrame with columns [Year, cluster_id, mean_irrigation, n_agents].
    """
    years_arr = np.array(years)
    rows = []
    for cluster_id in sorted(assignments["Cluster"].unique()):
        cids = assignments[assignments["Cluster"] == cluster_id]["AgentID"].values
        mask = np.isin(agent_ids, cids)
        cluster_vals = values[mask]
        cluster_mean = cluster_vals.mean(axis=0)
        n_agents = mask.sum()
        for i, yr in enumerate(years_arr):
            rows.append({
                "Year": yr,
                "cluster_id": cluster_id,
                "mean_irrigation": cluster_mean[i],
                "n_agents": n_agents,
            })
    return pd.DataFrame(rows)


def run_bocpd(y_values, prior_p, alpha0, beta0, kappa0, mu0):
    """Run BOCPD and return padded probability array aligned with input length.

    Prepends 0 so result has length T (same as y_values).
    """
    probs = run_offline_detection(
        y_values, prior_p=prior_p,
        alpha0=alpha0, beta0=beta0, kappa0=kappa0, mu0=mu0,
    )
    return np.insert(probs, 0, 0.0)


def summarize_cp(prob_cp, years, threshold, ignore_years=None):
    """Summarize changepoint detection results.

    Returns dict with cp_year_map, cp_prob_max, flag, mass_window_pm1.

    Parameters
    ----------
    ignore_years : list[int] or None
        Years to zero out before finding argmax (e.g. boundary artifacts).
    """
    years_arr = np.array(years)
    prob_masked = prob_cp.copy()
    if ignore_years is not None:
        for yr in ignore_years:
            prob_masked[years_arr == yr] = -1.0
    max_idx = int(np.argmax(prob_masked))
    cp_year = int(years_arr[max_idx])
    cp_prob = float(prob_cp[max_idx])

    # Sum of probabilities within ±1 year of the max
    window_mask = np.abs(years_arr - cp_year) <= 1
    mass = float(prob_cp[window_mask].sum())

    return {
        "cp_year_map": cp_year,
        "cp_prob_max": round(cp_prob, 6),
        "flag": 1 if cp_prob >= threshold else 0,
        "mass_window_pm1": round(mass, 6),
    }


def find_all_cps(prob_cp, years, threshold, ignore_years=None):
    """Find all years where posterior probability exceeds threshold.

    Returns list of (year, prob) tuples sorted by probability descending.
    """
    years_arr = np.array(years)
    mask = np.ones(len(prob_cp), dtype=bool)
    if ignore_years is not None:
        for yr in ignore_years:
            mask[years_arr == yr] = False
    cps = []
    for i in range(len(prob_cp)):
        if mask[i] and prob_cp[i] >= threshold:
            cps.append((int(years_arr[i]), round(float(prob_cp[i]), 6)))
    return sorted(cps, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Approach 1: Prior/Threshold Sensitivity
# ---------------------------------------------------------------------------

def run_sensitivity(cluster_means_df, years, prior_p_grid, threshold_grid, prior_sets):
    """Sweep hyperparameters and collect changepoint summaries."""
    rows = []
    clusters = sorted(cluster_means_df["cluster_id"].unique())

    for cid in clusters:
        cdf = cluster_means_df[cluster_means_df["cluster_id"] == cid]
        y_vals = cdf.sort_values("Year")["mean_irrigation"].values
        n_agents = int(cdf["n_agents"].iloc[0])

        for ps in prior_sets:
            for prior_p in prior_p_grid:
                prob_cp = run_bocpd(
                    y_vals, prior_p,
                    ps["alpha0"], ps["beta0"], ps["kappa0"], ps["mu0"],
                )
                for thr in threshold_grid:
                    summary = summarize_cp(prob_cp, years, thr,
                                           ignore_years=[years[0]])
                    rows.append({
                        "cluster_id": cid,
                        "n_agents": n_agents,
                        "prior_p": round(prior_p, 6),
                        "threshold": thr,
                        "prior_label": ps["label"],
                        "alpha0": ps["alpha0"],
                        "beta0": ps["beta0"],
                        "kappa0": ps["kappa0"],
                        "mu0": ps["mu0"],
                        **summary,
                    })

    return pd.DataFrame(rows)


def plot_sensitivity_heatmaps(df_sens, output_pdf):
    """Heatmaps of cp_year_map and cp_prob_max for default prior set."""
    mpl.rcParams["font.family"] = "Arial"
    df_def = df_sens[df_sens["prior_label"] == "default"]
    clusters = sorted(df_def["cluster_id"].unique())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(
        n_clusters, 2, figsize=(12, 4.5 * n_clusters), squeeze=False,
    )

    for row_idx, cid in enumerate(clusters):
        cdf = df_def[df_def["cluster_id"] == cid]
        n_agents = int(cdf["n_agents"].iloc[0])

        for col_idx, (metric, label, fmt, cmap) in enumerate([
            ("cp_year_map", "Changepoint Year (MAP)", ".0f", "YlOrRd"),
            ("cp_prob_max", "Max Posterior Probability", ".3f", "YlGnBu"),
        ]):
            ax = axes[row_idx, col_idx]
            pivot = cdf.pivot_table(
                index="prior_p", columns="threshold", values=metric, aggfunc="first",
            )
            pivot = pivot.sort_index(ascending=True)

            im = ax.imshow(
                pivot.values, aspect="auto", cmap=cmap,
                origin="lower",
            )
            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if col_idx == 0 else "black")

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=11)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.4f}" for v in pivot.index], fontsize=11)
            ax.set_xlabel("Threshold", fontsize=13)
            ax.set_ylabel("Prior p", fontsize=13)
            ax.set_title(f"Cluster {cid} (n={n_agents}): {label}",
                         fontsize=13, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout(pad=2.0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


# ---------------------------------------------------------------------------
# Approach 1b: Per-Agent Sensitivity (individual time series)
# ---------------------------------------------------------------------------

def run_agent_sensitivity(agent_series, years, prior_p_grid, threshold_grid,
                          prior_sets):
    """Sweep hyperparameters on individual agent time series.

    Runs both level (StudentT) and slope (linear regression) detection for
    each combination of agent × prior_p × threshold × prior_set.

    Parameters
    ----------
    agent_series : dict
        agent_id -> (years_array, values_array) from ``compute_agent_series()``.
    years : list[int]
        Year labels (used in ``summarize_cp``).
    prior_p_grid, threshold_grid : list[float]
        Hyperparameter grids to sweep.
    prior_sets : list[dict]
        Prior parameter sets (label, alpha0, beta0, kappa0, mu0).

    Returns
    -------
    pd.DataFrame with one row per agent × prior_p × threshold × prior_set.
    """
    rows = []
    for aid in sorted(agent_series.keys()):
        yrs, y_vals = agent_series[aid]
        for ps in prior_sets:
            for prior_p in prior_p_grid:
                # Level detection (StudentT)
                prob_level = run_bocpd(
                    y_vals, prior_p,
                    ps["alpha0"], ps["beta0"], ps["kappa0"], ps["mu0"],
                )
                # Slope detection (linear regression)
                slope_raw = run_offline_slope_detection(
                    y_vals, prior_p=prior_p,
                    a0=ps["alpha0"], b0=ps["beta0"],
                )
                prob_slope = np.insert(slope_raw, 0, 0.0)

                for thr in threshold_grid:
                    s_level = summarize_cp(prob_level, yrs, thr,
                                           ignore_years=[yrs[0]])
                    s_slope = summarize_cp(prob_slope, yrs, thr,
                                           ignore_years=[yrs[0]])
                    # Multi-CP detection
                    all_level = find_all_cps(prob_level, yrs, thr,
                                             ignore_years=[yrs[0]])
                    all_slope = find_all_cps(prob_slope, yrs, thr,
                                             ignore_years=[yrs[0]])
                    rows.append({
                        "agent_id": aid,
                        "prior_p": round(prior_p, 6),
                        "threshold": thr,
                        "prior_label": ps["label"],
                        "alpha0": ps["alpha0"],
                        "beta0": ps["beta0"],
                        "kappa0": ps["kappa0"],
                        "mu0": ps["mu0"],
                        "cp_year_map": s_level["cp_year_map"],
                        "cp_prob_max": s_level["cp_prob_max"],
                        "flag": s_level["flag"],
                        "mass_window_pm1": s_level["mass_window_pm1"],
                        "n_cps": len(all_level),
                        "all_cp_years": ",".join(str(y) for y, _ in all_level),
                        "all_cp_probs": ",".join(f"{p:.6f}" for _, p in all_level),
                        "slope_cp_year_map": s_slope["cp_year_map"],
                        "slope_cp_prob_max": s_slope["cp_prob_max"],
                        "slope_flag": s_slope["flag"],
                        "slope_mass_window_pm1": s_slope["mass_window_pm1"],
                        "slope_n_cps": len(all_slope),
                        "slope_all_cp_years": ",".join(str(y) for y, _ in all_slope),
                        "slope_all_cp_probs": ",".join(f"{p:.6f}" for _, p in all_slope),
                    })

    return pd.DataFrame(rows)


def plot_agent_sensitivity_heatmaps(df_sens, output_pdf):
    """Heatmaps of cp_year_map and cp_prob_max per agent (default prior only).

    Layout: one row per agent, 2 columns (CP year MAP, max probability).
    """
    mpl.rcParams["font.family"] = "Arial"
    df_def = df_sens[df_sens["prior_label"] == "default"]
    agents = sorted(df_def["agent_id"].unique())
    n_agents = len(agents)

    fig, axes = plt.subplots(
        n_agents, 2, figsize=(12, 4.5 * n_agents), squeeze=False,
    )

    for row_idx, aid in enumerate(agents):
        adf = df_def[df_def["agent_id"] == aid]

        for col_idx, (metric, label, fmt, cmap) in enumerate([
            ("cp_year_map", "Changepoint Year (MAP) — Levels", ".0f", "YlOrRd"),
            ("cp_prob_max", "Max Posterior Probability — Levels", ".3f", "YlGnBu"),
        ]):
            ax = axes[row_idx, col_idx]
            pivot = adf.pivot_table(
                index="prior_p", columns="threshold", values=metric,
                aggfunc="first",
            )
            pivot = pivot.sort_index(ascending=True)

            im = ax.imshow(
                pivot.values, aspect="auto", cmap=cmap, origin="lower",
            )
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if col_idx == 0 else "black")

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=11)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.4f}" for v in pivot.index], fontsize=11)
            ax.set_xlabel("Threshold", fontsize=13)
            ax.set_ylabel("Prior p", fontsize=13)
            ax.set_title(f"Agent {aid}: {label}",
                         fontsize=13, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout(pad=2.0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def print_agent_summary(df_sens):
    """Print a concise per-agent robustness summary."""
    agents = sorted(df_sens["agent_id"].unique())
    print("\n" + "=" * 70)
    print("PER-AGENT ROBUSTNESS SUMMARY")
    print("=" * 70)

    for aid in agents:
        adf = df_sens[df_sens["agent_id"] == aid]
        # Use default prior, mid threshold as reference
        ref = adf[(adf["prior_label"] == "default") & (adf["threshold"] == 0.3)]
        if len(ref) > 0:
            ref_row = ref.iloc[0]
            ref_yr = ref_row["cp_year_map"]
        else:
            ref_yr = int(adf["cp_year_map"].mode().iloc[0])

        print(f"\n--- Agent {aid} (reference CP = {ref_yr}) ---")

        # Multi-CP summary for default prior at threshold 0.3
        if len(ref) > 0:
            ref_row = ref.iloc[0]
            for det_type, col_n, col_yrs, col_probs in [
                ("Levels", "n_cps", "all_cp_years", "all_cp_probs"),
                ("Slopes", "slope_n_cps", "slope_all_cp_years", "slope_all_cp_probs"),
            ]:
                if col_n in ref_row.index:
                    n_multi = ref_row[col_n]
                    yrs_str = ref_row[col_yrs] if ref_row[col_yrs] else "none"
                    probs_str = ref_row[col_probs] if ref_row[col_probs] else ""
                    print(f"  {det_type} multi-CP (default, thr=0.3): "
                          f"{n_multi} CP(s) — years [{yrs_str}] probs [{probs_str}]")

        # Level detection stability
        n_total = len(adf)
        n_stable = (np.abs(adf["cp_year_map"] - ref_yr) <= 1).sum()
        n_flagged = adf["flag"].sum()
        print(f"  Levels:  {n_stable}/{n_total} combos have CP year within "
              f"±1 yr ({n_stable / n_total:.0%})")
        print(f"           {n_flagged}/{n_total} combos flag a detection")
        unique_yrs = sorted(adf["cp_year_map"].unique())
        print(f"           Unique CP years: {unique_yrs}")

        # Slope detection stability
        n_slope_stable = (np.abs(adf["slope_cp_year_map"] - ref_yr) <= 1).sum()
        n_slope_flagged = adf["slope_flag"].sum()
        slope_ref = int(adf["slope_cp_year_map"].mode().iloc[0])
        print(f"  Slopes:  mode CP year = {slope_ref}, "
              f"{n_slope_flagged}/{n_total} combos flag a detection")
        slope_unique = sorted(adf["slope_cp_year_map"].unique())
        print(f"           Unique CP years: {slope_unique}")

        # Per-prior-set breakdown (levels)
        for ps_label in ["default", "weak", "strong"]:
            ps_df = adf[adf["prior_label"] == ps_label]
            yrs = sorted(ps_df["cp_year_map"].unique())
            probs = ps_df["cp_prob_max"].values
            print(f"  Prior '{ps_label}': CP years = {yrs}, "
                  f"prob range [{probs.min():.3f}, {probs.max():.3f}]")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Approach 2: Resampling Robustness
# ---------------------------------------------------------------------------

def _compute_mean_for_agents(values, years, agent_ids, selected_ids):
    """Compute mean time series for a subset of agents."""
    mask = np.isin(agent_ids, selected_ids)
    return values[mask].mean(axis=0)


def run_jackknife(values, years, agent_ids, assignments,
                  prior_p, alpha0, beta0, kappa0, mu0, threshold):
    """Leave-one-agent-out jackknife for each cluster."""
    rows = []
    clusters = sorted(assignments["Cluster"].unique())

    for cid in clusters:
        cluster_agent_ids = assignments[
            assignments["Cluster"] == cid
        ]["AgentID"].values

        for drop_id in cluster_agent_ids:
            remaining = cluster_agent_ids[cluster_agent_ids != drop_id]
            mean_ts = _compute_mean_for_agents(values, years, agent_ids, remaining)
            prob_cp = run_bocpd(mean_ts, prior_p, alpha0, beta0, kappa0, mu0)
            summary = summarize_cp(prob_cp, years, threshold,
                                    ignore_years=[years[0]])
            rows.append({
                "cluster_id": cid,
                "dropped_agent": int(drop_id),
                "n_agents_remaining": len(remaining),
                **summary,
            })

    return pd.DataFrame(rows)


def run_bootstrap(values, years, agent_ids, assignments, n_boot,
                  prior_p, alpha0, beta0, kappa0, mu0, threshold, seed=42):
    """Bootstrap resampling of agents within each cluster."""
    rng = np.random.default_rng(seed)
    rows = []
    clusters = sorted(assignments["Cluster"].unique())

    for cid in clusters:
        cluster_agent_ids = assignments[
            assignments["Cluster"] == cid
        ]["AgentID"].values
        n = len(cluster_agent_ids)

        for b in range(n_boot):
            sampled = rng.choice(cluster_agent_ids, size=n, replace=True)
            mean_ts = _compute_mean_for_agents(values, years, agent_ids, sampled)
            prob_cp = run_bocpd(mean_ts, prior_p, alpha0, beta0, kappa0, mu0)
            summary = summarize_cp(prob_cp, years, threshold,
                                    ignore_years=[years[0]])
            rows.append({
                "cluster_id": cid,
                "boot_iter": b,
                **summary,
            })

    return pd.DataFrame(rows)


def _get_baseline_cp(cluster_means_df, years, cid,
                     prior_p, alpha0, beta0, kappa0, mu0, threshold):
    """Return baseline changepoint year for a cluster."""
    cdf = cluster_means_df[cluster_means_df["cluster_id"] == cid]
    y_vals = cdf.sort_values("Year")["mean_irrigation"].values
    prob_cp = run_bocpd(y_vals, prior_p, alpha0, beta0, kappa0, mu0)
    return summarize_cp(prob_cp, years, threshold,
                        ignore_years=[years[0]])["cp_year_map"]


def plot_resampling_results(df_jack, df_boot, years, cluster_means_df,
                            prior_p, alpha0, beta0, kappa0, mu0, threshold,
                            output_pdf):
    """Jackknife scatter + bootstrap histogram per cluster."""
    mpl.rcParams["font.family"] = "Arial"
    clusters = sorted(df_jack["cluster_id"].unique())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(
        n_clusters, 2, figsize=(14, 4.5 * n_clusters), squeeze=False,
    )

    for row_idx, cid in enumerate(clusters):
        baseline_yr = _get_baseline_cp(
            cluster_means_df, years, cid,
            prior_p, alpha0, beta0, kappa0, mu0, threshold,
        )
        n_agents = int(
            cluster_means_df[cluster_means_df["cluster_id"] == cid]["n_agents"].iloc[0]
        )

        # --- Left: Jackknife scatter ---
        ax_j = axes[row_idx, 0]
        jdf = df_jack[df_jack["cluster_id"] == cid]
        ax_j.scatter(
            range(len(jdf)), jdf["cp_year_map"],
            c="#1f77b4", s=50, edgecolors="black", linewidths=0.5, zorder=3,
        )
        ax_j.axhline(baseline_yr, color="red", linestyle="--", linewidth=1.5,
                      label=f"Baseline ({baseline_yr})")
        ax_j.axhspan(baseline_yr - 1, baseline_yr + 1, color="red", alpha=0.08)
        ax_j.set_xticks(range(len(jdf)))
        ax_j.set_xticklabels(jdf["dropped_agent"].values, fontsize=8, rotation=90)
        ax_j.set_xlabel("Dropped Agent ID", fontsize=12)
        ax_j.set_ylabel("Changepoint Year (MAP)", fontsize=12)
        ax_j.set_title(
            f"Cluster {cid} (n={n_agents}) — Jackknife", fontsize=13, fontweight="bold",
        )
        ax_j.legend(fontsize=10)
        ax_j.tick_params(axis="y", labelsize=11)

        # --- Right: Bootstrap histogram ---
        ax_b = axes[row_idx, 1]
        bdf = df_boot[df_boot["cluster_id"] == cid]
        years_range = sorted(bdf["cp_year_map"].unique())
        ax_b.hist(
            bdf["cp_year_map"], bins=np.arange(min(years) - 0.5, max(years) + 1.5, 1),
            color="#ff7f0e", edgecolor="black", alpha=0.7,
        )
        ax_b.axvline(baseline_yr, color="red", linestyle="--", linewidth=1.5,
                      label=f"Baseline ({baseline_yr})")
        ax_b.set_xlabel("Changepoint Year (MAP)", fontsize=12)
        ax_b.set_ylabel("Count", fontsize=12)
        ax_b.set_title(
            f"Cluster {cid} (n={n_agents}) — Bootstrap", fontsize=13, fontweight="bold",
        )
        ax_b.legend(fontsize=10)
        ax_b.tick_params(axis="both", labelsize=11)

    fig.tight_layout(pad=2.0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df_sens, df_jack, df_boot, cluster_means_df, years,
                  prior_p, alpha0, beta0, kappa0, mu0, threshold):
    """Print a concise robustness summary for each cluster."""
    clusters = sorted(df_sens["cluster_id"].unique())
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    for cid in clusters:
        baseline_yr = _get_baseline_cp(
            cluster_means_df, years, cid,
            prior_p, alpha0, beta0, kappa0, mu0, threshold,
        )
        n_agents = int(
            cluster_means_df[cluster_means_df["cluster_id"] == cid]["n_agents"].iloc[0]
        )
        print(f"\n--- Cluster {cid} (n={n_agents} agents, baseline CP = {baseline_yr}) ---")

        # Sensitivity: fraction of grid combos where cp_year is within ±1 of baseline
        sdf = df_sens[df_sens["cluster_id"] == cid]
        n_total = len(sdf)
        n_stable = (np.abs(sdf["cp_year_map"] - baseline_yr) <= 1).sum()
        print(f"  Sensitivity: {n_stable}/{n_total} grid combos have "
              f"CP year within ±1 yr of baseline ({n_stable / n_total:.0%})")

        # Bootstrap
        bdf = df_boot[df_boot["cluster_id"] == cid] if df_boot is not None else None
        if bdf is not None and len(bdf) > 0:
            detect_rate = bdf["flag"].mean()
            mode_yr = int(sp_stats.mode(bdf["cp_year_map"], keepdims=False).mode)
            print(f"  Bootstrap: detection rate = {detect_rate:.0%}, "
                  f"mode CP year = {mode_yr}")
        else:
            print("  Bootstrap: skipped")

        # Jackknife
        jdf = df_jack[df_jack["cluster_id"] == cid]
        shifts = np.abs(jdf["cp_year_map"] - baseline_yr)
        any_big_shift = (shifts > 1).any()
        if any_big_shift:
            shifted = jdf[shifts > 1]
            agents_str = ", ".join(str(a) for a in shifted["dropped_agent"].values)
            print(f"  Jackknife: dropping agent(s) [{agents_str}] shifts "
                  f"CP year by >1 yr")
        else:
            print(f"  Jackknife: all leave-one-out CP years within ±1 yr — stable")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BOCPD robustness evaluation (sensitivity + resampling)",
    )
    parser.add_argument("--k", type=int, default=2,
                        help="Number of clusters (default: 2)")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Number of bootstrap iterations (default: 200)")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip bootstrap (faster run)")
    parser.add_argument("--agent-ids", type=int, nargs="+", default=None,
                        help="Run robustness on individual agent time series "
                             "(sensitivity sweep only, skips jackknife/bootstrap)")
    args = parser.parse_args()

    k = args.k
    n_boot = args.n_boot

    # 1. Load data
    agent_ids, years, values = load_data()

    if args.agent_ids is not None:
        # ---------------------------------------------------------------
        # Per-agent mode: sensitivity sweep only
        # ---------------------------------------------------------------
        ids_tag = "_".join(str(a) for a in sorted(args.agent_ids))
        csv_out = RESULTS_DIR / f"changepoint_sensitivity_agents_{ids_tag}.csv"
        pdf_out = RESULTS_DIR / f"changepoint_sensitivity_agents_{ids_tag}.pdf"

        agent_series = compute_agent_series(
            values, years, agent_ids, args.agent_ids,
        )
        print(f"Loaded {len(agent_ids)} agents, {len(years)} years "
              f"({years[0]}-{years[-1]})")
        for aid in sorted(agent_series.keys()):
            _, vals = agent_series[aid]
            print(f"  Agent {aid}: range [{vals.min():.1f}, {vals.max():.1f}] mm")

        print("\n--- Running per-agent sensitivity analysis ---")
        df_sens = run_agent_sensitivity(
            agent_series, years, PRIOR_P_GRID, THRESHOLD_GRID, PRIOR_SETS,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df_sens.to_csv(csv_out, index=False)
        print(f"Saved: {csv_out}  ({len(df_sens)} rows)")

        plot_agent_sensitivity_heatmaps(df_sens, pdf_out)

        print("\n--- Jackknife/bootstrap skipped (per-agent mode) ---")
        print_agent_summary(df_sens)
        print("\nDone.")
        return

    # -------------------------------------------------------------------
    # Cluster mode (original)
    # -------------------------------------------------------------------
    assignments = load_cluster_assignments(k)

    # Output paths
    csv_sens = RESULTS_DIR / f"changepoint_sensitivity_k{k}.csv"
    pdf_sens = RESULTS_DIR / f"changepoint_sensitivity_k{k}.pdf"
    csv_jack = RESULTS_DIR / f"changepoint_jackknife_k{k}.csv"
    csv_boot = RESULTS_DIR / f"changepoint_bootstrap_k{k}.csv"
    pdf_resamp = RESULTS_DIR / f"changepoint_resampling_k{k}.pdf"

    print(f"Loaded {len(agent_ids)} agents, {len(years)} years "
          f"({years[0]}-{years[-1]}), k={k}")

    # 2. Compute cluster means
    cluster_means_df = compute_cluster_mean(values, years, agent_ids, assignments)
    for cid in sorted(cluster_means_df["cluster_id"].unique()):
        n = int(cluster_means_df[cluster_means_df["cluster_id"] == cid]["n_agents"].iloc[0])
        print(f"  Cluster {cid}: {n} agents")

    # 3. Sensitivity analysis
    print("\n--- Running sensitivity analysis ---")
    df_sens = run_sensitivity(
        cluster_means_df, years, PRIOR_P_GRID, THRESHOLD_GRID, PRIOR_SETS,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_sens.to_csv(csv_sens, index=False)
    print(f"Saved: {csv_sens}  ({len(df_sens)} rows)")

    plot_sensitivity_heatmaps(df_sens, pdf_sens)

    # 4. Jackknife
    print("\n--- Running jackknife (leave-one-agent-out) ---")
    df_jack = run_jackknife(
        values, years, agent_ids, assignments,
        DEFAULT_PRIOR_P, DEFAULT_ALPHA0, DEFAULT_BETA0,
        DEFAULT_KAPPA0, DEFAULT_MU0, DEFAULT_THRESHOLD,
    )
    df_jack.to_csv(csv_jack, index=False)
    print(f"Saved: {csv_jack}  ({len(df_jack)} rows)")

    # 5. Bootstrap
    df_boot = None
    if not args.skip_bootstrap:
        print(f"\n--- Running bootstrap ({n_boot} iterations) ---")
        df_boot = run_bootstrap(
            values, years, agent_ids, assignments, n_boot,
            DEFAULT_PRIOR_P, DEFAULT_ALPHA0, DEFAULT_BETA0,
            DEFAULT_KAPPA0, DEFAULT_MU0, DEFAULT_THRESHOLD,
        )
        df_boot.to_csv(csv_boot, index=False)
        print(f"Saved: {csv_boot}  ({len(df_boot)} rows)")
    else:
        print("\n--- Bootstrap skipped (--skip-bootstrap) ---")

    # 6. Resampling plots
    if df_boot is not None:
        plot_resampling_results(
            df_jack, df_boot, years, cluster_means_df,
            DEFAULT_PRIOR_P, DEFAULT_ALPHA0, DEFAULT_BETA0,
            DEFAULT_KAPPA0, DEFAULT_MU0, DEFAULT_THRESHOLD,
            pdf_resamp,
        )
    else:
        # Plot jackknife only (create dummy bootstrap df)
        print("Skipping resampling plot (no bootstrap data)")

    # 7. Console summary
    print_summary(
        df_sens, df_jack, df_boot, cluster_means_df, years,
        DEFAULT_PRIOR_P, DEFAULT_ALPHA0, DEFAULT_BETA0,
        DEFAULT_KAPPA0, DEFAULT_MU0, DEFAULT_THRESHOLD,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
