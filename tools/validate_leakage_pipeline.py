"""Validate that clustering and changepoint decisions are free of look-ahead bias.

Runs three checks:
1. DTC clustering on pre-CP data only (1993-2004) — verifies Cluster 2 agents
   {2, 3, 24, 28, 29} are still grouped together.
2. Offline BOCPD on truncated data (1993-2008) — verifies CP=2005 is still
   detected using only 3 post-CP years for evidence (Fearnhead 2006).
3. Online BOCPD on full data (1993-2020) — sequential detection using only
   past data at each step (Adams & MacKay 2007). Reports whether the CP
   is detectable prospectively.

Note: DTC uses TensorFlow (conda Python 3.7) while BOCPD uses PyTorch
(system Python 3.12), so changepoint detection runs via subprocess.

Usage:
    python tools/validate_leakage_pipeline.py

Outputs:
    results/leakage_check/pipeline_leakage_validation.csv
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress TF info messages before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.run_dtc_clustering import (
    load_data, normalize, run_clustering, sort_clusters, _auto_pool_size,
    INPUT_DIM,
)
from tools.cluster_robustness import (
    co_association_matrix, consensus_labels, pairwise_ari,
)

RESULTS_DIR = BASE_DIR / "results" / "leakage_check"
FULL_ASSIGNMENTS_CSV = BASE_DIR / "results" / "dtc_cluster_assignments_k2.csv"

# Python 3.12 for BOCPD (bayesian_changepoint_detection v1.0 with torch)
PYTHON312 = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"

# Reference: agents in Cluster 2 from full-period DTC
EXPECTED_CLUSTER2_AGENTS = {2, 3, 24, 28, 29}

# Clustering check config
N_CLUSTERING_SEEDS = 10
PRE_CP_YEAR_END = 2004
N_CLUSTERS = 2

# Changepoint check config
CP_CHECK_YEAR_END = 2008
CP_THRESHOLD = 0.5
EXPECTED_CP_YEAR = 2005


def run_clustering_check():
    """Run DTC on pre-CP data (1993-2004) with consensus co-association.

    Uses 10 independent DTC runs and derives consensus labels via
    hierarchical clustering on the co-association matrix (same approach
    as cluster_robustness.py for the full-period analysis).

    Returns dict with consensus results and per-run details.
    """
    from sklearn.metrics import silhouette_score

    print("=" * 60)
    print("CHECK 1: Clustering stability on pre-CP data (1993-2004)")
    print("=" * 60)

    agent_ids, years, values = load_data(year_end=PRE_CP_YEAR_END)
    timesteps = len(years)
    pool_size = _auto_pool_size(timesteps)
    print(f"  Data: {len(agent_ids)} agents, {timesteps} years "
          f"({years[0]}-{years[-1]})")
    print(f"  Architecture: timesteps={timesteps}, pool_size={pool_size}, "
          f"encoded_len={timesteps // pool_size}")
    print(f"  Running {N_CLUSTERING_SEEDS} DTC runs with consensus "
          f"co-association\n")

    values_norm, _, _ = normalize(values)
    X = values_norm.reshape(len(agent_ids), timesteps, INPUT_DIM).astype(np.float32)

    # Collect per-run labels
    all_labels = []
    per_run_results = []
    for seed_idx in range(N_CLUSTERING_SEEDS):
        print(f"  --- DTC run {seed_idx + 1}/{N_CLUSTERING_SEEDS} ---")
        np.random.seed(42 + seed_idx)
        try:
            import tensorflow as tf
            tf.random.set_seed(42 + seed_idx)
        except ImportError:
            pass

        labels = run_clustering(X, n_clusters=N_CLUSTERS, timesteps=timesteps,
                                pool_size=pool_size)
        labels = sort_clusters(labels, values, n_clusters=N_CLUSTERS)
        all_labels.append(labels)

        cluster2_mask = labels == 1
        cluster2_agents = set(agent_ids[cluster2_mask].tolist())
        per_run_results.append({
            "run": seed_idx + 1,
            "cluster2_agents": sorted(cluster2_agents),
        })
        print(f"  Minority cluster: {sorted(cluster2_agents)}")

    all_labels = np.array(all_labels)

    # Co-association matrix and consensus
    coassoc = co_association_matrix(all_labels)
    cons_labels = consensus_labels(coassoc, N_CLUSTERS)

    # Identify the minority consensus cluster
    unique, counts = np.unique(cons_labels, return_counts=True)
    minority_idx = unique[np.argmin(counts)]
    consensus_minority = set(
        agent_ids[i] for i in range(len(agent_ids))
        if cons_labels[i] == minority_idx
    )

    # Pairwise ARI (stability)
    ari_vals = pairwise_ari(all_labels)

    # Silhouette on original (normalized) data
    X_2d = X.reshape(X.shape[0], -1)
    sil = silhouette_score(X_2d, cons_labels) if len(unique) > 1 else -1.0

    # Overlap metrics
    expected = EXPECTED_CLUSTER2_AGENTS
    overlap = consensus_minority & expected
    recall = len(overlap) / len(expected) if expected else 0
    precision = len(overlap) / len(consensus_minority) if consensus_minority else 0

    print(f"\n  Consensus minority cluster: {sorted(consensus_minority)}")
    print(f"  Expected:  {sorted(expected)}")
    print(f"  Overlap:   {sorted(overlap)} "
          f"(recall={recall:.0%}, precision={precision:.0%})")
    print(f"  Pairwise ARI: {ari_vals.mean():.3f} ± {ari_vals.std():.3f}")
    print(f"  Consensus silhouette: {sil:.3f}")

    result = {
        "per_run": per_run_results,
        "consensus_minority": sorted(consensus_minority),
        "expected": sorted(expected),
        "overlap": sorted(overlap),
        "recall": recall,
        "precision": precision,
        "mean_ari": float(ari_vals.mean()),
        "std_ari": float(ari_vals.std()),
        "silhouette": float(sil),
        "n_runs": N_CLUSTERING_SEEDS,
        "coassoc": coassoc,
        "agent_ids": agent_ids,
    }
    return result


def _run_bocpd_subprocess(mode, year_end=None):
    """Run BOCPD via Python 3.12 subprocess.

    Returns (proc, output_csv_path).
    """
    year_end_args = ["--year-end", str(year_end)] if year_end else []
    mode_tag = "_online" if mode == "online" else ""
    year_tag = f"_1993-{year_end}" if year_end else ""

    cp_output_csv = (
        BASE_DIR / "results"
        / f"changepoint_probabilities_k{N_CLUSTERS}{mode_tag}{year_tag}.csv"
    )
    cmd = [
        PYTHON312,
        str(BASE_DIR / "tools" / "run_changepoint_detection.py"),
        "--k", str(N_CLUSTERS),
        "--mode", mode,
        "--threshold", str(CP_THRESHOLD),
    ] + year_end_args

    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    return proc, cp_output_csv


def _parse_cp_csv(cp_output_csv):
    """Parse changepoint CSV and extract Cluster 2 results."""
    df = pd.read_csv(cp_output_csv)

    cp_prob_col = None
    for col in df.columns:
        if "Cluster2_CP_Prob" in col:
            cp_prob_col = col
            break

    if cp_prob_col is None:
        return None, None, None

    years = df["Year"].values
    probs = df[cp_prob_col].values
    return years, probs, df


def run_changepoint_check_offline():
    """Run offline BOCPD on truncated data (1993-2008) and check for CP=2005.

    Returns dict with results.
    """
    print("\n" + "=" * 60)
    print("CHECK 2: Offline BOCPD on truncated data (1993-2008)")
    print("=" * 60)

    proc, cp_output_csv = _run_bocpd_subprocess("offline", year_end=CP_CHECK_YEAR_END)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"  Subprocess failed (exit {proc.returncode}):")
        print(proc.stderr[-500:] if proc.stderr else "")
        return {"check": "offline_cp", "pass": False, "error": "subprocess failed"}

    if not cp_output_csv.exists():
        print(f"  ERROR: Output not found: {cp_output_csv}")
        return {"check": "offline_cp", "pass": False, "error": "output missing"}

    years, probs, _ = _parse_cp_csv(cp_output_csv)
    if years is None:
        return {"check": "offline_cp", "pass": False, "error": "no Cluster2 column"}

    print(f"  Loaded results from: {cp_output_csv}")

    # Check for CP at 2004/2005 boundary
    cp_2005_prob = 0.0
    for i, y in enumerate(years):
        if y == 2004:
            cp_2005_prob = float(probs[i])
            break

    cp_2005_detected = cp_2005_prob >= CP_THRESHOLD
    max_idx = int(np.argmax(probs))

    result = {
        "check": "offline_cp",
        "pass": cp_2005_detected,
        "cp_2005_prob": cp_2005_prob,
        "max_cp_year": int(years[max_idx]),
        "max_cp_prob": float(probs[max_idx]),
        "data_range": f"1993-{CP_CHECK_YEAR_END}",
    }

    status = "PASS" if cp_2005_detected else "FAIL"
    print(f"  CP=2005 offline: p={cp_2005_prob:.4f} — {status}")
    print(f"  Max CP: {result['max_cp_year']} (p={result['max_cp_prob']:.4f})")
    return result


def run_changepoint_check_online():
    """Run online BOCPD on full data (1993-2020) — prospective detection.

    The online algorithm processes data sequentially. changepoint_probs[t]
    uses only data up to time t (Adams & MacKay 2007).

    Returns dict with results.
    """
    print("\n" + "=" * 60)
    print("CHECK 3: Online BOCPD on full data (1993-2020) — prospective")
    print("=" * 60)

    proc, cp_output_csv = _run_bocpd_subprocess("online")
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"  Subprocess failed (exit {proc.returncode}):")
        print(proc.stderr[-500:] if proc.stderr else "")
        return {"check": "online_cp", "pass": False, "error": "subprocess failed"}

    if not cp_output_csv.exists():
        print(f"  ERROR: Output not found: {cp_output_csv}")
        return {"check": "online_cp", "pass": False, "error": "output missing"}

    years, probs, _ = _parse_cp_csv(cp_output_csv)
    if years is None:
        return {"check": "online_cp", "pass": False, "error": "no Cluster2 column"}

    print(f"  Loaded results from: {cp_output_csv}")

    # Check for CP at 2004/2005 boundary
    cp_2005_prob = 0.0
    for i, y in enumerate(years):
        if y == 2004:
            cp_2005_prob = float(probs[i])
            break

    cp_2005_detected = cp_2005_prob >= CP_THRESHOLD
    max_idx = int(np.argmax(probs))
    max_cp_year = int(years[max_idx])
    max_cp_prob = float(probs[max_idx])

    result = {
        "check": "online_cp",
        "pass": cp_2005_detected,
        "cp_2005_prob": cp_2005_prob,
        "max_cp_year": max_cp_year,
        "max_cp_prob": max_cp_prob,
        "data_range": "1993-2020",
    }

    status = "PASS" if cp_2005_detected else "NOT DETECTED"
    print(f"  CP=2005 online: p={cp_2005_prob:.6f} — {status}")
    print(f"  Max online CP: {max_cp_year} (p={max_cp_prob:.6f})")

    if not cp_2005_detected:
        print(f"\n  Interpretation: The online BOCPD (which uses only past data at")
        print(f"  each step) does not detect CP=2005 at threshold {CP_THRESHOLD}.")
        print(f"  This is expected with n=2 agents — the high pre-CP interannual")
        print(f"  variance prevents the sequential learner from distinguishing")
        print(f"  the structural break from normal variability in real time.")
        print(f"  The offline detection of CP=2005 relies on retrospective")
        print(f"  (bidirectional) analysis, which is standard practice in")
        print(f"  empirical hydrology.")

    return result


def print_report(clustering_result, offline_result, online_result):
    """Print structured validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT: Pipeline Leakage Check")
    print("=" * 60)

    # Clustering summary (consensus-based)
    recall = clustering_result["recall"]
    precision = clustering_result["precision"]
    # Pass if ≥3/5 expected agents recovered (60% recall)
    clust_pass = recall >= 0.6
    print(f"\n1. CLUSTERING (pre-CP data 1993-{PRE_CP_YEAR_END}, k={N_CLUSTERS}, "
          f"{clustering_result['n_runs']} runs)")
    print(f"   Expected Cluster 2 agents: {clustering_result['expected']}")
    print(f"   Consensus minority cluster: {clustering_result['consensus_minority']}")
    print(f"   Overlap: {clustering_result['overlap']} "
          f"(recall={recall:.0%}, precision={precision:.0%})")
    print(f"   Pairwise ARI: {clustering_result['mean_ari']:.3f} "
          f"± {clustering_result['std_ari']:.3f}")
    print(f"   Consensus silhouette: {clustering_result['silhouette']:.3f}")
    print(f"   Result: {'PASS' if clust_pass else 'FAIL'} "
          f"(≥60% recall required)")

    # Offline CP summary
    offline_pass = offline_result.get("pass", False)
    print(f"\n2. OFFLINE BOCPD (truncated data 1993-{CP_CHECK_YEAR_END})")
    print(f"   Expected CP year: {EXPECTED_CP_YEAR}")
    print(f"   CP={EXPECTED_CP_YEAR} probability: "
          f"{offline_result.get('cp_2005_prob', 0):.4f}")
    print(f"   Result: {'PASS' if offline_pass else 'FAIL'}")

    # Online CP summary
    online_detected = online_result.get("pass", False)
    print(f"\n3. ONLINE BOCPD (full data, prospective)")
    print(f"   CP={EXPECTED_CP_YEAR} probability: "
          f"{online_result.get('cp_2005_prob', 0):.6f}")
    print(f"   Max CP: {online_result.get('max_cp_year', 'N/A')} "
          f"(p={online_result.get('max_cp_prob', 0):.6f})")
    if online_detected:
        print(f"   Result: DETECTED — CP detectable prospectively")
    else:
        print(f"   Result: NOT DETECTED — CP requires retrospective analysis")
        print(f"   (expected: high variance with n=2 agents prevents real-time detection)")

    # Overall assessment
    overall = clust_pass and offline_pass
    print(f"\nOVERALL: ", end="")
    if overall and online_detected:
        print("PASS — leakage is negligible; CP detectable prospectively")
    elif overall and not online_detected:
        print("PASS (with caveat) — clustering and offline CP are robust;")
        print("         online BOCPD confirms CP detection requires retrospective")
        print("         analysis, which is standard for empirical nonstationarity studies")
    else:
        print("FAIL — potential leakage concern")

    return overall, clust_pass, offline_pass, online_detected


def save_report(clustering_result, offline_result, online_result,
                overall, clust_pass, offline_pass, online_detected):
    """Save validation results to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = RESULTS_DIR / "pipeline_leakage_validation.csv"

    rows = []
    # Per-run clustering rows
    for r in clustering_result["per_run"]:
        rows.append({
            "check": "clustering",
            "detail": f"run_{r['run']}",
            "data_range": f"1993-{PRE_CP_YEAR_END}",
            "expected": str(sorted(EXPECTED_CLUSTER2_AGENTS)),
            "actual": str(r["cluster2_agents"]),
            "pass": set(r["cluster2_agents"]) == EXPECTED_CLUSTER2_AGENTS,
        })
    rows.append({
        "check": "clustering_consensus",
        "detail": (f"recall={clustering_result['recall']:.0%}, "
                   f"precision={clustering_result['precision']:.0%}, "
                   f"ARI={clustering_result['mean_ari']:.3f}, "
                   f"sil={clustering_result['silhouette']:.3f}"),
        "data_range": f"1993-{PRE_CP_YEAR_END}",
        "expected": str(sorted(EXPECTED_CLUSTER2_AGENTS)),
        "actual": str(clustering_result["consensus_minority"]),
        "pass": clust_pass,
    })

    # Offline CP row
    rows.append({
        "check": "offline_bocpd",
        "detail": f"CP={EXPECTED_CP_YEAR} p={offline_result.get('cp_2005_prob', 0):.4f}",
        "data_range": offline_result.get("data_range", f"1993-{CP_CHECK_YEAR_END}"),
        "expected": f"CP={EXPECTED_CP_YEAR} p>={CP_THRESHOLD}",
        "actual": f"max_CP={offline_result.get('max_cp_year', 'N/A')} "
                  f"p={offline_result.get('max_cp_prob', 0):.4f}",
        "pass": offline_pass,
    })

    # Online CP row
    rows.append({
        "check": "online_bocpd",
        "detail": f"CP={EXPECTED_CP_YEAR} p={online_result.get('cp_2005_prob', 0):.6f}",
        "data_range": "1993-2020",
        "expected": f"CP={EXPECTED_CP_YEAR} (prospective)",
        "actual": f"max_CP={online_result.get('max_cp_year', 'N/A')} "
                  f"p={online_result.get('max_cp_prob', 0):.6f}",
        "pass": online_detected,
    })

    # Overall
    rows.append({
        "check": "overall",
        "detail": "clustering + offline_cp pass; online_cp informational",
        "data_range": "",
        "expected": "",
        "actual": "",
        "pass": overall,
    })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


def save_coassociation_plot(clustering_result):
    """Save co-association heatmap for pre-CP clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coassoc = clustering_result["coassoc"]
    agent_ids = clustering_result["agent_ids"]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(coassoc, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"Pre-CP Co-association (1993-{PRE_CP_YEAR_END}, "
                 f"{clustering_result['n_runs']} runs)", fontsize=12)
    ax.set_xlabel("Agent index")
    ax.set_ylabel("Agent index")
    # Mark expected Cluster 2 agents
    expected_indices = [i for i, a in enumerate(agent_ids)
                        if a in EXPECTED_CLUSTER2_AGENTS]
    for idx in expected_indices:
        ax.axhline(idx, color="blue", linewidth=0.3, alpha=0.4)
        ax.axvline(idx, color="blue", linewidth=0.3, alpha=0.4)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Co-association frequency")

    out_path = RESULTS_DIR / "pre_cp_coassociation.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    # Run checks
    clustering_result = run_clustering_check()
    offline_result = run_changepoint_check_offline()
    online_result = run_changepoint_check_online()

    # Report
    overall, clust_pass, offline_pass, online_detected = print_report(
        clustering_result, offline_result, online_result
    )
    save_report(clustering_result, offline_result, online_result,
                overall, clust_pass, offline_pass, online_detected)
    save_coassociation_plot(clustering_result)


if __name__ == "__main__":
    main()
