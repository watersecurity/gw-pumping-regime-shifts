"""Baseline clustering comparison for DTC validation (Reviewer MC#3).

Runs k-means (with DTW distance via tslearn) and hierarchical clustering
(Ward's method with Euclidean distance) on the same 43×28 irrigation depth
matrix used by DTC, then compares silhouette scores and cluster assignments.

Usage:
    python tools/run_clustering_baselines.py
    python tools/run_clustering_baselines.py --k 2 --n-runs 20

Outputs:
    results/clustering_baselines/comparison_table.csv
    results/clustering_baselines/cluster_assignments.csv
    results/clustering_baselines/comparison_figure.pdf
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw as tslearn_dtw

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.run_dtc_clustering import load_data

RESULTS_DIR = BASE_DIR / "results" / "clustering_baselines"
MASTER_SEED = 42

# DTC reference cluster assignments (k=2, from findings.md)
DTC_CLUSTER2_AGENTS = {2, 3, 24, 28, 29}


def dtw_distance_matrix(X_2d):
    """Compute pairwise DTW distance matrix for 2D array (n_samples, timesteps)."""
    n = X_2d.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = tslearn_dtw(X_2d[i], X_2d[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def run_kmeans_dtw(X_3d, k, n_runs, seed):
    """Run TimeSeriesKMeans with DTW metric, return best labels and metrics."""
    all_labels = []
    all_inertias = []
    for i in range(n_runs):
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            max_iter=100,
            random_state=seed + i,
            n_init=1,
            verbose=0,
        )
        labels = model.fit_predict(X_3d)
        all_labels.append(labels)
        all_inertias.append(model.inertia_)

    # Pick run with lowest inertia as "best"
    best_idx = np.argmin(all_inertias)
    best_labels = all_labels[best_idx]

    # Pairwise ARI across runs
    ari_values = []
    for i, j in combinations(range(n_runs), 2):
        ari_values.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    ari_values = np.array(ari_values) if ari_values else np.array([1.0])

    return best_labels, all_labels, ari_values


def run_hierarchical(X_2d, k, method="ward"):
    """Run hierarchical clustering with given linkage method."""
    Z = linkage(X_2d, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
    return labels, Z


def run_hierarchical_dtw(dtw_dist_matrix, k, method="average"):
    """Run hierarchical clustering on precomputed DTW distance matrix."""
    condensed = squareform(dtw_dist_matrix)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    return labels, Z


def align_labels_to_dtc(labels, agent_ids, dtc_c2_agents=DTC_CLUSTER2_AGENTS):
    """Relabel to maximize agreement with DTC reference partition.

    Tests both possible label assignments (for k=2) and picks the one
    with higher agreement with the DTC reference.
    """
    agent_list = list(agent_ids)
    dtc_labels = np.array([1 if a in dtc_c2_agents else 0 for a in agent_list])

    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        # Fallback for degenerate cases
        return np.zeros_like(labels)

    # Try both possible mappings and pick the one with higher agreement
    mapping_a = np.where(labels == unique_labels[0], 0, 1)
    mapping_b = np.where(labels == unique_labels[0], 1, 0)

    agree_a = np.mean(mapping_a == dtc_labels)
    agree_b = np.mean(mapping_b == dtc_labels)

    return mapping_a if agree_a >= agree_b else mapping_b


def compute_agreement_with_dtc(labels, agent_ids, dtc_c2_agents=DTC_CLUSTER2_AGENTS):
    """Compute fraction of agents assigned same as DTC reference partition."""
    agent_list = list(agent_ids)
    dtc_labels = np.array([1 if a in dtc_c2_agents else 0 for a in agent_list])

    aligned = align_labels_to_dtc(labels, agent_ids, dtc_c2_agents)
    agreement = np.mean(aligned == dtc_labels)
    ari = adjusted_rand_score(dtc_labels, aligned)
    return agreement, ari


def plot_comparison(results, agent_ids, years, values, output_path):
    """Generate comparison figure with cluster assignments and summary."""
    rc = {
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
    }
    old_rc = {k: plt.rcParams.get(k) for k in rc}
    plt.rcParams.update(rc)

    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5),
                              sharey=True)
    if n_methods == 1:
        axes = [axes]

    colors = {0: "tab:blue", 1: "tab:red"}

    for ax, (name, res) in zip(axes, results.items()):
        labels = res["aligned_labels"]
        for c in [0, 1]:
            mask = labels == c
            for row in values[mask]:
                ax.plot(years, row, color=colors[c], alpha=0.5, linewidth=0.7)
            cluster_mean = values[mask].mean(axis=0)
            label = f"Cluster {c+1} (n={mask.sum()})"
            ax.plot(years, cluster_mean, color=colors[c], linewidth=2.5,
                    linestyle="--", label=label)

        ax.set_title(f"{name}\nSil={res['silhouette']:.3f}, "
                     f"ARI vs DTC={res['ari_vs_dtc']:.3f}",
                     fontsize=13)
        ax.set_xlabel("Year")
        if ax == axes[0]:
            ax.set_ylabel("Irrigation Depth (mm)")
        ticks = [y for y in range(years[0], years[-1] + 1) if y % 5 == 0]
        ax.set_xticks(ticks)
        ax.set_xlim(years[0], years[-1])
        ax.legend(fontsize=11, loc="upper left", frameon=False)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.tick_params(length=4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(old_rc)
    print(f"Saved comparison figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline clustering comparison for DTC validation")
    parser.add_argument("--k", type=int, default=2, help="Number of clusters")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of runs for k-means methods (stability)")
    args = parser.parse_args()

    k = args.k
    n_runs = args.n_runs

    # Load data
    agent_ids, years, values = load_data()
    print(f"Loaded {len(agent_ids)} agents, {len(years)} years "
          f"({years[0]}-{years[-1]})")

    # Normalize for clustering
    X_raw = values.astype(np.float64)
    X_norm = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min() + 1e-12)
    X_3d = X_norm.reshape(len(agent_ids), len(years), 1)  # for tslearn

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # ---- Method 1: K-Means + DTW (tslearn) ----
    print(f"\n{'='*60}")
    print(f"Running K-Means + DTW (k={k}, {n_runs} runs)")
    print(f"{'='*60}")
    best_labels, all_labels, ari_vals = run_kmeans_dtw(
        X_3d, k, n_runs, MASTER_SEED)

    sil_dtw = silhouette_score(X_norm, best_labels)
    aligned_dtw = align_labels_to_dtc(best_labels, agent_ids)
    agreement_dtw, ari_dtc_dtw = compute_agreement_with_dtc(
        best_labels, agent_ids)

    c2_agents_dtw = sorted(agent_ids[aligned_dtw == 1])
    print(f"  Silhouette: {sil_dtw:.3f}")
    print(f"  Stability ARI: {ari_vals.mean():.3f} ± {ari_vals.std():.3f}")
    print(f"  Agreement with DTC: {agreement_dtw:.1%}")
    print(f"  ARI vs DTC: {ari_dtc_dtw:.3f}")
    print(f"  Minority cluster agents: {c2_agents_dtw}")

    results["K-Means+DTW"] = {
        "labels": best_labels,
        "aligned_labels": aligned_dtw,
        "silhouette": sil_dtw,
        "mean_ari_stability": ari_vals.mean(),
        "std_ari_stability": ari_vals.std(),
        "agreement_with_dtc": agreement_dtw,
        "ari_vs_dtc": ari_dtc_dtw,
        "minority_agents": c2_agents_dtw,
    }

    # ---- Method 2: Hierarchical (Ward, Euclidean) ----
    print(f"\n{'='*60}")
    print(f"Running Hierarchical Clustering (Ward, Euclidean, k={k})")
    print(f"{'='*60}")
    labels_ward, Z_ward = run_hierarchical(X_norm, k, method="ward")

    sil_ward = silhouette_score(X_norm, labels_ward)
    aligned_ward = align_labels_to_dtc(labels_ward, agent_ids)
    agreement_ward, ari_dtc_ward = compute_agreement_with_dtc(
        labels_ward, agent_ids)

    c2_agents_ward = sorted(agent_ids[aligned_ward == 1])
    print(f"  Silhouette: {sil_ward:.3f}")
    print(f"  Agreement with DTC: {agreement_ward:.1%}")
    print(f"  ARI vs DTC: {ari_dtc_ward:.3f}")
    print(f"  Minority cluster agents: {c2_agents_ward}")

    results["Hierarchical (Ward)"] = {
        "labels": labels_ward,
        "aligned_labels": aligned_ward,
        "silhouette": sil_ward,
        "mean_ari_stability": 1.0,  # deterministic
        "std_ari_stability": 0.0,
        "agreement_with_dtc": agreement_ward,
        "ari_vs_dtc": ari_dtc_ward,
        "minority_agents": c2_agents_ward,
    }

    # ---- Method 3: Hierarchical (Average linkage, DTW distance) ----
    print(f"\n{'='*60}")
    print(f"Computing DTW distance matrix for hierarchical clustering...")
    print(f"{'='*60}")
    dtw_dist = dtw_distance_matrix(X_norm)
    labels_hdtw, Z_hdtw = run_hierarchical_dtw(dtw_dist, k, method="average")

    sil_hdtw = silhouette_score(dtw_dist, labels_hdtw, metric="precomputed")
    aligned_hdtw = align_labels_to_dtc(labels_hdtw, agent_ids)
    agreement_hdtw, ari_dtc_hdtw = compute_agreement_with_dtc(
        labels_hdtw, agent_ids)

    c2_agents_hdtw = sorted(agent_ids[aligned_hdtw == 1])
    print(f"  Silhouette (DTW): {sil_hdtw:.3f}")
    print(f"  Agreement with DTC: {agreement_hdtw:.1%}")
    print(f"  ARI vs DTC: {ari_dtc_hdtw:.3f}")
    print(f"  Minority cluster agents: {c2_agents_hdtw}")

    results["Hierarchical (DTW)"] = {
        "labels": labels_hdtw,
        "aligned_labels": aligned_hdtw,
        "silhouette": sil_hdtw,
        "mean_ari_stability": 1.0,
        "std_ari_stability": 0.0,
        "agreement_with_dtc": agreement_hdtw,
        "ari_vs_dtc": ari_dtc_hdtw,
        "minority_agents": c2_agents_hdtw,
    }

    # ---- Method 4: Standard K-Means (Euclidean) ----
    print(f"\n{'='*60}")
    print(f"Running Standard K-Means (Euclidean, k={k}, {n_runs} runs)")
    print(f"{'='*60}")
    all_labels_km = []
    all_inertias_km = []
    for i in range(n_runs):
        model = KMeans(n_clusters=k, random_state=MASTER_SEED + i, n_init=1)
        labels_km = model.fit_predict(X_norm)
        all_labels_km.append(labels_km)
        all_inertias_km.append(model.inertia_)

    best_idx_km = np.argmin(all_inertias_km)
    best_labels_km = all_labels_km[best_idx_km]

    ari_km = []
    for i, j in combinations(range(n_runs), 2):
        ari_km.append(adjusted_rand_score(all_labels_km[i], all_labels_km[j]))
    ari_km = np.array(ari_km) if ari_km else np.array([1.0])

    sil_km = silhouette_score(X_norm, best_labels_km)
    aligned_km = align_labels_to_dtc(best_labels_km, agent_ids)
    agreement_km, ari_dtc_km = compute_agreement_with_dtc(
        best_labels_km, agent_ids)

    c2_agents_km = sorted(agent_ids[aligned_km == 1])
    print(f"  Silhouette: {sil_km:.3f}")
    print(f"  Stability ARI: {ari_km.mean():.3f} ± {ari_km.std():.3f}")
    print(f"  Agreement with DTC: {agreement_km:.1%}")
    print(f"  ARI vs DTC: {ari_dtc_km:.3f}")
    print(f"  Minority cluster agents: {c2_agents_km}")

    results["K-Means (Euclidean)"] = {
        "labels": best_labels_km,
        "aligned_labels": aligned_km,
        "silhouette": sil_km,
        "mean_ari_stability": ari_km.mean(),
        "std_ari_stability": ari_km.std(),
        "agreement_with_dtc": agreement_km,
        "ari_vs_dtc": ari_dtc_km,
        "minority_agents": c2_agents_km,
    }

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"DTC reference: Cluster 2 = {sorted(DTC_CLUSTER2_AGENTS)}, "
          f"Silhouette = 0.455")

    summary_rows = []
    for name, res in results.items():
        row = {
            "Method": name,
            "Silhouette": round(res["silhouette"], 3),
            "Stability_ARI_mean": round(res["mean_ari_stability"], 3),
            "Stability_ARI_std": round(res["std_ari_stability"], 3),
            "Agreement_with_DTC": round(res["agreement_with_dtc"], 3),
            "ARI_vs_DTC": round(res["ari_vs_dtc"], 3),
            "Minority_Agents": str(res["minority_agents"]),
        }
        summary_rows.append(row)
        print(f"\n{name}:")
        print(f"  Silhouette: {res['silhouette']:.3f}")
        print(f"  Stability: {res['mean_ari_stability']:.3f} ± "
              f"{res['std_ari_stability']:.3f}")
        print(f"  Agreement with DTC: {res['agreement_with_dtc']:.1%}")
        print(f"  ARI vs DTC: {res['ari_vs_dtc']:.3f}")
        print(f"  Minority agents: {res['minority_agents']}")

    # Add DTC reference row
    summary_rows.insert(0, {
        "Method": "DTC (reference)",
        "Silhouette": 0.455,
        "Stability_ARI_mean": 0.344,
        "Stability_ARI_std": 0.339,
        "Agreement_with_DTC": 1.0,
        "ARI_vs_DTC": 1.0,
        "Minority_Agents": str(sorted(DTC_CLUSTER2_AGENTS)),
    })

    # Save comparison table
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"\nSaved comparison table to {RESULTS_DIR / 'comparison_table.csv'}")

    # Save per-agent cluster assignments
    assign_data = {"AgentID": agent_ids}
    assign_data["DTC_Cluster"] = [
        2 if a in DTC_CLUSTER2_AGENTS else 1 for a in agent_ids]
    for name, res in results.items():
        col = name.replace(" ", "_").replace("(", "").replace(")", "")
        assign_data[col] = res["aligned_labels"] + 1  # 1-indexed
    df_assign = pd.DataFrame(assign_data)
    df_assign.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)
    print(f"Saved assignments to {RESULTS_DIR / 'cluster_assignments.csv'}")

    # Plot comparison
    plot_comparison(results, agent_ids, years, values,
                    RESULTS_DIR / "comparison_figure.pdf")

    print(f"\n{'='*60}")
    print("DONE — Baseline clustering comparison complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
