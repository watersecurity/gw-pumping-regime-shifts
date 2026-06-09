"""Evaluate robustness of DTC cluster configuration.

Runs four evaluation methods:
1. Multi-run stability (10 runs, pairwise ARI, co-association matrix)
2. Silhouette analysis on latent representations
3. Optimal k selection (k=2..6, 5 runs each)
4. Perturbation robustness (Gaussian noise at 3 levels, 5 runs each)

Outputs:
    results/dtc_robustness_metrics.csv  — summary metrics
    results/dtc_robustness.pdf          — 4-panel visualization
"""

import os
import sys
import random
import warnings

# Suppress TF info/warning messages before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from itertools import combinations

import tensorflow as tf
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# Suppress TF logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.run_dtc_clustering import load_data, normalize
from tools.dtc.DeepTemporalClustering import DTC

# --- Paths ---
INPUT_CSV = BASE_DIR / "data" / "annual_irrigation_depth.csv"
RESULTS_DIR = BASE_DIR / "results"
TMP_DIR = BASE_DIR / ".tmp" / "dtc_robustness"
OUTPUT_CSV = RESULTS_DIR / "dtc_robustness_metrics.csv"
OUTPUT_PDF = RESULTS_DIR / "dtc_robustness.pdf"

# --- DTC hyperparameters (from run_dtc_clustering.py) ---
INPUT_DIM = 1
TIMESTEPS = 28
N_FILTERS = 16
KERNEL_SIZE = 5
POOL_SIZE = 7
N_UNITS = [10, 1]
DIST_METRIC = "eucl"
CLUSTER_INIT = "kmeans"
PRETRAIN_EPOCHS = 100
EPOCHS = 200
BATCH_SIZE = 43
GAMMA = 1.0
EVAL_EPOCHS = 1
TOL = 0.001
PATIENCE = 10


def _train_dtc(X, n_clusters, seed, save_dir, verbose=0):
    """Train a DTC model with a fixed random seed.

    Returns (labels, dtc_model) so the caller can access dtc.encode(X).
    """
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    dtc = DTC(
        n_clusters=n_clusters,
        input_dim=INPUT_DIM,
        timesteps=TIMESTEPS,
        n_filters=N_FILTERS,
        kernel_size=KERNEL_SIZE,
        strides=1,
        pool_size=POOL_SIZE,
        n_units=N_UNITS,
        alpha=1.0,
        dist_metric=DIST_METRIC,
        cluster_init=CLUSTER_INIT,
        heatmap=False,
    )
    dtc.initialize()
    dtc.compile(gamma=GAMMA, optimizer="adam")

    os.makedirs(save_dir, exist_ok=True)

    # Pretrain autoencoder (suppress output)
    dtc.pretrain(
        X=X,
        optimizer="adam",
        epochs=PRETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=save_dir,
        verbose=verbose,
    )

    # Initialize cluster centers
    dtc.init_cluster_weights(X)

    # Joint clustering + reconstruction training
    dtc.fit(
        X_train=X,
        y_train=None,
        epochs=EPOCHS,
        eval_epochs=EVAL_EPOCHS,
        save_epochs=9999,  # avoid checkpoint I/O
        batch_size=BATCH_SIZE,
        tol=TOL,
        patience=PATIENCE,
        save_dir=save_dir,
    )

    labels = dtc.predict(X)
    return labels, dtc


def _get_latent(dtc, X):
    """Extract flattened latent representations from DTC encoder.

    dtc.encode(X) returns shape (n_samples, 4, 1); flatten to (n_samples, 4).
    """
    encoded = dtc.encode(X)
    return encoded.reshape(encoded.shape[0], -1)


def run_multirun_stability(X, n_runs=10, n_clusters=3):
    """Run DTC n_runs times and compute pairwise ARI and co-association matrix."""
    print(f"\n{'='*60}")
    print(f"[1/4] Multi-run stability: {n_runs} runs at k={n_clusters}")
    print(f"{'='*60}")

    n_samples = X.shape[0]
    all_labels = []
    all_dtc_models = []
    save_dir = str(TMP_DIR / "multirun")

    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs} (seed={i}) ...", end=" ", flush=True)
        labels, dtc_model = _train_dtc(X, n_clusters, seed=i, save_dir=save_dir)
        all_labels.append(labels)
        all_dtc_models.append(dtc_model)
        counts = [np.sum(labels == c) for c in range(n_clusters)]
        print(f"clusters: {counts}")

    # Pairwise ARI
    ari_values = []
    for (i, j) in combinations(range(n_runs), 2):
        ari_values.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    mean_ari = np.mean(ari_values)
    std_ari = np.std(ari_values)
    print(f"\n  Pairwise ARI: mean={mean_ari:.4f}, std={std_ari:.4f}")

    # Co-association matrix
    coassoc = np.zeros((n_samples, n_samples))
    for labels in all_labels:
        for c in range(n_clusters):
            mask = labels == c
            indices = np.where(mask)[0]
            for ii in range(len(indices)):
                for jj in range(ii, len(indices)):
                    coassoc[indices[ii], indices[jj]] += 1
                    coassoc[indices[jj], indices[ii]] += 1
    # Diagonal is always n_runs (agent always co-clusters with itself)
    np.fill_diagonal(coassoc, n_runs)
    coassoc /= n_runs

    return all_labels, all_dtc_models, mean_ari, std_ari, coassoc


def run_silhouette_analysis(X, all_labels, all_dtc_models):
    """Compute silhouette scores on latent representations across runs."""
    print(f"\n{'='*60}")
    print("[2/4] Silhouette analysis on latent representations")
    print(f"{'='*60}")

    n_runs = len(all_labels)
    all_sil_samples = []
    all_sil_scores = []

    for i in range(n_runs):
        latent = _get_latent(all_dtc_models[i], X)
        labels = all_labels[i]
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            print(f"  Run {i+1}: only {n_unique} cluster(s), skipping silhouette")
            continue
        sil_samples = silhouette_samples(latent, labels)
        sil_score = np.mean(sil_samples)
        all_sil_samples.append(sil_samples)
        all_sil_scores.append(sil_score)
        print(f"  Run {i+1}: silhouette={sil_score:.4f}")

    mean_sil = np.mean(all_sil_scores)
    std_sil = np.std(all_sil_scores)
    print(f"\n  Silhouette: mean={mean_sil:.4f}, std={std_sil:.4f}")

    # Identify agents with consistently negative silhouette
    if all_sil_samples:
        stacked = np.stack(all_sil_samples, axis=0)  # (n_runs, n_samples)
        mean_per_agent = stacked.mean(axis=0)
        n_negative = np.sum(mean_per_agent < 0)
        print(f"  Agents with mean negative silhouette: {n_negative}")
    else:
        mean_per_agent = None
        n_negative = 0

    return mean_sil, std_sil, n_negative, all_sil_samples


def run_optimal_k(X, k_range=range(2, 7), n_runs_per_k=5):
    """Evaluate internal validity indices across k values."""
    print(f"\n{'='*60}")
    print(f"[3/4] Optimal k selection: k={list(k_range)}, {n_runs_per_k} runs each")
    print(f"{'='*60}")

    results = {k: {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
               for k in k_range}

    for k in k_range:
        print(f"\n  k={k}:")
        save_dir = str(TMP_DIR / f"k{k}")
        for run in range(n_runs_per_k):
            seed = 100 + k * 10 + run  # unique seeds per k and run
            print(f"    Run {run+1}/{n_runs_per_k} (seed={seed}) ...", end=" ", flush=True)
            labels, dtc_model = _train_dtc(X, k, seed=seed, save_dir=save_dir)
            latent = _get_latent(dtc_model, X)

            n_unique = len(np.unique(labels))
            if n_unique < 2:
                print(f"degenerate ({n_unique} cluster), skipping")
                continue

            sil = silhouette_score(latent, labels)
            db = davies_bouldin_score(latent, labels)
            ch = calinski_harabasz_score(latent, labels)
            results[k]["silhouette"].append(sil)
            results[k]["davies_bouldin"].append(db)
            results[k]["calinski_harabasz"].append(ch)
            counts = [np.sum(labels == c) for c in range(k)]
            print(f"sil={sil:.3f}, DB={db:.3f}, CH={ch:.1f}, clusters={counts}")

    # Find optimal k
    k_vals = list(k_range)
    mean_sil = [np.mean(results[k]["silhouette"]) if results[k]["silhouette"] else -1
                for k in k_vals]
    mean_db = [np.mean(results[k]["davies_bouldin"]) if results[k]["davies_bouldin"] else 999
               for k in k_vals]

    optimal_k_sil = k_vals[np.argmax(mean_sil)]
    optimal_k_db = k_vals[np.argmin(mean_db)]
    print(f"\n  Optimal k (silhouette): {optimal_k_sil}")
    print(f"  Optimal k (Davies-Bouldin): {optimal_k_db}")

    return results, optimal_k_sil, optimal_k_db


def run_perturbation_robustness(X, noise_levels=(0.01, 0.03, 0.05), n_runs=5, n_clusters=3):
    """Test robustness to Gaussian noise perturbation."""
    print(f"\n{'='*60}")
    print(f"[4/4] Perturbation robustness: noise={noise_levels}")
    print(f"{'='*60}")

    # Get reference labels (seed=0)
    print("  Training reference model (seed=0) ...", end=" ", flush=True)
    save_dir = str(TMP_DIR / "perturbation")
    ref_labels, _ = _train_dtc(X, n_clusters, seed=0, save_dir=save_dir)
    print("done")

    global_std = X.std()
    perturbation_results = {}

    for sigma_frac in noise_levels:
        sigma = sigma_frac * global_std
        ari_values = []
        print(f"\n  Noise level σ={sigma_frac*100:.0f}% (σ={sigma:.6f}):")
        for run in range(n_runs):
            seed = 200 + int(sigma_frac * 100) * 10 + run
            random.seed(seed)
            np.random.seed(seed)
            noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
            X_noisy = np.clip(X + noise, 0, 1)  # keep in [0, 1] range

            tf.random.set_seed(seed)
            print(f"    Run {run+1}/{n_runs} (seed={seed}) ...", end=" ", flush=True)
            labels, _ = _train_dtc(X_noisy, n_clusters, seed=seed, save_dir=save_dir)
            ari = adjusted_rand_score(ref_labels, labels)
            ari_values.append(ari)
            print(f"ARI={ari:.4f}")

        mean_ari = np.mean(ari_values)
        perturbation_results[sigma_frac] = ari_values
        print(f"    Mean ARI at σ={sigma_frac*100:.0f}%: {mean_ari:.4f}")

    return perturbation_results


def save_metrics(mean_ari, std_ari, mean_sil, std_sil, n_negative,
                 optimal_k_sil, optimal_k_db, perturbation_results):
    """Save summary metrics to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        ("multi_run_mean_ari", f"{mean_ari:.4f}"),
        ("multi_run_std_ari", f"{std_ari:.4f}"),
        ("silhouette_mean", f"{mean_sil:.4f}"),
        ("silhouette_std", f"{std_sil:.4f}"),
        ("n_negative_silhouette_agents", str(n_negative)),
        ("optimal_k_silhouette", str(optimal_k_sil)),
        ("optimal_k_davies_bouldin", str(optimal_k_db)),
    ]
    for sigma_frac, ari_vals in sorted(perturbation_results.items()):
        rows.append((f"perturbation_ari_noise_{sigma_frac}", f"{np.mean(ari_vals):.4f}"))

    df = pd.DataFrame(rows, columns=["metric", "value"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved metrics to {OUTPUT_CSV}")


def plot_results(coassoc, all_labels, all_sil_samples, k_results, perturbation_results):
    """Generate 4-panel PDF figure."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mpl.rcParams["font.family"] = "Arial"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- Panel A: Co-association matrix heatmap ---
    ax = axes[0, 0]
    # Order by most frequent cluster assignment (mode across runs)
    label_matrix = np.stack(all_labels, axis=0)  # (n_runs, n_samples)
    # Mode label for each sample
    from scipy import stats
    mode_labels = stats.mode(label_matrix, axis=0).mode.ravel()
    sort_order = np.argsort(mode_labels)
    coassoc_sorted = coassoc[np.ix_(sort_order, sort_order)]

    im = ax.imshow(coassoc_sorted, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_title("A. Co-association matrix", fontsize=14)
    ax.set_xlabel("Agent index (sorted)", fontsize=14)
    ax.set_ylabel("Agent index (sorted)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-association frequency", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # --- Panel B: Silhouette bar chart ---
    ax = axes[0, 1]
    if all_sil_samples:
        # Use mean silhouette across runs for each sample
        stacked = np.stack(all_sil_samples, axis=0)
        mean_sil_per_agent = stacked.mean(axis=0)

        # Sort within each cluster
        colors_map = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
        # Use mode labels for coloring
        cluster_order = []
        for c in sorted(np.unique(mode_labels)):
            agents_in_c = np.where(mode_labels == c)[0]
            sils_in_c = mean_sil_per_agent[agents_in_c]
            sorted_idx = agents_in_c[np.argsort(sils_in_c)[::-1]]
            cluster_order.extend(sorted_idx)
        cluster_order = np.array(cluster_order)

        bar_colors = [colors_map.get(mode_labels[i], "#888888") for i in cluster_order]
        ax.bar(range(len(cluster_order)), mean_sil_per_agent[cluster_order],
               color=bar_colors, edgecolor="none", width=1.0)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
        ax.axhline(y=np.mean(mean_sil_per_agent), color="red", linewidth=1, linestyle="--",
                    label=f"Mean={np.mean(mean_sil_per_agent):.3f}")
        ax.legend(fontsize=12)
    ax.set_title("B. Per-agent silhouette scores", fontsize=14)
    ax.set_xlabel("Agent (sorted by cluster & silhouette)", fontsize=14)
    ax.set_ylabel("Mean silhouette coefficient", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # --- Panel C: Optimal k line plots ---
    ax = axes[1, 0]
    k_vals = sorted(k_results.keys())

    # Silhouette (left y-axis)
    sil_means = [np.mean(k_results[k]["silhouette"]) if k_results[k]["silhouette"] else np.nan
                 for k in k_vals]
    sil_stds = [np.std(k_results[k]["silhouette"]) if k_results[k]["silhouette"] else 0
                for k in k_vals]
    db_means = [np.mean(k_results[k]["davies_bouldin"]) if k_results[k]["davies_bouldin"] else np.nan
                for k in k_vals]
    db_stds = [np.std(k_results[k]["davies_bouldin"]) if k_results[k]["davies_bouldin"] else 0
               for k in k_vals]
    ch_means = [np.mean(k_results[k]["calinski_harabasz"]) if k_results[k]["calinski_harabasz"] else np.nan
                for k in k_vals]
    ch_stds = [np.std(k_results[k]["calinski_harabasz"]) if k_results[k]["calinski_harabasz"] else 0
               for k in k_vals]

    color_sil = "#1f77b4"
    color_db = "#ff7f0e"
    color_ch = "#2ca02c"

    ax.errorbar(k_vals, sil_means, yerr=sil_stds, marker="o", color=color_sil,
                label="Silhouette", capsize=3, linewidth=2)
    ax.set_xlabel("Number of clusters (k)", fontsize=14)
    ax.set_ylabel("Silhouette score", fontsize=14, color=color_sil)
    ax.tick_params(axis="y", labelcolor=color_sil, labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xticks(k_vals)

    ax2 = ax.twinx()
    ax2.errorbar(k_vals, db_means, yerr=db_stds, marker="s", color=color_db,
                 label="Davies-Bouldin", capsize=3, linewidth=2)
    ax2.set_ylabel("Davies-Bouldin index", fontsize=14, color=color_db)
    ax2.tick_params(axis="y", labelcolor=color_db, labelsize=12)

    # Add CH on a third axis (inset text since matplotlib doesn't do triple axes well)
    # Instead, plot CH as normalized values on the left axis legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

    ax.set_title("C. Internal validity vs k", fontsize=14)

    # --- Panel D: Perturbation ARI box plots ---
    ax = axes[1, 1]
    noise_levels = sorted(perturbation_results.keys())
    box_data = [perturbation_results[nl] for nl in noise_levels]
    noise_pct = [f"{nl*100:.0f}%" for nl in noise_levels]

    bp = ax.boxplot(box_data, labels=noise_pct, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a8d8ea")
        patch.set_edgecolor("#1f77b4")
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)

    ax.axhline(y=0.7, color="green", linewidth=1, linestyle="--", alpha=0.7,
               label="Stability threshold (0.7)")
    ax.set_title("D. Perturbation robustness", fontsize=14)
    ax.set_xlabel("Noise level (% of global std)", fontsize=14)
    ax.set_ylabel("ARI vs reference", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=11)

    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF}")


def print_summary(mean_ari, std_ari, mean_sil, std_sil, n_negative,
                  optimal_k_sil, optimal_k_db, perturbation_results):
    """Print interpretive summary of all metrics."""
    print(f"\n{'='*60}")
    print("ROBUSTNESS EVALUATION SUMMARY")
    print(f"{'='*60}")

    # Multi-run stability
    if mean_ari > 0.7:
        stability = "STABLE"
    elif mean_ari > 0.5:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"
    print(f"\n1. Multi-run stability: {stability}")
    print(f"   Mean pairwise ARI = {mean_ari:.4f} ± {std_ari:.4f}")
    print(f"   (>0.7 = stable, 0.5-0.7 = moderate, <0.5 = unstable)")

    # Silhouette
    if mean_sil > 0.5:
        sil_quality = "GOOD"
    elif mean_sil > 0.25:
        sil_quality = "FAIR"
    else:
        sil_quality = "POOR"
    print(f"\n2. Silhouette analysis: {sil_quality}")
    print(f"   Mean silhouette = {mean_sil:.4f} ± {std_sil:.4f}")
    print(f"   Agents with negative silhouette: {n_negative}")
    print(f"   (>0.5 = good, 0.25-0.5 = fair, <0.25 = poor)")

    # Optimal k
    print(f"\n3. Optimal k selection:")
    print(f"   Best k by silhouette: {optimal_k_sil}")
    print(f"   Best k by Davies-Bouldin: {optimal_k_db}")
    if optimal_k_sil == optimal_k_db:
        print(f"   Both metrics agree on k={optimal_k_sil}")
    else:
        print(f"   Metrics disagree — consider k={optimal_k_sil} or k={optimal_k_db}")

    # Perturbation
    print(f"\n4. Perturbation robustness:")
    for sigma_frac in sorted(perturbation_results.keys()):
        ari_vals = perturbation_results[sigma_frac]
        mean = np.mean(ari_vals)
        robust = "ROBUST" if mean > 0.7 else "SENSITIVE"
        print(f"   σ={sigma_frac*100:.0f}%: ARI={mean:.4f} ({robust})")
    print(f"   (>0.7 at σ=3% indicates robustness)")

    print(f"\n{'='*60}")


def main():
    # Load and prepare data
    agent_ids, years, values = load_data()
    print(f"Loaded {len(agent_ids)} agents, {len(years)} years ({years[0]}-{years[-1]})")

    values_norm, vmin, vmax = normalize(values)
    X = values_norm.reshape(len(agent_ids), TIMESTEPS, INPUT_DIM).astype(np.float32)
    print(f"Input shape: {X.shape}")

    # 1. Multi-run stability
    all_labels, all_dtc_models, mean_ari, std_ari, coassoc = run_multirun_stability(X)

    # 2. Silhouette analysis
    mean_sil, std_sil, n_negative, all_sil_samples = run_silhouette_analysis(
        X, all_labels, all_dtc_models
    )

    # 3. Optimal k selection
    k_results, optimal_k_sil, optimal_k_db = run_optimal_k(X)

    # 4. Perturbation robustness
    perturbation_results = run_perturbation_robustness(X)

    # Save outputs
    save_metrics(mean_ari, std_ari, mean_sil, std_sil, n_negative,
                 optimal_k_sil, optimal_k_db, perturbation_results)

    plot_results(coassoc, all_labels, all_sil_samples, k_results, perturbation_results)

    print_summary(mean_ari, std_ari, mean_sil, std_sil, n_negative,
                  optimal_k_sil, optimal_k_db, perturbation_results)


if __name__ == "__main__":
    main()
