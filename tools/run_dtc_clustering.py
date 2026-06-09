"""Run Deep Temporal Clustering on annual irrigation depth time series.

Clusters 43 irrigation agents based on their 1993-2020
annual irrigation depth using the DTC method (Forest et al.).

Usage:
    python tools/run_dtc_clustering.py          # default k=3
    python tools/run_dtc_clustering.py --k 2    # specify k
    python tools/run_dtc_clustering.py --k 2 --year-end 2004  # pre-CP only

Outputs:
    results/dtc_cluster_assignments_k{K}.csv  — agent IDs with cluster labels
    results/dtc_clustered_agents_k{K}.pdf     — visualization (one panel per cluster)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# Suppress TF info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.dtc.DeepTemporalClustering import DTC

# --- Paths ---
INPUT_CSV = BASE_DIR / "data" / "annual_irrigation_depth.csv"
RESULTS_DIR = BASE_DIR / "results"
TMP_DIR = BASE_DIR / ".tmp" / "dtc"
# --- Architecture parameters ---
N_CLUSTERS = 3
INPUT_DIM = 1         # univariate
TIMESTEPS = 28        # 1993-2020
N_FILTERS = 16        # reduced from 50 for 43 samples
KERNEL_SIZE = 5       # captures ~5-year patterns
POOL_SIZE = 7         # must divide 28; gives encoded length 4
N_UNITS = [10, 1]     # reduced from [50, 1]
DIST_METRIC = "eucl"
CLUSTER_INIT = "kmeans"
PRETRAIN_EPOCHS = 100
EPOCHS = 200
BATCH_SIZE = 43       # full-batch (dataset is tiny)
GAMMA = 1.0
EVAL_EPOCHS = 1
SAVE_EPOCHS = 50
TOL = 0.001
PATIENCE = 10


def load_data(year_start=None, year_end=None):
    """Load CSV and return agent IDs, year labels, and raw values array.

    Parameters
    ----------
    year_start : int, optional
        First year to include (default: first year in data).
    year_end : int, optional
        Last year to include (default: last year in data).
    """
    df = pd.read_csv(INPUT_CSV)
    agent_ids = df.iloc[:, 0].values
    all_years = [int(c) for c in df.columns[1:]]

    # Filter year columns
    if year_start is not None or year_end is not None:
        ys = year_start if year_start is not None else all_years[0]
        ye = year_end if year_end is not None else all_years[-1]
        col_mask = [y for y in all_years if ys <= y <= ye]
        col_indices = [all_years.index(y) for y in col_mask]
        years = col_mask
        values = df.iloc[:, [i + 1 for i in col_indices]].values.astype(np.float64)
    else:
        years = all_years
        values = df.iloc[:, 1:].values.astype(np.float64)

    return agent_ids, years, values


def _auto_pool_size(timesteps, target_encoded_len=4):
    """Find a pool_size that divides timesteps and gives the target encoded length.

    For 28 years: pool_size=7, encoded_len=4
    For 12 years: pool_size=3, encoded_len=4
    Falls back to the largest divisor that gives encoded_len >= target.
    """
    # First try exact match
    if timesteps % target_encoded_len == 0:
        return timesteps // target_encoded_len

    # Find divisors and pick the one closest to target_encoded_len
    divisors = [d for d in range(1, timesteps + 1) if timesteps % d == 0]
    # pool_size candidates that give encoded_len = timesteps // pool_size
    best = None
    for d in divisors:
        encoded_len = timesteps // d
        if best is None or abs(encoded_len - target_encoded_len) < abs((timesteps // best) - target_encoded_len):
            best = d
    return best


def normalize(values):
    """Global min-max normalization to [0, 1]."""
    vmin = values.min()
    vmax = values.max()
    return (values - vmin) / (vmax - vmin), vmin, vmax


def run_clustering(X, n_clusters=N_CLUSTERS, timesteps=TIMESTEPS,
                    pool_size=POOL_SIZE, seed=None):
    """Initialize DTC, pretrain autoencoder, and run joint clustering.

    Parameters
    ----------
    seed : int, optional
        If provided, fixes the random/numpy/tensorflow seeds for a
        reproducible fit. Default ``None`` keeps the original
        (non-deterministic) behavior.
    """
    if seed is not None:
        import random
        import tensorflow as tf
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    dtc = DTC(
        n_clusters=n_clusters,
        input_dim=INPUT_DIM,
        timesteps=timesteps,
        n_filters=N_FILTERS,
        kernel_size=KERNEL_SIZE,
        strides=1,
        pool_size=pool_size,
        n_units=N_UNITS,
        alpha=1.0,
        dist_metric=DIST_METRIC,
        cluster_init=CLUSTER_INIT,
        heatmap=False,
    )
    dtc.initialize()
    dtc.model.summary()
    dtc.compile(gamma=GAMMA, optimizer="adam")

    save_dir = str(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Pretrain autoencoder
    dtc.pretrain(
        X=X,
        optimizer="adam",
        epochs=PRETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=save_dir,
        verbose=1,
    )

    # Initialize cluster centers
    dtc.init_cluster_weights(X)

    # Joint clustering + reconstruction training
    dtc.fit(
        X_train=X,
        y_train=None,
        epochs=EPOCHS,
        eval_epochs=EVAL_EPOCHS,
        save_epochs=SAVE_EPOCHS,
        batch_size=BATCH_SIZE,
        tol=TOL,
        patience=PATIENCE,
        save_dir=save_dir,
    )

    labels = dtc.predict(X)
    return labels


def sort_clusters(labels, values, n_clusters=N_CLUSTERS):
    """Re-label clusters so that Cluster 1 has the lowest mean irrigation."""
    means = []
    for c in range(n_clusters):
        mask = labels == c
        means.append(values[mask].mean())
    order = np.argsort(means)  # ascending mean irrigation
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels])


def save_assignments(agent_ids, labels, output_csv):
    """Save cluster assignments to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"AgentID": agent_ids, "Cluster": labels + 1})  # 1-indexed
    df.to_csv(output_csv, index=False)
    print(f"Saved cluster assignments to {output_csv}")


def plot_clusters(agent_ids, years, values, labels, n_clusters, output_pdf):
    """Generate figure with one panel per cluster."""
    mpl.rcParams["font.family"] = "Arial"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5), sharey=True)
    if n_clusters == 1:
        axes = [axes]

    for c in range(n_clusters):
        ax = axes[c]
        mask = labels == c
        n_agents = mask.sum()

        # Individual agent lines
        for row in values[mask]:
            ax.plot(years, row, color=colors[c % len(colors)], alpha=0.6, linewidth=0.8)

        # Cluster mean
        cluster_mean = values[mask].mean(axis=0)
        ax.plot(years, cluster_mean, color="black", linewidth=2, linestyle="--")

        ax.set_title(f"Cluster {c + 1} (n={n_agents})", fontsize=16)
        ax.set_xlabel("Year", fontsize=16)
        if c == 0:
            ax.set_ylabel("Annual Irrigation Depth (mm)", fontsize=16)
        # Dynamic x-axis ticks based on year range
        yr_min, yr_max = years[0], years[-1]
        ticks = [y for y in range(yr_min, yr_max + 1) if y % 5 == 0]
        ax.set_xticks(ticks)
        ax.set_xlim(yr_min, yr_max)
        ax.tick_params(axis="both", labelsize=14)

    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved figure to {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Run DTC clustering")
    parser.add_argument("--k", type=int, default=N_CLUSTERS,
                        help=f"Number of clusters (default: {N_CLUSTERS})")
    parser.add_argument("--year-start", type=int, default=None,
                        help="First year to include (default: first year in data)")
    parser.add_argument("--year-end", type=int, default=None,
                        help="Last year to include (default: last year in data)")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Pool size for DTC (default: auto-select to match encoded_len=4)")
    args = parser.parse_args()
    n_clusters = args.k

    # Build output filename tag
    agent_ids, years, values = load_data(
        year_start=args.year_start, year_end=args.year_end
    )
    year_tag = f"_{years[0]}-{years[-1]}" if (args.year_start or args.year_end) else ""
    output_csv = RESULTS_DIR / f"dtc_cluster_assignments_k{n_clusters}{year_tag}.csv"
    output_pdf = RESULTS_DIR / f"dtc_clustered_agents_k{n_clusters}{year_tag}.pdf"

    print(f"Loaded {len(agent_ids)} agents, {len(years)} years ({years[0]}-{years[-1]})")

    # Determine architecture parameters
    timesteps = len(years)
    if args.pool_size is not None:
        pool_size = args.pool_size
    else:
        pool_size = _auto_pool_size(timesteps)
    encoded_len = timesteps // pool_size
    print(f"Architecture: timesteps={timesteps}, pool_size={pool_size}, "
          f"encoded_len={encoded_len}")

    # Normalize and reshape for DTC: (n_samples, timesteps, 1)
    values_norm, vmin, vmax = normalize(values)
    X = values_norm.reshape(len(agent_ids), timesteps, INPUT_DIM).astype(np.float32)
    print(f"Input shape: {X.shape}, min={X.min():.4f}, max={X.max():.4f}")
    print(f"Running DTC with k={n_clusters}")

    # Run DTC
    labels = run_clustering(X, n_clusters=n_clusters, timesteps=timesteps,
                            pool_size=pool_size)

    # Sort clusters by mean irrigation depth (ascending)
    labels = sort_clusters(labels, values, n_clusters=n_clusters)

    # Save and plot (using original un-normalized values)
    save_assignments(agent_ids, labels, output_csv)
    plot_clusters(agent_ids, years, values, labels, n_clusters, output_pdf)

    # Summary
    for c in range(n_clusters):
        mask = labels == c
        ids = agent_ids[mask]
        print(f"Cluster {c + 1}: {len(ids)} agents — {sorted(ids)}")


if __name__ == "__main__":
    main()
