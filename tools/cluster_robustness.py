"""
Cluster robustness analysis for DTC clustering.

Runs DTC multiple times at each k value and computes:
1. Co-association matrix (pairwise co-clustering frequency)
2. Consensus silhouette scores
3. Stability metrics (adjusted Rand index between runs)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.run_dtc_clustering import load_data, run_clustering

# Root seed for reproducibility (matches the pipeline-wide MASTER_SEED=42).
MASTER_SEED = 42


def run_multiple(k, n_runs, X, timesteps, pool_size, verbose=False,
                 master_seed=MASTER_SEED):
    """Run DTC clustering n_runs times (seeded) and collect label arrays.

    Each run uses a fixed seed ``master_seed + i`` so the full sweep is
    reproducible on this machine.
    """
    all_labels = []
    for i in range(n_runs):
        if verbose:
            print(f"  k={k}, run {i+1}/{n_runs} (seed={master_seed + i})")
        # Suppress TF output
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        labels = run_clustering(X, n_clusters=k, timesteps=timesteps,
                                pool_size=pool_size, seed=master_seed + i)
        all_labels.append(labels)
    return np.array(all_labels)  # shape (n_runs, n_agents)


def co_association_matrix(all_labels):
    """Compute co-association matrix: fraction of runs where each pair clusters together."""
    n_runs, n_agents = all_labels.shape
    coassoc = np.zeros((n_agents, n_agents))
    for labels in all_labels:
        for i in range(n_agents):
            for j in range(i, n_agents):
                if labels[i] == labels[j]:
                    coassoc[i, j] += 1
                    coassoc[j, i] += 1
    coassoc /= n_runs
    np.fill_diagonal(coassoc, 1.0)
    return coassoc


def pairwise_ari(all_labels):
    """Compute pairwise Adjusted Rand Index between all runs."""
    n_runs = len(all_labels)
    ari_values = []
    for i, j in combinations(range(n_runs), 2):
        ari_values.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    return np.array(ari_values)


def consensus_labels(coassoc, k):
    """Derive consensus labels from co-association matrix using hierarchical clustering."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = 1 - coassoc
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)  # numerical safety
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=k, criterion="maxclust")


def main():
    parser = argparse.ArgumentParser(description="Cluster robustness analysis")
    parser.add_argument("--k-range", type=int, nargs=2, default=[2, 5],
                        help="Range of k values (inclusive)")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of runs per k")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load data
    agent_ids_arr, years, values = load_data()
    agent_ids = list(agent_ids_arr)
    X = values.astype(np.float64)
    X_min, X_max = X.min(), X.max()
    X = (X - X_min) / (X_max - X_min + 1e-12)
    timesteps = X.shape[1]
    pool_size = max(1, timesteps // 4)
    X = X.reshape(X.shape[0], timesteps, 1)

    k_min, k_max = args.k_range
    results = []
    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    for k in range(k_min, k_max + 1):
        print(f"\n{'='*60}")
        print(f"Running k={k} ({args.n_runs} runs)")
        print(f"{'='*60}")

        all_labels = run_multiple(k, args.n_runs, X, timesteps, pool_size,
                                  verbose=args.verbose)

        # Co-association matrix
        coassoc = co_association_matrix(all_labels)

        # Pairwise ARI (stability)
        ari_vals = pairwise_ari(all_labels)
        mean_ari = ari_vals.mean()
        std_ari = ari_vals.std()

        # Consensus labels
        cons_labels = consensus_labels(coassoc, k)

        # Silhouette on original data using consensus labels
        X_2d = X.reshape(X.shape[0], -1)
        if len(np.unique(cons_labels)) > 1:
            sil = silhouette_score(X_2d, cons_labels)
        else:
            sil = -1.0

        # Cluster sizes from consensus
        unique, counts = np.unique(cons_labels, return_counts=True)
        size_str = ", ".join(f"C{u}:{c}" for u, c in zip(unique, counts))
        # Plain "38, 5" form (descending) for the Table S2 column
        size_list = ", ".join(str(c) for c in sorted(counts, reverse=True))

        # --- Medoid (most representative) single run ---
        # Pick the run with the highest mean pairwise ARI to the other runs.
        # Average-linkage consensus peels off singleton outliers when stability
        # is low; the medoid is an actual DTC partition, matching how the
        # canonical k=2 partition is defined (a single representative run).
        n_runs = len(all_labels)
        ari_mat = np.ones((n_runs, n_runs))
        for a, b in combinations(range(n_runs), 2):
            v = adjusted_rand_score(all_labels[a], all_labels[b])
            ari_mat[a, b] = ari_mat[b, a] = v
        mean_to_others = (ari_mat.sum(axis=1) - 1.0) / max(n_runs - 1, 1)
        medoid_idx = int(np.argmax(mean_to_others))
        medoid_seed = MASTER_SEED + medoid_idx
        medoid_labels = all_labels[medoid_idx]
        if len(np.unique(medoid_labels)) > 1:
            medoid_sil = silhouette_score(X_2d, medoid_labels)
        else:
            medoid_sil = -1.0
        m_unique, m_counts = np.unique(medoid_labels, return_counts=True)
        medoid_size_list = ", ".join(str(c) for c in sorted(m_counts, reverse=True))
        medoid_members = {
            int(c): [agent_ids[i] for i in range(len(agent_ids))
                     if medoid_labels[i] == c]
            for c in m_unique
        }
        print(f"  Medoid run: idx={medoid_idx} (seed={medoid_seed}), "
              f"silhouette={medoid_sil:.3f}, sizes={medoid_size_list}")

        # Persist raw per-run labels so any partition rule can be recomputed
        # offline without re-fitting DTC.
        pd.DataFrame(all_labels, columns=list(agent_ids)).to_csv(
            os.path.join(outdir, f"cluster_runs_labels_k{k}.csv"), index=False)

        # Which agents in each cluster
        cluster_members = {}
        for c in unique:
            members = [agent_ids[i] for i in range(len(agent_ids))
                       if cons_labels[i] == c]
            cluster_members[c] = members

        print(f"\nk={k} Summary:")
        print(f"  Mean ARI (stability): {mean_ari:.3f} ± {std_ari:.3f}")
        print(f"  Consensus silhouette: {sil:.3f}")
        print(f"  Cluster sizes: {size_str}")
        for c, members in cluster_members.items():
            print(f"  Cluster {c}: {members}")

        # Check if agents 4, 11 are together
        a4_cluster = cons_labels[agent_ids.index(4)]
        a11_cluster = cons_labels[agent_ids.index(11)]
        together = a4_cluster == a11_cluster
        print(f"  Agents 4 & 11 together: {together} (cluster {a4_cluster}, {a11_cluster})")

        results.append({
            "k": k,
            "mean_ari": mean_ari,
            "std_ari": std_ari,
            "silhouette": sil,
            "sizes": size_str,
            "size_list": size_list,
            "medoid_idx": medoid_idx,
            "medoid_seed": medoid_seed,
            "medoid_silhouette": medoid_sil,
            "medoid_size_list": medoid_size_list,
            "medoid_members": medoid_members,
            "agents_4_11_together": together,
            "coassoc": coassoc,
            "cons_labels": cons_labels,
            "cluster_members": cluster_members,
        })

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'k':>3} {'ARI (stab)':>12} {'Silhouette':>12} {'Sizes':>25} {'4&11 together':>15}")
    for r in results:
        print(f"{r['k']:>3} {r['mean_ari']:>8.3f}±{r['std_ari']:.3f} {r['silhouette']:>12.3f} {r['sizes']:>25} {str(r['agents_4_11_together']):>15}")

    # Save results (outdir defined above)
    summary_rows = []
    for r in results:
        summary_rows.append({
            "k": r["k"],
            "mean_ari": round(r["mean_ari"], 4),
            "std_ari": round(r["std_ari"], 4),
            "silhouette": round(r["silhouette"], 4),
            "cluster_sizes": r["sizes"],
            "agents_4_11_together": r["agents_4_11_together"],
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(outdir, "cluster_robustness_summary.csv"), index=False)

    # --- Authoritative source for SI Table S2 (one row per k) ---
    # The Table S2 columns use the MEDOID (representative single run): silhouette
    # and cluster sizes of an actual DTC partition, paired with the ensemble
    # stability ARI (mean ± SD pairwise across the 10 seeded runs). Consensus
    # columns are kept alongside for diagnostic comparison.
    s2_rows = []
    for r in results:
        s2_rows.append({
            "k": r["k"],
            "medoid_silhouette": round(r["medoid_silhouette"], 3),
            "medoid_sizes": r["medoid_size_list"],
            "medoid_seed": r["medoid_seed"],
            "mean_ari": round(r["mean_ari"], 3),
            "std_ari": round(r["std_ari"], 3),
            "consensus_silhouette": round(r["silhouette"], 3),
            "consensus_sizes": r["size_list"],
        })
    pd.DataFrame(s2_rows).to_csv(
        os.path.join(outdir, "cluster_quality_sweep.csv"), index=False)
    print("Saved Table S2 source to results/cluster_quality_sweep.csv")

    # --- Membership per k (medoid + consensus, so partitions are auditable) ---
    member_rows = []
    for r in results:
        for c, members in r["medoid_members"].items():
            member_rows.append({
                "k": r["k"], "method": "medoid", "cluster": int(c),
                "size": len(members),
                "agents": " ".join(str(a) for a in sorted(members)),
            })
        for c, members in r["cluster_members"].items():
            member_rows.append({
                "k": r["k"], "method": "consensus", "cluster": int(c),
                "size": len(members),
                "agents": " ".join(str(a) for a in sorted(members)),
            })
    pd.DataFrame(member_rows).to_csv(
        os.path.join(outdir, "cluster_quality_membership.csv"), index=False)
    print("Saved membership to results/cluster_quality_membership.csv")

    # Plot co-association matrices
    n_k = len(results)
    fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 4.5), constrained_layout=True)
    if n_k == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        im = ax.imshow(r["coassoc"], cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"k={r['k']}\nARI={r['mean_ari']:.3f}, Sil={r['silhouette']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Agent index")
        ax.set_ylabel("Agent index")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Co-association frequency")
    fig.savefig(os.path.join(outdir, "cluster_robustness_coassociation.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "cluster_robustness_coassociation.png"),
                dpi=150, bbox_inches="tight")

    # Save raw co-association matrix as CSV for each k
    for r in results:
        coassoc_df = pd.DataFrame(r["coassoc"],
                                   index=agent_ids, columns=agent_ids)
        coassoc_path = os.path.join(outdir, f"cluster_coassociation_k{r['k']}.csv")
        coassoc_df.to_csv(coassoc_path)
        print(f"Saved co-association matrix: {coassoc_path}")

    print(f"\nSaved co-association plot to results/cluster_robustness_coassociation.pdf")
    print(f"Saved summary to results/cluster_robustness_summary.csv")


if __name__ == "__main__":
    main()
