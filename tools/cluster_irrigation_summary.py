"""Summarize cluster membership, mean annual irrigation depths, and changepoints.

Loads DTC k=2 cluster assignments, computes per-agent and cluster-mean annual
irrigation depth, runs BOCPD on cluster-mean time series, and produces summary
CSVs and a two-panel figure.

Usage:
    python tools/cluster_irrigation_summary.py
    python tools/cluster_irrigation_summary.py --threshold 0.3

Outputs:
    results/cluster_irrigation_summary.csv   — per-agent annual irrigation + cluster means
    results/cluster_mean_changepoint.csv     — cluster-mean time series with BOCPD probs
    results/cluster_irrigation_summary.pdf   — two-panel figure
    results/cluster_irrigation_summary.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.run_changepoint_detection import (
    run_offline_detection,
    run_offline_slope_detection,
)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ASSIGNMENTS_CSV = RESULTS_DIR / "dtc_cluster_assignments_k2.csv"
DEFAULT_THRESHOLD = 0.5


def load_agent_annual_irrigation(agent_id):
    """Load agent CSV and compute total annual irrigation depth (mm/year)."""
    path = DATA_DIR / f"agentdata_{agent_id}.csv"
    df = pd.read_csv(path)
    annual = df.groupby("Year")["Irrigation_Depth"].sum().sort_index()
    return annual


def build_summary(assignments_df, threshold=DEFAULT_THRESHOLD):
    """Build per-agent annual irrigation and cluster-mean summaries.

    Returns
    -------
    agent_annual : pd.DataFrame
        Columns: Year, Agent_{id} for each agent, ClusterMean_{cid} for each cluster
    cluster_means : dict
        {cluster_id: pd.Series} of cluster-mean annual irrigation indexed by Year
    agent_overall : pd.DataFrame
        Per-agent overall mean irrigation and cluster assignment
    """
    agent_ids = assignments_df["AgentID"].values
    clusters = assignments_df.set_index("AgentID")["Cluster"]

    # Collect per-agent annual irrigation
    agent_series = {}
    for aid in sorted(agent_ids):
        agent_series[aid] = load_agent_annual_irrigation(aid)

    # Build wide DataFrame
    frames = {f"Agent_{aid}": s for aid, s in agent_series.items()}
    agent_annual = pd.DataFrame(frames)
    agent_annual.index.name = "Year"

    # Compute cluster means
    cluster_means = {}
    for cid in sorted(clusters.unique()):
        cid_agents = sorted(clusters[clusters == cid].index.tolist())
        cols = [f"Agent_{a}" for a in cid_agents]
        cluster_means[cid] = agent_annual[cols].mean(axis=1)
        agent_annual[f"ClusterMean_{cid}"] = cluster_means[cid]

    agent_annual = agent_annual.reset_index()

    # Per-agent overall mean + cluster assignment
    rows = []
    for aid in sorted(agent_ids):
        rows.append({
            "AgentID": aid,
            "Cluster": clusters[aid],
            "MeanAnnualIrrigation_mm": agent_series[aid].mean(),
        })
    agent_overall = pd.DataFrame(rows)

    return agent_annual, cluster_means, agent_overall


def run_bocpd_on_clusters(cluster_means, threshold=DEFAULT_THRESHOLD):
    """Run offline BOCPD (levels + slope) on each cluster-mean time series.

    Returns
    -------
    cp_results : dict
        {cluster_id: dict with keys years, values, level_probs, slope_probs, combined_probs}
    """
    cp_results = {}
    for cid in sorted(cluster_means.keys()):
        series = cluster_means[cid]
        years = series.index.values
        values = series.values

        print(f"\nCluster {cid}: running BOCPD on mean irrigation "
              f"({len(years)} years, range [{values.min():.1f}, {values.max():.1f}] mm)")

        level_probs = run_offline_detection(values)
        slope_probs = run_offline_slope_detection(values)
        combined = np.maximum(level_probs, slope_probs)

        # Report detections
        for name, probs in [("level", level_probs), ("slope", slope_probs)]:
            detected = [
                (years[i], years[i + 1], probs[i])
                for i in range(len(probs))
                if probs[i] >= threshold
            ]
            if detected:
                print(f"  {name.capitalize()} changepoints (threshold={threshold}):")
                for y1, y2, p in detected:
                    print(f"    Between {y1} and {y2}: p={p:.4f}")
            else:
                print(f"  No {name} changepoints above threshold {threshold}")

        cp_results[cid] = {
            "years": years,
            "values": values,
            "level_probs": level_probs,
            "slope_probs": slope_probs,
            "combined_probs": combined,
        }

    return cp_results


def save_changepoint_csv(cp_results, threshold, output_path):
    """Save cluster-mean changepoint results to CSV."""
    frames = []
    for cid in sorted(cp_results.keys()):
        r = cp_results[cid]
        years = r["years"]
        # Pad probs (length T-1) to T by appending 0
        level_padded = np.append(r["level_probs"], 0.0)
        slope_padded = np.append(r["slope_probs"], 0.0)
        combined_padded = np.append(r["combined_probs"], 0.0)

        df = pd.DataFrame({
            "Year": years,
            f"Cluster{cid}_MeanIrrigation_mm": np.round(r["values"], 2),
            f"Cluster{cid}_CP_Prob_Level": np.round(level_padded, 6),
            f"Cluster{cid}_CP_Prob_Slope": np.round(slope_padded, 6),
            f"Cluster{cid}_CP_Prob_Combined": np.round(combined_padded, 6),
            f"Cluster{cid}_CP_Detected": (combined_padded >= threshold).astype(int),
        })
        frames.append(df)

    # Merge on Year
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="Year", how="outer")
    result = result.sort_values("Year").reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return result


def plot_summary(cp_results, threshold, output_pdf):
    """Two-panel figure: cluster-mean irrigation with BOCPD changepoint bars."""
    from matplotlib.patches import Patch

    orig_rc = mpl.rcParams.copy()
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
    })

    n_clusters = len(cp_results)
    fig, axes = plt.subplots(
        n_clusters, 1,
        figsize=(12, 5 * n_clusters),
        constrained_layout=True,
        squeeze=False,
    )

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for row, cid in enumerate(sorted(cp_results.keys())):
        r = cp_results[cid]
        years = r["years"]
        values = r["values"]
        level_probs = r["level_probs"]
        slope_probs = r["slope_probs"]

        # Pad probs to match years length (prepend 0)
        if len(level_probs) == len(years) - 1:
            level_padded = np.insert(level_probs, 0, 0.0)
            slope_padded = np.insert(slope_probs, 0, 0.0)
        else:
            level_padded = level_probs
            slope_padded = slope_probs

        ax = axes[row, 0]

        # Left axis: irrigation time series
        line_irr, = ax.plot(
            years, values, "ko-", linewidth=1.2, markersize=5,
            label="Mean Irrigation Depth", zorder=5,
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Irrigation Depth (mm)")
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)

        # Right axis: BOCPD probabilities
        ax_r = ax.twinx()
        bar_width = 0.4
        ax_r.bar(
            years - bar_width / 2, level_padded, width=bar_width,
            color="#ff7f0e", alpha=0.7, label="Mean/Var Shift",
        )
        ax_r.bar(
            years + bar_width / 2, slope_padded, width=bar_width,
            color="#2ca02c", alpha=0.7, label="Slope Change",
        )
        ax_r.set_ylabel("Posterior Probability")
        ax_r.set_ylim(0.0, 1.05)

        # Threshold line
        thresh_line = ax_r.axhline(
            y=threshold, color="red", linestyle="--", linewidth=1.5,
        )

        # Mark detected changepoints (combined)
        combined_padded = np.maximum(level_padded, slope_padded)
        cp_line = None
        for i in range(len(r["combined_probs"])):
            if r["combined_probs"][i] >= threshold:
                cp_line = ax.axvline(
                    x=years[i] + 0.5, color="red", linestyle=":",
                    linewidth=1.5, zorder=1,
                )

        # Legend
        bar_level = Patch(facecolor="#ff7f0e", alpha=0.7)
        bar_slope = Patch(facecolor="#2ca02c", alpha=0.7)
        handles = [line_irr, bar_level, bar_slope, thresh_line]
        labels = ["Mean Irrigation Depth", "Mean/Var Shift", "Slope Change",
                   f"Threshold = {threshold}"]
        if cp_line is not None:
            handles.append(cp_line)
            labels.append("Detected Changepoint")
        ax.legend(handles, labels, loc="upper left", fontsize=11,
                  frameon=False, bbox_to_anchor=(0.0, 1.01))

        # Line on top of bars
        ax.set_zorder(ax_r.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.set_title(f"Cluster {cid}", fontsize=14, fontweight="bold")
        ax.text(0.01, 0.99, panel_labels[row], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

        # Grid on left axis
        ax.grid(True, alpha=0.18, linewidth=0.6)
        ax.tick_params(length=4)

    output_png = output_pdf.with_suffix(".png")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")

    mpl.rcParams.update(orig_rc)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster irrigation summary with BOCPD changepoint analysis"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="BOCPD probability threshold (default: 0.5)")
    args = parser.parse_args()
    threshold = args.threshold

    # 1. Load cluster assignments
    assignments = pd.read_csv(ASSIGNMENTS_CSV)
    print(f"Loaded {len(assignments)} agents from {ASSIGNMENTS_CSV.name}")
    for cid in sorted(assignments["Cluster"].unique()):
        agents = sorted(assignments[assignments["Cluster"] == cid]["AgentID"].tolist())
        print(f"  Cluster {cid}: {len(agents)} agents — {agents}")

    # 2. Build per-agent and cluster-mean summaries
    agent_annual, cluster_means, agent_overall = build_summary(assignments, threshold)

    # Save per-agent summary CSV
    summary_csv = RESULTS_DIR / "cluster_irrigation_summary.csv"
    agent_annual.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    # Print console summary
    print("\n" + "=" * 70)
    print("PER-AGENT MEAN ANNUAL IRRIGATION DEPTH (mm)")
    print("=" * 70)
    for cid in sorted(agent_overall["Cluster"].unique()):
        subset = agent_overall[agent_overall["Cluster"] == cid].sort_values("AgentID")
        cluster_mean = subset["MeanAnnualIrrigation_mm"].mean()
        print(f"\nCluster {cid} (n={len(subset)} agents, cluster mean = {cluster_mean:.1f} mm):")
        for _, row in subset.iterrows():
            print(f"  Agent {int(row['AgentID']):3d}: {row['MeanAnnualIrrigation_mm']:.1f} mm")

    # 3. Run BOCPD on cluster means
    cp_results = run_bocpd_on_clusters(cluster_means, threshold)

    # 4. Save changepoint CSV
    cp_csv = RESULTS_DIR / "cluster_mean_changepoint.csv"
    save_changepoint_csv(cp_results, threshold, cp_csv)

    # 5. Plot
    output_pdf = RESULTS_DIR / "cluster_irrigation_summary.pdf"
    plot_summary(cp_results, threshold, output_pdf)

    # Final summary
    print("\n" + "=" * 70)
    print("CHANGEPOINT SUMMARY")
    print("=" * 70)
    for cid in sorted(cp_results.keys()):
        r = cp_results[cid]
        years = r["years"]
        combined = r["combined_probs"]
        detected = [(years[i], years[i + 1], combined[i])
                    for i in range(len(combined)) if combined[i] >= threshold]
        n_agents = (assignments["Cluster"] == cid).sum()
        mean_irr = r["values"].mean()
        print(f"\nCluster {cid} ({n_agents} agents, overall mean = {mean_irr:.1f} mm):")
        if detected:
            for y1, y2, p in detected:
                print(f"  Changepoint between {y1}-{y2}: combined p={p:.4f}")
        else:
            print("  No changepoints detected")


if __name__ == "__main__":
    main()
