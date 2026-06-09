#!/usr/bin/env python3
"""
Fig 5 (main): Transition-Window Robustness Analysis (per-agent CP design).

Panel (a): Heatmap of M2 KGE wins across CP-1, CP, CP+1 per agent.
Panel (b): Per-agent M2 KGE across the CP transition window.
Panel (c): Per-agent KGE comparison (M1 vs M2) at CP*.
Panel (d): Per-agent KGE improvement (M2 − M1) at CP*.

Data sources:
  results/transition_window/summary.csv
  results/transition_window/per_agent_detail.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "results" / "figures"

AGENT_CP = {
    2: 2013, 24: 2003, 28: 2011, 29: 2011,
    12: 2003, 14: 2004, 18: 2003, 20: 2004,
}
AGENT_ORDER = [2, 12, 14, 18, 20, 24, 28, 29]

RC = {
    "font.family": "Arial", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(BASE / "results" / "transition_window" / "summary.csv")
    detail = pd.read_csv(BASE / "results" / "transition_window" / "per_agent_detail.csv")

    orig_rc = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(RC)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    # ── Panel (a): Heatmap of M2 KGE wins across the CP window ──────
    ax = axes[0, 0]
    offsets = [-1, 0, 1]
    matrix = np.full((len(AGENT_ORDER), 3), np.nan)
    for i, aid in enumerate(AGENT_ORDER):
        for j, off in enumerate(offsets):
            cp_cand = AGENT_CP[aid] + off
            row = detail[(detail["AgentID"] == aid) & (detail["CP_Year"] == cp_cand)]
            if len(row) > 0:
                matrix[i, j] = row.iloc[0]["M2_wins_KGE"]

    im = ax.imshow(matrix, cmap=ListedColormap(["tab:red", "tab:blue"]),
                   vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["CP−1", "CP", "CP+1"])
    ax.set_yticks(range(len(AGENT_ORDER)))
    ax.set_yticklabels([f"A{a}" for a in AGENT_ORDER])
    ax.set_xlabel("Transition window offset")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                val = int(matrix[i, j])
                ax.text(j, i, "+" if val else "−",
                        ha="center", va="center", fontsize=14,
                        color="black" if val else "white",
                        fontweight="bold")
    ax.tick_params(length=4)
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # ── Panel (b): M2 KGE across the CP window per agent ────────────
    ax = axes[0, 1]
    for i, aid in enumerate(AGENT_ORDER):
        kge_vals = []
        cp_labels = []
        for off in offsets:
            cp_cand = AGENT_CP[aid] + off
            row = detail[(detail["AgentID"] == aid) & (detail["CP_Year"] == cp_cand)]
            if len(row) > 0:
                kge_vals.append(row.iloc[0]["M2_KGE"])
                cp_labels.append(off)
        ax.plot(cp_labels, kge_vals, "o-", linewidth=1.5, markersize=6,
                label=f"A{aid}")
    ax.set_xticks(offsets)
    ax.set_xticklabels(["CP−1", "CP", "CP+1"])
    ax.set_xlabel("Transition window offset")
    ax.set_ylabel("M2 KGE")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(frameon=False, fontsize=13, ncol=2)
    ax.grid(False)
    ax.tick_params(length=4)
    ax.text(0.01, 0.99, "(b)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # ── Panel (c): Per-agent KGE at CP* (M1 vs M2) ───────────────────
    ax = axes[1, 0]
    summary_sorted = summary.set_index("AgentID").loc[AGENT_ORDER].reset_index()
    x = np.arange(len(AGENT_ORDER))
    width = 0.35
    ax.bar(x - width/2, summary_sorted["M1_KGE"], width,
           color="tab:blue", alpha=0.7, label="M1")
    ax.bar(x + width/2, summary_sorted["M2_KGE"], width,
           color="tab:red", alpha=0.7, label="M2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{a}" for a in AGENT_ORDER])
    ax.set_ylabel("KGE")
    ax.set_xlabel("Agent (KGE at CP*)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(frameon=False)
    ax.grid(False)
    ax.tick_params(length=4)
    ax.text(0.01, 0.99, "(c)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # ── Panel (d): KGE improvement at CP* (M2 − M1) ──────────────────
    ax = axes[1, 1]
    kge_diff = (summary_sorted["M2_KGE"] - summary_sorted["M1_KGE"]).values
    colors = ["tab:green" if v > 0 else "tab:red" for v in kge_diff]
    ax.barh(x, kge_diff, color=colors, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"A{a}" for a in AGENT_ORDER])
    ax.set_xlabel("KGE improvement at CP* (M2 − M1)")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(False)
    ax.tick_params(length=4)
    # Add CP* year labels
    for i, row in summary_sorted.iterrows():
        cp_label = f"CP*={int(row['Best_CP'])}"
        if kge_diff[i] > 0.45:
            # Place inside bar, right-aligned near the end
            ax.text(kge_diff[i] - 0.01, i, cp_label,
                    va="center", ha="right", fontsize=13)
        else:
            offset = 0.01 if kge_diff[i] >= 0 else -0.01
            ax.text(kge_diff[i] + offset, i, cp_label,
                    va="center", ha="left" if kge_diff[i] >= 0 else "right",
                    fontsize=13)
    ax.text(0.01, 0.99, "(d)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    for fmt, dpi in [("pdf", 300), ("png", 150)]:
        path = OUTPUT_DIR / f"Fig5_transition_window.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    matplotlib.rcParams.update(orig_rc)


if __name__ == "__main__":
    main()
