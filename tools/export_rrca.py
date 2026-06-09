#!/usr/bin/env python3
"""Export RRCA-format irrigation depth CSV.

Produces a unified CSV with all 43 agents for the post-CP period (2005-2020):
  - All agents: observed annual irrigation depth (baseline)
  - Cluster 2 (agents 2, 3, 24, 28, 29): additionally, XGBoost point estimates
    (2005-2016 train) + 100-member ensemble predictions (2017-2020 test)

Output columns: year, agent, depth_inches, source, member
  - source="observed": baseline observed depth (all agents, all years)
  - source="xgboost":  model point estimate (Cluster 2; 2005-2016)
  - source="ensemble": ensemble member prediction (Cluster 2; 2017-2020)
  - member is blank except for ensemble rows (integer 0-99)
  - All depths in inches (mm / 25.4), rounded to 1 decimal
"""

from pathlib import Path
import glob

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "two_regime"
MM_TO_INCHES = 25.4
YEARS = range(2005, 2021)  # 2005-2020 inclusive


def load_all_observed():
    """Load observed annual irrigation depth for all agents, 2005-2020."""
    frames = []
    for fp in sorted(glob.glob(str(DATA_DIR / "agentdata_*.csv"))):
        frames.append(pd.read_csv(fp))

    monthly = pd.concat(frames, ignore_index=True)
    annual = (monthly.groupby(["AgentID", "Year"])["Irrigation_Depth"]
              .sum().reset_index())
    annual = annual[annual["Year"].isin(YEARS)].copy()

    return pd.DataFrame({
        "year": annual["Year"].astype(int),
        "agent": annual["AgentID"].astype(int),
        "depth_inches": (annual["Irrigation_Depth"] / MM_TO_INCHES).round(1),
        "source": "observed",
        "member": pd.array([pd.NA] * len(annual), dtype="Int64"),
    })


def load_cluster2_predictions():
    """Load XGBoost point estimates + ensemble member predictions."""
    # Training-period point estimates (2005-2016)
    point_df = pd.read_csv(RESULTS_DIR / "point_predictions_post.csv")
    train_df = point_df[point_df["Split"] == "train"].copy()

    train_out = pd.DataFrame({
        "year": train_df["Year"].astype(int),
        "agent": train_df["AgentID"].astype(int),
        "depth_inches": (train_df["Pred"] / MM_TO_INCHES).round(1),
        "source": "xgboost",
        "member": pd.array([pd.NA] * len(train_df), dtype="Int64"),
    })

    # Ensemble predictions (2017-2020)
    member_df = pd.read_csv(
        RESULTS_DIR / "ensemble_member_predictions_post.csv")

    ens_out = pd.DataFrame({
        "year": member_df["Year"].astype(int),
        "agent": member_df["AgentID"].astype(int),
        "depth_inches": (member_df["Pred_mm"] / MM_TO_INCHES).round(1),
        "source": "ensemble",
        "member": member_df["Member"].astype("Int64"),
    })

    return pd.concat([train_out, ens_out], ignore_index=True)


def main():
    obs = load_all_observed()
    c2 = load_cluster2_predictions()

    out = pd.concat([obs, c2], ignore_index=True)
    out = out.sort_values(["year", "agent", "source", "member"],
                          ignore_index=True)

    out_path = RESULTS_DIR / "rrca_irrigation_depth.csv"
    out.to_csv(out_path, index=False)

    n_obs = len(obs)
    n_c2_train = len(c2[c2["source"] == "xgboost"])
    n_c2_ens = len(c2[c2["source"] == "ensemble"])
    print(f"Observed baseline:    {n_obs} rows "
          f"({obs['agent'].nunique()} agents × {len(YEARS)} years)")
    print(f"Cluster 2 xgboost:   {n_c2_train} rows")
    print(f"Cluster 2 ensemble:  {n_c2_ens} rows")
    print(f"Total:               {len(out)} rows")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
