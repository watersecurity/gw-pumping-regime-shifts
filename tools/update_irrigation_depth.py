#!/usr/bin/env python3
"""Update agentdata files and annual_irrigation_depth.csv with recalculated
irrigation depth values.

Sources:
  - data/irrigation_depth_monthly_1993_2020.csv  (long: year, month, agent_id, depth_mm)
  - data/irrigation_depth_annual_1993_2020.csv   (long: year, agent_id, depth_mm)

Targets:
  - data/agentdata_#.csv   (43 files) -- replace Irrigation_Depth column
  - data/annual_irrigation_depth.csv   -- rebuild wide-format pivot (46 agents)
"""

import os
import glob
import re

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
SANITY_YEAR = 2005  # year used for before/after comparison


def update_agentdata_files(monthly_src: pd.DataFrame) -> None:
    """Replace Irrigation_Depth in each agentdata_#.csv with new monthly depth_mm."""

    # Build lookup: (agent_id, year, month) -> depth_mm
    lookup = monthly_src.set_index(["agent_id", "year", "month"])["depth_mm"]

    pattern = os.path.join(DATA_DIR, "agentdata_*.csv")
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r"agentdata_(\d+)", f).group(1)))

    print(f"Found {len(files)} agentdata files to update.\n")
    print(f"{'Agent':>6}  {'Old 2005 total':>14}  {'New 2005 total':>14}  {'Diff':>10}")
    print("-" * 52)

    for fpath in files:
        aid = int(re.search(r"agentdata_(\d+)", fpath).group(1))
        df = pd.read_csv(fpath)

        # Capture old total for sanity year
        mask_sanity = df["Year"] == SANITY_YEAR
        old_total = df.loc[mask_sanity, "Irrigation_Depth"].sum()

        # Replace Irrigation_Depth via lookup
        new_depths = []
        for _, row in df.iterrows():
            key = (aid, int(row["Year"]), int(row["Month"]))
            if key in lookup.index:
                new_depths.append(lookup[key])
            else:
                # Keep original if no source data (should not happen)
                new_depths.append(row["Irrigation_Depth"])
                print(f"  WARNING: no source data for agent {aid}, "
                      f"year {int(row['Year'])}, month {int(row['Month'])}")

        df["Irrigation_Depth"] = new_depths

        # New total for sanity year
        new_total = df.loc[mask_sanity, "Irrigation_Depth"].sum()

        print(f"{aid:>6}  {old_total:>14.2f}  {new_total:>14.2f}  "
              f"{new_total - old_total:>+10.2f}")

        df.to_csv(fpath, index=False)

    print(f"\nAll {len(files)} agentdata files updated.")


def rebuild_annual_file(annual_src: pd.DataFrame) -> None:
    """Rebuild data/annual_irrigation_depth.csv as wide-format pivot."""

    pivot = annual_src.pivot(index="agent_id", columns="year", values="depth_mm")
    pivot.index.name = "AgentID"
    pivot.columns.name = None  # remove "year" label from column axis
    pivot = pivot.sort_index()

    out_path = os.path.join(DATA_DIR, "annual_irrigation_depth.csv")
    pivot.to_csv(out_path)

    print(f"\nRebuilt {out_path}")
    print(f"  Shape: {pivot.shape[0]} agents x {pivot.shape[1]} years")
    print(f"  Agent IDs: {list(pivot.index)}")


def cross_check(monthly_src: pd.DataFrame, annual_src: pd.DataFrame) -> None:
    """Verify monthly sums match annual totals for a sample of agents."""

    monthly_annual = (monthly_src
                      .groupby(["agent_id", "year"])["depth_mm"]
                      .sum()
                      .reset_index())

    merged = monthly_annual.merge(annual_src[["agent_id", "year", "depth_mm"]],
                                  on=["agent_id", "year"],
                                  suffixes=("_monthly_sum", "_annual"))

    max_diff = (merged["depth_mm_monthly_sum"] - merged["depth_mm_annual"]).abs().max()
    print(f"\nCross-check: max |monthly_sum - annual| = {max_diff:.6f} mm")
    if max_diff < 0.01:
        print("  PASS: monthly and annual sources are consistent.")
    else:
        print("  WARNING: discrepancy detected between monthly and annual sources.")


def main():
    monthly_path = os.path.join(DATA_DIR, "irrigation_depth_monthly_1993_2020.csv")
    annual_path = os.path.join(DATA_DIR, "irrigation_depth_annual_1993_2020.csv")

    print("Loading source data...")
    monthly_src = pd.read_csv(monthly_path)
    annual_src = pd.read_csv(annual_path)
    print(f"  Monthly: {len(monthly_src)} rows, "
          f"{monthly_src['agent_id'].nunique()} agents")
    print(f"  Annual:  {len(annual_src)} rows, "
          f"{annual_src['agent_id'].nunique()} agents\n")

    # Step 1: Update agentdata files
    print("=" * 52)
    print("Step 1: Update agentdata_#.csv files")
    print("=" * 52)
    update_agentdata_files(monthly_src)

    # Step 2: Rebuild annual file
    print("\n" + "=" * 52)
    print("Step 2: Rebuild annual_irrigation_depth.csv")
    print("=" * 52)
    rebuild_annual_file(annual_src)

    # Step 3: Cross-check
    print("\n" + "=" * 52)
    print("Step 3: Cross-check monthly vs annual consistency")
    print("=" * 52)
    cross_check(monthly_src, annual_src)


if __name__ == "__main__":
    main()
