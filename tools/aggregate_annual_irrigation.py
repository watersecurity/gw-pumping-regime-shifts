"""Aggregate monthly irrigation depth to annual totals for each agent.

Reads all data/agentdata_*.csv files, sums Irrigation_Depth by AgentID
and Year, and outputs a wide-format CSV to data/annual_irrigation_depth.csv.
"""

import glob
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    files = sorted(glob.glob(str(DATA_DIR / "agentdata_*.csv")))
    if not files:
        raise FileNotFoundError(f"No agentdata_*.csv files found in {DATA_DIR}")

    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["AgentID", "Year", "Irrigation_Depth"])
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    annual = (
        all_data.groupby(["AgentID", "Year"])["Irrigation_Depth"]
        .sum()
        .reset_index()
    )

    pivot = annual.pivot(index="AgentID", columns="Year", values="Irrigation_Depth")
    pivot.columns = [str(int(c)) for c in pivot.columns]
    pivot = pivot.sort_index().reset_index()

    out_path = DATA_DIR / "annual_irrigation_depth.csv"
    pivot.to_csv(out_path, index=False)
    print(f"Saved {len(pivot)} agents x {len(pivot.columns) - 1} years to {out_path}")


if __name__ == "__main__":
    main()
