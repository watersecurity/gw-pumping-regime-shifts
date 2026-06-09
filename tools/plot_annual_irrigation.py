"""Plot annual irrigation depth (mm) for all agents, 1993-2020."""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams["font.family"] = "Arial"

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "annual_irrigation_depth.csv"
OUTPUT_PDF = BASE_DIR / "results" / "annual_irrigation_depth_all_agents.pdf"


def main():
    df = pd.read_csv(INPUT_CSV)
    years = [int(c) for c in df.columns[1:]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.plot(years, row.iloc[1:].values, linewidth=0.8)

    ax.set_title("All Agents", fontsize=16)
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Annual Irrigation Depth (mm)", fontsize=16)
    ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020])
    ax.set_xlim(1993, 2020)
    ax.tick_params(axis="both", labelsize=14)

    fig.tight_layout()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
