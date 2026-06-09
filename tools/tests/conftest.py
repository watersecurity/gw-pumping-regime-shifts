"""Shared pytest fixtures for MODFLOW ensemble tests."""

import numpy as np
import pandas as pd
import pytest

AGENT_IDS_8 = [2, 12, 14, 18, 20, 24, 28, 29]  # Agent 3 dropped (no CP ≥ 0.3)
YEARS = list(range(1993, 2021))  # 28 years


@pytest.fixture
def synthetic_annual():
    """Synthetic 8-agent annual DataFrame for testing.

    Returns a DataFrame with 224 rows (28 years x 8 agents) with realistic
    column structure matching the aggregated annual data format.
    """
    rng = np.random.default_rng(99)
    rows = []
    for aid in AGENT_IDS_8:
        for year in YEARS:
            rows.append({
                "AgentID": aid,
                "Year": year,
                "Irrigation_Depth": float(rng.uniform(50, 500)),
                "Precipitation": float(rng.uniform(300, 800)),
                "Temperature": float(rng.uniform(10, 25)),
                "Corn": float(rng.uniform(2, 6)),
                "Wheat": float(rng.uniform(3, 7)),
                "Soybeans": float(rng.uniform(5, 12)),
                "Sorghum": float(rng.uniform(2, 5)),
                "Diesel": float(rng.uniform(1, 4)),
            })
    df = pd.DataFrame(rows)
    assert len(df) == 224, f"Expected 224 rows, got {len(df)}"
    return df
