"""Unit and smoke tests for run_modflow_ensemble.py.

Covers:
  REQ-MF-1: pooled training set sizes, eval construction, bootstrap shape
  REQ-MF-2: RRCA columns, unit conversion, observed replacement, summary columns
  REQ-MF-3: summary agents, PI positive
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure tools/ directory is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Skip all tests gracefully if run_modflow_ensemble is not yet implemented
pytest.importorskip("run_modflow_ensemble")

from run_modflow_ensemble import (
    AGENT_IDS_8,
    PER_AGENT_CP,
    PER_AGENT_EVAL_START,
    build_pooled_eval,
    build_pooled_train,
    build_rrca_member_rows,
    build_rrca_observed_rows,
    build_rrca_summary_rows,
    compute_agent_pi_summary,
    run_pooled_bootstrap,
)


# ---------------------------------------------------------------------------
# REQ-MF-1: Pooled training set construction
# ---------------------------------------------------------------------------

def test_pooled_train_m1_sizes(synthetic_annual):
    """M1 training set: per-agent CP-filtered row counts are correct."""
    df = build_pooled_train(synthetic_annual, model="M1")

    # Agent 14: CP=2004, M1 gets Year < 2004 → 1993-2003 → 11 rows
    agent14 = df[df["AgentID"] == 14]
    assert len(agent14) == 11, f"Agent 14 M1 rows: expected 11, got {len(agent14)}"

    # Agent 12: CP=2004, M1 gets Year < 2004 → 1993-2003 → 11 rows
    agent12 = df[df["AgentID"] == 12]
    assert len(agent12) == 11, f"Agent 12 M1 rows: expected 11, got {len(agent12)}"

    # Agent 2: CP=2012, M1 gets Year < 2012 → 1993-2011 → 19 rows
    agent2 = df[df["AgentID"] == 2]
    assert len(agent2) == 19, f"Agent 2 M1 rows: expected 19, got {len(agent2)}"

    # Agent 29: CP=2011, M1 gets Year < 2011 → 1993-2010 → 18 rows
    agent29 = df[df["AgentID"] == 29]
    assert len(agent29) == 18, f"Agent 29 M1 rows: expected 18, got {len(agent29)}"

    # Total: 19+11+11+11+12+11+19+18 = 112
    assert len(df) == 112, f"Total M1 rows: expected 112, got {len(df)}"


def test_pooled_train_m2_sizes(synthetic_annual):
    """M2 training set: per-agent CP-filtered row counts are correct."""
    df = build_pooled_train(synthetic_annual, model="M2")

    # Agent 14: CP=2004, M2 gets Year <= 2004 → 1993-2004 → 12 rows
    agent14 = df[df["AgentID"] == 14]
    assert len(agent14) == 12, f"Agent 14 M2 rows: expected 12, got {len(agent14)}"

    # Agent 12: CP=2004, M2 gets Year <= 2004 → 1993-2004 → 12 rows
    agent12 = df[df["AgentID"] == 12]
    assert len(agent12) == 12, f"Agent 12 M2 rows: expected 12, got {len(agent12)}"

    # Agent 2: CP=2012, M2 gets Year <= 2012 → 1993-2012 → 20 rows
    agent2 = df[df["AgentID"] == 2]
    assert len(agent2) == 20, f"Agent 2 M2 rows: expected 20, got {len(agent2)}"

    # Agent 29: CP=2011, M2 gets Year <= 2011 → 1993-2011 → 19 rows
    agent29 = df[df["AgentID"] == 29]
    assert len(agent29) == 19, f"Agent 29 M2 rows: expected 19, got {len(agent29)}"

    # Total: 20+12+12+12+13+12+20+19 = 120
    assert len(df) == 120, f"Total M2 rows: expected 120, got {len(df)}"


def test_pooled_eval_construction(synthetic_annual):
    """Eval set: per-agent eval start from PER_AGENT_EVAL_START is respected."""
    df = build_pooled_eval(synthetic_annual)

    # Agent 14: eval_start=2005, eval_end=2020 → 16 rows
    agent14 = df[df["AgentID"] == 14]
    assert len(agent14) == 16, f"Agent 14 eval rows: expected 16, got {len(agent14)}"

    # Agent 12: eval_start=2005, eval_end=2020 → 16 rows
    agent12 = df[df["AgentID"] == 12]
    assert len(agent12) == 16, f"Agent 12 eval rows: expected 16, got {len(agent12)}"

    # Agent 2: eval_start=2013, eval_end=2020 → 8 rows
    agent2 = df[df["AgentID"] == 2]
    assert len(agent2) == 8, f"Agent 2 eval rows: expected 8, got {len(agent2)}"

    # Agent 29: eval_start=2012, eval_end=2020 → 9 rows
    agent29 = df[df["AgentID"] == 29]
    assert len(agent29) == 9, f"Agent 29 eval rows: expected 9, got {len(agent29)}"

    # Total: 8+16+16+16+15+16+8+9 = 104
    assert len(df) == 104, f"Total eval rows: expected 104, got {len(df)}"


# ---------------------------------------------------------------------------
# REQ-MF-2: RRCA export columns and values
# ---------------------------------------------------------------------------

def test_rrca_columns():
    """RRCA member rows have exactly the required column set."""
    # Create minimal mock data: 3 agents, 2 years, 2 members
    eval_df = pd.DataFrame({
        "Year": [2006, 2007, 2006, 2007, 2006, 2007],
        "AgentID": [2, 2, 12, 12, 14, 14],
        "Irrigation_Depth": [100.0, 110.0, 90.0, 95.0, 200.0, 210.0],
    })
    member_preds = np.array([
        [100.0, 110.0, 90.0, 95.0, 200.0, 210.0],
        [105.0, 115.0, 85.0, 90.0, 195.0, 205.0],
    ])  # shape (2, 6)

    result = build_rrca_member_rows(eval_df, member_preds, model_label="M1")

    expected_cols = {"year", "agent", "depth_mm", "depth_inches", "source", "model", "member"}
    assert set(result.columns) == expected_cols, (
        f"Columns mismatch. Got: {set(result.columns)}, Expected: {expected_cols}"
    )


def test_unit_conversion():
    """depth_inches = depth_mm / 25.4 with 1 decimal rounding."""
    eval_df = pd.DataFrame({
        "Year": [2006],
        "AgentID": [2],
        "Irrigation_Depth": [254.0],
    })
    member_preds = np.array([[254.0]])  # shape (1, 1)

    result = build_rrca_member_rows(eval_df, member_preds, model_label="M1")
    assert len(result) == 1
    assert result.iloc[0]["depth_mm"] == pytest.approx(254.0, abs=0.01)
    assert result.iloc[0]["depth_inches"] == pytest.approx(10.0, abs=0.05), (
        f"Expected 10.0 inches for 254 mm, got {result.iloc[0]['depth_inches']}"
    )


# ---------------------------------------------------------------------------
# REQ-MF-3: Uncertainty summary
# ---------------------------------------------------------------------------

def test_summary_agents(synthetic_annual):
    """compute_agent_pi_summary returns a row for each of the 8 agents."""
    eval_df = build_pooled_eval(synthetic_annual)
    eval_agents = eval_df["AgentID"].values
    y_eval = eval_df["Irrigation_Depth"].values
    n_eval = len(eval_df)  # 104

    # Synthetic predictions: random values
    rng = np.random.default_rng(42)
    member_preds = rng.uniform(50, 500, size=(5, n_eval))

    result = compute_agent_pi_summary(member_preds, y_eval, eval_agents, model_label="M1")

    assert len(result) == 8, f"Expected 8 agents in summary, got {len(result)}"
    result_agents = {r["agent"] for r in result}
    assert result_agents == set(AGENT_IDS_8), (
        f"Missing agents: {set(AGENT_IDS_8) - result_agents}"
    )


def test_pi_positive(synthetic_annual):
    """All agents have pi_width_mean_mm > 0."""
    eval_df = build_pooled_eval(synthetic_annual)
    eval_agents = eval_df["AgentID"].values
    y_eval = eval_df["Irrigation_Depth"].values
    n_eval = len(eval_df)  # 104

    rng = np.random.default_rng(42)
    member_preds = rng.uniform(50, 500, size=(5, n_eval))

    result = compute_agent_pi_summary(member_preds, y_eval, eval_agents, model_label="M1")

    for row in result:
        assert row["pi_width_mean_mm"] > 0, (
            f"Agent {row['agent']} has non-positive pi_width_mean_mm: "
            f"{row['pi_width_mean_mm']}"
        )


# ---------------------------------------------------------------------------
# REQ-MF-2: Observed replacement boundary
# ---------------------------------------------------------------------------

def test_rrca_observed_replacement(synthetic_annual):
    """Observed rows use source='observed' for year <= agent CP only."""
    result = build_rrca_observed_rows(synthetic_annual)

    # All rows should be source='observed'
    assert (result["source"] == "observed").all()

    # Agent 14: CP=2004, only years 1993-2004 included
    agent14 = result[result["agent"] == 14]
    assert (agent14["year"] <= 2004).all(), (
        f"Agent 14 observed rows should be <= 2004, got max {agent14['year'].max()}"
    )
    assert 1993 in agent14["year"].values, "Agent 14 should have 1993 in observed rows"
    assert 2004 in agent14["year"].values, "Agent 14 should have 2004 in observed rows"
    assert 2005 not in agent14["year"].values, "Agent 14 should not have 2005 in observed rows"

    # Agent 2: CP=2012, only years up through 2012
    agent2 = result[result["agent"] == 2]
    assert (agent2["year"] <= 2012).all(), (
        f"Agent 2 observed rows should be <= 2012, got max {agent2['year'].max()}"
    )
    assert 2012 in agent2["year"].values, "Agent 2 should have 2012 in observed rows"
    assert 2013 not in agent2["year"].values, "Agent 2 should not have 2013 in observed rows"

    # Scope: 8 non-stationary agents only
    assert set(result["agent"].unique()) == set(AGENT_IDS_8), (
        "build_rrca_observed_rows should cover exactly the 8 non-stationary agents"
    )


# ---------------------------------------------------------------------------
# REQ-MF-2: RRCA summary rows
# ---------------------------------------------------------------------------

def test_rrca_summary_columns():
    """build_rrca_summary_rows has all required columns and correct row count."""
    # Mock eval_df: 3 agents, 2 years each = 6 rows
    eval_df = pd.DataFrame({
        "Year": [2006, 2007, 2006, 2007, 2006, 2007],
        "AgentID": [2, 2, 12, 12, 14, 14],
        "Irrigation_Depth": [100.0, 110.0, 90.0, 95.0, 200.0, 210.0],
    })

    # Synthetic ensemble predictions: 5 members, 6 eval rows
    rng = np.random.default_rng(7)
    m1_member_preds = rng.uniform(80, 120, size=(5, 6))
    m2_member_preds = rng.uniform(90, 130, size=(5, 6))

    result = build_rrca_summary_rows(eval_df, m1_member_preds, m2_member_preds)

    required_cols = {
        "year", "agent", "observed_mm", "observed_inches",
        "m1_median_mm", "m2_median_mm",
        "m1_pi_lo_mm", "m1_pi_hi_mm",
        "m2_pi_lo_mm", "m2_pi_hi_mm",
    }
    missing = required_cols - set(result.columns)
    assert not missing, f"Missing columns: {missing}"

    assert len(result) == 6, f"Expected 6 rows, got {len(result)}"

    # All median and PI values should be numeric (not NaN)
    for col in ["m1_median_mm", "m2_median_mm", "m1_pi_lo_mm",
                "m1_pi_hi_mm", "m2_pi_lo_mm", "m2_pi_hi_mm"]:
        assert result[col].notna().all(), f"Column {col} has NaN values"


# ---------------------------------------------------------------------------
# REQ-MF-1: Bootstrap output shape
# ---------------------------------------------------------------------------

def test_bootstrap_shape(synthetic_annual):
    """run_pooled_bootstrap returns shape (n_boot, n_eval_rows)."""
    xgb = pytest.importorskip("xgboost")  # noqa: F841 — skip if no xgboost

    from run_transition_window import prepare_features

    df_m1_train = build_pooled_train(synthetic_annual, model="M1")
    df_eval = build_pooled_eval(synthetic_annual)
    X_eval, y_eval = prepare_features(df_eval, agent_ids=AGENT_IDS_8)

    best_params = {
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }

    n_boot = 2
    result = run_pooled_bootstrap(
        df_m1_train, X_eval, best_params,
        n_boot=n_boot, master_seed=42, seed_offset=0,
    )

    expected_eval_rows = 104
    assert result.shape == (n_boot, expected_eval_rows), (
        f"Expected shape ({n_boot}, {expected_eval_rows}), got {result.shape}"
    )
