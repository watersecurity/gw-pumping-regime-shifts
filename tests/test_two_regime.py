"""Tests for the two-regime experiment utilities and config."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add tools/ to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from run_xgboost_abm import (
    REGIME_A,
    REGIME_B,
    make_constrained_moving_block_year_sequence,
    bootstrap_panel_by_year_sequence,
)


# ── TestRegimeConfig ─────────────────────────────────────────────────────────
class TestRegimeConfig:
    """Validate regime dict consistency."""

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_trainval_test_no_overlap(self, regime):
        overlap = set(regime["trainval_years"]) & set(regime["test_years"])
        assert overlap == set(), f"trainval/test overlap: {overlap}"

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_anchor_subset_of_trainval(self, regime):
        assert set(regime["anchor_years"]).issubset(set(regime["trainval_years"]))

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_anchor_is_tail_of_trainval(self, regime):
        tv = sorted(regime["trainval_years"])
        anchor = regime["anchor_years"]
        assert tv[-len(anchor):] == anchor

    def test_regimes_dont_overlap(self):
        all_a = set(REGIME_A["full_years"])
        all_b = set(REGIME_B["full_years"])
        assert all_a & all_b == set()

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_full_equals_trainval_union_test(self, regime):
        expected = set(regime["trainval_years"]) | set(regime["test_years"])
        assert set(regime["full_years"]) == expected

    def test_regime_a_test_years(self):
        assert REGIME_A["test_years"] == [2002, 2003, 2004]

    def test_regime_b_test_years(self):
        assert REGIME_B["test_years"] == [2017, 2018, 2019, 2020]


# ── TestMakeConstrainedSequence ──────────────────────────────────────────────
class TestMakeConstrainedSequence:
    """Tests for make_constrained_moving_block_year_sequence."""

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_tail_matches_anchor(self, regime):
        """Over 50 replicates, tail always matches anchor."""
        rng = np.random.default_rng(123)
        anchor = regime["anchor_years"]
        for _ in range(50):
            result = make_constrained_moving_block_year_sequence(
                regime["trainval_years"], anchor, 3, rng)
            assert result[-len(anchor):] == anchor

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_output_length(self, regime):
        """Output length == len(trainval_years)."""
        rng = np.random.default_rng(42)
        result = make_constrained_moving_block_year_sequence(
            regime["trainval_years"], regime["anchor_years"], 3, rng)
        assert len(result) == len(regime["trainval_years"])

    @pytest.mark.parametrize("regime", [REGIME_A, REGIME_B])
    def test_prefix_years_in_range(self, regime):
        """Prefix years come from prefix pool (trainval minus anchor)."""
        rng = np.random.default_rng(99)
        anchor = regime["anchor_years"]
        prefix_pool = set(regime["trainval_years"]) - set(anchor)
        for _ in range(50):
            result = make_constrained_moving_block_year_sequence(
                regime["trainval_years"], anchor, 3, rng)
            prefix = result[:-len(anchor)]
            assert all(yr in prefix_pool for yr in prefix), (
                f"Out-of-range years in prefix: {prefix}")

    def test_reproducibility(self):
        rng1 = np.random.default_rng(42)
        r1 = make_constrained_moving_block_year_sequence(
            REGIME_A["trainval_years"], REGIME_A["anchor_years"], 3, rng1)
        rng2 = np.random.default_rng(42)
        r2 = make_constrained_moving_block_year_sequence(
            REGIME_A["trainval_years"], REGIME_A["anchor_years"], 3, rng2)
        assert r1 == r2

    def test_invalid_anchor_raises(self):
        """Anchor that isn't the tail of trainval raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="must be the tail"):
            make_constrained_moving_block_year_sequence(
                list(range(1993, 2002)),  # trainval 1993-2001
                [1993, 1994, 1995],       # anchor is the head, not tail
                3, rng)


# ── TestBootstrapPanel ───────────────────────────────────────────────────────
class TestBootstrapPanel:
    """Tests for bootstrap_panel_by_year_sequence."""

    @pytest.fixture
    def sample_df(self):
        """Small DataFrame with 2 agents, 3 years."""
        rows = []
        for yr in [2000, 2001, 2002]:
            for aid in [4, 11]:
                rows.append({"Year": yr, "AgentID": aid, "Value": yr * 10 + aid})
        return pd.DataFrame(rows)

    def test_correct_row_count_no_repeats(self, sample_df):
        result = bootstrap_panel_by_year_sequence(sample_df, [2000, 2001, 2002])
        assert len(result) == 6  # 3 years x 2 agents

    def test_correct_row_count_with_repeats(self, sample_df):
        result = bootstrap_panel_by_year_sequence(sample_df, [2000, 2000, 2001])
        assert len(result) == 6  # 2 * 2 + 1 * 2 agents

    def test_missing_year_raises(self, sample_df):
        with pytest.raises(ValueError, match="Year 1999 not found"):
            bootstrap_panel_by_year_sequence(sample_df, [1999, 2000])
