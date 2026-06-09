"""Tests for constrained_block_bootstrap_years and _anchor_tail_for_cp."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tools/ to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from run_xgboost_abm import (
    constrained_block_bootstrap_years,
    _anchor_tail_for_cp,
)


# ── TestAnchorTail ───────────────────────────────────────────────────────────
class TestAnchorTail:
    """Verify _anchor_tail_for_cp returns correct tails."""

    def test_cp_equals_base(self):
        assert _anchor_tail_for_cp(2005, base_cp=2005) == [2005]

    def test_cp_2007(self):
        assert _anchor_tail_for_cp(2007, base_cp=2005) == [2005, 2006, 2007]

    def test_cp_2009(self):
        assert _anchor_tail_for_cp(2009, base_cp=2005) == [2005, 2006, 2007, 2008, 2009]

    def test_raises_on_cp_below_base(self):
        with pytest.raises(ValueError, match="must be >= base_cp"):
            _anchor_tail_for_cp(2004, base_cp=2005)


# ── TestTailConstraint ───────────────────────────────────────────────────────
class TestTailConstraint:
    """Over 50 replicates × 3 anchor sizes, last N years always match anchor."""

    @pytest.mark.parametrize("cp", [2005, 2007, 2009])
    def test_tail_always_matches(self, cp):
        pre = np.arange(1993, 2005)
        anchor = _anchor_tail_for_cp(cp)
        rng = np.random.default_rng(123)

        for _ in range(50):
            result = constrained_block_bootstrap_years(pre, anchor, 3, rng)
            assert result[-len(anchor):] == anchor


# ── TestOutputLength ─────────────────────────────────────────────────────────
class TestOutputLength:
    """len(result) == len(pre_cp) + len(anchor)."""

    @pytest.mark.parametrize("cp", [2005, 2007, 2009])
    def test_length(self, cp):
        pre = np.arange(1993, 2005)
        anchor = _anchor_tail_for_cp(cp)
        rng = np.random.default_rng(42)

        result = constrained_block_bootstrap_years(pre, anchor, 3, rng)
        assert len(result) == len(pre) + len(anchor)


# ── TestYearRanges ───────────────────────────────────────────────────────────
class TestYearRanges:
    """Bootstrapped portion only contains years from pre_cp_years."""

    @pytest.mark.parametrize("cp", [2005, 2007, 2009])
    def test_bootstrapped_years_in_range(self, cp):
        pre = np.arange(1993, 2005)
        anchor = _anchor_tail_for_cp(cp)
        rng = np.random.default_rng(99)

        for _ in range(50):
            result = constrained_block_bootstrap_years(pre, anchor, 3, rng)
            bootstrapped = result[:len(pre)]
            assert all(yr in pre for yr in bootstrapped), (
                f"Found out-of-range year in bootstrapped portion: {bootstrapped}")


# ── TestMovingBlockDiversity ─────────────────────────────────────────────────
class TestMovingBlockDiversity:
    """Over 500 replicates, all 10 start positions (1993-2002) are observed."""

    def test_all_start_positions_observed(self):
        pre = np.arange(1993, 2005)  # 12 years
        anchor = [2005]
        block_size = 3
        rng = np.random.default_rng(7)

        # Possible start positions: 0..9 (n - block_size + 1 = 12 - 3 + 1 = 10)
        # Corresponding start years: 1993..2002
        observed_starts = set()
        for _ in range(500):
            result = constrained_block_bootstrap_years(pre, anchor, block_size, rng)
            bootstrapped = result[:len(pre)]
            # Check which starting years appeared by looking at blocks
            for i in range(0, len(bootstrapped), block_size):
                block = bootstrapped[i:i + block_size]
                if len(block) == block_size:
                    start_yr = block[0]
                    if start_yr in pre:
                        observed_starts.add(start_yr)

        expected_starts = set(range(1993, 2003))  # 1993 through 2002
        assert expected_starts.issubset(observed_starts), (
            f"Missing start years: {expected_starts - observed_starts}")


# ── TestReproducibility ──────────────────────────────────────────────────────
class TestReproducibility:
    """Same RNG seed → identical output."""

    def test_same_seed_same_result(self):
        pre = np.arange(1993, 2005)
        anchor = [2005, 2006, 2007]

        rng1 = np.random.default_rng(42)
        result1 = constrained_block_bootstrap_years(pre, anchor, 3, rng1)

        rng2 = np.random.default_rng(42)
        result2 = constrained_block_bootstrap_years(pre, anchor, 3, rng2)

        assert result1 == result2


# ── TestEdgeCases ────────────────────────────────────────────────────────────
class TestEdgeCases:
    """Edge case handling."""

    def test_block_size_equals_pre_length(self):
        """block_size == len(pre) works (single block covers all years)."""
        pre = np.arange(1993, 1996)  # 3 years
        anchor = [2005]
        rng = np.random.default_rng(42)

        result = constrained_block_bootstrap_years(pre, anchor, 3, rng)
        assert len(result) == 4
        assert result[-1] == 2005
        # With block_size == n, only one start position (0), so bootstrapped
        # portion is always [1993, 1994, 1995]
        assert result[:3] == [1993, 1994, 1995]

    def test_block_size_greater_than_pre_raises(self):
        """block_size > len(pre) raises ValueError."""
        pre = np.arange(1993, 1995)  # 2 years
        anchor = [2005]
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="must be >= block_size"):
            constrained_block_bootstrap_years(pre, anchor, 3, rng)

    def test_overlapping_pre_anchor_raises(self):
        """Overlapping pre_cp_years and anchor_tail_years raises ValueError."""
        pre = np.arange(1993, 2006)  # includes 2005
        anchor = [2005, 2006, 2007]
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="must not overlap"):
            constrained_block_bootstrap_years(pre, anchor, 3, rng)
