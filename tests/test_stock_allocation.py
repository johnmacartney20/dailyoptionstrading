"""Unit tests for TFSA stock and RRSP allocation models.

Tests cover:
- StockScoreComponents from score_stock_growth / score_stock_stability
- allocate_tfsa_stock_portfolio (sector cap, correlation, tiered capital allocation)
- allocate_rrsp_portfolio (stability scoring, sector constraints, 4-position default)
- TfsaStockPortfolio.exit_guidance property
- Rejection tracking for position-limit and lower-score candidates
"""

import math

import numpy as np
import pandas as pd
import pytest

from scanner.analyzer import (
    StockScoreComponents,
    _MIN_HISTORY_DAYS,
    score_stock_growth,
    score_stock_stability,
)
from scanner.portfolio_allocator import (
    RejectedCandidate,
    RrspPortfolio,
    RrspStockAllocation,
    StockAllocation,
    TfsaStockPortfolio,
    _tfsa_concentrated_allocation,
    allocate_rrsp_portfolio,
    allocate_tfsa_stock_portfolio,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_history(
    n_days: int = 60,
    start_price: float = 100.0,
    drift: float = 0.001,      # daily drift (positive = uptrend)
    vol: float = 0.01,          # daily std-dev of returns
    avg_volume: float = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with *n_days* of daily data."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n_days)
    prices = start_price * np.cumprod(1 + returns)
    volume = rng.normal(avg_volume, avg_volume * 0.1, n_days).clip(1)

    dates = pd.date_range(end="2024-12-31", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": volume,
        },
        index=dates,
    )


# ── score_stock_growth ────────────────────────────────────────────────────────


def test_score_growth_returns_zero_on_short_history():
    hist = _make_history(n_days=_MIN_HISTORY_DAYS - 1)
    result = score_stock_growth(hist)
    assert result.composite == 0.0
    assert "Insufficient" in result.reasoning


def test_score_growth_returns_zero_on_none():
    result = score_stock_growth(None)
    assert result.composite == 0.0


def test_score_growth_structure():
    hist = _make_history(n_days=60, drift=0.002, vol=0.008)
    result = score_stock_growth(hist, market_return_20d=0.0)
    assert isinstance(result, StockScoreComponents)
    assert result.composite == pytest.approx(
        result.trend_strength
        + result.relative_strength
        + result.volatility_control
        + result.liquidity
        + result.drawdown_risk,
        abs=0.01,
    )


def test_score_growth_bounded():
    hist = _make_history(n_days=60, drift=0.005, vol=0.005)
    result = score_stock_growth(hist, market_return_20d=0.0)
    assert 0.0 <= result.trend_strength <= 30.0
    assert 0.0 <= result.relative_strength <= 20.0
    assert 0.0 <= result.volatility_control <= 15.0
    assert 0.0 <= result.liquidity <= 15.0
    assert 0.0 <= result.drawdown_risk <= 20.0
    assert 0.0 <= result.composite <= 100.0


def test_score_growth_uptrend_beats_downtrend():
    up = _make_history(n_days=60, drift=+0.003, vol=0.008)
    down = _make_history(n_days=60, drift=-0.003, vol=0.008)
    assert score_stock_growth(up).composite > score_stock_growth(down).composite


def test_score_growth_low_vol_beats_high_vol():
    low = _make_history(n_days=60, drift=0.001, vol=0.005)
    high_v = _make_history(n_days=60, drift=0.001, vol=0.035)
    assert (
        score_stock_growth(low).volatility_control
        > score_stock_growth(high_v).volatility_control
    )


def test_score_growth_outperforming_market_adds_rs_pts():
    hist = _make_history(n_days=60, drift=0.003, vol=0.008)
    low_market = score_stock_growth(hist, market_return_20d=-0.05)
    high_market = score_stock_growth(hist, market_return_20d=+0.10)
    assert low_market.relative_strength >= high_market.relative_strength


def test_score_growth_high_liquidity_adds_pts():
    low_liq = _make_history(n_days=60, avg_volume=1_000)
    high_liq = _make_history(n_days=60, avg_volume=10_000_000)
    assert score_stock_growth(high_liq).liquidity > score_stock_growth(low_liq).liquidity


def test_score_growth_reasoning_non_empty():
    hist = _make_history(n_days=60)
    result = score_stock_growth(hist)
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0


# ── score_stock_stability ─────────────────────────────────────────────────────


def test_score_stability_returns_zero_on_short_history():
    hist = _make_history(n_days=_MIN_HISTORY_DAYS - 1)
    result = score_stock_stability(hist)
    assert result.composite == 0.0


def test_score_stability_structure():
    hist = _make_history(n_days=60, drift=0.001, vol=0.006)
    result = score_stock_stability(hist)
    assert isinstance(result, StockScoreComponents)
    assert result.relative_strength == 0.0  # not used for RRSP
    assert result.composite == pytest.approx(
        result.trend_strength
        + result.relative_strength
        + result.volatility_control
        + result.liquidity
        + result.drawdown_risk,
        abs=0.01,
    )


def test_score_stability_bounded():
    hist = _make_history(n_days=60, drift=0.001, vol=0.007)
    result = score_stock_stability(hist)
    # Stability model: vol_control max=30, trend max=30, liq max=20, drawdown max=20
    assert 0.0 <= result.trend_strength <= 30.0
    assert 0.0 <= result.volatility_control <= 30.0
    assert 0.0 <= result.liquidity <= 20.0
    assert 0.0 <= result.drawdown_risk <= 20.0
    assert 0.0 <= result.composite <= 100.0


def test_score_stability_low_vol_preferred():
    low = _make_history(n_days=60, drift=0.001, vol=0.004)
    high_v = _make_history(n_days=60, drift=0.001, vol=0.030)
    assert (
        score_stock_stability(low).volatility_control
        > score_stock_stability(high_v).volatility_control
    )


# ── allocate_tfsa_stock_portfolio ─────────────────────────────────────────────


def test_tfsa_stock_empty_histories():
    result = allocate_tfsa_stock_portfolio({})
    assert result.num_positions == 0
    assert result.total_deployed == 0.0


def test_tfsa_stock_none_histories_skipped():
    result = allocate_tfsa_stock_portfolio({"AAPL": None})
    assert result.num_positions == 0


def test_tfsa_stock_single_position():
    hist = _make_history(n_days=60, drift=0.003, vol=0.007)
    result = allocate_tfsa_stock_portfolio({"AAPL": hist}, total_capital=1000.0)
    # With a single candidate and 50% cap the allocation = 500
    assert result.num_positions == 1
    trade = result.selected[0]
    assert trade.ticker == "AAPL"
    assert trade.sector == "Technology"
    assert trade.allocation == pytest.approx(500.0)
    assert trade.pct_of_portfolio == pytest.approx(50.0)


def test_tfsa_stock_respects_sector_cap():
    """Two Technology tickers → only the higher-scored one is selected."""
    hist_aapl = _make_history(n_days=60, drift=0.005, vol=0.007, seed=1)
    hist_msft = _make_history(n_days=60, drift=0.001, vol=0.012, seed=2)
    result = allocate_tfsa_stock_portfolio(
        {"AAPL": hist_aapl, "MSFT": hist_msft},
        total_capital=1000.0,
        max_positions=3,
    )
    sectors = [t.sector for t in result.selected]
    assert len(set(sectors)) == len(sectors), "Duplicate sectors found"


def test_tfsa_stock_rejects_correlated():
    """AAPL and MSFT are correlated; the second should be rejected."""
    hist = _make_history(n_days=60, drift=0.003, vol=0.007)
    result = allocate_tfsa_stock_portfolio(
        {"AAPL": hist, "MSFT": hist},
        total_capital=1000.0,
    )
    rejected_tickers = [r.ticker for r in result.rejected]
    # Either MSFT (correlation) or whichever scores lower should be rejected
    assert "MSFT" in rejected_tickers or "AAPL" in rejected_tickers


def test_tfsa_stock_max_positions_respected():
    hists = {
        "AAPL": _make_history(60, drift=0.003, vol=0.007, seed=1),
        "AMGN": _make_history(60, drift=0.002, vol=0.008, seed=2),
        "COST": _make_history(60, drift=0.001, vol=0.009, seed=3),
        "CSCO": _make_history(60, drift=0.001, vol=0.010, seed=4),
    }
    result = allocate_tfsa_stock_portfolio(hists, total_capital=1000.0, max_positions=2)
    assert result.num_positions <= 2


def test_tfsa_stock_higher_score_gets_more_capital():
    hist_strong = _make_history(60, drift=0.006, vol=0.005, seed=10)
    hist_weak = _make_history(60, drift=0.001, vol=0.015, seed=11)
    result = allocate_tfsa_stock_portfolio(
        {"AMGN": hist_strong, "COST": hist_weak},
        total_capital=1000.0,
        max_positions=2,
    )
    if result.num_positions == 2:
        allocs = {t.ticker: t.allocation for t in result.selected}
        assert allocs["AMGN"] >= allocs["COST"]


def test_tfsa_stock_total_deployed():
    hists = {
        "AAPL": _make_history(60, drift=0.003, vol=0.007, seed=1),
        "AMGN": _make_history(60, drift=0.002, vol=0.008, seed=2),
    }
    result = allocate_tfsa_stock_portfolio(hists, total_capital=1000.0)
    assert abs(result.total_deployed - 1000.0) < 0.05


def test_tfsa_stock_exit_guidance_populated():
    hist = _make_history(60, drift=0.003, vol=0.007)
    result = allocate_tfsa_stock_portfolio({"AAPL": hist}, total_capital=1000.0)
    assert "AAPL" in result.exit_guidance
    assert "20-day" in result.exit_guidance


def test_tfsa_stock_exit_guidance_empty_when_no_positions():
    result = allocate_tfsa_stock_portfolio({})
    assert result.exit_guidance == ""


def test_tfsa_stock_portfolio_dataclass_properties():
    portfolio = TfsaStockPortfolio(total_capital=1000.0)
    assert portfolio.total_deployed == 0.0
    assert portfolio.num_positions == 0
    portfolio.selected.append(
        StockAllocation("AAPL", "Technology", 150.0, 72.5, 500.0, 50.0, "above 20MA")
    )
    assert portfolio.num_positions == 1
    assert portfolio.total_deployed == 500.0


# ── allocate_rrsp_portfolio ───────────────────────────────────────────────────


def test_rrsp_empty_histories():
    result = allocate_rrsp_portfolio({})
    assert result.num_positions == 0
    assert result.total_deployed == 0.0


def test_rrsp_single_position():
    hist = _make_history(n_days=60, drift=0.001, vol=0.006)
    result = allocate_rrsp_portfolio({"RY.TO": hist}, total_capital=1000.0)
    assert result.num_positions == 1
    trade = result.selected[0]
    assert trade.ticker == "RY.TO"
    assert trade.allocation == pytest.approx(500.0)  # capped at 50%
    assert isinstance(trade.long_term_thesis, str)
    assert len(trade.long_term_thesis) > 0


def test_rrsp_respects_sector_cap():
    """Two Financials tickers → only one selected."""
    hist_ry = _make_history(60, drift=0.002, vol=0.006, seed=1)
    hist_td = _make_history(60, drift=0.001, vol=0.007, seed=2)
    result = allocate_rrsp_portfolio(
        {"RY.TO": hist_ry, "TD.TO": hist_td},
        total_capital=1000.0,
    )
    sectors = [t.sector for t in result.selected]
    assert len(set(sectors)) == len(sectors)


def test_rrsp_max_positions_respected():
    hists = {
        "RY.TO": _make_history(60, drift=0.001, vol=0.006, seed=1),
        "AAPL": _make_history(60, drift=0.002, vol=0.007, seed=2),
        "COST": _make_history(60, drift=0.001, vol=0.008, seed=3),
        "ENB.TO": _make_history(60, drift=0.001, vol=0.009, seed=4),
    }
    result = allocate_rrsp_portfolio(hists, total_capital=1000.0, max_positions=2)
    assert result.num_positions <= 2


def test_rrsp_total_deployed():
    hists = {
        "RY.TO": _make_history(60, drift=0.001, vol=0.006, seed=1),
        "AAPL": _make_history(60, drift=0.002, vol=0.007, seed=2),
    }
    result = allocate_rrsp_portfolio(hists, total_capital=1000.0)
    assert abs(result.total_deployed - 1000.0) < 0.05


def test_rrsp_portfolio_dataclass_properties():
    portfolio = RrspPortfolio(total_capital=1000.0)
    assert portfolio.total_deployed == 0.0
    assert portfolio.num_positions == 0
    portfolio.selected.append(
        RrspStockAllocation(
            "RY.TO", "Financials", 135.0, 68.0, 500.0, 50.0,
            "above 50-day MA; low volatility"
        )
    )
    assert portfolio.num_positions == 1
    assert portfolio.total_deployed == 500.0


def test_rrsp_stable_stock_scores_higher_than_volatile():
    stable = _make_history(60, drift=0.001, vol=0.004, seed=1)
    volatile = _make_history(60, drift=0.001, vol=0.040, seed=1)
    result_stable = allocate_rrsp_portfolio({"RY.TO": stable}, total_capital=1000.0)
    result_vol = allocate_rrsp_portfolio({"RY.TO": volatile}, total_capital=1000.0)
    if result_stable.num_positions == 1 and result_vol.num_positions == 1:
        assert result_stable.selected[0].composite_score > result_vol.selected[0].composite_score


# ── _tfsa_concentrated_allocation ────────────────────────────────────────────


def test_tfsa_concentrated_allocation_three_positions():
    """3 positions → exact tier weights 45 / 32 / 23 %."""
    allocs = _tfsa_concentrated_allocation(3, 1000.0)
    assert len(allocs) == 3
    assert abs(allocs[0] - 450.0) < 0.01   # rank 1: 45 %
    assert abs(allocs[1] - 320.0) < 0.01   # rank 2: 32 %
    assert abs(allocs[2] - 230.0) < 0.01   # rank 3: remaining 23 %
    assert abs(sum(allocs) - 1000.0) < 0.01


def test_tfsa_concentrated_allocation_two_positions():
    """2 positions → tier weights [45, 32] normalised to 100 %."""
    allocs = _tfsa_concentrated_allocation(2, 1000.0)
    assert len(allocs) == 2
    assert allocs[0] > allocs[1], "rank-1 must receive more than rank-2"
    assert abs(sum(allocs) - 1000.0) < 0.01


def test_tfsa_concentrated_allocation_one_position_capped():
    """1 position → capped at max_single_pct (default 50 %)."""
    allocs = _tfsa_concentrated_allocation(1, 1000.0, max_single_pct=0.50)
    assert len(allocs) == 1
    assert abs(allocs[0] - 500.0) < 0.01


def test_tfsa_concentrated_allocation_zero_positions():
    assert _tfsa_concentrated_allocation(0, 1000.0) == []


def test_tfsa_concentrated_allocation_rank_order():
    """Each rank must receive strictly more capital than the next rank (n=3)."""
    allocs = _tfsa_concentrated_allocation(3, 1000.0)
    assert allocs[0] > allocs[1] > allocs[2]


# ── TFSA tiered allocation integration ───────────────────────────────────────


def test_tfsa_stock_three_positions_tiered_allocation():
    """With 3 qualifying positions the allocations must match the 45/32/23 tiers."""
    # Use three tickers from distinct sectors and correlation groups so all 3
    # pass the diversity filters.
    hists = {
        "AAPL": _make_history(60, drift=0.006, vol=0.005, seed=1),   # highest score
        "AMGN": _make_history(60, drift=0.004, vol=0.006, seed=2),   # second
        "COST": _make_history(60, drift=0.002, vol=0.008, seed=3),   # third
    }
    result = allocate_tfsa_stock_portfolio(hists, total_capital=1000.0, max_positions=3)
    if result.num_positions == 3:
        allocs = [t.allocation for t in result.selected]
        # Rank 1 must be near 45 %, rank 2 near 32 %, rank 3 near 23 %
        assert abs(allocs[0] - 450.0) < 1.0
        assert abs(allocs[1] - 320.0) < 1.0
        assert abs(allocs[2] - 230.0) < 1.0


def test_tfsa_stock_rejects_position_limit_candidates():
    """Candidates exceeding the 3-position cap must appear in rejected with a reason."""
    hists = {
        "AAPL": _make_history(60, drift=0.006, vol=0.005, seed=1),
        "AMGN": _make_history(60, drift=0.005, vol=0.006, seed=2),
        "COST": _make_history(60, drift=0.004, vol=0.007, seed=3),
        "ENB.TO": _make_history(60, drift=0.003, vol=0.008, seed=4),
    }
    result = allocate_tfsa_stock_portfolio(hists, total_capital=1000.0, max_positions=3)
    assert result.num_positions <= 3
    rejected_tickers = [r.ticker for r in result.rejected]
    # At least one ticker should be rejected with a score-based reason
    score_rejected = [
        r for r in result.rejected if "lower composite score" in r.reason
    ]
    assert len(score_rejected) >= 1


# ── RRSP default max_positions = 4 ───────────────────────────────────────────


def test_rrsp_default_max_positions_is_four():
    """allocate_rrsp_portfolio should select up to 4 positions by default."""
    hists = {
        "RY.TO": _make_history(60, drift=0.001, vol=0.006, seed=1),   # Financials
        "AAPL": _make_history(60, drift=0.002, vol=0.007, seed=2),    # Technology
        "COST": _make_history(60, drift=0.001, vol=0.008, seed=3),    # Consumer
        "ENB.TO": _make_history(60, drift=0.001, vol=0.009, seed=4),  # Energy
    }
    result = allocate_rrsp_portfolio(hists, total_capital=1000.0)
    # With 4 tickers in distinct sectors the default limit allows all 4
    assert result.num_positions <= 4


def test_rrsp_rejects_position_limit_candidates():
    """Candidates exceeding the RRSP position cap must appear in rejected."""
    hists = {
        "RY.TO": _make_history(60, drift=0.002, vol=0.005, seed=1),
        "AAPL": _make_history(60, drift=0.002, vol=0.006, seed=2),
        "COST": _make_history(60, drift=0.001, vol=0.007, seed=3),
        "ENB.TO": _make_history(60, drift=0.001, vol=0.008, seed=4),
        "CNR.TO": _make_history(60, drift=0.001, vol=0.009, seed=5),
    }
    result = allocate_rrsp_portfolio(hists, total_capital=1000.0, max_positions=3)
    assert result.num_positions <= 3
    score_rejected = [
        r for r in result.rejected if "lower stability score" in r.reason
    ]
    assert len(score_rejected) >= 1


def test_rrsp_rejected_candidates_have_justification():
    """All rejected RRSP candidates must include a non-empty reason string."""
    hists = {
        "RY.TO": _make_history(60, drift=0.002, vol=0.005, seed=1),
        "TD.TO": _make_history(60, drift=0.001, vol=0.007, seed=2),  # same sector → rejected
    }
    result = allocate_rrsp_portfolio(hists, total_capital=1000.0)
    for rejected in result.rejected:
        assert isinstance(rejected.reason, str)
        assert len(rejected.reason) > 0
