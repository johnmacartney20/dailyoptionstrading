"""Unit tests for scanner.portfolio_allocator."""

from datetime import date, timedelta

import pandas as pd
import pytest

from scanner.portfolio_allocator import (
    PortfolioAllocation,
    TradeAllocation,
    RejectedCandidate,
    TfsaTradeAllocation,
    TfsaAllocation,
    _are_correlated,
    _parse_long_strike,
    _parse_sell_strike_tfsa,
    _score_weighted_allocation,
    allocate_portfolio,
    allocate_tfsa_portfolio,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expiry(days: int = 30) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


def _make_put_row(
    ticker: str,
    strike: float = 95.0,
    bid: float = 1.5,
    score: float = 50.0,
    open_interest: int = 1000,
    spread_structure: str = "Sell 95P / Buy 90P",
    max_spread_loss: float = 350.0,
    expiry: str | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "option_type": "put",
        "strike": strike,
        "bid": bid,
        "score": score,
        "openInterest": open_interest,
        "spread_structure": spread_structure,
        "max_spread_loss": max_spread_loss,
        "expiry": expiry or _expiry(),
    }


def _make_suggestions(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── _are_correlated ────────────────────────────────────────────────────────────

def test_correlated_same_group():
    assert _are_correlated("AAPL", "MSFT") is True


def test_correlated_chip_stocks():
    assert _are_correlated("AMD", "NVDA") is True


def test_not_correlated_different_groups():
    assert _are_correlated("AAPL", "AMGN") is False


def test_not_correlated_unknown_ticker():
    assert _are_correlated("AAPL", "UNKNOWN") is False


# ── _parse_long_strike ─────────────────────────────────────────────────────────

def test_parse_long_strike_put():
    assert _parse_long_strike("Sell 95P / Buy 90P") == 90.0


def test_parse_long_strike_call():
    assert _parse_long_strike("Sell 105C / Buy 110C") == 110.0


def test_parse_long_strike_decimal():
    assert _parse_long_strike("Sell 97.5P / Buy 95.0P") == 95.0


def test_parse_long_strike_no_match():
    assert _parse_long_strike("invalid") is None


# ── _score_weighted_allocation ─────────────────────────────────────────────────

def test_allocation_proportional_to_score():
    allocs = _score_weighted_allocation([60.0, 40.0], total_capital=1000.0, max_position_pct=1.0)
    assert len(allocs) == 2
    assert abs(allocs[0] - 600.0) < 0.01
    assert abs(allocs[1] - 400.0) < 0.01


def test_allocation_cap_applied():
    # Both scores equal → 50/50, but cap is 40 % → both get 400, 200 unallocated
    # The redistribution is not triggered (both are at cap) so deployed = 800.
    allocs = _score_weighted_allocation([50.0, 50.0], total_capital=1000.0, max_position_pct=0.40)
    assert all(a <= 400.0 + 0.01 for a in allocs)


def test_allocation_three_trades_total_equals_capital():
    allocs = _score_weighted_allocation([70.0, 50.0, 30.0], total_capital=1000.0, max_position_pct=0.50)
    assert abs(sum(allocs) - 1000.0) < 0.05


def test_allocation_zero_scores_equal_weight():
    allocs = _score_weighted_allocation([0.0, 0.0], total_capital=1000.0, max_position_pct=1.0)
    assert abs(allocs[0] - 500.0) < 0.01
    assert abs(allocs[1] - 500.0) < 0.01


def test_allocation_single_trade():
    allocs = _score_weighted_allocation([80.0], total_capital=1000.0, max_position_pct=0.50)
    assert abs(allocs[0] - 500.0) < 0.01  # capped at 50%


# ── allocate_portfolio ─────────────────────────────────────────────────────────

def test_allocate_empty_suggestions():
    result = allocate_portfolio(pd.DataFrame())
    assert result.num_open_trades == 0
    assert result.total_deployed == 0.0


def test_allocate_no_puts():
    df = _make_suggestions(_make_put_row("AAPL"))
    df["option_type"] = "call"
    result = allocate_portfolio(df)
    assert result.num_open_trades == 0


def test_allocate_single_put():
    df = _make_suggestions(_make_put_row("AAPL", score=60.0))
    result = allocate_portfolio(df, total_capital=1000.0)
    assert result.num_open_trades == 1
    trade = result.selected[0]
    assert trade.ticker == "AAPL"
    assert trade.sector == "Technology"
    assert trade.strategy_type == "Bull Put Spread"
    # Single trade with 50% cap → allocation = 500
    assert trade.allocation == 500.0
    assert trade.pct_of_portfolio == 50.0


def test_allocate_respects_sector_cap():
    """Two tech tickers should result in only one being selected."""
    df = _make_suggestions(
        _make_put_row("AAPL", score=70.0),
        _make_put_row("MSFT", score=60.0, strike=290.0, spread_structure="Sell 290P / Buy 285P"),
    )
    result = allocate_portfolio(df, total_capital=1000.0)
    tickers = [t.ticker for t in result.selected]
    sectors = [t.sector for t in result.selected]
    # Both are Technology; only one should be accepted
    assert len(set(sectors)) == len(sectors), "Duplicate sectors found"


def test_allocate_rejects_correlated():
    """AAPL and MSFT are in the same correlation group; second should be rejected."""
    df = _make_suggestions(
        _make_put_row("AAPL", score=70.0),
        _make_put_row("MSFT", score=60.0, strike=290.0, spread_structure="Sell 290P / Buy 285P"),
    )
    result = allocate_portfolio(df, total_capital=1000.0)
    rejected_tickers = [r.ticker for r in result.rejected]
    # MSFT should be rejected (either correlation or duplicate sector)
    assert "MSFT" in rejected_tickers


def test_allocate_max_three_trades():
    rows = [
        _make_put_row("AAPL", score=80.0),
        _make_put_row("AMGN", score=70.0, strike=200.0, spread_structure="Sell 200P / Buy 190P"),
        _make_put_row("COST", score=60.0, strike=700.0, spread_structure="Sell 700P / Buy 690P"),
        _make_put_row("CSCO", score=50.0, strike=45.0, spread_structure="Sell 45P / Buy 42P"),
    ]
    df = _make_suggestions(*rows)
    result = allocate_portfolio(df, total_capital=1000.0, max_trades=3)
    assert result.num_open_trades <= 3


def test_allocate_total_deployed():
    rows = [
        _make_put_row("AAPL", score=80.0),
        _make_put_row("AMGN", score=70.0, strike=200.0, spread_structure="Sell 200P / Buy 190P"),
    ]
    df = _make_suggestions(*rows)
    result = allocate_portfolio(df, total_capital=1000.0)
    assert abs(result.total_deployed - 1000.0) < 0.05


def test_allocate_trade_fields_populated():
    df = _make_suggestions(
        _make_put_row("AAPL", strike=95.0, bid=1.5, score=60.0, max_spread_loss=350.0)
    )
    result = allocate_portfolio(df, total_capital=1000.0)
    assert result.num_open_trades == 1
    trade = result.selected[0]
    assert trade.short_strike == 95.0
    assert trade.long_strike == 90.0          # parsed from "Sell 95P / Buy 90P"
    assert trade.max_profit == 150.0          # 1.5 * 100
    assert trade.max_loss == 350.0


def test_allocate_portfolio_dataclass_properties():
    pa = PortfolioAllocation(total_capital=1000.0)
    assert pa.total_deployed == 0.0
    assert pa.num_open_trades == 0
    pa.selected.append(
        TradeAllocation("AAPL", "Technology", "Bull Put Spread", 95.0, 90.0,
                        "2025-05-16", 60.0, 150.0, 350.0, 500.0, 50.0)
    )
    assert pa.num_open_trades == 1
    assert pa.total_deployed == 500.0


# ── _parse_sell_strike_tfsa ───────────────────────────────────────────────────

def test_parse_sell_strike_tfsa_call():
    assert _parse_sell_strike_tfsa("Buy 105C / Sell 110C") == 110.0


def test_parse_sell_strike_tfsa_decimal():
    assert _parse_sell_strike_tfsa("Buy 97.5C / Sell 102.5C") == 102.5


def test_parse_sell_strike_tfsa_no_match():
    assert _parse_sell_strike_tfsa("invalid") is None


# ── allocate_tfsa_portfolio ────────────────────────────────────────────────────

def _make_call_row(
    ticker: str,
    strike: float = 105.0,
    bid: float = 1.0,
    ask: float = 1.1,
    score: float = 50.0,
    tfsa_score: float = 55.0,
    open_interest: int = 1000,
    tfsa_spread: str = "Buy 105C / Sell 110C",
    max_spread_loss: float = 390.0,
    expiry: str | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "option_type": "call",
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "score": score,
        "tfsa_score": tfsa_score,
        "openInterest": open_interest,
        "tfsa_spread": tfsa_spread,
        "max_spread_loss": max_spread_loss,
        "expiry": expiry or _expiry(),
    }


def test_tfsa_allocate_empty_suggestions():
    result = allocate_tfsa_portfolio(pd.DataFrame())
    assert result.num_open_trades == 0
    assert result.total_deployed == 0.0


def test_tfsa_allocate_no_calls():
    """Only put rows → no TFSA trades selected."""
    df = _make_suggestions(_make_put_row("AAPL"))
    result = allocate_tfsa_portfolio(df)
    assert result.num_open_trades == 0


def test_tfsa_allocate_single_call():
    df = _make_suggestions(_make_call_row("AAPL", strike=105.0, ask=1.0))
    result = allocate_tfsa_portfolio(df, total_capital=1000.0)
    assert result.num_open_trades == 1
    trade = result.selected[0]
    assert trade.ticker == "AAPL"
    assert trade.sector == "Technology"
    assert trade.strategy_type == "Call Debit Spread"
    assert trade.buy_strike == 105.0
    assert trade.sell_strike == 110.0  # 105 + 5 width
    # max_loss = ask * 100 = 100
    assert trade.max_loss == pytest.approx(100.0)
    # max_profit = (width - ask) * 100 = (5 - 1) * 100 = 400
    assert trade.max_profit == pytest.approx(400.0)


def test_tfsa_allocate_max_two_trades():
    """TFSA allocator should select at most 2 trades by default."""
    rows = [
        _make_call_row("AAPL", tfsa_score=80.0),
        _make_call_row("AMGN", strike=200.0, tfsa_score=70.0,
                       tfsa_spread="Buy 200C / Sell 210C"),
        _make_call_row("COST", strike=700.0, tfsa_score=60.0,
                       tfsa_spread="Buy 700C / Sell 710C"),
    ]
    df = _make_suggestions(*rows)
    result = allocate_tfsa_portfolio(df, total_capital=1000.0)
    assert result.num_open_trades <= 2


def test_tfsa_allocate_rejects_correlated():
    """AAPL and MSFT are correlated; second should be rejected."""
    df = _make_suggestions(
        _make_call_row("AAPL", tfsa_score=70.0),
        _make_call_row("MSFT", strike=300.0, tfsa_score=65.0,
                       tfsa_spread="Buy 300C / Sell 310C"),
    )
    result = allocate_tfsa_portfolio(df, total_capital=1000.0)
    rejected_tickers = [r.ticker for r in result.rejected]
    assert "MSFT" in rejected_tickers


def test_tfsa_allocate_rejects_duplicate_sector():
    """Two Technology tickers → only one selected."""
    df = _make_suggestions(
        _make_call_row("AAPL", tfsa_score=70.0),
        _make_call_row("NVDA", strike=800.0, tfsa_score=65.0,
                       tfsa_spread="Buy 800C / Sell 810C"),
    )
    result = allocate_tfsa_portfolio(df, total_capital=1000.0)
    sectors = [t.sector for t in result.selected]
    assert len(set(sectors)) == len(sectors)


def test_tfsa_allocate_sorted_by_tfsa_score():
    """Higher tfsa_score trade should be selected over lower one."""
    df = _make_suggestions(
        _make_call_row("AAPL", tfsa_score=40.0),
        _make_call_row("AMGN", strike=200.0, tfsa_score=75.0,
                       tfsa_spread="Buy 200C / Sell 210C"),
    )
    result = allocate_tfsa_portfolio(df, total_capital=1000.0, max_trades=1)
    assert result.num_open_trades == 1
    assert result.selected[0].ticker == "AMGN"


def test_tfsa_allocate_fallback_spread_without_tfsa_spread():
    """When tfsa_spread is missing, spread structure is inferred from strike."""
    row = _make_call_row("AAPL", strike=105.0, ask=1.0, tfsa_score=60.0)
    del row["tfsa_spread"]
    df = pd.DataFrame([row])
    result = allocate_tfsa_portfolio(df, total_capital=1000.0)
    assert result.num_open_trades == 1
    assert result.selected[0].sell_strike == 110.0


def test_tfsa_allocation_dataclass_properties():
    ta = TfsaAllocation(total_capital=1000.0)
    assert ta.total_deployed == 0.0
    assert ta.num_open_trades == 0
    ta.selected.append(
        TfsaTradeAllocation("AAPL", "Technology", "Call Debit Spread",
                            105.0, 110.0, "2025-05-16", 60.0, 400.0, 100.0, 600.0, 60.0)
    )
    assert ta.num_open_trades == 1
    assert ta.total_deployed == 600.0
