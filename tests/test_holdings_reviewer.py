"""Tests for options mark-to-market tracking in holdings_reviewer."""

import pandas as pd
import pytest

from scanner.holdings_reviewer import track_options_performance


def _chain(calls_rows=None, puts_rows=None):
    calls = pd.DataFrame(calls_rows or [])
    puts = pd.DataFrame(puts_rows or [])
    return calls, puts


def test_track_options_performance_long_call_updates_metadata(monkeypatch):
    """Long call mark/P&L should be computed and stored in metadata history."""

    monkeypatch.setattr("scanner.holdings_reviewer.get_stock_price", lambda ticker: 100.0)
    monkeypatch.setattr(
        "scanner.holdings_reviewer.get_options_chain",
        lambda ticker, expiry: _chain(
            calls_rows=[{"strike": 100.0, "bid": 2.0, "ask": 2.2}],
            puts_rows=[],
        ),
    )

    position = {
        "ticker": "AAPL",
        "account_type": "TFSA",
        "sub_portfolio": "long-call",
        "entry_price": 1.5,
        "quantity": 2,
        "status": "HOLD",
        "metadata": {
            "option_type": "call",
            "expiry": "2026-08-21",
            "strike": 100.0,
        },
    }

    df = track_options_performance([position], as_of="2026-07-06")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["mark"] == pytest.approx(2.1)
    # (2.1 - 1.5) * 100 * 2 contracts
    assert row["unrealized_pnl"] == pytest.approx(120.0)

    history = position["metadata"]["performance_history"]
    assert len(history) == 1
    assert history[0]["date"] == "2026-07-06"
    assert history[0]["mark"] == pytest.approx(2.1)


def test_track_options_performance_put_spread_uses_short_minus_long(monkeypatch):
    """Credit spread mark should use short leg minus long leg."""

    monkeypatch.setattr("scanner.holdings_reviewer.get_stock_price", lambda ticker: 50.0)
    monkeypatch.setattr(
        "scanner.holdings_reviewer.get_options_chain",
        lambda ticker, expiry: _chain(
            calls_rows=[],
            puts_rows=[
                {"strike": 45.0, "bid": 2.0, "ask": 2.2},
                {"strike": 40.0, "bid": 0.6, "ask": 0.8},
            ],
        ),
    )

    position = {
        "ticker": "RY.TO",
        "account_type": "OPTIONS",
        "sub_portfolio": "put-spread",
        "entry_price": 1.5,  # entry credit
        "quantity": 1,
        "status": "HOLD",
        "metadata": {
            "option_type": "put",
            "expiry": "2026-07-17",
            "strike": 45.0,
            "long_strike": 40.0,
        },
    }

    df = track_options_performance([position], as_of="2026-07-06")

    row = df.iloc[0]
    # short mid 2.1 minus long mid 0.7 = 1.4
    assert row["mark"] == pytest.approx(1.4)
    # credit spread P&L = (entry - mark) * 100
    assert row["unrealized_pnl"] == pytest.approx(10.0)


def test_track_options_performance_overwrites_same_day_snapshot(monkeypatch):
    """Running twice on the same date should not duplicate history rows."""

    monkeypatch.setattr("scanner.holdings_reviewer.get_stock_price", lambda ticker: 110.0)
    monkeypatch.setattr(
        "scanner.holdings_reviewer.get_options_chain",
        lambda ticker, expiry: _chain(
            calls_rows=[{"strike": 100.0, "bid": 3.0, "ask": 3.2}],
            puts_rows=[],
        ),
    )

    position = {
        "ticker": "MSFT",
        "account_type": "TFSA",
        "sub_portfolio": "long-call",
        "entry_price": 2.5,
        "quantity": 1,
        "status": "HOLD",
        "metadata": {
            "option_type": "call",
            "expiry": "2026-08-21",
            "strike": 100.0,
            "performance_history": [
                {"date": "2026-07-05", "mark": 2.8},
                {"date": "2026-07-06", "mark": 3.0},
            ],
        },
    }

    df = track_options_performance([position], as_of="2026-07-06")

    history = position["metadata"]["performance_history"]
    assert len(history) == 2
    assert history[-1]["date"] == "2026-07-06"
    # New mark should be midpoint of 3.0 and 3.2 -> 3.1
    assert history[-1]["mark"] == pytest.approx(3.1)
    # Day change should be vs prior day (2.8), not stale same-day row
    assert df.iloc[0]["daily_change"] == pytest.approx(0.3)
