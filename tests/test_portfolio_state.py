"""Unit tests for portfolio_state helpers and weekly summary."""

import pytest

from scanner.portfolio_state import (
    backfill_legacy_holdings_in_state,
    weekly_options_performance_summary,
)


def test_backfill_legacy_holdings_in_state_adds_missing_accounts_once():
    state = {
        "positions": [
            {
                "ticker": "RY.TO",
                "account_type": "RRSP",
                "sub_portfolio": "stability",
                "entry_date": "2026-07-01",
                "entry_price": 0.0,
                "quantity": 1,
                "entry_composite_score": 0.0,
                "entry_thesis_tags": ["legacy-seed"],
                "status": "HOLD",
                "metadata": {},
            }
        ],
        "closed_positions": [],
    }

    added = backfill_legacy_holdings_in_state(
        state,
        rrsp_holdings=["RY.TO"],
        tfsa_holdings=["SHOP.TO"],
        fhsa_holdings=["AAPL"],
    )
    assert added == 2

    keys = {
        (
            str(p.get("ticker", "")).upper(),
            str(p.get("account_type", "")).upper(),
            str(p.get("sub_portfolio", "")).lower(),
        )
        for p in state["positions"]
    }
    assert ("RY.TO", "RRSP", "stability") in keys
    assert ("SHOP.TO", "TFSA", "growth") in keys
    assert ("AAPL", "FHSA", "growth") in keys

    # Second pass is idempotent.
    added_again = backfill_legacy_holdings_in_state(
        state,
        rrsp_holdings=["RY.TO"],
        tfsa_holdings=["SHOP.TO"],
        fhsa_holdings=["AAPL"],
    )
    assert added_again == 0


def test_weekly_options_summary_filters_high_conviction_and_calculates_pnl_change():
    state = {
        "positions": [
            {
                "ticker": "AAPL",
                "account_type": "TFSA",
                "sub_portfolio": "long-call",
                "entry_price": 2.0,
                "quantity": 2,
                "entry_composite_score": 9.5,
                "status": "HOLD",
                "metadata": {
                    "option_type": "call",
                    "expiry": "2026-08-21",
                    "performance_history": [
                        {"date": "2026-06-25", "mark": 1.8, "pnl_total": -40.0},
                        {"date": "2026-07-02", "mark": 2.1, "pnl_total": 20.0},
                        {"date": "2026-07-06", "mark": 2.3, "pnl_total": 60.0},
                    ],
                },
            },
            {
                "ticker": "MSFT",
                "account_type": "OPTIONS",
                "sub_portfolio": "put-spread",
                "entry_price": 1.5,
                "quantity": 1,
                "entry_composite_score": 7.9,  # below min_entry_score -> excluded
                "status": "HOLD",
                "metadata": {
                    "option_type": "put",
                    "expiry": "2026-07-17",
                    "performance_history": [
                        {"date": "2026-07-03", "mark": 1.2, "pnl_total": 30.0},
                        {"date": "2026-07-06", "mark": 1.1, "pnl_total": 40.0},
                    ],
                },
            },
            {
                "ticker": "RY.TO",
                "account_type": "RRSP",
                "sub_portfolio": "stability",  # non-option
                "entry_price": 100.0,
                "quantity": 1,
                "entry_composite_score": 80.0,
                "status": "HOLD",
                "metadata": {},
            },
        ],
        "closed_positions": [],
    }

    summary = weekly_options_performance_summary(
        state,
        min_entry_score=8.0,
        lookback_days=7,
        as_of="2026-07-06",
    )

    assert summary["tracked_positions"] == 1
    assert summary["lookback_days"] == 7
    assert summary["min_entry_score"] == pytest.approx(8.0)

    row = summary["rows"][0]
    assert row["ticker"] == "AAPL"
    # Lookback window includes 2026-07-02 and 2026-07-06
    assert row["weekly_pnl_change"] == pytest.approx(40.0)
    assert row["unrealized_pnl"] == pytest.approx(60.0)
    # Basis = entry(2.0) * 100 * qty(2) = 400
    assert row["weekly_return_pct"] == pytest.approx(10.0)
    assert row["days_captured"] == 2

    assert summary["total_weekly_pnl_change"] == pytest.approx(40.0)
    assert summary["total_unrealized_pnl"] == pytest.approx(60.0)


def test_weekly_options_summary_handles_missing_history():
    state = {
        "positions": [
            {
                "ticker": "AAPL",
                "account_type": "TFSA",
                "sub_portfolio": "long-call",
                "entry_price": 2.0,
                "quantity": 1,
                "entry_composite_score": 9.0,
                "status": "HOLD",
                "metadata": {"option_type": "call", "expiry": "2026-08-21"},
            }
        ],
        "closed_positions": [],
    }

    summary = weekly_options_performance_summary(
        state,
        min_entry_score=8.0,
        lookback_days=7,
        as_of="2026-07-06",
    )

    assert summary["tracked_positions"] == 0
    assert summary["rows"] == []
    assert summary["total_weekly_pnl_change"] == pytest.approx(0.0)
