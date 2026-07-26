"""Unit tests for portfolio_state helpers."""

from scanner.portfolio_state import backfill_legacy_holdings_in_state


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
