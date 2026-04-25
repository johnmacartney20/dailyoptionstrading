"""Unit tests for scanner.risk."""

import pandas as pd
import pytest

from scanner.risk import add_position_sizing_columns, allocate_under_total_notional


def test_add_position_sizing_columns_put_cash_cap():
    df = pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "option_type": "put",
                "strike": 50.0,
                "stock_price": 55.0,
                "score": 10.0,
            }
        ]
    )
    out = add_position_sizing_columns(df, account_cash=10000.0, max_notional_per_trade=None)
    assert "notional_per_contract" in out.columns
    assert "max_contracts" in out.columns
    assert out.loc[0, "notional_per_contract"] == 5000.0
    assert int(out.loc[0, "max_contracts"]) == 2


def test_allocate_under_total_notional_respects_budget_and_ticker_cap():
    df = pd.DataFrame(
        [
            {"ticker": "AAA", "option_type": "put", "strike": 10.0, "stock_price": 11.0, "score": 100.0},
            {"ticker": "AAA", "option_type": "put", "strike": 10.0, "stock_price": 11.0, "score": 90.0},
            {"ticker": "BBB", "option_type": "put", "strike": 10.0, "stock_price": 11.0, "score": 80.0},
        ]
    ).sort_values("score", ascending=False).reset_index(drop=True)

    # Each row is 10*100 = 1000 notional per contract.
    picked = allocate_under_total_notional(df, max_total_notional=2000.0, max_trades_per_ticker=1)

    assert not picked.empty
    assert picked["selected_notional"].sum() <= 2000.0

    # Only one trade per ticker, so we should get AAA (top one) + BBB.
    assert picked["ticker"].tolist() == ["AAA", "BBB"]
    assert picked["selected_contracts"].tolist() == [1, 1]


def test_notional_uses_max_spread_loss_when_available():
    """When max_spread_loss is present, it should be used as notional — not strike × 100.

    A high-priced stock (e.g. strike=$200) has a notional of $20 000 per contract
    when computed as strike × 100.  But for a defined-risk $5-wide bull put spread
    the actual capital at risk is only max_spread_loss ≈ $450, allowing far more
    contracts to fit within a given cash budget.
    """
    # strike=$200 → strike×100 = $20 000; max_spread_loss=$450 (spread-based)
    df = pd.DataFrame(
        [
            {
                "ticker": "EXPNS",
                "option_type": "put",
                "strike": 200.0,
                "stock_price": 210.0,
                "score": 50.0,
                "max_spread_loss": 450.0,
            }
        ]
    )
    out = add_position_sizing_columns(df, account_cash=10_000.0)

    assert "notional_per_contract" in out.columns
    assert out.loc[0, "notional_per_contract"] == pytest.approx(450.0)
    # floor(10000 / 450) = 22
    assert int(out.loc[0, "max_contracts"]) == 22
