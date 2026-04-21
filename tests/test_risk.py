"""Unit tests for scanner.risk."""

import pandas as pd

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
