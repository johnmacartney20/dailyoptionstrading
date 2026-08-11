"""Unit tests for helper functions in main.py."""

import pandas as pd

from main import _available_account_cash, _record_new_entries
from scanner.risk import add_position_sizing_columns, filter_unaffordable_trades
from scanner.portfolio_allocator import (
    PortfolioAllocation,
    RrspPortfolio,
    RrspStockAllocation,
    StockAllocation,
    TfsaAllocation,
    TfsaStockPortfolio,
    TradeAllocation,
)


def test_available_account_cash_ignores_cash_rows_and_exits():
    state = {
        "positions": [
            {
                "account_type": "TFSA",
                "status": "HOLD",
                "entry_price": 100.0,
                "quantity": 2,
                "metadata": {},
            },
            {
                "account_type": "TFSA",
                "status": "FLAG",
                "entry_price": 50.0,
                "quantity": 1,
                "metadata": {},
            },
            {
                "account_type": "TFSA",
                "status": "EXIT",
                "entry_price": 500.0,
                "quantity": 1,
                "metadata": {},
            },
            {
                "account_type": "TFSA",
                "status": "HOLD",
                "entry_price": 1.0,
                "quantity": 100,
                "metadata": {"is_cash": True},
            },
        ]
    }
    # Deployed non-cash HOLD/FLAG = 250, so available = 750
    assert _available_account_cash(state, "TFSA", 1000.0) == 750.0


def test_available_account_cash_never_negative():
    state = {
        "positions": [
            {
                "account_type": "RRSP",
                "status": "HOLD",
                "entry_price": 600.0,
                "quantity": 2,
                "metadata": {},
            }
        ]
    }
    assert _available_account_cash(state, "RRSP", 1000.0) == 0.0


def test_record_new_entries_skips_non_finite_allocations_and_prices():
    state = {"positions": []}

    portfolio = PortfolioAllocation(
        total_capital=1000.0,
        selected=[
            TradeAllocation(
                ticker="AAPL",
                sector="Technology",
                strategy_type="Bull Put Spread",
                short_strike=95.0,
                long_strike=90.0,
                expiration="2026-08-21",
                score=60.0,
                max_profit=150.0,
                max_loss=350.0,
                allocation=float("nan"),
                pct_of_portfolio=50.0,
            )
        ],
    )
    options_stock = TfsaStockPortfolio(
        total_capital=1000.0,
        selected=[
            StockAllocation(
                ticker="MSFT",
                sector="Technology",
                current_price=250.0,
                composite_score=80.0,
                allocation=float("nan"),
                pct_of_portfolio=40.0,
                reasoning="strong trend",
            )
        ],
    )
    tfsa_opts = TfsaAllocation(total_capital=1000.0)
    tfsa_stock = TfsaStockPortfolio(
        total_capital=1000.0,
        selected=[
            StockAllocation(
                ticker="NVDA",
                sector="Semiconductors",
                current_price=float("nan"),
                composite_score=82.0,
                allocation=400.0,
                pct_of_portfolio=40.0,
                reasoning="strong trend",
            )
        ],
    )
    rrsp = RrspPortfolio(
        total_capital=1000.0,
        selected=[
            RrspStockAllocation(
                ticker="COST",
                sector="Consumer",
                current_price=100.0,
                composite_score=75.0,
                allocation=300.0,
                pct_of_portfolio=30.0,
                long_term_thesis="stable compounder",
            )
        ],
    )
    fhsa_stock = TfsaStockPortfolio(
        total_capital=1000.0,
        selected=[
            StockAllocation(
                ticker="AMZN",
                sector="Consumer",
                current_price=float("nan"),
                composite_score=78.0,
                allocation=250.0,
                pct_of_portfolio=25.0,
                reasoning="strong trend",
            )
        ],
    )

    _record_new_entries(
        state,
        portfolio,
        options_stock,
        tfsa_opts,
        tfsa_stock,
        rrsp,
        fhsa_stock,
    )

    assert [pos["ticker"] for pos in state["positions"]] == ["COST"]


# ── Always-on per-contract sizing tests ──────────────────────────────────────

def _make_options_suggestions(*dicts) -> pd.DataFrame:
    return pd.DataFrame(list(dicts))


def test_always_on_sizing_filters_unaffordable_puts():
    """Suggestions whose notional exceeds available cash are dropped."""
    # A single put spread with max_spread_loss = 6000 (capital at risk per contract).
    # Available OPTIONS-spreads cash is only 4000, so 0 contracts fit → filtered out.
    df = _make_options_suggestions(
        {
            "ticker": "AAPL",
            "option_type": "put",
            "strike": 200.0,
            "stock_price": 210.0,
            "score": 80.0,
            "max_spread_loss": 6000.0,
        }
    )
    options_spreads_capital = 4000.0
    sized = add_position_sizing_columns(df, account_cash=options_spreads_capital)
    filtered = filter_unaffordable_trades(sized)
    assert filtered.empty, "Suggestion exceeding available cash should be filtered out"


def test_always_on_sizing_keeps_affordable_puts():
    """Suggestions whose notional fits within available cash are kept and sized correctly."""
    # max_spread_loss = 500 per contract; 3000 available → up to 6 contracts.
    df = _make_options_suggestions(
        {
            "ticker": "MSFT",
            "option_type": "put",
            "strike": 100.0,
            "stock_price": 105.0,
            "score": 75.0,
            "max_spread_loss": 500.0,
        }
    )
    options_spreads_capital = 3000.0
    sized = add_position_sizing_columns(df, account_cash=options_spreads_capital)
    filtered = filter_unaffordable_trades(sized)
    assert len(filtered) == 1
    assert int(filtered.loc[0, "max_contracts"]) == 6


def test_always_on_sizing_mixed_affordability():
    """Only affordable suggestions survive; expensive ones are dropped."""
    df = _make_options_suggestions(
        {
            "ticker": "NVDA",
            "option_type": "put",
            "strike": 150.0,
            "stock_price": 155.0,
            "score": 90.0,
            "max_spread_loss": 400.0,   # affordable: 2500 / 400 = 6 contracts
        },
        {
            "ticker": "TSLA",
            "option_type": "put",
            "strike": 300.0,
            "stock_price": 310.0,
            "score": 85.0,
            "max_spread_loss": 3000.0,  # not affordable: 2500 < 3000
        },
    )
    options_spreads_capital = 2500.0
    sized = add_position_sizing_columns(df, account_cash=options_spreads_capital)
    filtered = filter_unaffordable_trades(sized)
    assert list(filtered["ticker"]) == ["NVDA"]
    assert int(filtered.loc[filtered["ticker"] == "NVDA", "max_contracts"].iloc[0]) == 6


def test_always_on_sizing_with_zero_available_cash():
    """When no cash is available, all suggestions are filtered out."""
    df = _make_options_suggestions(
        {
            "ticker": "AMGN",
            "option_type": "put",
            "strike": 50.0,
            "stock_price": 55.0,
            "score": 70.0,
            "max_spread_loss": 200.0,
        }
    )
    sized = add_position_sizing_columns(df, account_cash=0.0)
    filtered = filter_unaffordable_trades(sized)
    assert filtered.empty, "No cash available should result in no suggestions"

