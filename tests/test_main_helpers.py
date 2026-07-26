"""Unit tests for helper functions in main.py."""

from main import _available_account_cash, _record_new_entries
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
