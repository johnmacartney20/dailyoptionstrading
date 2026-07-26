"""Unit tests for helper functions in main.py."""

from main import _available_account_cash


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
