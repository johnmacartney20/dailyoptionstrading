"""Risk controls and position sizing helpers.

These helpers are intentionally simple and conservative:
- Compute a per-contract notional exposure.
- Optionally cap contracts per trade (by cash and/or notional).
- Optionally select a top-ranked subset to fit a total notional budget.

They do not attempt to model margin, assignment probability, fees, or slippage.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import pandas as pd


def add_position_sizing_columns(
    suggestions: pd.DataFrame,
    account_cash: Optional[float] = None,
    max_notional_per_trade: Optional[float] = None,
) -> pd.DataFrame:
    """Return a copy of *suggestions* with sizing columns added.

    Added columns (when computable):
    - ``notional_per_contract``: exposure proxy per 1 contract
    - ``max_contracts``: min of applicable caps (cash + per-trade notional)

    If no caps are provided, only ``notional_per_contract`` is added.
    """
    if suggestions.empty:
        return suggestions

    out = suggestions.copy()

    def notional_per_contract(row: pd.Series) -> float:
        option_type = str(row.get("option_type", ""))
        if option_type == "put":
            strike = float(row.get("strike", 0.0) or 0.0)
            return max(strike, 0.0) * 100.0
        if option_type == "call":
            stock_price = float(row.get("stock_price", 0.0) or 0.0)
            return max(stock_price, 0.0) * 100.0
        return float("nan")

    out["notional_per_contract"] = out.apply(notional_per_contract, axis=1)

    caps = []
    if account_cash is not None:
        cash = float(account_cash)
        caps.append((cash / out["notional_per_contract"]).apply(_safe_floor_int))
    if max_notional_per_trade is not None:
        cap = float(max_notional_per_trade)
        caps.append((cap / out["notional_per_contract"]).apply(_safe_floor_int))

    if caps:
        max_contracts = caps[0]
        for c in caps[1:]:
            max_contracts = max_contracts.where(max_contracts < c, c)
        out["max_contracts"] = max_contracts

    return out


def filter_unaffordable_trades(suggestions: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows with ``max_contracts`` < 1 when that column is present."""
    if suggestions.empty:
        return suggestions
    if "max_contracts" not in suggestions.columns:
        return suggestions

    out = suggestions.copy()
    out = out[out["max_contracts"].fillna(1).astype(int) >= 1]
    return out.reset_index(drop=True)


def allocate_under_total_notional(
    suggestions: pd.DataFrame,
    max_total_notional: float,
    max_trades_per_ticker: Optional[int] = None,
) -> pd.DataFrame:
    """Greedily allocate contracts to highest-ranked rows within a notional budget.

    Expects *suggestions* to already be sorted best-to-worst (e.g. by ``score``).

    Returns a filtered copy with:
    - ``selected_contracts``
    - ``selected_notional``

    Notes
    -----
    - This is *not* an optimizer; it's a simple greedy allocator.
    - If ``max_contracts`` is missing, each row is treated as max 1 contract.
    """
    if suggestions.empty:
        return suggestions

    if max_total_notional <= 0:
        return suggestions.iloc[0:0].copy()

    out = suggestions.copy()

    if "notional_per_contract" not in out.columns:
        out = add_position_sizing_columns(out, account_cash=None, max_notional_per_trade=None)

    max_per_row = (
        out["max_contracts"].fillna(1).astype(int)
        if "max_contracts" in out.columns
        else pd.Series([1] * len(out), index=out.index)
    )

    ticker_cap = int(max_trades_per_ticker) if max_trades_per_ticker else None
    ticker_counts: dict[str, int] = defaultdict(int)

    selected_contracts = []
    total_used = 0.0

    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", ""))
        if ticker_cap is not None and ticker:
            if ticker_counts[ticker] >= ticker_cap:
                selected_contracts.append(0)
                continue

        notional = float(row.get("notional_per_contract", float("nan")))
        if not math.isfinite(notional) or notional <= 0:
            selected_contracts.append(0)
            continue

        remaining = float(max_total_notional) - total_used
        if remaining < notional:
            selected_contracts.append(0)
            continue

        row_cap = int(max_per_row.loc[idx])
        row_cap = max(row_cap, 0)
        if row_cap < 1:
            selected_contracts.append(0)
            continue

        feasible = int(remaining // notional)
        contracts = min(row_cap, feasible)
        if contracts < 1:
            selected_contracts.append(0)
            continue

        selected_contracts.append(contracts)
        total_used += contracts * notional
        if ticker_cap is not None and ticker:
            ticker_counts[ticker] += 1

    out["selected_contracts"] = selected_contracts
    out["selected_notional"] = out["selected_contracts"] * out["notional_per_contract"]

    out = out[out["selected_contracts"] > 0].reset_index(drop=True)
    return out


def _safe_floor_int(value: float) -> int:
    try:
        if not math.isfinite(float(value)):
            return 0
        return int(math.floor(float(value)))
    except Exception:  # noqa: BLE001
        return 0
