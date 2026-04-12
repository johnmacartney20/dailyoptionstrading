"""Options analytics module.

Pure-function helpers for computing metrics used to rank and filter options:
days-to-expiration, out-of-the-money percentage, annualized return on capital,
and a composite score.
"""

import math
from datetime import date, datetime

import pandas as pd


def calculate_dte(expiry_str: str) -> int:
    """Return calendar days from today to *expiry_str* (``"YYYY-MM-DD"`` format).

    Returns 0 if the expiration date is today or in the past.
    """
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    delta = (expiry - date.today()).days
    return max(delta, 0)


def calculate_otm_pct(strike: float, stock_price: float, option_type: str) -> float:
    """Return the out-of-the-money percentage for an option.

    Positive values mean the option is OTM; negative values mean ITM.

    * **put**:  ``(stock_price - strike) / stock_price``
      – a put with strike < stock_price is OTM.
    * **call**: ``(strike - stock_price) / stock_price``
      – a call with strike > stock_price is OTM.
    """
    if stock_price <= 0:
        return float("nan")
    if option_type == "put":
        return (stock_price - strike) / stock_price
    # call
    return (strike - stock_price) / stock_price


def calculate_annualized_return(bid: float, strike: float, dte: int) -> float:
    """Return the annualised return on capital (as a percentage) for a short option.

    Formula: ``(bid / strike) × (365 / dte) × 100``

    This represents the premium received as a percentage of the capital at risk
    (the strike price for a cash-secured put, or the effective cost basis for a
    covered call), scaled to a one-year horizon.

    Returns ``0.0`` when inputs are invalid.
    """
    if dte <= 0 or strike <= 0 or bid <= 0:
        return 0.0
    return (bid / strike) * (365.0 / dte) * 100.0


def score_option(
    bid: float,
    strike: float,
    dte: int,
    open_interest: int,
    implied_volatility: float,
) -> float:
    """Return a composite score for ranking options (higher is better).

    Components:
    * **Annualized return** – primary driver of score (already reflects IV).
    * **Liquidity bonus** – log-scaled open-interest bonus (max +20 points)
      so that highly liquid options rank above similarly-priced illiquid ones.
    """
    ann_return = calculate_annualized_return(bid, strike, dte)
    if ann_return == 0.0:
        return 0.0

    # Liquidity bonus: log10(OI+1) gives 0 for OI=0, ≈3 for OI=1000
    oi_bonus = min(math.log10(max(open_interest, 1)) * 5.0, 20.0)

    return ann_return + oi_bonus


def enrich_options(
    df: pd.DataFrame,
    stock_price: float,
    option_type: str,
    expiry: str,
    ticker: str,
) -> pd.DataFrame:
    """Add computed columns to a raw options DataFrame.

    Added columns: ``ticker``, ``option_type``, ``expiry``, ``dte``,
    ``stock_price``, ``otm_pct``, ``annualized_return``, ``score``.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    dte = calculate_dte(expiry)

    out["ticker"] = ticker
    out["option_type"] = option_type
    out["expiry"] = expiry
    out["dte"] = dte
    out["stock_price"] = stock_price

    out["otm_pct"] = out["strike"].apply(
        lambda s: calculate_otm_pct(s, stock_price, option_type)
    )
    out["annualized_return"] = out.apply(
        lambda row: calculate_annualized_return(row["bid"], row["strike"], dte),
        axis=1,
    )
    out["score"] = out.apply(
        lambda row: score_option(
            row["bid"],
            row["strike"],
            dte,
            int(row.get("openInterest", 0) or 0),
            float(row.get("impliedVolatility", 0.0) or 0.0),
        ),
        axis=1,
    )
    return out
