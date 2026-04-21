"""Options screening and suggestion module.

Applies the configurable screening criteria to enriched options DataFrames and
produces a ranked list of trade suggestions.
"""

import logging
from typing import List

import pandas as pd

from .analyzer import enrich_options
from .config import SCREENING_PARAMS

logger = logging.getLogger(__name__)

# Columns to include in the final suggestions output (in display order).
OUTPUT_COLUMNS = [
    "ticker",
    "option_type",
    "expiry",
    "dte",
    "strike",
    "stock_price",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "openInterest",
    "volume",
    "impliedVolatility",
    "otm_pct",
    "annualized_return",
    "bid_ask_spread_pct",
    "risk_adjusted_return",
    "max_spread_loss",
    "spread_structure",
    "score",
    "tfsa_score",
    "tfsa_spread",
    "inTheMoney",
    "contractSymbol",
]


def screen_options(
    options_df: pd.DataFrame,
    stock_price: float,
    option_type: str,
    expiry: str,
    ticker: str,
) -> pd.DataFrame:
    """Enrich and filter *options_df* returning only qualifying candidates.

    Filtering rules (all from :data:`~scanner.config.SCREENING_PARAMS`):

    * DTE within the configured window.
    * Bid ≥ ``min_bid``.
    * Open interest ≥ ``min_open_interest`` (500 – enforces liquidity).
    * Bid-ask spread ≤ ``max_bid_ask_spread_pct`` of the bid price.
    * OTM % between ``min_otm_pct`` (3 %) and ``max_otm_pct`` (15 %).
      Trades within 3 % of the stock price are excluded as too close to ATM.
    * Annualised return ≥ ``min_annualized_return_pct`` (floor only).
    * Max spread loss ≤ ``max_spread_loss`` (small-account compatibility).

    Returns an empty :class:`~pandas.DataFrame` when no options qualify.
    """
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    params = SCREENING_PARAMS

    # Enrich with computed metrics
    df = enrich_options(options_df, stock_price, option_type, expiry, ticker)

    # ── DTE filter ────────────────────────────────────────────────────────────
    df = df[
        (df["dte"] >= params["min_dte"]) & (df["dte"] <= params["max_dte"])
    ]
    if df.empty:
        return pd.DataFrame()

    # ── Numeric filters ───────────────────────────────────────────────────────
    df = df[df["bid"] >= params["min_bid"]]
    df = df[
        df["openInterest"].fillna(0).astype(int) >= params["min_open_interest"]
    ]
    df = df[df["annualized_return"] >= params["min_annualized_return_pct"]]

    # ── Liquidity: bid-ask spread filter ─────────────────────────────────────
    df = df[
        df["bid_ask_spread_pct"].fillna(float("inf")) <= params["max_bid_ask_spread_pct"]
    ]

    # ── Moneyness filter: require min_otm_pct ≤ OTM % ≤ max_otm_pct ─────────
    df = df[
        (df["otm_pct"] >= params["min_otm_pct"])
        & (df["otm_pct"] <= params["max_otm_pct"])
    ]

    # ── Small-account: max spread loss filter ─────────────────────────────────
    df = df[df["max_spread_loss"] <= params["max_spread_loss"]]

    if df.empty:
        return pd.DataFrame()

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def generate_suggestions(screened_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge all per-ticker/expiry screened DataFrames into a ranked table.

    Parameters
    ----------
    screened_frames:
        List of DataFrames produced by :func:`screen_options`.  May contain
        empty DataFrames – they are silently ignored.

    Returns
    -------
    A single DataFrame sorted by ``score`` descending, with only the standard
    :data:`OUTPUT_COLUMNS` (any missing columns are skipped gracefully).
    """
    non_empty = [f for f in screened_frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)

    # Keep only output columns that are actually present
    cols = [c for c in OUTPUT_COLUMNS if c in combined.columns]
    combined = combined[cols].sort_values("score", ascending=False).reset_index(drop=True)
    return combined
