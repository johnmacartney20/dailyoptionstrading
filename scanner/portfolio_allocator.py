"""Portfolio allocation module.

Selects the top 2–3 put-spread trades from the screened suggestions and
allocates a fixed capital budget ($1,000 by default) across them, respecting
per-trade and per-sector concentration limits and inter-trade correlation
exclusions.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .analyzer import _spread_width, suggest_call_debit_spread

# ── Sector mapping ─────────────────────────────────────────────────────────────
TICKER_SECTORS: Dict[str, str] = {
    # TSX – Financials
    "RY.TO": "Financials",
    "TD.TO": "Financials",
    "BNS.TO": "Financials",
    "BMO.TO": "Financials",
    "CM.TO": "Financials",
    "MFC.TO": "Financials",
    "SLF.TO": "Financials",
    "BAM.TO": "Financials",
    # TSX – Energy
    "ENB.TO": "Energy",
    "SU.TO": "Energy",
    "CNQ.TO": "Energy",
    "TRP.TO": "Energy",
    # TSX – Industrials
    "CNR.TO": "Industrials",
    "CP.TO": "Industrials",
    # TSX – Technology
    "SHOP.TO": "Technology",
    "CSU.TO": "Technology",
    # TSX – Telecom
    "BCE.TO": "Telecom",
    "T.TO": "Telecom",
    # TSX – Materials
    "ABX.TO": "Materials",
    "AEM.TO": "Materials",
    # TSX – Consumer
    "L.TO": "Consumer",
    "ATD.TO": "Consumer",
    # NASDAQ – Technology (Magnificent 7, excluding NVDA/TSLA which are Semiconductors/Consumer)
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Technology",
    "META": "Technology",
    "TSLA": "Technology",
    # NASDAQ – Semiconductors
    "NVDA": "Semiconductors",
    "AMD": "Semiconductors",
    "INTC": "Semiconductors",
    "QCOM": "Semiconductors",
    "AVGO": "Semiconductors",
    "MU": "Semiconductors",
    # NASDAQ – Software / Cloud
    "NFLX": "Software",
    "ADBE": "Software",
    "CRM": "Software",
    "CSCO": "Software",
    "ORCL": "Software",
    "NOW": "Software",
    # NASDAQ – Consumer / Services
    "COST": "Consumer",
    "SBUX": "Consumer",
    "PYPL": "Consumer",
    # NASDAQ – Healthcare / Biotech
    "AMGN": "Healthcare",
    "GILD": "Healthcare",
    "MRNA": "Healthcare",
}

# Groups of tickers considered highly correlated with each other.
# Only one trade per group is allowed in the portfolio.
_CORRELATION_GROUPS: List[frozenset] = [
    frozenset({"AAPL", "MSFT", "GOOGL", "AMZN", "META"}),           # Mega-cap tech
    frozenset({"NVDA", "AMD", "INTC", "AVGO", "MU", "QCOM"}),       # Chips / semiconductors
    frozenset({"AMGN", "GILD", "MRNA"}),                             # Biotech
    frozenset({"RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO"}),     # Canadian big banks
    frozenset({"ENB.TO", "TRP.TO", "SU.TO", "CNQ.TO"}),             # Canadian energy
    frozenset({"ABX.TO", "AEM.TO"}),                                  # Gold miners
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TradeAllocation:
    """Details for a single accepted trade in the portfolio."""

    ticker: str
    sector: str
    strategy_type: str
    short_strike: float
    long_strike: float
    expiration: str
    score: float
    max_profit: float       # per contract (dollars)
    max_loss: float         # per contract (dollars)
    allocation: float       # dollar amount allocated
    pct_of_portfolio: float  # 0–100


@dataclass
class RejectedCandidate:
    """A top candidate that was excluded from the portfolio."""

    ticker: str
    score: float
    reason: str


@dataclass
class PortfolioAllocation:
    """Full portfolio allocation result."""

    total_capital: float
    selected: List[TradeAllocation] = field(default_factory=list)
    rejected: List[RejectedCandidate] = field(default_factory=list)

    @property
    def total_deployed(self) -> float:
        return round(sum(t.allocation for t in self.selected), 2)

    @property
    def num_open_trades(self) -> int:
        return len(self.selected)


@dataclass
class TfsaTradeAllocation:
    """Details for a single accepted TFSA trade (long call / call debit spread)."""

    ticker: str
    sector: str
    strategy_type: str      # "Call Debit Spread"
    buy_strike: float       # lower-strike call we buy
    sell_strike: float      # higher-strike call we sell (defines the cap)
    expiration: str
    tfsa_score: float
    max_profit: float       # per contract (dollars) = (width − ask) × 100
    max_loss: float         # per contract (dollars) = ask × 100
    allocation: float       # dollar amount allocated
    pct_of_portfolio: float  # 0–100


@dataclass
class TfsaAllocation:
    """Full TFSA portfolio allocation result."""

    total_capital: float
    selected: List[TfsaTradeAllocation] = field(default_factory=list)
    rejected: List[RejectedCandidate] = field(default_factory=list)

    @property
    def total_deployed(self) -> float:
        return round(sum(t.allocation for t in self.selected), 2)

    @property
    def num_open_trades(self) -> int:
        return len(self.selected)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_sector(ticker: str) -> str:
    return TICKER_SECTORS.get(ticker, "Unknown")


def _are_correlated(ticker_a: str, ticker_b: str) -> bool:
    """Return True if *ticker_a* and *ticker_b* share a correlation group."""
    for group in _CORRELATION_GROUPS:
        if ticker_a in group and ticker_b in group:
            return True
    return False


def _parse_long_strike(spread_structure: str) -> Optional[float]:
    """Extract the long-leg strike price from a spread structure string.

    E.g. ``"Sell 95P / Buy 90P"`` → ``90.0``
    """
    match = re.search(r"Buy\s+([\d.]+)[PC]", spread_structure, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _parse_sell_strike_tfsa(spread_structure: str) -> Optional[float]:
    """Extract the sell-leg strike from a TFSA call debit spread string.

    E.g. ``"Buy 105C / Sell 110C"`` → ``110.0``
    """
    match = re.search(r"Sell\s+([\d.]+)[PC]", spread_structure, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _score_weighted_allocation(
    scores: List[float],
    total_capital: float,
    max_position_pct: float,
) -> List[float]:
    """Return score-weighted capital allocations capped at *max_position_pct*.

    Any capital that would exceed the per-position cap is redistributed once to
    the remaining un-capped positions (proportional to their scores).
    """
    n = len(scores)
    if n == 0:
        return []

    cap = total_capital * max_position_pct
    total_score = sum(scores)

    if total_score <= 0:
        # Equal weight fallback
        raw = [total_capital / n] * n
    else:
        raw = [s / total_score * total_capital for s in scores]

    capped = [min(a, cap) for a in raw]
    excess = sum(r - c for r, c in zip(raw, capped))  # total redistributable

    if excess > 0.01:
        uncapped_idx = [i for i, (r, c) in enumerate(zip(raw, capped)) if r < cap]
        uncapped_scores = [scores[i] for i in uncapped_idx]
        total_uncapped = sum(uncapped_scores)
        if total_uncapped > 0:
            for idx, i in enumerate(uncapped_idx):
                capped[i] = min(capped[i] + uncapped_scores[idx] / total_uncapped * excess, cap)

    return capped


# ── Public API ─────────────────────────────────────────────────────────────────

def allocate_portfolio(
    suggestions: pd.DataFrame,
    total_capital: float = 1000.0,
    max_trades: int = 3,
    max_position_pct: float = 0.50,
    max_sector_pct: float = 0.50,
) -> PortfolioAllocation:
    """Select the top 2–*max_trades* put-spread trades and allocate *total_capital*.

    Selection rules (applied in order):
    1. Put spreads only, ranked by score descending.
    2. Skip duplicate tickers.
    3. Skip tickers highly correlated with an already-selected trade.
    4. Skip tickers whose sector is already represented (enforces the 50 %
       sector cap conservatively for a 2–3 trade portfolio).

    Capital is then allocated proportionally to score, with each position
    capped at *max_position_pct* × *total_capital*.

    Parameters
    ----------
    suggestions:
        Full ranked suggestions DataFrame from :func:`~scanner.suggester.generate_suggestions`.
    total_capital:
        Total dollars to deploy (default ``1000.0``).
    max_trades:
        Maximum number of simultaneous open trades (default ``3``).
    max_position_pct:
        Maximum fraction of *total_capital* per single trade (default ``0.50``).
    max_sector_pct:
        Maximum fraction of *total_capital* for a single sector (default ``0.50``).
        With 2–3 trades the practical limit is one trade per sector.

    Returns
    -------
    :class:`PortfolioAllocation`
    """
    result = PortfolioAllocation(total_capital=total_capital)

    if suggestions.empty:
        return result

    puts = suggestions[suggestions["option_type"] == "put"].reset_index(drop=True)
    if puts.empty:
        return result

    selected_rows: List[pd.Series] = []
    selected_tickers: List[str] = []
    selected_sectors: List[str] = []

    # Inspect a generous pool to surface useful rejection messages.
    candidate_pool = puts.head(max(max_trades * 5, 15))

    for _, row in candidate_pool.iterrows():
        if len(selected_rows) >= max_trades:
            break

        ticker = str(row["ticker"])
        sector = _get_sector(ticker)

        if ticker in selected_tickers:
            result.rejected.append(
                RejectedCandidate(ticker, float(row["score"]), "duplicate ticker")
            )
            continue

        correlated_with = next(
            (t for t in selected_tickers if _are_correlated(ticker, t)), None
        )
        if correlated_with:
            result.rejected.append(
                RejectedCandidate(
                    ticker,
                    float(row["score"]),
                    f"high correlation with {correlated_with}",
                )
            )
            continue

        if sector in selected_sectors:
            result.rejected.append(
                RejectedCandidate(ticker, float(row["score"]), "duplicate sector exposure")
            )
            continue

        selected_rows.append(row)
        selected_tickers.append(ticker)
        selected_sectors.append(sector)

    if not selected_rows:
        return result

    # ── Capital allocation ────────────────────────────────────────────────────
    scores = [float(r["score"]) for r in selected_rows]
    allocations = _score_weighted_allocation(scores, total_capital, max_position_pct)

    for row, alloc in zip(selected_rows, allocations):
        ticker = str(row["ticker"])
        sector = _get_sector(ticker)
        spread_struct = str(row.get("spread_structure", ""))
        long_strike = _parse_long_strike(spread_struct) or 0.0
        max_profit_per_contract = float(row.get("bid", 0.0)) * 100.0
        max_loss_per_contract = float(row.get("max_spread_loss", 0.0))

        result.selected.append(
            TradeAllocation(
                ticker=ticker,
                sector=sector,
                strategy_type="Bull Put Spread",
                short_strike=float(row["strike"]),
                long_strike=long_strike,
                expiration=str(row.get("expiry", "")),
                score=float(row.get("score", 0.0)),
                max_profit=round(max_profit_per_contract, 2),
                max_loss=round(max_loss_per_contract, 2),
                allocation=round(alloc, 2),
                pct_of_portfolio=round(alloc / total_capital * 100, 1),
            )
        )

    return result


def allocate_tfsa_portfolio(
    suggestions: pd.DataFrame,
    total_capital: float = 1000.0,
    max_trades: int = 2,
    max_position_pct: float = 0.60,
) -> TfsaAllocation:
    """Select 1–2 high-conviction call debit spread trades for a TFSA account.

    TFSA constraints applied here:
    * **Calls only** – no put selling or credit spreads.
    * **Ranked by ``tfsa_score``** (expected upside potential) rather than
      probability of profit.
    * **Defined risk**: max loss per trade is capped at the net debit paid.
    * At most *max_trades* (default 2) simultaneous positions.

    Selection rules (same diversity filters as :func:`allocate_portfolio`):
    1. Calls only, ranked by ``tfsa_score`` descending (falls back to ``score``
       when ``tfsa_score`` is absent).
    2. Skip duplicate tickers.
    3. Skip tickers highly correlated with an already-selected trade.
    4. Skip tickers whose sector is already represented.

    Parameters
    ----------
    suggestions:
        Full ranked suggestions DataFrame from
        :func:`~scanner.suggester.generate_suggestions`.
    total_capital:
        Total dollars to deploy (default ``1000.0``).
    max_trades:
        Maximum number of simultaneous TFSA trades (default ``2``).
    max_position_pct:
        Maximum fraction of *total_capital* per single trade (default ``0.60``).

    Returns
    -------
    :class:`TfsaAllocation`
    """
    result = TfsaAllocation(total_capital=total_capital)

    if suggestions.empty:
        return result

    calls = suggestions[suggestions["option_type"] == "call"].copy()
    if calls.empty:
        return result

    sort_col = "tfsa_score" if "tfsa_score" in calls.columns else "score"
    calls = calls.sort_values(sort_col, ascending=False).reset_index(drop=True)

    selected_rows: List[pd.Series] = []
    selected_tickers: List[str] = []
    selected_sectors: List[str] = []

    candidate_pool = calls.head(max(max_trades * 5, 10))

    for _, row in candidate_pool.iterrows():
        if len(selected_rows) >= max_trades:
            break

        ticker = str(row["ticker"])
        sector = _get_sector(ticker)
        row_score = float(row.get(sort_col, 0.0))

        if ticker in selected_tickers:
            result.rejected.append(
                RejectedCandidate(ticker, row_score, "duplicate ticker")
            )
            continue

        correlated_with = next(
            (t for t in selected_tickers if _are_correlated(ticker, t)), None
        )
        if correlated_with:
            result.rejected.append(
                RejectedCandidate(
                    ticker, row_score, f"high correlation with {correlated_with}"
                )
            )
            continue

        if sector in selected_sectors:
            result.rejected.append(
                RejectedCandidate(ticker, row_score, "duplicate sector exposure")
            )
            continue

        selected_rows.append(row)
        selected_tickers.append(ticker)
        selected_sectors.append(sector)

    if not selected_rows:
        return result

    # ── Capital allocation ────────────────────────────────────────────────────
    scores = [float(r.get(sort_col, 0.0)) for r in selected_rows]
    allocations = _score_weighted_allocation(scores, total_capital, max_position_pct)

    for row, alloc in zip(selected_rows, allocations):
        ticker = str(row["ticker"])
        sector = _get_sector(ticker)

        # Use the pre-computed TFSA spread structure when available
        spread_struct = str(row.get("tfsa_spread", ""))
        if not spread_struct:
            spread_struct = suggest_call_debit_spread(float(row["strike"]))

        buy_strike = float(row["strike"])
        sell_strike = _parse_sell_strike_tfsa(spread_struct) or (
            buy_strike + _spread_width(buy_strike)
        )

        ask = float(row.get("ask", 0.0))
        width = sell_strike - buy_strike
        max_profit_per_contract = max((width - ask) * 100.0, 0.0)
        max_loss_per_contract = ask * 100.0

        result.selected.append(
            TfsaTradeAllocation(
                ticker=ticker,
                sector=sector,
                strategy_type="Call Debit Spread",
                buy_strike=buy_strike,
                sell_strike=sell_strike,
                expiration=str(row.get("expiry", "")),
                tfsa_score=float(row.get(sort_col, 0.0)),
                max_profit=round(max_profit_per_contract, 2),
                max_loss=round(max_loss_per_contract, 2),
                allocation=round(alloc, 2),
                pct_of_portfolio=round(alloc / total_capital * 100, 1),
            )
        )

    return result
