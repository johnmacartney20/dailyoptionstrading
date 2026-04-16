"""Portfolio allocation module.

Selects the top 2–3 put-spread trades from the screened suggestions and
allocates a fixed capital budget ($1,000 by default) across them, respecting
per-trade and per-sector concentration limits and inter-trade correlation
exclusions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .analyzer import (
    StockScoreComponents,
    score_stock_growth,
    score_stock_stability,
)

logger = logging.getLogger(__name__)

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

# Maximum allowed DTE spread (in calendar days) between any two selected trades.
# Keeps all positions near the same expiry cycle (approximately 7–10 days window).
_MAX_DTE_SPREAD: int = 10

# Target DTE window for TFSA long calls (approximately 30 days out).
# Expiries outside this window are excluded before candidate selection.
_TFSA_LONG_CALL_MIN_DTE: int = 21
_TFSA_LONG_CALL_MAX_DTE: int = 42

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
    """Details for a single accepted TFSA trade (long call – single-leg)."""

    ticker: str
    sector: str
    strategy_type: str      # "Long Call"
    buy_strike: float       # strike of the call we buy
    sell_strike: float      # 0.0 – unused for single-leg long calls
    expiration: str
    tfsa_score: float
    max_profit: float       # 0.0 – unlimited upside for a long call
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
    """Extract the sell-leg strike from a spread structure string.

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


def _tfsa_concentrated_allocation(
    n: int,
    total_capital: float,
    max_single_pct: float = 0.50,
) -> List[float]:
    """Return tiered concentrated capital allocations for TFSA stock positions.

    Applies strict conviction-based concentration (higher rank = more capital):

    * Rank 1 (highest composite score): targets 40–50 % → midpoint **45 %**
    * Rank 2: targets 30–35 % → midpoint **32 %**
    * Rank 3: receives the **remaining** allocation → ~23 %

    For fewer than 3 selected positions the tier weights are normalised to
    sum to 100 % of *total_capital*.  A single selected position is
    additionally capped at *max_single_pct* (default 50 %) to avoid
    over-concentration in a single name.
    """
    _TIER_WEIGHTS = [0.45, 0.32, 0.23]
    if n == 0:
        return []
    weights = _TIER_WEIGHTS[:n]
    total_weight = sum(weights)
    allocs = [total_capital * w / total_weight for w in weights]
    if n == 1:
        allocs[0] = min(allocs[0], total_capital * max_single_pct)
    return allocs


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
    reference_dte: Optional[int] = None

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

        candidate_dte = row.get("dte")
        if reference_dte is not None and candidate_dte is not None:
            if abs(int(candidate_dte) - reference_dte) > _MAX_DTE_SPREAD:
                result.rejected.append(
                    RejectedCandidate(
                        ticker,
                        float(row["score"]),
                        f"DTE mismatch ({int(candidate_dte)}d vs reference {reference_dte}d)",
                    )
                )
                continue

        selected_rows.append(row)
        selected_tickers.append(ticker)
        selected_sectors.append(sector)
        if reference_dte is None and candidate_dte is not None:
            reference_dte = int(candidate_dte)

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
    """Select 1–2 high-conviction long call trades for a TFSA account.

    TFSA constraints applied here:
    * **Single-leg calls only** – no spreads, no short premium, no multi-leg
      strategies.  Brokers often do not permit short options inside registered
      accounts; a plain long call is universally supported.
    * **Ranked by ``tfsa_score``** (expected upside potential) rather than
      probability of profit.
    * **Defined risk**: max loss per trade is strictly the net premium paid.
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

    # ── DTE pre-filter: prefer options close to 30 days to expiry ─────────────
    if "dte" in calls.columns:
        dte_filtered = calls[
            (calls["dte"] >= _TFSA_LONG_CALL_MIN_DTE)
            & (calls["dte"] <= _TFSA_LONG_CALL_MAX_DTE)
        ]
        # Fall back to the full call list only if the filter yields nothing.
        if not dte_filtered.empty:
            calls = dte_filtered.reset_index(drop=True)
        else:
            logger.debug(
                "No TFSA long calls with DTE in [%d, %d]; using full range.",
                _TFSA_LONG_CALL_MIN_DTE,
                _TFSA_LONG_CALL_MAX_DTE,
            )

    selected_rows: List[pd.Series] = []
    selected_tickers: List[str] = []
    selected_sectors: List[str] = []
    reference_dte: Optional[int] = None

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

        candidate_dte = row.get("dte")
        if reference_dte is not None and candidate_dte is not None:
            if abs(int(candidate_dte) - reference_dte) > _MAX_DTE_SPREAD:
                result.rejected.append(
                    RejectedCandidate(
                        ticker,
                        row_score,
                        f"DTE mismatch ({int(candidate_dte)}d vs reference {reference_dte}d)",
                    )
                )
                continue

        selected_rows.append(row)
        selected_tickers.append(ticker)
        selected_sectors.append(sector)
        if reference_dte is None and candidate_dte is not None:
            reference_dte = int(candidate_dte)

    if not selected_rows:
        return result

    # ── Capital allocation ────────────────────────────────────────────────────
    scores = [float(r.get(sort_col, 0.0)) for r in selected_rows]
    allocations = _score_weighted_allocation(scores, total_capital, max_position_pct)

    for row, alloc in zip(selected_rows, allocations):
        ticker = str(row["ticker"])
        sector = _get_sector(ticker)

        ask = float(row.get("ask", 0.0))
        buy_strike = float(row["strike"])
        # Single-leg long call: max loss is the premium paid; upside is unlimited.
        max_loss_per_contract = ask * 100.0

        result.selected.append(
            TfsaTradeAllocation(
                ticker=ticker,
                sector=sector,
                strategy_type="Long Call",
                buy_strike=buy_strike,
                sell_strike=0.0,          # not applicable for single-leg
                expiration=str(row.get("expiry", "")),
                tfsa_score=float(row.get(sort_col, 0.0)),
                max_profit=0.0,           # unlimited upside for a long call
                max_loss=round(max_loss_per_contract, 2),
                allocation=round(alloc, 2),
                pct_of_portfolio=round(alloc / total_capital * 100, 1),
            )
        )

    return result


# ── TFSA stock allocation (growth model) ─────────────────────────────────────

@dataclass
class StockAllocation:
    """A single stock position selected for the TFSA growth portfolio."""

    ticker: str
    sector: str
    current_price: float
    composite_score: float
    allocation: float        # dollar amount allocated
    pct_of_portfolio: float  # 0–100
    reasoning: str           # key scoring factors in plain language


@dataclass
class TfsaStockPortfolio:
    """TFSA stock portfolio allocation result (growth focus)."""

    total_capital: float
    selected: List[StockAllocation] = field(default_factory=list)
    rejected: List[RejectedCandidate] = field(default_factory=list)

    @property
    def total_deployed(self) -> float:
        return round(sum(t.allocation for t in self.selected), 2)

    @property
    def num_positions(self) -> int:
        return len(self.selected)

    @property
    def exit_guidance(self) -> str:
        if not self.selected:
            return ""
        tickers_str = ", ".join(t.ticker for t in self.selected)
        return (
            f"Exit guidance for {tickers_str}: "
            "take partial profits at +15\u201325%; "
            "allow remaining position to run up to +40\u201360% if trend remains intact; "
            "exit early if price breaks below the 20-day moving average."
        )


def allocate_tfsa_stock_portfolio(
    price_histories: Dict[str, Optional[pd.DataFrame]],
    total_capital: float = 1000.0,
    max_positions: int = 3,
    max_position_pct: float = 0.50,
    max_sector_pct: float = 0.50,
    market_return_20d: float = 0.0,
) -> TfsaStockPortfolio:
    """Select up to *max_positions* high-conviction stocks for TFSA growth.

    Applies **strict selection pressure**: all candidates are ranked by composite
    score and only the top *max_positions* (default 3) qualify.  Every
    non-selected candidate is recorded in :attr:`TfsaStockPortfolio.rejected`
    with an explicit reason.

    Composite scoring (higher = better, max ≈ 100 pts):

    * **Trend Strength** (0–30 pts): price above 20/50-day MAs + momentum
    * **Relative Strength** (0–20 pts): 20-day return vs broader market
    * **Volatility Control** (0–15 pts): penalises excessively volatile stocks
    * **Liquidity** (0–15 pts): log-scaled 20-day average daily volume
    * **Drawdown Risk** (0–20 pts): penalises stocks extended above support

    Capital is allocated with conviction-based concentration via
    :func:`_tfsa_concentrated_allocation`:

    * Rank 1 (highest score): 40–50 % → target 45 %
    * Rank 2: 30–35 % → target 32 %
    * Rank 3: remaining allocation → ~23 %

    Portfolio constraints:
    - At most *max_positions* stocks (default 3)
    - At most 1 stock per sector
    - At most one stock from each correlation group

    Parameters
    ----------
    price_histories:
        Mapping of ticker → OHLCV DataFrame (from ``get_price_history``).
        ``None`` values are silently skipped.
    total_capital:
        Total dollars to deploy.
    max_positions:
        Maximum number of stock positions (default ``3``).
    max_position_pct:
        Maximum fraction of *total_capital* for a single position when only
        one position is selected (default ``0.50``).
    max_sector_pct:
        Maximum fraction of *total_capital* for a single sector (default
        ``0.50``).  With ≤ 3 positions the practical rule is one per sector.
    market_return_20d:
        Trailing 20-day benchmark return used for relative-strength scoring.

    Returns
    -------
    :class:`TfsaStockPortfolio`
    """
    result = TfsaStockPortfolio(total_capital=total_capital)
    if not price_histories:
        return result

    # Score all candidates
    candidates: List[tuple] = []
    for ticker, hist in price_histories.items():
        if hist is None or hist.empty:
            continue
        score = score_stock_growth(hist, market_return_20d)
        if score.composite > 0:
            price = float(hist["Close"].iloc[-1])
            candidates.append((ticker, price, score))

    # Sort by composite score descending
    candidates.sort(key=lambda x: x[2].composite, reverse=True)

    selected_tickers: List[str] = []
    selected_sectors: List[str] = []
    selected_items: List[tuple] = []

    # Inspect a generous pool to surface useful rejection messages
    candidate_pool = candidates[: max(max_positions * 5, 15)]

    for ticker, price, score in candidate_pool:
        if len(selected_items) >= max_positions:
            # Position limit reached – reject all remaining pool candidates explicitly.
            result.rejected.append(
                RejectedCandidate(
                    ticker,
                    score.composite,
                    f"lower composite score – outside top-{max_positions} selection",
                )
            )
            continue

        sector = _get_sector(ticker)

        if ticker in selected_tickers:
            result.rejected.append(
                RejectedCandidate(ticker, score.composite, "duplicate ticker")
            )
            continue

        correlated_with = next(
            (t for t in selected_tickers if _are_correlated(ticker, t)), None
        )
        if correlated_with:
            result.rejected.append(
                RejectedCandidate(
                    ticker, score.composite, f"high correlation with {correlated_with}"
                )
            )
            continue

        if sector in selected_sectors:
            result.rejected.append(
                RejectedCandidate(ticker, score.composite, "duplicate sector exposure")
            )
            continue

        selected_tickers.append(ticker)
        selected_sectors.append(sector)
        selected_items.append((ticker, price, score, sector))

    # Candidates beyond the inspected pool are also rejected (lower score)
    for ticker, price, score in candidates[len(candidate_pool):]:
        result.rejected.append(
            RejectedCandidate(
                ticker,
                score.composite,
                f"lower composite score – outside top-{max_positions} selection",
            )
        )

    if not selected_items:
        return result

    # ── Capital allocation (conviction-based tiers) ───────────────────────────
    allocations = _tfsa_concentrated_allocation(
        len(selected_items), total_capital, max_single_pct=max_position_pct
    )

    for (ticker, price, score, sector), alloc in zip(selected_items, allocations):
        result.selected.append(
            StockAllocation(
                ticker=ticker,
                sector=sector,
                current_price=round(price, 2),
                composite_score=round(score.composite, 2),
                allocation=round(alloc, 2),
                pct_of_portfolio=round(alloc / total_capital * 100, 1),
                reasoning=score.reasoning,
            )
        )

    return result


# ── RRSP allocation (stability model) ─────────────────────────────────────────

@dataclass
class RrspStockAllocation:
    """A single RRSP stock or ETF holding (stability focus)."""

    ticker: str
    sector: str
    current_price: float
    composite_score: float
    allocation: float        # dollar amount allocated
    pct_of_portfolio: float  # 0–100
    long_term_thesis: str    # plain-language long-term reasoning


@dataclass
class RrspPortfolio:
    """RRSP portfolio allocation result (stability focus)."""

    total_capital: float
    selected: List[RrspStockAllocation] = field(default_factory=list)
    rejected: List[RejectedCandidate] = field(default_factory=list)

    @property
    def total_deployed(self) -> float:
        return round(sum(t.allocation for t in self.selected), 2)

    @property
    def num_positions(self) -> int:
        return len(self.selected)


def allocate_rrsp_portfolio(
    price_histories: Dict[str, Optional[pd.DataFrame]],
    total_capital: float = 1000.0,
    max_positions: int = 4,
    max_position_pct: float = 0.50,
) -> RrspPortfolio:
    """Select up to *max_positions* positions for RRSP stability allocation.

    Applies **strict selection pressure**: all candidates are ranked by
    stability score and only the top *max_positions* (default 4) that also
    meet the diversity constraints are accepted.  Every non-selected candidate
    is recorded in :attr:`RrspPortfolio.rejected` with an explicit reason.

    Prioritises large-cap stocks and ETFs using a stability-focused score:

    * **Consistency** (0–30 pts): steady upward trend above long-term MAs
    * **Low Volatility** (0–30 pts): directly rewards low annualised vol
    * **Liquidity** (0–20 pts): high average daily volume
    * **Trend Protection** (0–20 pts): not in steep drawdown

    Each selected position includes a ``long_term_thesis`` that serves as the
    required justification for inclusion.  Portfolio constraints are the same
    as :func:`allocate_tfsa_stock_portfolio` (no duplicate sectors, no
    correlated tickers, score-weighted capital).

    Parameters
    ----------
    price_histories:
        Mapping of ticker → OHLCV DataFrame (from ``get_price_history``).
    total_capital:
        Total dollars to deploy.
    max_positions:
        Maximum number of holdings (default ``4``).
    max_position_pct:
        Maximum fraction of *total_capital* per position (default ``0.50``).

    Returns
    -------
    :class:`RrspPortfolio`
    """
    result = RrspPortfolio(total_capital=total_capital)
    if not price_histories:
        return result

    candidates: List[tuple] = []
    for ticker, hist in price_histories.items():
        if hist is None or hist.empty:
            continue
        score = score_stock_stability(hist)
        if score.composite > 0:
            price = float(hist["Close"].iloc[-1])
            candidates.append((ticker, price, score))

    candidates.sort(key=lambda x: x[2].composite, reverse=True)

    selected_tickers: List[str] = []
    selected_sectors: List[str] = []
    selected_items: List[tuple] = []

    candidate_pool = candidates[: max(max_positions * 5, 15)]

    for ticker, price, score in candidate_pool:
        if len(selected_items) >= max_positions:
            # Position limit reached – reject remaining pool candidates explicitly.
            result.rejected.append(
                RejectedCandidate(
                    ticker,
                    score.composite,
                    f"lower stability score – outside top-{max_positions} selection",
                )
            )
            continue

        sector = _get_sector(ticker)

        if ticker in selected_tickers:
            result.rejected.append(
                RejectedCandidate(ticker, score.composite, "duplicate ticker")
            )
            continue

        correlated_with = next(
            (t for t in selected_tickers if _are_correlated(ticker, t)), None
        )
        if correlated_with:
            result.rejected.append(
                RejectedCandidate(
                    ticker, score.composite, f"high correlation with {correlated_with}"
                )
            )
            continue

        if sector in selected_sectors:
            result.rejected.append(
                RejectedCandidate(ticker, score.composite, "duplicate sector exposure")
            )
            continue

        selected_tickers.append(ticker)
        selected_sectors.append(sector)
        selected_items.append((ticker, price, score, sector))

    # Candidates beyond the inspected pool are also rejected (lower score)
    for ticker, price, score in candidates[len(candidate_pool):]:
        result.rejected.append(
            RejectedCandidate(
                ticker,
                score.composite,
                f"lower stability score – outside top-{max_positions} selection",
            )
        )

    if not selected_items:
        return result

    scores = [item[2].composite for item in selected_items]
    allocations = _score_weighted_allocation(scores, total_capital, max_position_pct)

    for (ticker, price, score, sector), alloc in zip(selected_items, allocations):
        result.selected.append(
            RrspStockAllocation(
                ticker=ticker,
                sector=sector,
                current_price=round(price, 2),
                composite_score=round(score.composite, 2),
                allocation=round(alloc, 2),
                pct_of_portfolio=round(alloc / total_capital * 100, 1),
                long_term_thesis=score.reasoning,
            )
        )

    return result
