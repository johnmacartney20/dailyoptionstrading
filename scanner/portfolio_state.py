"""Persistent portfolio state helpers.

This module manages the JSON file that stores live portfolio positions across
runs so the model can behave like a daily portfolio manager instead of a
stateless idea generator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


STATUS_HOLD = "HOLD"
STATUS_FLAG = "FLAG"
STATUS_EXIT = "EXIT"
_VALID_STATUSES = {STATUS_HOLD, STATUS_FLAG, STATUS_EXIT}


@dataclass
class PositionIdentity:
    """Natural key used to identify positions for updates."""

    ticker: str
    account_type: str
    sub_portfolio: str


def _today_str() -> str:
    return date.today().isoformat()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _empty_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "generated_at": _now_iso(),
        "positions": [],
        "closed_positions": [],
    }


def _new_position(
    ticker: str,
    account_type: str,
    sub_portfolio: str,
    entry_date: str,
    entry_price: float,
    quantity: int,
    entry_composite_score: float,
    entry_thesis_tags: Sequence[str],
    status: str = STATUS_HOLD,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status_norm = status.upper()
    if status_norm not in _VALID_STATUSES:
        raise ValueError(f"Unsupported status: {status}")

    return {
        "ticker": _normalize_ticker(ticker),
        "account_type": str(account_type),
        "sub_portfolio": str(sub_portfolio),
        "entry_date": str(entry_date),
        "entry_price": float(entry_price),
        "quantity": int(max(quantity, 1)),
        "entry_composite_score": float(entry_composite_score),
        "entry_thesis_tags": [str(t) for t in list(entry_thesis_tags)[:3]],
        "status": status_norm,
        "last_review_date": None,
        "last_review_score": None,
        "last_review_reason": "",
        "review_history": [],
        "metadata": metadata or {},
    }


def _position_identity(pos: Dict[str, Any]) -> PositionIdentity:
    return PositionIdentity(
        ticker=_normalize_ticker(pos.get("ticker", "")),
        account_type=str(pos.get("account_type", "")),
        sub_portfolio=str(pos.get("sub_portfolio", "")),
    )


def _position_sort_key(pos: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(pos.get("account_type", "")),
        str(pos.get("sub_portfolio", "")),
        _normalize_ticker(pos.get("ticker", "")),
        str(pos.get("entry_date", "")),
    )


def load_portfolio_state(state_path: str) -> Dict[str, Any]:
    """Load state JSON from disk; returns an empty state when missing."""
    path = Path(state_path)
    if not path.exists():
        return _empty_state()

    try:
        state = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse portfolio state %s: %s", path, exc)
        return _empty_state()

    if not isinstance(state, dict):
        return _empty_state()

    state.setdefault("version", 1)
    state.setdefault("generated_at", _now_iso())
    state.setdefault("positions", [])
    state.setdefault("closed_positions", [])
    return state


def save_portfolio_state(state: Dict[str, Any], state_path: str) -> None:
    """Write state JSON in a stable, human-readable format."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    positions = list(state.get("positions", []))
    closed = list(state.get("closed_positions", []))
    state["positions"] = sorted(positions, key=_position_sort_key)
    state["closed_positions"] = sorted(closed, key=_position_sort_key)
    state["generated_at"] = _now_iso()

    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def migrate_from_legacy_holdings(
    state_path: str,
    rrsp_holdings: Sequence[str],
    tfsa_holdings: Sequence[str],
) -> Dict[str, Any]:
    """Seed initial state from legacy config holdings lists."""
    state = _empty_state()

    for ticker in rrsp_holdings:
        if not str(ticker).strip():
            continue
        state["positions"].append(
            _new_position(
                ticker=ticker,
                account_type="RRSP",
                sub_portfolio="stability",
                entry_date=_today_str(),
                entry_price=0.0,
                quantity=1,
                entry_composite_score=0.0,
                entry_thesis_tags=["legacy-seed", "manual-holding", "rrsp"],
                status=STATUS_HOLD,
            )
        )

    for ticker in tfsa_holdings:
        if not str(ticker).strip():
            continue
        state["positions"].append(
            _new_position(
                ticker=ticker,
                account_type="TFSA",
                sub_portfolio="growth",
                entry_date=_today_str(),
                entry_price=0.0,
                quantity=1,
                entry_composite_score=0.0,
                entry_thesis_tags=["legacy-seed", "manual-holding", "tfsa"],
                status=STATUS_HOLD,
            )
        )

    # Remove duplicates by natural key while preserving first seen record.
    seen: set[Tuple[str, str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for pos in state["positions"]:
        ident = _position_identity(pos)
        key = (ident.ticker, ident.account_type, ident.sub_portfolio)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pos)
    state["positions"] = deduped

    save_portfolio_state(state, state_path)
    return state


def load_or_initialize_state(
    state_path: str,
    rrsp_holdings: Sequence[str],
    tfsa_holdings: Sequence[str],
) -> Tuple[Dict[str, Any], bool]:
    """Load state if present, otherwise seed it from legacy lists.

    Returns ``(state, seeded)`` where ``seeded`` is True only on first run.
    """
    path = Path(state_path)
    if path.exists():
        return load_portfolio_state(state_path), False
    return migrate_from_legacy_holdings(state_path, rrsp_holdings, tfsa_holdings), True


def get_positions(
    state: Dict[str, Any],
    account_type: Optional[str] = None,
    sub_portfolio: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return filtered active positions from state."""
    status_set = {s.upper() for s in statuses} if statuses else None
    out: List[Dict[str, Any]] = []
    for pos in state.get("positions", []):
        if account_type and str(pos.get("account_type", "")).upper() != account_type.upper():
            continue
        if sub_portfolio and str(pos.get("sub_portfolio", "")).lower() != sub_portfolio.lower():
            continue
        if status_set and str(pos.get("status", "")).upper() not in status_set:
            continue
        out.append(pos)
    return out


def get_holding_tickers(
    state: Dict[str, Any],
    account_type: Optional[str] = None,
    sub_portfolio: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return unique tickers for filtered positions."""
    seen: set[str] = set()
    tickers: List[str] = []
    for pos in get_positions(
        state,
        account_type=account_type,
        sub_portfolio=sub_portfolio,
        statuses=statuses,
    ):
        ticker = _normalize_ticker(pos.get("ticker", ""))
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def add_or_update_position(state: Dict[str, Any], position: Dict[str, Any]) -> None:
    """Upsert an active position by (ticker, account_type, sub_portfolio)."""
    ident = _position_identity(position)
    for idx, existing in enumerate(state.get("positions", [])):
        cur = _position_identity(existing)
        if (
            cur.ticker == ident.ticker
            and cur.account_type.upper() == ident.account_type.upper()
            and cur.sub_portfolio.lower() == ident.sub_portfolio.lower()
        ):
            state["positions"][idx] = position
            return
    state.setdefault("positions", []).append(position)


def record_review(
    position: Dict[str, Any],
    review_date: str,
    current_score: float,
    verdict: str,
    reason: str,
) -> None:
    """Append one review snapshot and update current status fields."""
    verdict_norm = verdict.upper()
    if verdict_norm not in _VALID_STATUSES:
        verdict_norm = STATUS_FLAG

    snapshot = {
        "date": str(review_date),
        "score": float(current_score),
        "status": verdict_norm,
        "reason": str(reason),
    }
    position.setdefault("review_history", []).append(snapshot)
    position["last_review_date"] = str(review_date)
    position["last_review_score"] = float(current_score)
    position["last_review_reason"] = str(reason)
    position["status"] = verdict_norm

    if verdict_norm == STATUS_EXIT:
        position["exit_date"] = str(review_date)
        position["exit_score"] = float(current_score)


def move_exited_positions(state: Dict[str, Any]) -> None:
    """Move EXIT positions from active list to closed_positions list."""
    still_open: List[Dict[str, Any]] = []
    for pos in state.get("positions", []):
        if str(pos.get("status", "")).upper() == STATUS_EXIT:
            state.setdefault("closed_positions", []).append(pos)
        else:
            still_open.append(pos)
    state["positions"] = still_open


def build_position(
    ticker: str,
    account_type: str,
    sub_portfolio: str,
    entry_price: float,
    quantity: int,
    entry_composite_score: float,
    entry_thesis_tags: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a new active HOLD position using today's date."""
    return _new_position(
        ticker=ticker,
        account_type=account_type,
        sub_portfolio=sub_portfolio,
        entry_date=_today_str(),
        entry_price=entry_price,
        quantity=quantity,
        entry_composite_score=entry_composite_score,
        entry_thesis_tags=entry_thesis_tags,
        status=STATUS_HOLD,
        metadata=metadata,
    )


def portfolio_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return high-level counts for reporting."""
    active = state.get("positions", [])
    by_account: Dict[str, int] = {}
    by_status: Dict[str, int] = {STATUS_HOLD: 0, STATUS_FLAG: 0, STATUS_EXIT: 0}

    for pos in active:
        account = str(pos.get("account_type", "unknown"))
        by_account[account] = by_account.get(account, 0) + 1
        status = str(pos.get("status", STATUS_HOLD)).upper()
        if status in by_status:
            by_status[status] += 1

    return {
        "total_positions": len(active),
        "by_account": by_account,
        "by_status": by_status,
    }
