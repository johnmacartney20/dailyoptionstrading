"""Tests for holdings review lifecycle and consolidation policies."""

from scanner.holdings_reviewer import account_health_summary_lines, exited_capital_by_bucket, review_holdings


def test_review_holdings_tier1_two_day_flag_and_immediate_exit(monkeypatch):
    score_map = {
        "CORE": 70.0,
        "TRIM": 63.0,
        "FLAG2": 58.0,
        "NOWEXIT": 42.0,
    }

    monkeypatch.setattr(
        "scanner.holdings_reviewer._score_position",
        lambda pos, market_return_20d: (score_map[str(pos.get("ticker", "")).upper()], "mock"),
    )

    positions = [
        {
            "ticker": "CORE",
            "account_type": "TFSA",
            "sub_portfolio": "growth",
            "entry_price": 100.0,
            "quantity": 80,
            "entry_composite_score": 66.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {},
        },
        {
            "ticker": "TRIM",
            "account_type": "TFSA",
            "sub_portfolio": "growth",
            "entry_price": 100.0,
            "quantity": 70,
            "entry_composite_score": 58.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {},
        },
        {
            "ticker": "FLAG2",
            "account_type": "TFSA",
            "sub_portfolio": "growth",
            "entry_price": 100.0,
            "quantity": 60,
            "entry_composite_score": 50.0,
            "entry_date": "2026-07-01",
            "review_history": [{"date": "2026-07-08", "status": "FLAG", "score": 47.0, "reason": "prior"}],
            "metadata": {},
        },
        {
            "ticker": "NOWEXIT",
            "account_type": "RRSP",
            "sub_portfolio": "stability",
            "entry_price": 100.0,
            "quantity": 50,
            "entry_composite_score": 40.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {},
        },
    ]

    reviews = review_holdings(
        positions,
        thresholds={},
        market_return_20d=0.0,
        account_capitals={"TFSA": 65_000.0, "RRSP": 24_000.0},
    )
    by_ticker = {r.ticker: r for r in reviews}

    assert by_ticker["CORE"].verdict == "HOLD"
    assert by_ticker["CORE"].verdict_tag == "HOLD"
    assert by_ticker["TRIM"].verdict == "HOLD"
    assert by_ticker["FLAG2"].verdict == "EXIT"
    assert by_ticker["FLAG2"].verdict_tag == "EXIT (score)"
    assert "2/2" in by_ticker["FLAG2"].reason
    assert by_ticker["NOWEXIT"].verdict == "EXIT"
    assert by_ticker["NOWEXIT"].verdict_tag == "EXIT (score)"


def test_review_holdings_enforces_options_sleeve_caps_and_cross_account_note(monkeypatch):
    score_map = {
        "S1": 80.0,
        "S2": 78.0,
        "S3": 76.0,
        "S4": 74.0,
        "P1": 75.0,
        "P2": 73.0,
        "P3": 71.0,
        "AMGN": 72.0,
    }

    monkeypatch.setattr(
        "scanner.holdings_reviewer._score_position",
        lambda pos, market_return_20d: (score_map[str(pos.get("ticker", "")).upper()], "mock"),
    )

    positions = []
    for ticker in ["S1", "S2", "S3", "S4"]:
        positions.append(
            {
                "ticker": ticker,
                "account_type": "OPTIONS",
                "sub_portfolio": "growth",
                "entry_price": 10.0,
                "quantity": 100,
                "entry_composite_score": 60.0,
                "entry_date": "2026-07-01",
                "review_history": [],
                "metadata": {},
            }
        )
    for ticker in ["P1", "P2", "P3"]:
        positions.append(
            {
                "ticker": ticker,
                "account_type": "OPTIONS",
                "sub_portfolio": "put-spread",
                "entry_price": 2.0,
                "quantity": 10,
                "entry_composite_score": 60.0,
                "entry_date": "2026-07-01",
                "review_history": [],
                "metadata": {"option_type": "put", "expiry": "2026-08-21", "strike": 100.0},
            }
        )
    positions.extend(
        [
            {
                "ticker": "AMGN",
                "account_type": "TFSA",
                "sub_portfolio": "growth",
                "entry_price": 100.0,
                "quantity": 40,
                "entry_composite_score": 60.0,
                "entry_date": "2026-07-01",
                "review_history": [],
                "metadata": {},
            },
            {
                "ticker": "AMGN",
                "account_type": "FHSA",
                "sub_portfolio": "growth",
                "entry_price": 100.0,
                "quantity": 30,
                "entry_composite_score": 60.0,
                "entry_date": "2026-07-01",
                "review_history": [],
                "metadata": {},
            },
        ]
    )

    reviews = review_holdings(
        positions,
        thresholds={},
        market_return_20d=0.0,
        account_capitals={"OPTIONS": 20_000.0, "TFSA": 65_000.0, "FHSA": 36_000.0},
    )

    options_stock = [r for r in reviews if r.account_type == "OPTIONS" and r.sub_portfolio == "growth"]
    options_spreads = [r for r in reviews if r.account_type == "OPTIONS" and r.sub_portfolio == "put-spread"]

    assert sum(1 for r in options_stock if r.verdict == "EXIT" and r.verdict_tag == "EXIT (cap)") == 2
    assert sum(1 for r in options_spreads if r.verdict == "EXIT" and r.verdict_tag == "EXIT (cap)") == 1

    amgn_rows = [r for r in reviews if r.ticker == "AMGN"]
    assert len(amgn_rows) == 2
    assert all("cross-account exposure AMGN" in r.reason for r in amgn_rows)

    freed = exited_capital_by_bucket(reviews)
    assert freed["OPTIONS"] > 0
    assert freed["OPTIONS_stock"] > 0
    assert freed["OPTIONS_spreads"] > 0

    lines = account_health_summary_lines(reviews, account_capitals={"OPTIONS": 20_000.0, "TFSA": 65_000.0, "FHSA": 36_000.0, "RRSP": 24_000.0})
    assert any(line.startswith("TFSA: $65,000") for line in lines)
    assert any(line.startswith("OPTIONS: $20,000") for line in lines)


def test_review_holdings_enforces_options_5050_capital_split(monkeypatch):
    score_map = {
        "P1": 80.0,
        "P2": 70.0,
        "G1": 75.0,
        "G2": 72.0,
    }

    monkeypatch.setattr(
        "scanner.holdings_reviewer._score_position",
        lambda pos, market_return_20d: (score_map[str(pos.get("ticker", "")).upper()], "mock"),
    )

    positions = [
        {
            "ticker": "P1",
            "account_type": "OPTIONS",
            "sub_portfolio": "put-spread",
            "entry_price": 1.0,
            "quantity": 1,
            "entry_composite_score": 70.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {"cad_equiv": 7000.0},
        },
        {
            "ticker": "P2",
            "account_type": "OPTIONS",
            "sub_portfolio": "put-spread",
            "entry_price": 1.0,
            "quantity": 1,
            "entry_composite_score": 65.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {"cad_equiv": 6000.0},
        },
        {
            "ticker": "G1",
            "account_type": "OPTIONS",
            "sub_portfolio": "growth",
            "entry_price": 1.0,
            "quantity": 1,
            "entry_composite_score": 65.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {"cad_equiv": 4000.0},
        },
        {
            "ticker": "G2",
            "account_type": "OPTIONS",
            "sub_portfolio": "growth",
            "entry_price": 1.0,
            "quantity": 1,
            "entry_composite_score": 64.0,
            "entry_date": "2026-07-01",
            "review_history": [],
            "metadata": {"cad_equiv": 3000.0},
        },
    ]

    reviews = review_holdings(
        positions,
        thresholds={},
        market_return_20d=0.0,
        account_capitals={"OPTIONS": 20_000.0},
    )
    by_ticker = {r.ticker: r for r in reviews}

    # Put-spread sleeve totals 13k > 10k, so the lower-score spread should be trimmed.
    assert by_ticker["P2"].verdict == "EXIT"
    assert by_ticker["P2"].verdict_tag == "EXIT (cap)"
    assert by_ticker["P1"].verdict == "HOLD"
