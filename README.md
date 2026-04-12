# dailyoptionstrading

Scans public **free** data sources (Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)) for daily options-trading suggestions on the **TSX** (Toronto Stock Exchange) and **NASDAQ**.

## Features

* Screens for **covered-call** and **cash-secured-put** opportunities.
* Filters by days-to-expiration, bid price, open interest, OTM%, and minimum annualised return.
* Ranks candidates with a composite score (annualised return + liquidity bonus).
* Displays a formatted summary table; optionally exports full results to CSV.
* Configurable ticker lists and screening parameters in `scanner/config.py`.

## Quick Start

```bash
# 1 – Install dependencies
pip install -r requirements.txt

# 2 – Scan all exchanges (TSX + NASDAQ) and show top 10 suggestions per strategy
python main.py

# 3 – NASDAQ only, top 5
python main.py --exchange nasdaq --top 5

# 4 – TSX, cash-secured puts only, save to CSV
python main.py --exchange tsx --strategy put --output tsx_puts.csv
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--exchange` | `all` | `tsx`, `nasdaq`, or `all` |
| `--strategy` | `all` | `call`, `put`, or `all` |
| `--top` | `10` | Number of suggestions per strategy to print |
| `--output FILE` | – | Save full results to a CSV file |

## Project Layout

```
dailyoptionstrading/
├── main.py                  # CLI entry point
├── requirements.txt
├── scanner/
│   ├── config.py            # Ticker lists & screening parameters
│   ├── data_fetcher.py      # Yahoo Finance data retrieval
│   ├── analyzer.py          # DTE, OTM%, annualised return, scoring
│   └── suggester.py         # Screening filters & suggestion ranking
└── tests/
    ├── test_analyzer.py
    └── test_suggester.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Screening Logic

| Filter | Default |
|--------|---------|
| Days to expiry | 7 – 60 days |
| Min open interest | 10 contracts |
| Min bid | $0.05 |
| Min annualised return | 10 % |
| Max OTM | 10 % |

Options that pass all filters are ranked by a **composite score**:

```
score = annualised_return  +  min(log10(open_interest + 1) × 5, 20)
```

The liquidity bonus (capped at +20 pts) ensures that highly-liquid options rank
above equally-priced illiquid ones.

## Data Sources

| Source | API | Cost |
|--------|-----|------|
| Yahoo Finance | `yfinance` | Free |

> **Disclaimer:** The suggestions produced by this tool are **not** financial
> advice.  Always do your own research and consult a licensed financial adviser
> before trading options.

