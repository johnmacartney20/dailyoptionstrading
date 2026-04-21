# dailyoptionstrading

Scans public **free** data sources (Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)) for daily options-trading suggestions on the **TSX** (Toronto Stock Exchange) and **NASDAQ**.

## Features

* Screens for **covered-call** and **cash-secured-put** opportunities.
* Filters by days-to-expiration, bid price, open interest, OTM%, and minimum annualised return.
* Ranks candidates with a composite score (annualised return + liquidity bonus).
* Displays a formatted summary table; optionally exports full results to CSV.
* **Emails an HTML summary** to any address via Gmail (or any SMTP relay).
* **Runs automatically every weekday** via a GitHub Actions scheduled workflow.
* Computes **TFSA (growth)** and **RRSP (stability)** stock allocations for longer-term accounts.
* Optional **monthly TFSA + RRSP portfolio review email** via GitHub Actions.
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

# 5 – Scan and email results to yourself
SMTP_USER=you@gmail.com SMTP_PASSWORD="your-app-password" \
  python main.py --email you@gmail.com
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--exchange` | `all` | `tsx`, `nasdaq`, or `all` |
| `--strategy` | `all` | `call`, `put`, or `all` |
| `--top` | `10` | Number of suggestions per strategy to print/email |
| `--output FILE` | – | Save full results to a CSV file |
| `--email ADDRESS` | – | Send HTML summary to this address (comma-separate for multiple) |
| `--smtp-host` | `smtp.gmail.com` | SMTP server hostname |
| `--smtp-port` | `587` | SMTP server port (STARTTLS) |
| `--smtp-user` | `$SMTP_USER` | Sender email address |
| `--smtp-password` | `$SMTP_PASSWORD` | SMTP / App Password |
| `--monthly-email ADDRESS` | – | Send a separate monthly TFSA + RRSP portfolio review email to this address |

### Optional: risk controls & run logging

These flags are **optional**. If you don't use them, the scanner behaves exactly as before.

| Flag | Default | Description |
|------|---------|-------------|
| `--account-cash` | – | Account cash used to compute `max_contracts` per suggestion |
| `--max-notional-per-trade` | – | Notional cap per trade idea (per row); used to compute `max_contracts` |
| `--max-total-notional` | – | Greedily selects a subset of top-ranked ideas to fit within a total notional budget |
| `--max-trades-per-ticker` | – | Limits ideas per ticker (only applied with `--max-total-notional`) |
| `--run-log-dir` | `runs` | Directory to write run logs (CSV + JSON metadata) |
| `--run-log` | `false` | Enable run logging |

### Optional: monthly review tuning

These flags only affect the **monthly** TFSA + RRSP review email (triggered via `--monthly-email`).

| Flag | Default | Description |
|------|---------|-------------|
| `--monthly-tfsa-capital` | `25000` | TFSA capital used for monthly allocation |
| `--monthly-rrsp-capital` | `25000` | RRSP capital used for monthly allocation |
| `--monthly-tfsa-max-positions` | `5` | Max TFSA stock positions in the monthly review |
| `--monthly-tfsa-max-position-pct` | `0.35` | Max per-position fraction (TFSA monthly) |
| `--monthly-tfsa-max-sector-pct` | `0.40` | Max sector fraction (TFSA monthly) |
| `--monthly-rrsp-max-positions` | `6` | Max RRSP positions in the monthly review |
| `--monthly-rrsp-max-position-pct` | `0.30` | Max per-position fraction (RRSP monthly) |

## Automated Daily Email via GitHub Actions

The workflow at `.github/workflows/daily_scan.yml` runs the scanner automatically
every weekday at **9:30 AM Eastern Time** (13:30 UTC) and emails you the results.

### One-time setup (5 minutes)

**Step 1 – Create a Gmail App Password**

> You need this because Gmail blocks plain passwords for scripts.
> Your regular Gmail password will *not* work.

1. Go to **myaccount.google.com/security** and enable **2-Step Verification** if not already on.
2. Go to **myaccount.google.com/apppasswords**.
3. Choose app = *Mail*, device = *Other* → type `dailyoptionstrading` → click **Generate**.
4. Copy the 16-character password shown — you won't see it again.

**Step 2 – Add three GitHub repository secrets**

Go to your repo on GitHub → **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret name | Value |
|---|---|
| `SMTP_USER` | your Gmail address, e.g. `yourname@gmail.com` |
| `SMTP_PASSWORD` | the 16-character App Password from Step 1 |
| `EMAIL_TO` | the address to send results to (can be the same as `SMTP_USER`) |

**Step 3 – Enable Actions (if disabled)**

Go to your repo → **Actions** tab → click **"I understand my workflows, go ahead and enable them"** if prompted.

**That's it.** The scanner will run automatically on the next weekday morning.

## Automated Monthly TFSA + RRSP Email

The workflow at `.github/workflows/monthly_portfolio_email.yml` runs the monthly portfolio review
on a schedule and emails you the TFSA + RRSP allocation summary.

### Running manually

You can also trigger the workflow on demand:

1. Go to **Actions → Daily Options Scan → Run workflow**.
2. Optionally override the exchange, strategy, or top-N values.
3. Click **Run workflow**.

### Changing the schedule

Edit the `cron` line in `.github/workflows/daily_scan.yml`:

```yaml
# Current: 13:30 UTC = 9:30 AM ET
- cron: "30 13 * * 1-5"

# Examples:
# "30 14 * * 1-5"  = 10:30 AM ET
# "0  21 * * 1-5"  = 5:00 PM ET (after-hours recap)
# "0  14 * * 1-5"  = 10:00 AM ET
```

## Project Layout

```
dailyoptionstrading/
├── .github/workflows/daily_scan.yml  # Scheduled email workflow
├── main.py                           # CLI entry point
├── requirements.txt
├── scanner/
│   ├── config.py      # Ticker lists & screening parameters
│   ├── data_fetcher.py # Yahoo Finance data retrieval
│   ├── analyzer.py    # DTE, OTM%, annualised return, scoring
│   ├── emailer.py     # HTML email builder and SMTP sender
│   └── suggester.py   # Screening filters & suggestion ranking
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
