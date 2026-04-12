"""Configuration for the Daily Options Trading Scanner.

Ticker lists and screening parameters for TSX and NASDAQ options scanning.
Data source: Yahoo Finance (free public data via yfinance).
"""

# ── TSX (Toronto Stock Exchange) tickers ──────────────────────────────────────
# Yahoo Finance uses the ".TO" suffix for TSX-listed stocks.
TSX_TICKERS = [
    # Banks / Financials
    "RY.TO",   # Royal Bank of Canada
    "TD.TO",   # Toronto-Dominion Bank
    "BNS.TO",  # Bank of Nova Scotia
    "BMO.TO",  # Bank of Montreal
    "CM.TO",   # CIBC
    "MFC.TO",  # Manulife Financial
    "SLF.TO",  # Sun Life Financial
    # Energy
    "ENB.TO",  # Enbridge
    "SU.TO",   # Suncor Energy
    "CNQ.TO",  # Canadian Natural Resources
    "TRP.TO",  # TC Energy (TransCanada)
    # Railways / Industrials
    "CNR.TO",  # Canadian National Railway
    "CP.TO",   # Canadian Pacific Kansas City
    # Technology
    "SHOP.TO", # Shopify
    "CSU.TO",  # Constellation Software
    # Telecom
    "BCE.TO",  # BCE Inc.
    "T.TO",    # TELUS
    # Mining / Materials
    "ABX.TO",  # Barrick Gold
    "AEM.TO",  # Agnico Eagle Mines
    # Consumer / Retail
    "L.TO",    # Loblaw Companies
    "ATD.TO",  # Alimentation Couche-Tard
    # Infrastructure / Alternative Assets
    "BAM.TO",  # Brookfield Asset Management
]

# ── NASDAQ tickers ─────────────────────────────────────────────────────────────
NASDAQ_TICKERS = [
    # Big Tech (Magnificent 7)
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta Platforms
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    # Semiconductors
    "AMD",    # Advanced Micro Devices
    "INTC",   # Intel
    "QCOM",   # Qualcomm
    "AVGO",   # Broadcom
    "MU",     # Micron Technology
    # Software / Cloud
    "NFLX",   # Netflix
    "ADBE",   # Adobe
    "CRM",    # Salesforce
    "CSCO",   # Cisco Systems
    "ORCL",   # Oracle
    "NOW",    # ServiceNow
    # Consumer / Services
    "COST",   # Costco
    "SBUX",   # Starbucks
    "PYPL",   # PayPal
    # Biotech / Healthcare
    "AMGN",   # Amgen
    "GILD",   # Gilead Sciences
    "MRNA",   # Moderna
]

# ── Screening parameters ───────────────────────────────────────────────────────
SCREENING_PARAMS = {
    # Days to expiration window (inclusive)
    "min_dte": 7,
    "max_dte": 60,
    # Minimum open interest (contracts) – ensures sufficient liquidity
    "min_open_interest": 10,
    # Minimum bid price – filters out near-worthless options
    "min_bid": 0.05,
    # Minimum annualized return on capital (percentage)
    "min_annualized_return_pct": 10.0,
    # Maximum out-of-the-money % relative to stock price
    # 0.10 = options up to 10% OTM are considered
    "max_otm_pct": 0.10,
}
