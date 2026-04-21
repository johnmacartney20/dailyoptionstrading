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

# ── RRSP tickers ───────────────────────────────────────────────────────────────
# Curated list of large-cap stocks and broad-market ETFs suitable for a
# long-term, stability-focused RRSP portfolio.
RRSP_TICKERS = [
    # Canadian Blue-chip
    "RY.TO",    # Royal Bank of Canada – largest Canadian bank
    "TD.TO",    # Toronto-Dominion Bank
    "ENB.TO",   # Enbridge – pipeline/infrastructure income
    "CNR.TO",   # Canadian National Railway
    "BCE.TO",   # BCE Inc. – telecom dividend
    # US Large-cap (RRSP-eligible)
    "AAPL",     # Apple – mega-cap technology
    "MSFT",     # Microsoft – mega-cap technology / cloud
    "COST",     # Costco – defensive consumer staples
    "AMGN",     # Amgen – large-cap healthcare
    # Broad-market ETFs
    "SPY",      # SPDR S&P 500 ETF – US broad market
    "QQQ",      # Invesco NASDAQ-100 ETF – US growth
]

# ── Screening parameters ───────────────────────────────────────────────────────
SCREENING_PARAMS = {
    # Days to expiration window (inclusive)
    "min_dte": 7,
    "max_dte": 60,
    # Minimum open interest (contracts) – ensures sufficient liquidity
    "min_open_interest": 500,
    # Maximum bid-ask spread as a fraction of the bid price (liquidity filter)
    # 0.10 = spread must be less than 10 % of the bid
    "max_bid_ask_spread_pct": 0.10,
    # Minimum bid price – filters out near-worthless options
    "min_bid": 0.05,
    # Minimum annualized return on capital (percentage) – kept as a floor only;
    # primary ranking is now driven by the composite score
    "min_annualized_return_pct": 5.0,
    # OTM % window: favor strikes 3-15 % below (or above for calls) stock price.
    # Trades within 3 % of the stock price are excluded (too close to ATM).
    "min_otm_pct": 0.03,
    "max_otm_pct": 0.15,
    # Assumed spread width (dollars) used to compute max loss and spread structure.
    # A $5-wide spread caps max loss at ~$500 per contract (small-account friendly).
    "spread_width": 5.0,
    # Maximum allowable max loss per spread contract (dollars)
    "max_spread_loss": 1000.0,
}
