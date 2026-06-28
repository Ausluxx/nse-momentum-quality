"""
Central configuration for the NSE Momentum + Quality Factor Strategy.
All parameters live here — change once, applies everywhere.
"""

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS = [
    "HDFCBANK.NS",   "ICICIBANK.NS",  "SBIN.NS",       "AXISBANK.NS",   "KOTAKBANK.NS",
    "BAJFINANCE.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "BAJAJFINSV.NS", "JIOFIN.NS",
    "TCS.NS",        "INFY.NS",       "HCLTECH.NS",    "WIPRO.NS",      "TECHM.NS",
    "LTIMindtree.NS",
    "RELIANCE.NS",   "ONGC.NS",       "BPCL.NS",       "NTPC.NS",       "POWERGRID.NS",
    "COALINDIA.NS",
    "LT.NS",         "ADANIPORTS.NS", "HAL.NS",         "BEL.NS",        "ADANIENT.NS",
    "HAVELLS.NS",
    "HINDUNILVR.NS", "ITC.NS",        "BRITANNIA.NS",  "TATACONSUM.NS", "NESTLEIND.NS",
    "DMART.NS",
    "MARUTI.NS",     "BAJAJ-AUTO.NS", "M&M.NS",         "TITAN.NS",      "TRENT.NS",
    "TATAMOTORS.NS", "EICHERMOT.NS",
    "TATASTEEL.NS",  "JSWSTEEL.NS",   "HINDALCO.NS",   "GRASIM.NS",     "ULTRACEMCO.NS",
    "ASIANPAINT.NS", "PIDILITIND.NS", "BERGEPAINT.NS",
    "SUNPHARMA.NS",  "DRREDDY.NS",    "CIPLA.NS",      "DIVISLAB.NS",   "APOLLOHOSP.NS",
    "TORNTPHARM.NS",
    "BHARTIARTL.NS",
    "HDFCLIFE.NS",   "SBILIFE.NS",    "MUTHOOTFIN.NS",
]
TICKERS = list(dict.fromkeys(TICKERS))

SECTOR_MAP = {
    "HDFCBANK.NS":    "Financials",  "ICICIBANK.NS":   "Financials",  "SBIN.NS":       "Financials",
    "AXISBANK.NS":    "Financials",  "KOTAKBANK.NS":   "Financials",  "BAJFINANCE.NS": "Financials",
    "INDUSINDBK.NS":  "Financials",  "SHRIRAMFIN.NS":  "Financials",  "BAJAJFINSV.NS": "Financials",
    "JIOFIN.NS":      "Financials",  "HDFCLIFE.NS":    "Financials",  "SBILIFE.NS":    "Financials",
    "MUTHOOTFIN.NS":  "Financials",
    "TCS.NS":         "IT",          "INFY.NS":        "IT",          "HCLTECH.NS":    "IT",
    "WIPRO.NS":       "IT",          "TECHM.NS":       "IT",          "LTIMindtree.NS":"IT",
    "RELIANCE.NS":    "Energy",      "ONGC.NS":        "Energy",      "BPCL.NS":       "Energy",
    "NTPC.NS":        "Energy",      "POWERGRID.NS":   "Energy",      "COALINDIA.NS":  "Energy",
    "LT.NS":          "Industrials", "ADANIPORTS.NS":  "Industrials", "HAL.NS":        "Industrials",
    "BEL.NS":         "Industrials", "ADANIENT.NS":    "Industrials", "HAVELLS.NS":    "Industrials",
    "HINDUNILVR.NS":  "ConsStaples", "ITC.NS":         "ConsStaples", "BRITANNIA.NS":  "ConsStaples",
    "TATACONSUM.NS":  "ConsStaples", "NESTLEIND.NS":   "ConsStaples", "DMART.NS":      "ConsStaples",
    "MARUTI.NS":      "ConsDisc",    "BAJAJ-AUTO.NS":  "ConsDisc",    "M&M.NS":        "ConsDisc",
    "TITAN.NS":       "ConsDisc",    "TRENT.NS":       "ConsDisc",    "TATAMOTORS.NS": "ConsDisc",
    "EICHERMOT.NS":   "ConsDisc",
    "TATASTEEL.NS":   "Materials",   "JSWSTEEL.NS":    "Materials",   "HINDALCO.NS":   "Materials",
    "GRASIM.NS":      "Materials",   "ULTRACEMCO.NS":  "Materials",   "ASIANPAINT.NS": "Materials",
    "PIDILITIND.NS":  "Materials",   "BERGEPAINT.NS":  "Materials",
    "SUNPHARMA.NS":   "Healthcare",  "DRREDDY.NS":     "Healthcare",  "CIPLA.NS":      "Healthcare",
    "DIVISLAB.NS":    "Healthcare",  "APOLLOHOSP.NS":  "Healthcare",  "TORNTPHARM.NS": "Healthcare",
    "BHARTIARTL.NS":  "Telecom",
}

# ── Data ──────────────────────────────────────────────────────────────────────
START_DATE   = "2019-01-01"
END_DATE     = "2024-12-31"
NIFTY_TICKER = "^NSEI"
RANDOM_SEED  = 42

# ── Signal construction ───────────────────────────────────────────────────────
MOM_LOOKBACK = 252    # 12-month lookback in trading days
MOM_SKIP     = 21     # skip last month (short-term reversal avoidance)
MOM3_SKIP    = 63     # 3-month momentum lookback
MOM12_WEIGHT = 0.70   # weight of 12-1 signal in composite momentum
MOM3_WEIGHT  = 0.30   # weight of 3-month signal in composite momentum

# ── Portfolio construction ────────────────────────────────────────────────────
TRAIN_WINDOW    = 36    # months of data used for IC estimation
LAMBDAS         = [0.0, 0.05, 0.15]   # transaction cost penalty values to compare
DEFAULT_MAX_POS = 0.08  # default max weight per stock (8%)

# ── Transaction costs ─────────────────────────────────────────────────────────
TC_BPS       = 20   # round-trip transaction cost in basis points
SLIPPAGE_BPS = 5    # additional slippage for large trades (> 0.5% of portfolio)
