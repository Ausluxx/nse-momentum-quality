# Data

## Price Data

Daily adjusted close prices are downloaded automatically via **yfinance** using NSE tickers (`.NS` suffix). No manual download required — the pipeline fetches data on first run.

```python
import yfinance as yf
prices = yf.download("RELIANCE.NS", start="2019-01-01", end="2024-12-31", auto_adjust=True)
```

`auto_adjust=True` incorporates dividend adjustments into historical prices, so total returns are partially captured through adjusted prices.

## Fundamental Data (Quality Signal)

The quality signal (ROE, D/E ratio, ROE volatility) currently uses **synthetic data** generated with sector-aware distributions. This is sufficient for demonstrating the pipeline architecture but should be replaced with real data for production use.

### To Use Real Screener.in Data

1. Export quarterly fundamentals from [Screener.in](https://www.screener.in) for each ticker
2. Structure the CSV as:

```
date, ticker, roe, de_ratio, roe_std
2023-03-31, RELIANCE.NS, 14.2, 0.45, 3.1
2023-03-31, TCS.NS, 41.8, 0.08, 2.3
...
```

3. Replace `build_fundamentals()` in `nse_momentum_quality_backtest.py` with:

```python
def build_fundamentals(tickers, dates):
    df = pd.read_csv("data/screener_fundamentals.csv", parse_dates=["date"])
    return df.set_index(["date", "ticker"])[["roe", "de_ratio", "roe_std"]]
```

### Alternative: Capitaline / Bloomberg

For institutional-grade backtesting, use **point-in-time** fundamental data from Capitaline or Bloomberg. Point-in-time data records the value of each fundamental *as it was known on that date*, preventing lookahead bias from restated financials.

## Benchmark Data

Nifty 50 (`^NSEI`) is downloaded via yfinance alongside stock prices.

## Important Notes

- **Do not commit** raw CSV files or parquet data to this repository (covered by `.gitignore`)
- **Survivorship bias:** The current ticker list reflects *current* NSE 50 constituents. For a bias-free backtest, use historical index constituent files available from the NSE website
- Price data via yfinance is sufficient for research; for production, use a commercial data vendor (Refinitiv, Bloomberg, ICICI Direct API)
