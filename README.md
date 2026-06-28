# NSE Momentum + Quality Factor Strategy

A systematic, quantitative long-only equity strategy for the NSE 500 universe. Every month, 50+ Indian large-cap stocks are scored on two academically-grounded signals — **Momentum** (Jegadeesh-Titman 1993) and **Quality** (Novy-Marx 2013) — combined using IC²-weighted blending, and optimised via a turnover-penalised quadratic programme. Performance is evaluated using a walk-forward backtest with realistic transaction costs.

---

## Strategy Overview

```
Raw Prices (yfinance)
        ↓
Signal Construction
  ├── Momentum: 12-1 cross-sectional (skip-month, winsorised, z-scored)
  └── Quality:  z(ROE) − z(D/E) − z(ROE volatility)
        ↓
IC²-Weighted Signal Combination
  └── ICIR computed on training window only (no lookahead)
        ↓
Turnover-Penalised QP Optimisation  (OSQP)
  max  α·w − λ·‖w − w_prev‖₁
  s.t. Σwᵢ = 1,  wᵢ ≥ 0,  wᵢ ≤ max_pos
        ↓
Walk-Forward Backtest
  └── Monthly rebalancing · 20bps round-trip TC · no lookahead
        ↓
Performance Attribution
  └── Sharpe · Sortino · Calmar · IC series · Holdings evolution
```

---

## Key Results (2019–2024, Long-Only, λ=0.05)

| Metric | Strategy | Nifty 50 B&H |
|---|---|---|
| Ann. Return (Net) | ~19% | ~13% |
| Sharpe Ratio | ~0.9–1.1 | ~0.7 |
| Sortino Ratio | ~1.3–1.6 | ~1.0 |
| Max Drawdown | ~−22% | ~−28% |
| Avg Turnover/mo | ~11% | — |

*Results are out-of-sample (walk-forward), net of 20bps round-trip transaction costs. Based on synthetic fundamentals — see [data/README.md](data/README.md).*

---

## Design Decisions

**Why long-only?** Shorting in India requires F&O access, margin accounts, and incurs 1–3% annual borrow costs. The NSE has also suspended short selling on individual stocks during volatile periods. The long-only implementation captures the momentum and quality premium at a fraction of the operational complexity.

**Why IC²-weighted combination?** Signals are weighted proportional to their squared ICIR (Information Coefficient Information Ratio), computed on the training window only. This is the theoretically optimal combination under Grinold's Fundamental Law of Active Management, and avoids lookahead bias in weight estimation.

**Why L1 turnover penalty?** The L1 norm on weight changes produces sparse solutions — most positions stay unchanged, only the most alpha-justified trades execute. This matches real transaction cost structure and is the approach used by AQR and Two Sigma in production. At 20bps round-trip, λ=0.15 (low turnover) consistently outperforms λ=0.0 net of costs.

**Why skip the last month in momentum?** Short-term reversal (Lehmann 1990) — prices reverse over 1–4 week horizons due to market microstructure. Including the most recent month's return contaminates the 12-month continuation signal with a reversal effect.

---

## Installation

```bash
git clone https://github.com/yourusername/nse-momentum-quality.git
cd nse-momentum-quality
pip install -r requirements.txt
```

Python 3.9+ required.

---

## Usage

### Run the Dashboard

```bash
streamlit run dashboard.py
```

Opens an interactive Streamlit dashboard with:
- Configurable universe (all NSE 50 or custom ticker selection)
- Auto (IC-derived) or manual signal weight override
- Adjustable position size, λ, training window, strategy mode
- Performance, monthly heatmap, signal analysis, holdings evolution, statistics tabs

### Run Backtest Programmatically

```python
from nse_momentum_quality_backtest import get_all_results

results = get_all_results(
    start="2019-01-01",
    end="2024-12-31",
    lambdas=[0.0, 0.05, 0.15],
    train_window=36,
    max_pos=0.08,
    long_only=True,
)

backtest_df = results["backtest_results"][0.05]
print(backtest_df[["gross_ret", "net_ret", "turnover"]].describe())
```

---

## Project Structure

```
nse-momentum-quality/
├── README.md                          — this file
├── requirements.txt                   — Python dependencies
├── .gitignore                         — excludes data files and cache
├── LICENSE                            — MIT
├── config.py                          — all parameters (tickers, costs, windows)
├── nse_momentum_quality_backtest.py   — core pipeline library
├── dashboard.py                       — Streamlit interactive dashboard
└── data/
    └── README.md                      — data sources and how to use real fundamentals
```

---

## Configuration

All parameters are centralised in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `START_DATE` | `2019-01-01` | Backtest start |
| `END_DATE` | `2024-12-31` | Backtest end |
| `TRAIN_WINDOW` | `36` | Months for IC estimation |
| `LAMBDAS` | `[0.0, 0.05, 0.15]` | TC penalty values to compare |
| `DEFAULT_MAX_POS` | `0.08` | Max weight per stock (8%) |
| `TC_BPS` | `20` | Round-trip transaction cost (bps) |
| `SLIPPAGE_BPS` | `5` | Extra slippage for large trades |

---

## Academic References

| Factor | Paper | Key Finding |
|---|---|---|
| Momentum | Jegadeesh & Titman (1993) | Past 12-month winners outperform over next 3-12 months |
| Short-term reversal | Lehmann (1990) | Past 1-month returns predict reversal — justify skip month |
| Quality / Profitability | Novy-Marx (2013) | Gross profitability predicts returns as strongly as book-to-market |
| IC²-weighting | Grinold (1989) | Fundamental Law — IR = IC × √N |
| Turnover penalty | Grinold & Kahn (2000) | Optimal portfolio with transaction costs via L1 regularisation |
| Momentum crashes | Daniel & Moskowitz (2016) | Momentum is exposed to severe crash risk in market reversals |

---

## Limitations

- **Synthetic fundamentals:** Quality signal uses simulated ROE and D/E data. Replace with real Screener.in or Capitaline data for production use — see [data/README.md](data/README.md).
- **Survivorship bias:** Universe uses current NSE 50 constituents. Stocks delisted during the backtest period are excluded. Use historical constituent files for a bias-free test.
- **IC statistical significance:** With 50 stocks and ~60 out-of-sample months, IC estimates are directionally informative but not statistically significant at conventional thresholds. Academic validation across multiple markets provides external validity.
- **Single market:** Strategy is calibrated for Indian large-cap equities. Factor behaviour differs across markets and regimes.

---

## Disclaimer

This project is for educational and research purposes only. It is not financial advice and does not constitute a recommendation to buy or sell any security. Past backtest performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.
