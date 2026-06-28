import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import cvxpy as cp
import yfinance as yf

TICKERS = [
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "AXISBANK.NS",   "KOTAKBANK.NS",
    "BAJFINANCE.NS","INDUSINDBK.NS", "SHRIRAMFIN.NS", "BAJAJFINSV.NS", "JIOFIN.NS",
    "TCS.NS",       "INFY.NS",       "HCLTECH.NS",    "WIPRO.NS",      "TECHM.NS",
    "LTIMindtree.NS",
    "RELIANCE.NS",  "ONGC.NS",       "BPCL.NS",       "NTPC.NS",       "POWERGRID.NS",
    "COALINDIA.NS",
    "LT.NS",        "ADANIPORTS.NS", "HAL.NS",         "BEL.NS",        "ADANIENT.NS",
    "HAVELLS.NS",
    "HINDUNILVR.NS","ITC.NS",        "BRITANNIA.NS",  "TATACONSUM.NS", "NESTLEIND.NS",
    "DMART.NS",
    "MARUTI.NS",    "BAJAJ-AUTO.NS", "M&M.NS",         "TITAN.NS",      "TRENT.NS",
    "TATAMOTORS.NS","EICHERMOT.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS",   "HINDALCO.NS",   "GRASIM.NS",     "ULTRACEMCO.NS",
    "ASIANPAINT.NS","PIDILITIND.NS", "BERGEPAINT.NS",
    "SUNPHARMA.NS", "DRREDDY.NS",    "CIPLA.NS",      "DIVISLAB.NS",   "APOLLOHOSP.NS",
    "TORNTPHARM.NS",
    "BHARTIARTL.NS",
    "HDFCLIFE.NS",  "SBILIFE.NS",    "MUTHOOTFIN.NS",
]
TICKERS = list(dict.fromkeys(TICKERS))

SECTOR_MAP = {
    "HDFCBANK.NS":   "Financials",  "ICICIBANK.NS":  "Financials",  "SBIN.NS":       "Financials",
    "AXISBANK.NS":   "Financials",  "KOTAKBANK.NS":  "Financials",  "BAJFINANCE.NS": "Financials",
    "INDUSINDBK.NS": "Financials",  "SHRIRAMFIN.NS": "Financials",  "BAJAJFINSV.NS": "Financials",
    "JIOFIN.NS":     "Financials",  "HDFCLIFE.NS":   "Financials",  "SBILIFE.NS":    "Financials",
    "MUTHOOTFIN.NS": "Financials",
    "TCS.NS":        "IT",          "INFY.NS":       "IT",          "HCLTECH.NS":    "IT",
    "WIPRO.NS":      "IT",          "TECHM.NS":      "IT",          "LTIMindtree.NS":"IT",
    "RELIANCE.NS":   "Energy",      "ONGC.NS":       "Energy",      "BPCL.NS":       "Energy",
    "NTPC.NS":       "Energy",      "POWERGRID.NS":  "Energy",      "COALINDIA.NS":  "Energy",
    "LT.NS":         "Industrials", "ADANIPORTS.NS": "Industrials", "HAL.NS":        "Industrials",
    "BEL.NS":        "Industrials", "ADANIENT.NS":   "Industrials", "HAVELLS.NS":    "Industrials",
    "HINDUNILVR.NS": "ConsStaples", "ITC.NS":        "ConsStaples", "BRITANNIA.NS":  "ConsStaples",
    "TATACONSUM.NS": "ConsStaples", "NESTLEIND.NS":  "ConsStaples", "DMART.NS":      "ConsStaples",
    "MARUTI.NS":     "ConsDisc",    "BAJAJ-AUTO.NS": "ConsDisc",    "M&M.NS":        "ConsDisc",
    "TITAN.NS":      "ConsDisc",    "TRENT.NS":      "ConsDisc",    "TATAMOTORS.NS": "ConsDisc",
    "EICHERMOT.NS":  "ConsDisc",
    "TATASTEEL.NS":  "Materials",   "JSWSTEEL.NS":   "Materials",   "HINDALCO.NS":   "Materials",
    "GRASIM.NS":     "Materials",   "ULTRACEMCO.NS": "Materials",   "ASIANPAINT.NS": "Materials",
    "PIDILITIND.NS": "Materials",   "BERGEPAINT.NS": "Materials",
    "SUNPHARMA.NS":  "Healthcare",  "DRREDDY.NS":    "Healthcare",  "CIPLA.NS":      "Healthcare",
    "DIVISLAB.NS":   "Healthcare",  "APOLLOHOSP.NS": "Healthcare",  "TORNTPHARM.NS": "Healthcare",
    "BHARTIARTL.NS": "Telecom",
}

START_DATE   = "2019-01-01"
END_DATE     = "2024-12-31"
MOM_LOOKBACK = 252
MOM_SKIP     = 21
MOM3_SKIP    = 63
TRAIN_WINDOW = 36
LAMBDAS      = [0.0, 0.05, 0.15]
TC_BPS       = 20
SLIPPAGE_BPS = 5
NIFTY_TICKER = "^NSEI"
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)


def _synthetic_prices(tickers, start, end, seed=RANDOM_SEED):
    print("       [OFFLINE] Generating synthetic GBM prices …")
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n, s  = len(dates), len(tickers)
    drifts = rng.normal(0.12, 0.08, s)
    vols   = rng.uniform(0.15, 0.45, s)
    dt     = 1 / 252
    px     = np.zeros((n, s))
    px[0]  = rng.uniform(100, 5000, s)
    mkt    = rng.normal(0, 0.01, n)
    for t in range(1, n):
        shock = 0.5 * mkt[t] + 0.5 * rng.normal(0, 1, s) * vols * dt**0.5
        px[t] = px[t-1] * np.exp((drifts - 0.5*vols**2)*dt + shock)
    df = pd.DataFrame(px, index=dates, columns=tickers)
    df[rng.random((n, s)) < 0.03] = np.nan
    return df.ffill(limit=5)


def download_prices(tickers, start=START_DATE, end=END_DATE):
    print("━" * 62)
    print("STEP 1 │ Downloading price data …")
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(1)
        if raw.empty or raw.dropna(how="all").empty:
            raise ValueError("Empty response")
    except Exception as e:
        print(f"       yfinance unavailable ({e}). Using synthetic data.")
        raw = _synthetic_prices(tickers, start, end)
    raw = raw.ffill(limit=5).dropna(axis=1, thresh=int(0.75 * len(raw)))
    print(f"       {raw.shape[1]} stocks × {raw.shape[0]} trading days")
    return raw


def download_benchmark(ticker=NIFTY_TICKER, start=START_DATE, end=END_DATE):
    try:
        bm = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
        if isinstance(bm, pd.DataFrame):
            bm = bm.squeeze()
        if bm.empty:
            raise ValueError("Empty")
    except Exception:
        dates = pd.bdate_range(start, end)
        rng   = np.random.default_rng(RANDOM_SEED + 1)
        shk   = rng.normal(0.12/252, 0.15/252**0.5, len(dates))
        bm    = pd.Series(18000 * np.exp(np.cumsum(shk)), index=dates)
    bm.name = "NIFTY50"
    return bm


def build_fundamentals(tickers, dates):
    """
    Sector-aware synthetic fundamentals (ROE, D/E, ROE volatility).
    To use real data replace with:
        df = pd.read_csv("screener_fundamentals.csv", parse_dates=["date"])
        return df.set_index(["date","ticker"])[["roe","de_ratio","roe_std"]]
    """
    SECTOR_DEFAULTS = {
        "Financials": (18, 1.5, 4), "IT":          (25, 0.2, 3),
        "Energy":     (12, 1.8, 5), "Industrials": (14, 1.2, 4),
        "ConsStaples":(20, 0.5, 3), "Materials":   (10, 1.6, 6),
        "Healthcare": (22, 0.4, 3), "ConsDisc":    (15, 0.8, 4),
        "Telecom":    ( 8, 2.0, 5),
    }
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return pd.DataFrame(columns=["roe","de_ratio","roe_std","sector"])
    qtrs = pd.date_range(dates.min(), dates.max(), freq="QE")
    rng  = np.random.default_rng(RANDOM_SEED)
    records = []
    for tkr in tickers:
        sector = SECTOR_MAP.get(tkr, "Industrials")
        r_mu, de_mu, rs_mu = SECTOR_DEFAULTS.get(sector, (14, 1.0, 4))
        base_roe = rng.normal(r_mu, 5)
        base_de  = max(rng.normal(de_mu, 0.3), 0.05)
        base_rs  = max(rng.exponential(rs_mu), 0.5)
        for q in qtrs:
            records.append({
                "date":     q,  "ticker":   tkr,
                "roe":      max(base_roe + rng.normal(0, 2), -30),
                "de_ratio": max(base_de  + rng.normal(0, 0.1), 0),
                "roe_std":  max(base_rs  + rng.normal(0, 0.5), 0.1),
                "sector":   sector,
            })
    return pd.DataFrame(records).set_index(["date","ticker"])


def compute_momentum(prices):
    print("━" * 62)
    print("STEP 2 │ Computing momentum signals (12-1 + 3-month) …")

    def _cs_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(
               df.std(axis=1).replace(0, np.nan), axis=0)

    def _winsorise(df):
        return df.clip(lower=df.quantile(0.01, axis=1),
                       upper=df.quantile(0.99, axis=1), axis=0)

    mom12     = _cs_zscore(_winsorise(prices.shift(MOM_SKIP) / prices.shift(MOM_LOOKBACK) - 1))
    mom3      = _cs_zscore(_winsorise(prices.shift(MOM_SKIP) / prices.shift(MOM3_SKIP)    - 1))
    composite = _cs_zscore(0.70 * mom12.fillna(0) + 0.30 * mom3.fillna(0))
    print(f"       Signal shape: {composite.shape}")
    return composite


def compute_quality(fundamentals, prices):
    print("━" * 62)
    print("STEP 3 │ Computing Novy-Marx quality signal …")

    def _z(s):
        return (s - s.mean()) / (s.std() + 1e-8)

    f = fundamentals.copy()
    f["quality_raw"] = _z(f["roe"]) - _z(f["de_ratio"]) - _z(f["roe_std"])
    f["quality"]     = f.groupby(level="date")["quality_raw"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8))

    wide = f["quality"].unstack("ticker").reindex(prices.index, method="ffill")
    wide = wide.reindex(columns=prices.columns)
    print(f"       Signal shape: {wide.dropna(how='all').shape}")
    return wide


def _compute_ic_series(signal, fwd_returns):
    ics = {}
    for t in range(len(signal)):
        sig    = signal.iloc[t].dropna()
        ret    = fwd_returns.iloc[t].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 10:
            continue
        ic, _ = spearmanr(sig[common], ret[common])
        ics[signal.index[t]] = ic
    return pd.Series(ics)


def combine_signals(mom_z, qual_z, prices, train_window=TRAIN_WINDOW,
                    mom_weight=None, qual_weight=None):
    print("━" * 62)
    print("STEP 4 │ IC-weighting signal combination (train window only) …")

    mom_m  = mom_z.resample("ME").last()
    qual_m = qual_z.resample("ME").last()
    ret_m  = prices.resample("ME").last().pct_change()

    idx           = mom_m.index.intersection(qual_m.index).intersection(ret_m.index)
    mom_m, qual_m, ret_m = mom_m.loc[idx], qual_m.loc[idx], ret_m.loc[idx]
    fwd           = ret_m.shift(-1)

    # Full IC series for display
    ic_mom_full  = _compute_ic_series(mom_m,  fwd)
    ic_qual_full = _compute_ic_series(qual_m, fwd)

    # Training-only IC for weight estimation — no lookahead bias
    train_idx     = min(train_window, len(mom_m) - 1)
    ic_mom_train  = _compute_ic_series(mom_m.iloc[:train_idx],  fwd.iloc[:train_idx])
    ic_qual_train = _compute_ic_series(qual_m.iloc[:train_idx], fwd.iloc[:train_idx])

    def _icir(ic):
        return ic.mean() / (ic.std() + 1e-8) if len(ic) > 2 else 0.0

    icir_mom  = _icir(ic_mom_train)
    icir_qual = _icir(ic_qual_train)

    for name, ic, icir in [("Momentum", ic_mom_full, icir_mom),
                            ("Quality ", ic_qual_full, icir_qual)]:
        print(f"       {name} │ Mean IC = {ic.mean():+.4f} │ ICIR = {icir:+.3f} │ "
              f"{'✓ tradeable' if abs(icir) > 0.5 else '○ borderline'}")

    denom  = icir_mom**2 + icir_qual**2
    if denom < 1e-6:
        w_mom, w_qual = 0.5, 0.5
        print("       Insufficient IC data — using equal weights 50/50")
    else:
        w_mom  = icir_mom**2 / denom
        w_qual = icir_qual**2 / denom

    if mom_weight is not None and qual_weight is not None:
        w_mom, w_qual = float(mom_weight), float(qual_weight)
        print(f"       Blend (override) → Momentum: {w_mom:.1%}  Quality: {w_qual:.1%}")
    else:
        print(f"       Blend (IC-derived) → Momentum: {w_mom:.1%}  Quality: {w_qual:.1%}")

    valid_idx  = fwd.dropna(how="all").index
    mom_m_v    = mom_m.reindex(valid_idx)
    qual_m_v   = qual_m.reindex(valid_idx)
    composite  = w_mom * mom_m_v.fillna(0) + w_qual * qual_m_v.fillna(0)

    meta = {
        "w_mom": w_mom, "w_qual": w_qual,
        "weight_source": "override" if mom_weight is not None else "IC-derived",
        "icir_mom": icir_mom, "icir_qual": icir_qual,
        "ic_mom": ic_mom_full, "ic_qual": ic_qual_full,
        "fwd_ret": fwd.reindex(valid_idx),
        "mom_monthly": mom_m_v, "qual_monthly": qual_m_v,
        "train_end_date":    mom_m.index[train_idx - 1] if train_idx > 0 else mom_m.index[0],
        "test_start_date":   mom_m.index[train_idx] if train_idx < len(mom_m) else mom_m.index[-1],
        "train_window_used": train_idx,
        "total_months":      len(mom_m),
    }
    return composite, meta


def optimise_weights(alpha, w_prev, lam, n, max_pos=0.08, long_only=True):
    w = cp.Variable(n)
    if long_only:
        prob = cp.Problem(
            cp.Maximize(alpha @ w - lam * cp.norm1(w - w_prev)),
            [cp.sum(w) == 1, w >= 0, w <= max_pos]
        )
    else:
        prob = cp.Problem(
            cp.Maximize(alpha @ w - lam * cp.norm1(w - w_prev)),
            [cp.norm1(w) <= 1, cp.sum(w) == 0,
             w <=  max_pos, w >= -max_pos]
        )
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                   max_iter=10_000, eps_abs=1e-6, eps_rel=1e-6)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    return w.value if w.value is not None else w_prev


def run_backtest(composite, prices, lam, train_window=TRAIN_WINDOW,
                 max_pos=0.08, long_only=True):
    ret_m   = prices.resample("ME").last().pct_change()
    dates   = composite.index
    tickers = composite.columns.tolist()
    n       = len(tickers)
    w_prev  = np.ones(n) / n if long_only else np.zeros(n)
    results, weight_history = [], {}

    for t in range(train_window, len(dates) - 1):
        alpha     = composite.iloc[t].fillna(0).reindex(tickers).fillna(0).values
        w_opt     = optimise_weights(alpha, w_prev, lam, n, max_pos, long_only)
        next_date = dates[t + 1]
        r         = (ret_m.loc[next_date].reindex(tickers).fillna(0).values
                     if next_date in ret_m.index else np.zeros(n))
        gross     = float(w_opt @ r)
        trades    = np.abs(w_opt - w_prev)
        turnover  = trades.sum() / 2
        tc        = (turnover * TC_BPS / 10_000
                     + trades[trades > 0.005].sum() * SLIPPAGE_BPS / 10_000)
        results.append({
            "date":      next_date, "gross_ret": gross,
            "net_ret":   gross - tc, "turnover":  turnover,
            "tc_drag":   tc,          "lambda":    lam,
        })
        weight_history[next_date] = dict(zip(tickers, w_opt))
        w_prev = w_opt.copy()

    return pd.DataFrame(results).set_index("date"), weight_history


def run_naive_decile_backtest(composite, prices, train_window=TRAIN_WINDOW, long_only=True):
    ret_m = prices.resample("ME").last().pct_change()
    dates = composite.index
    rows  = []
    for t in range(train_window, len(dates) - 1):
        row = composite.iloc[t].dropna()
        nd  = dates[t + 1]
        if len(row) < 10 or nd not in ret_m.index:
            rows.append({"date": nd, "net_ret": 0.0})
            continue
        r = ret_m.loc[nd]
        if long_only:
            q_cutoff = row.quantile(0.70 if len(row) < 20 else 0.80)
            top      = row[row >= q_cutoff].index
            ret      = r.reindex(top).mean() if len(top) > 0 else 0.0
        else:
            top    = row[row >= row.quantile(0.90)].index
            bottom = row[row <= row.quantile(0.10)].index
            ret    = (r.reindex(top).mean() - r.reindex(bottom).mean()) / 2
        rows.append({"date": nd, "net_ret": float(ret) - 0.5 * TC_BPS / 10_000})
    return pd.DataFrame(rows).set_index("date")["net_ret"].rename(
        "Naive Top-Decile" if long_only else "Naive Decile L/S")


def compute_stats(returns):
    ret = returns.dropna()
    if len(ret) < 3:
        return {k: np.nan for k in ["ann_ret","ann_vol","sharpe","sortino",
                                    "max_dd","calmar","hit_rate",
                                    "rolling_sharpe","drawdown_series","cum_returns"]}
    ann_ret = (1 + ret).prod() ** (12 / len(ret)) - 1
    ann_vol = ret.std() * 12**0.5
    sharpe  = ann_ret / (ann_vol + 1e-8)
    down    = ret[ret < 0]
    sortino = ann_ret / (down.std() * 12**0.5 if len(down) > 1 else 1e-8)
    cum     = (1 + ret).cumprod()
    dd      = cum / cum.cummax() - 1
    roll_sh = ret.rolling(12).mean() * 12 / (ret.rolling(12).std() * 12**0.5 + 1e-8)
    return {
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe":  sharpe,
        "sortino": sortino, "max_dd":  dd.min(), "calmar":  ann_ret / (abs(dd.min()) + 1e-8),
        "hit_rate": (ret > 0).mean(),
        "rolling_sharpe": roll_sh, "drawdown_series": dd, "cum_returns": cum,
    }


def get_all_results(start=START_DATE, end=END_DATE,
                    lambdas=None, train_window=TRAIN_WINDOW,
                    max_pos=0.08, long_only=True,
                    mom_weight=None, qual_weight=None,
                    ticker_subset=None, progress_cb=None):
    if lambdas is None:
        lambdas = LAMBDAS

    def _cb(s, t, msg):
        if progress_cb:
            progress_cb(s, t, msg)

    _cb(0, 7, "Downloading prices …")
    prices    = download_prices(ticker_subset if ticker_subset else TICKERS, start, end)
    benchmark = download_benchmark(NIFTY_TICKER, start, end)

    _cb(1, 7, "Building fundamentals …")
    fund = build_fundamentals(prices.columns.tolist(),
                              prices.resample("ME").last().index)

    _cb(2, 7, "Computing momentum …")
    mom_z = compute_momentum(prices)

    _cb(3, 7, "Computing quality …")
    qual_z = compute_quality(fund, prices)

    _cb(4, 7, "Combining signals …")
    composite, meta = combine_signals(mom_z, qual_z, prices, train_window,
                                      mom_weight=mom_weight, qual_weight=qual_weight)

    _cb(5, 7, "Running backtests …")
    backtest_results, weight_histories = {}, {}
    for lam in lambdas:
        df, wh = run_backtest(composite, prices, lam, train_window, max_pos, long_only)
        backtest_results[lam] = df
        weight_histories[lam] = wh

    _cb(6, 7, "Running naïve baseline …")
    naive_returns = run_naive_decile_backtest(composite, prices, train_window, long_only)

    _cb(7, 7, "Done ✓")
    return {
        "prices":           prices,
        "benchmark":        benchmark,
        "composite":        composite,
        "meta":             meta,
        "backtest_results": backtest_results,
        "naive_returns":    naive_returns,
        "weight_histories": weight_histories,
    }