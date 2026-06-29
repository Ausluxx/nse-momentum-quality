import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date
import cvxpy as cp

from nse_momentum_quality_backtest import (
    get_all_results, compute_stats,
    compute_momentum, compute_quality, build_fundamentals,
    TICKERS, SECTOR_MAP, LAMBDAS, TRAIN_WINDOW,
)

st.set_page_config(page_title="NSE Factor Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

BG     = "#1C1C1C"; PANEL  = "#242424"; BORDER = "#333333"
CREAM  = "#F5E8D8"; MUTED  = "#9A8878"; CORAL  = "#FF6F61"
GOLD   = "#DAA520"; ORANGE = "#FF4500"; GREEN  = "#6DBF8A"; WHITE = "#FFFFFF"
CLR    = {0.0: CORAL, 0.05: GOLD, 0.10: ORANGE, 0.15: GREEN, 0.20: "#9B8EA8"}

def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def _fmt_inr(x):
    if abs(x) >= 1e7: return f"₹{x/1e7:.2f}Cr"
    if abs(x) >= 1e5: return f"₹{x/1e5:.2f}L"
    return f"₹{x:,.0f}"

# Base Plotly layout — no xaxis/yaxis/legend keys so callers can set freely
_PLOTLY_BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(family="JetBrains Mono, monospace", color=CREAM, size=11),
    margin=dict(l=52, r=20, t=44, b=40),
    hoverlabel=dict(bgcolor="#2e2e2e", bordercolor=BORDER, font_color=CREAM, font_size=11),
)
# Full layout including axes — use for most charts
def plotly_layout(**extra):
    base = dict(**_PLOTLY_BASE)
    base["xaxis"] = dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=10))
    base["yaxis"] = dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=10))
    base["legend"] = dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1, font=dict(color=CREAM, size=10))
    base.update(extra)
    return base

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"],[data-testid="stToolbar"]{{
    background-color:{BG}!important;color:{CREAM}!important;font-family:'Inter',sans-serif!important;}}
[data-testid="stSidebar"]{{background-color:{PANEL}!important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{CREAM}!important;}}
[data-testid="stMetric"]{{background:linear-gradient(135deg,{PANEL},{BG});border:1px solid {BORDER};
    border-left:3px solid {GOLD};border-radius:8px;padding:14px 16px;}}
[data-testid="stMetricValue"]{{color:{CREAM}!important;font-family:'JetBrains Mono',monospace!important;
    font-size:1.35rem!important;font-weight:500!important;}}
[data-testid="stMetricLabel"]{{color:{MUTED}!important;font-size:.68rem!important;
    text-transform:uppercase;letter-spacing:.08em;}}
[data-testid="stMetricDelta"]{{font-size:.76rem!important;}}
.stTabs [data-baseweb="tab-list"]{{background:{PANEL};border-bottom:1px solid {BORDER};gap:2px;}}
.stTabs [data-baseweb="tab"]{{color:{MUTED}!important;font-family:'Inter',sans-serif!important;
    font-size:.82rem;padding:8px 18px;border-radius:4px 4px 0 0;}}
.stTabs [aria-selected="true"]{{color:{CREAM}!important;border-bottom:2px solid {CORAL}!important;background:#2a2a2a;}}
.stButton>button{{background:linear-gradient(135deg,{CORAL}20,{ORANGE}20);border:1px solid {CORAL};
    color:{CREAM}!important;font-family:'JetBrains Mono',monospace;font-size:.83rem;
    border-radius:6px;letter-spacing:.04em;transition:all .18s;}}
.stButton>button:hover{{background:linear-gradient(135deg,{CORAL}44,{ORANGE}44);
    border-color:{ORANGE};transform:translateY(-1px);}}
.stDataFrame{{border:1px solid {BORDER};border-radius:8px;overflow:hidden;}}
h1,h2,h3{{font-family:'Inter',sans-serif!important;}}
h1{{font-weight:600!important;color:{CREAM}!important;letter-spacing:-.02em;}}
div[data-testid="stExpander"]{{border:1px solid {BORDER};border-radius:8px;background:{PANEL};}}
.slabel{{font-family:'Inter',sans-serif;font-size:.68rem;font-weight:600;color:{MUTED};
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;}}
hr.div{{border:none;border-top:1px solid {BORDER};margin:14px 0;}}
</style>""", unsafe_allow_html=True)

TODAY = date.today()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — single clean flow
# ─────────────────────────────────────────────────────────────────────────────
# Read mode from session state for the subtitle (safe before widget renders)
_sidebar_mode = "Long Only"

with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.05rem;font-weight:600;color:{CREAM};"
        f"margin-bottom:2px;font-family:Inter,sans-serif'>NSE Factor Strategy</div>"
        f"<div style='font-size:.73rem;color:{MUTED};margin-bottom:14px'>"
        f"Momentum · Quality · {_sidebar_mode}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 1. Period
    st.markdown("<div class='slabel'>Backtest Period</div>", unsafe_allow_html=True)
    cs, ce = st.columns(2)
    start_date = cs.date_input("From", value=date(2019, 1, 1),
                               min_value=date(2010, 1, 1),
                               max_value=date(TODAY.year - 1, 12, 31),
                               label_visibility="collapsed")
    end_date = ce.date_input("To", value=TODAY,
                             min_value=date(2011, 1, 1), max_value=TODAY,
                             label_visibility="collapsed")
    st.caption(f"{start_date.strftime('%d %b %Y')}  →  {end_date.strftime('%d %b %Y')}")
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 2. Universe — pick ALL or custom tickers
    st.markdown("<div class='slabel'>Universe</div>", unsafe_allow_html=True)
    universe_mode = st.radio("Universe mode", ["All NSE 50 stocks", "Custom selection"],
                             index=0, label_visibility="collapsed")
    custom_tickers = []
    if universe_mode == "Custom selection":
        custom_tickers = st.multiselect(
            "Pick stocks",
            options=sorted([t.replace(".NS", "") for t in TICKERS]),
            default=[],
            placeholder="Select stocks to include…",
        )
        if len(custom_tickers) < 10:
            st.warning("Pick at least 10 stocks for meaningful results.")
        else:
            st.caption(f"{len(custom_tickers)} stocks selected")
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 3. Signal weights — auto (IC-derived) or manual override
    st.markdown("<div class='slabel'>Signal Weights</div>", unsafe_allow_html=True)
    weight_mode = st.radio(
        "Weight mode",
        ["Auto (model decides)", "Manual override"],
        index=0,
        key="_weight_mode",
        label_visibility="collapsed",
        help="Auto: the model uses the training window to calculate which signal "
             "predicted returns better (IC²-weighted). It decides the blend itself.\n\n"
             "Manual: you force a specific Momentum / Quality split regardless of what "
             "the IC calculation says.",
    )
    auto_weights = (weight_mode == "Auto (model decides)")

    if auto_weights:
        # Pass None — combine_signals will use IC-derived weights
        mom_weight  = None
        qual_weight = None
        st.caption("Model will calculate optimal blend from training data.")
    else:
        mom_weight = st.number_input(
            "Momentum weight (0.00 – 1.00)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            format="%.2f",
            help="Quality weight = 1 − Momentum weight. "
                 "0.7 = 70% Momentum, 30% Quality.",
        )
        mom_weight  = round(float(mom_weight), 2)
        qual_weight = round(1.0 - mom_weight, 2)
        st.caption(f"Override: Momentum {mom_weight:.0%}  ·  Quality {qual_weight:.0%}")
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 4. Position size
    st.markdown("<div class='slabel'>Max Position Size per Stock</div>", unsafe_allow_html=True)
    max_pos_pct = st.number_input(
        "Max position per stock (%)",
        min_value=1, max_value=50, value=8, step=1,
        help="8% = at most ~12 stocks get meaningful weight. "
             "Lower = more diversified. Higher = more concentrated.",
    )
    max_pos_pct = int(max_pos_pct)
    st.caption(f"Max {max_pos_pct}% per stock  ·  Min ~{100//max_pos_pct} positions")
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 5. Cost penalty
    st.markdown("<div class='slabel'>Transaction Cost Penalty (λ)</div>", unsafe_allow_html=True)
    lam_choice = st.number_input(
        "λ value (0.00 – 0.50)",
        min_value=0.0, max_value=0.50, value=0.05, step=0.01,
        format="%.2f",
        help="0 = ignore trading costs (high turnover). "
             "0.05 = moderate. 0.15 = low turnover. "
             "Higher = optimiser avoids large weight changes.",
    )
    lam_choice = round(float(lam_choice), 2)
    run_all    = st.checkbox("Compare λ = 0, 0.05, 0.15 as well", value=True)
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 6. Strategy mode
    st.markdown("<div class='slabel'>Strategy Mode</div>", unsafe_allow_html=True)

    
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # 7. Training window (advanced)
    with st.expander("Advanced", expanded=False):
        train_w = st.number_input(
            "Training Window (months)",
            min_value=6, max_value=120, value=TRAIN_WINDOW, step=1,
            key="train_w_input",
            help="Months of history used to learn IC weights. "
                 "Minimum 12 recommended. More = more stable IC estimate.",
        )
        train_w = int(train_w)
    train_w = int(st.session_state.get("train_w_input", TRAIN_WINDOW))

    run_btn = st.button("▶  RUN BACKTEST", use_container_width=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:.68rem;color:{MUTED};line-height:1.9'>"
        f"<b style='color:{CREAM}'>Pipeline</b><br>"
        "1 · Prices via yfinance<br>2 · Momentum (12-1 + 3m)<br>"
        "3 · Quality (Novy-Marx)<br>4 · IC² signal blend<br>"
        "5 · QP optimisation (OSQP)<br>6 · Walk-forward backtest<br>"
        "7 · Performance attribution</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _run_pipeline(start_str, end_str, lambdas, tw, max_pos,
                  mom_w, qual_w, ticker_subset):
    prog = st.progress(0, text="Initialising …")
    def cb(s, t, msg): prog.progress(s / t, text=msg)
    result = get_all_results(
        start=start_str, end=end_str, lambdas=lambdas, train_window=tw,
        max_pos=max_pos, mom_weight=mom_w, qual_weight=qual_w,
        ticker_subset=ticker_subset, progress_cb=cb,
    )
    prog.empty()
    return result

# Build the ticker subset list (None = use all)
ticker_subset = None
if universe_mode == "Custom selection" and len(custom_tickers) >= 10:
    ticker_subset = [t + ".NS" for t in custom_tickers]

params_key = (f"{start_date}|{end_date}|{lam_choice}|{run_all}|{train_w}|"
              f"{weight_mode}|{mom_weight}|{max_pos_pct}|{sorted(custom_tickers)}")

_first_load = "results" not in st.session_state
if run_btn or _first_load:
    _base_lams = [0.0, 0.05, 0.15]
    if run_all:
        lambdas = sorted(set(_base_lams + [lam_choice]))
    else:
        lambdas = [lam_choice]
    with st.spinner("Running pipeline …"):
        st.session_state["results"] = _run_pipeline(
            str(start_date), str(end_date), lambdas, train_w,
            max_pos_pct / 100,
            mom_weight,   # None = auto IC-derived
            qual_weight,  # None = auto IC-derived
            ticker_subset,
        )
        st.session_state["params_key"] = params_key
        # Reset selected_lam to the sidebar choice after a fresh run
        st.session_state["selected_lam"] = lam_choice
elif st.session_state.get("params_key") != params_key:
    st.info("Parameters changed — click  ▶ RUN BACKTEST  to apply.", icon="ℹ️")

R                = st.session_state["results"]
backtest_results = R["backtest_results"]
naive_returns    = R["naive_returns"]
prices           = R["prices"]
benchmark        = R["benchmark"]
composite        = R["composite"]
meta             = R["meta"]
# primary_lam is chosen interactively via a selector below the header —
# initialise from sidebar, but user can switch without re-running
available_lams = list(backtest_results.keys())
if "selected_lam" not in st.session_state or st.session_state["selected_lam"] not in available_lams:
    st.session_state["selected_lam"] = (lam_choice if lam_choice in available_lams
                                        else available_lams[0])
primary_lam = st.session_state["selected_lam"]
primary_df       = backtest_results[primary_lam]
stats            = compute_stats(primary_df["net_ret"])
g_stats          = compute_stats(primary_df["gross_ret"])
naive_stats      = compute_stats(naive_returns)
ic_m             = meta["ic_mom"]
ic_q             = meta["ic_qual"]
actual_start     = primary_df.index.min().strftime("%b %Y")
actual_end       = primary_df.index.max().strftime("%b %Y")

# Explain λ ranking to avoid confusion — placed here so backtest_results is in scope
if len(backtest_results) > 1:
    _net_rets   = {l: compute_stats(df["net_ret"])["ann_ret"]   for l, df in backtest_results.items()}
    _gross_rets = {l: compute_stats(df["gross_ret"])["ann_ret"] for l, df in backtest_results.items()}
    _best_net   = max(_net_rets,   key=_net_rets.get)
    _best_gross = max(_gross_rets, key=_gross_rets.get)
    if _best_net != min(backtest_results.keys()):  # λ=0 rarely beats λ=0.05+ net of costs
        st.caption(
            f"ℹ️  λ={_best_net} has the best net return — this is normal. "
            f"λ=0 trades most aggressively (highest gross alpha) but pays the most in "
            f"transaction costs ({20}bps/rebalance × high turnover). "
            f"Higher λ = less trading = lower costs = better net return in most periods."
        )

# Pre-compute benchmark monthly returns once — reused across KPIs and charts
_bm_m_full = pd.Series(dtype=float)
if benchmark is not None and len(benchmark) > 5:
    _bm_m_full = benchmark.resample("ME").last().pct_change().dropna()

# Diagnostic warnings
_total_months = len(primary_df)
_n_stocks     = prices.shape[1]
_warns = []
if _total_months < train_w:
    _warns.append(
        f"Date range too short: only {_total_months} out-of-sample months "
        f"but training window = {train_w}m. Extend the date range or reduce training window.")
if _total_months < 24:
    _warns.append(
        f"Only {_total_months} backtest months — need at least 24 for reliable statistics.")
# Quality IC is often slightly negative at 1-month horizon — that is normal
# (quality is a slow factor, works over 1-3 years, not monthly)
# Only warn if Momentum IC is also deeply negative — that signals a real problem
if float(ic_m.mean()) < -0.02:
    _warns.append(
        f"Momentum IC is negative ({ic_m.mean():+.4f}). "
        f"This means the momentum signal is predicting the wrong direction — "
        f"likely caused by too short a training window or a bear-market-dominated period. "
        f"Try extending the start date or reducing the training window.")
if float(ic_q.mean()) < -0.05:
    _warns.append(
        f"Quality IC is notably negative ({ic_q.mean():+.4f}) even for a slow factor. "
        f"This can happen in strong bull markets where high-leverage names outperform quality. "
        f"Consider increasing the Momentum weight in Signal Weights.")
if _n_stocks < 10:
    _warns.append(f"Only {_n_stocks} stocks loaded — too few for a diversified factor portfolio.")
for w in _warns:
    st.warning(w, icon="⚠️")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + KPIs
# ─────────────────────────────────────────────────────────────────────────────

univ_str = f"{_n_stocks} stocks" if ticker_subset is None else f"{_n_stocks} stocks (custom)"
st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px'>"
    f"<span style='font-size:1.55rem;font-weight:600;color:{CREAM};"
    f"font-family:Inter,sans-serif'>NSE Factor Strategy</span>"
    f"<span style='font-size:.78rem;color:{MUTED};font-family:Inter,sans-serif'>"
    f"Momentum + Quality  ·  Long Only  ·  Walk-Forward</span></div>"
    f"<div style='font-size:.76rem;color:{MUTED};"
    f"font-family:JetBrains Mono,monospace;margin-bottom:6px'>"
    f"λ={primary_lam}  ·  Net  ·  Long Only  ·  {actual_start} – {actual_end}  ·  {univ_str}  ·  "
    f"Mom {meta['w_mom']:.0%} / Qual {meta['w_qual']:.0%}</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='div'>", unsafe_allow_html=True)

# ── λ selector — switch which run the KPIs display, no re-run needed ─────────
if len(available_lams) > 1:
    lam_cols = st.columns(len(available_lams) + 1)
    lam_cols[0].markdown(
        f"<div style='font-size:.7rem;color:{MUTED};text-transform:uppercase;"
        f"letter-spacing:.08em;padding-top:10px'>Viewing λ =</div>",
        unsafe_allow_html=True)
    for i, lam in enumerate(available_lams):
        s_lam  = compute_stats(backtest_results[lam]["net_ret"])
        active = (lam == primary_lam)
        border = CORAL if active else BORDER
        bg_val = "#2a2a2a" if active else PANEL
        ann_r  = s_lam["ann_ret"] * 100
        sharpe = s_lam["sharpe"]
        if lam_cols[i+1].button(
            f"λ={lam}  {ann_r:+.1f}%  SR {sharpe:.2f}",
            key=f"lam_btn_{lam}",
            use_container_width=True,
        ):
            st.session_state["selected_lam"] = lam
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
    # re-derive after potential switch
    primary_lam = st.session_state["selected_lam"]
    primary_df  = backtest_results[primary_lam]
    stats       = compute_stats(primary_df["net_ret"])
    g_stats     = compute_stats(primary_df["gross_ret"])

# Absolute profit figures
_initial   = 100_000
_total_ret = float((1 + primary_df["net_ret"]).prod() - 1)
_final_val = _initial * (1 + _total_ret)
_profit    = _final_val - _initial
_nifty_ret = 0.0
if len(_bm_m_full) > 5:
    common = primary_df.index.intersection(_bm_m_full.index)
    if len(common) > 5:
        _nifty_ret = float((1 + _bm_m_full.loc[common]).prod() - 1)
_alpha_ret = _total_ret - _nifty_ret

_ret_ser  = primary_df["net_ret"].copy()
_ret_ser.index = pd.to_datetime(_ret_ser.index)
import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore")
    _yr_rets = _ret_ser.groupby(_ret_ser.index.year).apply(lambda x: (1+x).prod()-1)
_best_yr  = int(_yr_rets.idxmax()) if len(_yr_rets) > 0 else "—"
_best_val = float(_yr_rets.max())  if len(_yr_rets) > 0 else 0.0
_wrst_yr  = int(_yr_rets.idxmin()) if len(_yr_rets) > 0 else "—"
_wrst_val = float(_yr_rets.min())  if len(_yr_rets) > 0 else 0.0

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Total Return",        f"{_total_ret*100:+.1f}%",
          f"vs Nifty {_nifty_ret*100:+.1f}%  (α {_alpha_ret*100:+.1f}%)")
c2.metric(f"₹1L → {_fmt_inr(_final_val)}", f"{_fmt_inr(_profit)}",
          f"{len(primary_df)} months")
c3.metric("Ann. Return (Net)",   f"{stats['ann_ret']*100:+.1f}%",
          f"{(stats['ann_ret']-naive_stats['ann_ret'])*100:+.1f}% vs Naïve")
c4.metric("Best Year",           f"{_best_val*100:+.1f}%", f"{_best_yr}")
c5.metric("Worst Year",          f"{_wrst_val*100:+.1f}%", f"{_wrst_yr}")
c6.metric("Max Drawdown",        f"{stats['max_dd']*100:.1f}%")
c7.metric("Avg Turnover / mo",   f"{primary_df['turnover'].mean()*100:.1f}%")

ci1,ci2,ci3,ci4,ci5,ci6 = st.columns(6)
ci1.metric("Sharpe (Net)",  f"{stats['sharpe']:.2f}",  f"{stats['sharpe']-g_stats['sharpe']:+.2f} gross→net")
ci2.metric("Sortino (Net)", f"{stats['sortino']:.2f}")
ci3.metric("Calmar Ratio",  f"{stats['calmar']:.2f}")
ci4.metric("Hit Rate",      f"{stats['hit_rate']*100:.1f}%")
_mom_ic_label = "✓ tradeable" if abs(meta["icir_mom"]) > 0.5 else "○ borderline"
_qual_ic_note = (
    "normal for slow factor" if float(ic_q.mean()) < 0
    else ("✓ positive" if float(ic_q.mean()) > 0.01 else "○ near zero")
)
ci5.metric("Mom IC (1m fwd)",  f"{ic_m.mean():+.4f}", _mom_ic_label)
ci6.metric("Qual IC (1m fwd)", f"{ic_q.mean():+.4f}", _qual_ic_note)

# ── Model Decisions panel ────────────────────────────────────────────────────
_weight_src   = meta.get("weight_source", "IC-derived")
_auto_mode    = (_weight_src == "IC-derived")
_ic_mom_v   = float(ic_m.mean())
_ic_qual_v  = float(ic_q.mean())
_icir_mom_v = float(meta["icir_mom"])
_icir_qual_v= float(meta["icir_qual"])
_w_mom_v    = float(meta["w_mom"])
_w_qual_v   = float(meta["w_qual"])
_train_months = train_w

_ic_derived_mom_pct = _icir_mom_v**2 / (_icir_mom_v**2 + _icir_qual_v**2 + 1e-8)
_ic_derived_qual_pct = 1 - _ic_derived_mom_pct

if _auto_mode:
    _blend_explain = (
        f"You chose Auto mode — the model calculated the optimal blend itself. "
        f"During the {_train_months}-month training window it measured: "
        f"Momentum IC = {_ic_mom_v:+.4f}, Quality IC = {_ic_qual_v:+.4f}. "
        f"Using IC² weighting, it decided: "
        f"Momentum {_w_mom_v:.0%} / Quality {_w_qual_v:.0%}. "
        f"This is the statistically optimal blend for this dataset and period. "
        f"This blend is then held fixed for every month of the out-of-sample period."
    )
    _override_note = ""
else:
    _blend_explain = (
        f"You chose Manual override — signal weights were forced to "
        f"Momentum {_w_mom_v:.0%} / Quality {_w_qual_v:.0%}. "
        f"The IC calculation ran but its output was ignored."
    )
    _override_note = (
        f"If Auto mode were used, the model would have chosen "
        f"Momentum {_ic_derived_mom_pct:.0%} / Quality {_ic_derived_qual_pct:.0%} "
        f"(Mom IC={_ic_mom_v:+.4f}, Qual IC={_ic_qual_v:+.4f}). "
        f"Your override {'increased' if _w_mom_v > _ic_derived_mom_pct else 'decreased'} "
        f"Momentum weight by {abs(_w_mom_v - _ic_derived_mom_pct)*100:.0f} percentage points."
    )

_quality_note = (
    "Quality IC being slightly negative is normal — quality is a slow factor that "
    "predicts returns over 1–3 years, not monthly. It works as a stock filter, not a timing signal."
    if _ic_qual_v < 0 else
    "Both signals show positive IC, meaning both are adding predictive value."
)

_icir_interpret = ""
if abs(_icir_mom_v) > 0.5:
    _icir_interpret = "Momentum ICIR > 0.5 — considered tradeable at institutional funds."
elif abs(_icir_mom_v) > 0.3:
    _icir_interpret = "Momentum ICIR borderline — signal exists but is weak."
else:
    _icir_interpret = "Momentum ICIR < 0.3 — signal is weak in this period. Longer training window may help."

st.markdown(
    f"<div style='background:{PANEL};border:1px solid {BORDER};border-left:3px solid {CORAL};"
    f"border-radius:8px;padding:14px 20px;margin-bottom:4px'>"
    f"<div style='font-size:.68rem;color:{MUTED};text-transform:uppercase;"
    f"letter-spacing:.1em;margin-bottom:10px'>What the Model Decided  "
    f"<span style='color:{GOLD};text-transform:none;letter-spacing:0'>"
    f"({'Model decided — Auto mode' if _auto_mode else 'You overrode — Manual mode'})"
    f"</span></div>"
    f"<div style='display:flex;gap:32px;flex-wrap:wrap'>"

    f"<div style='min-width:220px'>"
    f"<div style='font-size:.7rem;color:{GOLD};font-weight:600;margin-bottom:6px'>"
    f"Signal Blend Decision</div>"
    f"<div style='font-size:.78rem;color:{CREAM};line-height:1.7'>{_blend_explain}</div>"
    + (f"<div style='font-size:.72rem;color:{MUTED};margin-top:6px;font-style:italic'>{_override_note}</div>" if _override_note else "")
    + f"</div>"

    f"<div style='min-width:200px'>"
    f"<div style='font-size:.7rem;color:{GOLD};font-weight:600;margin-bottom:6px'>"
    f"Signal Quality</div>"
    f"<div style='font-size:.78rem;color:{CREAM};line-height:1.7'>{_quality_note}</div>"
    f"<div style='font-size:.72rem;color:{MUTED};margin-top:6px'>{_icir_interpret}</div>"
    f"</div>"

    f"<div style='min-width:160px'>"
    f"<div style='font-size:.7rem;color:{GOLD};font-weight:600;margin-bottom:6px'>"
    f"Training → Testing Split</div>"
    f"<div style='font-size:.78rem;color:{CREAM};line-height:1.7'>"
    f"First {_train_months} months: signal learning only, no returns generated.<br>"
    f"Remaining {_total_months} months: out-of-sample, monthly rebalancing.<br>"
    f"Total period: {actual_start} – {actual_end}."
    f"</div></div>"

    f"</div></div>",
    unsafe_allow_html=True,
)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Performance", "Monthly Returns", "Signal Analysis",
                "Holdings & Sectors", "Statistics"])

# ── TAB 0: Performance ────────────────────────────────────────────────────────
with tabs[0]:
    col_chart, col_cards = st.columns([4, 1])
    with col_chart:
        fig = go.Figure()
        for lam, df in backtest_results.items():
            c = CLR.get(lam, MUTED)
            cum_g = (1 + df["gross_ret"]).cumprod()
            cum_n = (1 + df["net_ret"]).cumprod()
            fig.add_trace(go.Scatter(x=cum_g.index, y=cum_g.values,
                line=dict(color=c, width=1, dash="dot"), opacity=0.3, showlegend=False,
                hovertemplate=f"λ={lam} Gross  %{{x|%b %Y}}<br>%{{y:.2f}}×<extra></extra>"))
            fig.add_trace(go.Scatter(x=cum_n.index, y=cum_n.values,
                line=dict(color=c, width=2), name=f"λ={lam}",
                hovertemplate=f"λ={lam} Net  %{{x|%b %Y}}<br>%{{y:.2f}}×<extra></extra>"))
        cum_naive = (1 + naive_returns).cumprod()
        fig.add_trace(go.Scatter(x=cum_naive.index, y=cum_naive.values, name="Naïve",
            line=dict(color=MUTED, width=1.5, dash="dash"),
            hovertemplate="Naïve  %{x|%b %Y}<br>%{y:.2f}×<extra></extra>"))
        if len(_bm_m_full) > 5:
            common = primary_df.index.intersection(_bm_m_full.index)
            if len(common) > 5:
                cum_bm = (1 + _bm_m_full.loc[common]).cumprod()
                fig.add_trace(go.Scatter(x=cum_bm.index, y=cum_bm.values, name="Nifty 50",
                    line=dict(color=WHITE, width=1, dash="dashdot"), opacity=0.35,
                    hovertemplate="Nifty 50  %{x|%b %Y}<br>%{y:.2f}×<extra></extra>"))
        fig.add_hline(y=1, line_color=BORDER, line_width=0.8)
        fig.update_layout(**plotly_layout(
            title=f"Cumulative Return  —  {actual_start} to {actual_end}",
            title_font=dict(size=13, color=CREAM), height=420,
            yaxis_tickformat=".2f", yaxis_ticksuffix="×"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    with col_cards:
        st.markdown(
            f"<div style='font-size:.68rem;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:.08em;margin-bottom:8px'>Period Summary</div>",
            unsafe_allow_html=True)
        # Nifty card first
        if len(_bm_m_full) > 5:
            _bm_common = primary_df.index.intersection(_bm_m_full.index)
            if len(_bm_common) > 5:
                _bm_total = float((1 + _bm_m_full.loc[_bm_common]).prod() - 1)
                _bm_ann   = (1 + _bm_m_full.loc[_bm_common]).prod() ** (12 / len(_bm_common)) - 1
                st.markdown(
                    f"<div style='background:{PANEL};border:1px solid {BORDER};"
                    f"border-left:3px solid {MUTED};border-radius:6px;"
                    f"padding:9px 12px;margin-bottom:7px'>"
                    f"<div style='font-size:.68rem;color:{MUTED};"
                    f"font-family:JetBrains Mono,monospace'>Nifty 50 B&H</div>"
                    f"<div style='font-size:1.05rem;color:{CREAM};"
                    f"font-family:JetBrains Mono,monospace;font-weight:500'>"
                    f"{_bm_ann*100:+.1f}% ann</div>"
                    f"<div style='font-size:.7rem;color:{MUTED}'>Benchmark</div>"
                    f"<div style='font-size:.68rem;color:{MUTED}'>"
                    f"Total {_bm_total*100:+.1f}%</div></div>",
                    unsafe_allow_html=True)
        for lam, df in backtest_results.items():
            s = compute_stats(df["net_ret"]); c = CLR.get(lam, MUTED)
            cum_t = (1 + df["net_ret"]).prod() - 1
            _active_border = f"border:2px solid {c}" if lam == primary_lam else f"border:1px solid {BORDER}"
            st.markdown(
                f"<div style='background:{PANEL};{_active_border};"
                f"border-left:3px solid {c};border-radius:6px;"
                f"padding:9px 12px;margin-bottom:7px'>"
                f"<div style='font-size:.68rem;color:{MUTED};"
                f"font-family:JetBrains Mono,monospace'>λ = {lam}"
                f"{'  ◀ viewing' if lam == primary_lam else ''}</div>"
                f"<div style='font-size:1.05rem;color:{CREAM};"
                f"font-family:JetBrains Mono,monospace;font-weight:500'>"
                f"{s['ann_ret']*100:+.1f}% ann</div>"
                f"<div style='font-size:.7rem;color:{MUTED}'>"
                f"Sharpe {s['sharpe']:.2f}  ·  DD {s['max_dd']*100:.1f}%</div>"
                f"<div style='font-size:.68rem;color:{MUTED}'>"
                f"Total {cum_t*100:+.1f}%</div></div>",
                unsafe_allow_html=True)

    col_dd, col_rs = st.columns(2)
    with col_dd:
        fig_dd = go.Figure()
        for lam, df in backtest_results.items():
            c = CLR.get(lam, MUTED); cum = (1+df["net_ret"]).cumprod()
            dd = (cum/cum.cummax()-1)*100
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=f"λ={lam}",
                line=dict(color=c, width=1.5),
                fill="tozeroy" if lam == primary_lam else None,
                fillcolor=rgba(c, 0.09),
                hovertemplate=f"λ={lam}  %{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>"))
        fig_dd.update_layout(**plotly_layout(
            title="Drawdown Profile (Net)",
            title_font=dict(size=12, color=CREAM), height=290, yaxis_ticksuffix="%"))
        st.plotly_chart(fig_dd, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    with col_rs:
        rs = stats["rolling_sharpe"].dropna()
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=rs.index, y=rs.values, name="Rolling 12m Sharpe",
            line=dict(color=GOLD, width=1.8), fill="tozeroy",
            fillcolor=rgba(GOLD, 0.08),
            hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>"))
        fig_rs.add_hline(y=0, line_color=BORDER, line_width=0.8)
        fig_rs.add_hline(y=1, line_color=GREEN, line_width=0.7, line_dash="dot",
                         annotation_text="1.0", annotation_font_color=GREEN,
                         annotation_font_size=9)
        fig_rs.update_layout(**plotly_layout(
            title=f"Rolling 12-Month Sharpe  (λ={primary_lam}, Net)",
            title_font=dict(size=12, color=CREAM), height=290))
        st.plotly_chart(fig_rs, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    col_sh, col_tov = st.columns(2)
    lv    = list(backtest_results.keys())
    gs_v  = [compute_stats(backtest_results[l]["gross_ret"])["sharpe"] for l in lv]
    ns_v  = [compute_stats(backtest_results[l]["net_ret"])["sharpe"]   for l in lv]
    tov_v = [backtest_results[l]["turnover"].mean() * 100              for l in lv]
    with col_sh:
        fig_sh = go.Figure()
        fig_sh.add_trace(go.Bar(x=[f"λ={l}" for l in lv], y=gs_v, name="Gross",
            marker_color=ORANGE, opacity=0.75,
            hovertemplate="λ=%{x}<br>Gross Sharpe: %{y:.2f}<extra></extra>"))
        fig_sh.add_trace(go.Bar(x=[f"λ={l}" for l in lv], y=ns_v, name="Net",
            marker_color=GREEN, opacity=0.75,
            hovertemplate="λ=%{x}<br>Net Sharpe: %{y:.2f}<extra></extra>"))
        for i, (g, n) in enumerate(zip(gs_v, ns_v)):
            mid_y = (g + n) / 2
            fig_sh.add_annotation(
                x=f"λ={lv[i]}", y=mid_y,
                text=f"gap {g-n:.2f}",
                font=dict(color=WHITE, size=9, family="JetBrains Mono"),
                bgcolor=rgba(CORAL, 0.75), borderpad=2,
                showarrow=False)
        fig_sh.update_layout(**plotly_layout(
            title="Gross vs Net Sharpe by λ",
            title_font=dict(size=12, color=CREAM), height=280, barmode="group"))
        st.plotly_chart(fig_sh, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    with col_tov:
        fig_tov = go.Figure(go.Bar(
            x=[f"λ={l}" for l in lv], y=tov_v,
            marker_color=[CLR.get(l, MUTED) for l in lv],
            marker_line_color=BG, marker_line_width=1,
            text=[f"{t:.1f}%" for t in tov_v],
            textposition="auto",
            textfont=dict(color=BG, size=11, family="JetBrains Mono"),
            hovertemplate="λ=%{x}<br>Turnover: %{y:.1f}%/mo<extra></extra>",
            constraintext="none"))
        _tov_max = max(tov_v) if tov_v else 1
        fig_tov.update_layout(**plotly_layout(
            title="Average Monthly Turnover",
            title_font=dict(size=12, color=CREAM), height=280,
            yaxis_ticksuffix="%", showlegend=False,
            yaxis_range=[0, _tov_max * 1.25]))
        st.plotly_chart(fig_tov, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

# ── TAB 1: Monthly Returns ────────────────────────────────────────────────────
with tabs[1]:
    hm_lam = primary_lam   # uses the λ chosen in sidebar — change there
    ser = backtest_results[hm_lam]["net_ret"].copy()
    ser.index = pd.to_datetime(ser.index)
    piv = ser.groupby([ser.index.year, ser.index.month]).sum().unstack(1)
    piv.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    v    = piv.values.astype(float) * 100
    vmax = float(np.nanpercentile(np.abs(v[~np.isnan(v)]), 95)) if v.size > 0 else 5.0
    fig_hm = go.Figure(go.Heatmap(
        z=v, x=list(piv.columns), y=[str(yr) for yr in piv.index],
        text=[[f"{v[i,j]:.1f}%" if not np.isnan(v[i,j]) else ""
               for j in range(v.shape[1])] for i in range(v.shape[0])],
        texttemplate="%{text}", textfont=dict(size=10, family="JetBrains Mono", color=WHITE),
        colorscale=[[0, "#CC3B2E"], [0.35, "#6B2020"], [0.5, "#1a1a1a"], [0.65, "#1a4a2a"], [1, "#2E8B4A"]],
        zmid=0, zmin=-vmax, zmax=vmax,
        hovertemplate="%{y}  %{x}<br>Return: %{z:.2f}%<extra></extra>",
        showscale=True,
        colorbar=dict(ticksuffix="%", tickfont=dict(color=MUTED, size=10),
                      outlinecolor=BORDER, outlinewidth=1),
    ))
    fig_hm.update_layout(**plotly_layout(
        title=f"Monthly Net Returns  (λ={hm_lam})",
        title_font=dict(size=13, color=CREAM),
        height=max(320, len(piv) * 52 + 100), xaxis_side="top"))
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
    ann_rows = []
    for yr, row in piv.iterrows():
        valid = row.dropna()
        if len(valid) == 0: continue
        ann_rows.append({
            "Year":            int(yr),
            "Ann Return":      f"{((1+valid/100).prod()-1)*100:+.1f}%",
            "Positive Months": f"{(valid>0).sum()} / {len(valid)}",
            "Best Month":      f"{valid.max():+.1f}%",
            "Worst Month":     f"{valid.min():+.1f}%",
            "Avg Monthly":     f"{valid.mean():+.2f}%",
        })
    st.dataframe(pd.DataFrame(ann_rows).set_index("Year"), use_container_width=True)

# ── TAB 2: Signal Analysis ────────────────────────────────────────────────────
with tabs[2]:
    col_q, col_ic = st.columns(2)
    with col_q:
        fwd = meta["fwd_ret"]; ca = composite.reindex(fwd.index)
        q_rets = {i: [] for i in range(1, 6)}
        for t in range(len(ca)):
            row = ca.iloc[t].dropna()
            if len(row) < 10: continue
            rr  = fwd.iloc[t].dropna(); co = row.index.intersection(rr.index)
            if len(co) < 10: continue
            lbl_ = pd.qcut(row[co], 5, labels=False, duplicates="drop")
            for q in range(5):
                m = lbl_ == q
                if m.sum() > 0: q_rets[q+1].append(rr[co][m].mean())
        qm = [np.mean(q_rets[q])*100 for q in range(1, 6)]
        ql = [np.std(q_rets[q])*100  for q in range(1, 6)]
        fig_q = go.Figure(go.Bar(
            x=["Q1 Short","Q2","Q3","Q4","Q5 Long"], y=qm,
            error_y=dict(type="data", array=ql, color=CREAM, thickness=1.2, width=5),
            marker_color=[CORAL,"#E08868",MUTED,"#7aab87",GREEN],
            marker_line_color=BG, marker_line_width=1,
            text=[f"{v:+.2f}%" for v in qm], textposition="outside",
            textfont=dict(color=CREAM, size=10, family="JetBrains Mono"),
            hovertemplate="%{x}<br>Avg Monthly: %{y:.3f}%<extra></extra>"))
        fig_q.add_hline(y=0, line_color=BORDER, line_width=0.8)
        fig_q.update_layout(**plotly_layout(
            title="Quintile Avg Monthly Returns ± 1σ",
            title_font=dict(size=12, color=CREAM), height=320,
            yaxis_ticksuffix="%", showlegend=False))
        st.plotly_chart(fig_q, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    with col_ic:
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(x=ic_m.index, y=ic_m.values,
            name=f"Momentum IC  (μ={ic_m.mean():+.3f})",
            line=dict(color=GOLD, width=1.5),
            hovertemplate="Mom IC  %{x|%b %Y}<br>%{y:.4f}<extra></extra>"))
        fig_ic.add_trace(go.Scatter(x=ic_q.index, y=ic_q.values,
            name=f"Quality IC  (μ={ic_q.mean():+.3f})",
            line=dict(color=CORAL, width=1.5),
            hovertemplate="Qual IC  %{x|%b %Y}<br>%{y:.4f}<extra></extra>"))
        if len(ic_m) > 6:
            rm = ic_m.rolling(6).mean()
            fig_ic.add_trace(go.Scatter(x=rm.index, y=rm.values,
                line=dict(color=GOLD, width=2.5, dash="dot"),
                opacity=0.45, showlegend=False))
        fig_ic.add_hline(y=0, line_color=BORDER, line_width=0.8)
        fig_ic.add_hline(y=0.05, line_color=GREEN, line_width=0.6, line_dash="dot",
                         annotation_text="IC=0.05", annotation_font_color=GREEN,
                         annotation_font_size=9)
        fig_ic.update_layout(**plotly_layout(
            title="IC vs 1-Month Forward Return",
            title_font=dict(size=12, color=CREAM), height=320))
        st.plotly_chart(fig_ic, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    col_dist, col_sc = st.columns(2)
    with col_dist:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=ic_m.dropna().values, name="Momentum IC",
            marker=dict(color=GOLD, line=dict(color=BG, width=0.8)),
            opacity=0.85, nbinsx=20, histnorm="probability",
            hovertemplate="Momentum IC: %{x:.3f}<br>Freq: %{y:.1%}<extra></extra>"))
        fig_dist.add_trace(go.Histogram(x=ic_q.dropna().values, name="Quality IC",
            marker=dict(color=CORAL, line=dict(color=BG, width=0.8)),
            opacity=0.85, nbinsx=20, histnorm="probability",
            hovertemplate="Quality IC: %{x:.3f}<br>Freq: %{y:.1%}<extra></extra>"))
        fig_dist.add_vline(x=0, line_color=MUTED, line_width=1,
                           annotation_text="IC=0", annotation_font_color=MUTED,
                           annotation_font_size=9)
        fig_dist.add_vline(x=0.05, line_color=GREEN, line_width=0.8, line_dash="dot",
                           annotation_text="IC=0.05", annotation_font_color=GREEN,
                           annotation_font_size=9)
        fig_dist.update_layout(**plotly_layout(
            title="IC Distribution — Momentum vs Quality",
            title_font=dict(size=12, color=CREAM),
            height=280, barmode="group", yaxis_tickformat=".0%"))
        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    with col_sc:
        common_idx = ic_m.index.intersection(ic_q.index)
        corr_val   = ic_m.loc[common_idx].corr(ic_q.loc[common_idx])
        fig_sc = go.Figure(go.Scatter(
            x=ic_m.loc[common_idx].values, y=ic_q.loc[common_idx].values,
            mode="markers",
            marker=dict(color=GOLD, size=6, opacity=0.5,
                        line=dict(color=BG, width=0.5)),
            hovertemplate="Mom: %{x:.3f}<br>Qual: %{y:.3f}<extra></extra>"))
        fig_sc.add_hline(y=0, line_color=BORDER, line_width=0.6)
        fig_sc.add_vline(x=0, line_color=BORDER, line_width=0.6)
        fig_sc.update_layout(**plotly_layout(
            title=f"Momentum IC vs Quality IC  (ρ={corr_val:.3f})",
            title_font=dict(size=12, color=CREAM), height=280,
            xaxis_title="Momentum IC", yaxis_title="Quality IC"))
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

# ── TAB 3: Holdings & Sectors ─────────────────────────────────────────────────
with tabs[3]:
    # weight_histories is stored separately in R (survives Streamlit caching)
    w_hist = R.get("weight_histories", {}).get(primary_lam, {})

    # Best stocks insight panel
    if w_hist:
        all_w_df  = pd.DataFrame(w_hist).T
        avg_long  = all_w_df.clip(lower=0).mean().sort_values(ascending=False)
        avg_short = all_w_df.clip(upper=0).mean().sort_values()
        top_longs  = avg_long[avg_long > 0.005].head(5)
        top_shorts = avg_short[avg_short < -0.005].head(5)

        long_html  = "".join([
            f"<div style='font-size:.8rem;color:{CREAM};"
            f"font-family:JetBrains Mono,monospace;margin-bottom:3px'>"
            f"{t.replace('.NS','')} "
            f"<span style='color:{MUTED}'>{w*100:.1f}% avg wt</span></div>"
            for t, w in top_longs.items()])
        short_html = "".join([
            f"<div style='font-size:.8rem;color:{CREAM};"
            f"font-family:JetBrains Mono,monospace;margin-bottom:3px'>"
            f"{t.replace('.NS','')} "
            f"<span style='color:{MUTED}'>{abs(w)*100:.1f}% avg wt</span></div>"
            for t, w in top_shorts.items()]) if False else ""
        short_section = (
            f"<div><div style='font-size:.72rem;color:{CORAL};font-weight:600;"
            f"margin-bottom:6px'>Consistent Shorts</div>{short_html}</div>"
        ) if False else ""

        explain = (
            "Fully invested in top-ranked stocks by momentum and quality. "
            "Returns come from owning better businesses at the right time."
            if long_only_bool else
            "Longs: high momentum, high quality. Shorts: low momentum, high leverage. "
            "Returns come from the spread between the two groups."
        )

        st.markdown(
            f"<div style='background:{PANEL};border:1px solid {BORDER};"
            f"border-radius:8px;padding:14px 20px;margin-bottom:16px'>"
            f"<div style='font-size:.68rem;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:.1em;margin-bottom:10px'>"
            f"Strategy Insight — {actual_start} to {actual_end}</div>"
            f"<div style='display:flex;gap:40px;flex-wrap:wrap'>"
            f"<div><div style='font-size:.72rem;color:{GREEN};font-weight:600;"
            f"margin-bottom:6px'>Consistent Longs</div>{long_html}</div>"
            f"{short_section}"
            f"<div style='border-left:1px solid {BORDER};padding-left:24px'>"
            f"<div style='font-size:.72rem;color:{GOLD};font-weight:600;"
            f"margin-bottom:6px'>What drives returns</div>"
            f"<div style='font-size:.78rem;color:{MUTED};max-width:280px;line-height:1.6'>"
            f"{explain}</div></div></div></div>",
            unsafe_allow_html=True,
        )

    col_wt, col_sec = st.columns([3, 2])
    with col_wt:
        if w_hist:
            sel_date = sorted(w_hist.keys())[-1]
            st.markdown(
                f"<div style='font-size:.78rem;color:{MUTED};"
                f"font-family:JetBrains Mono,monospace;margin-bottom:6px'>"
                f"Latest rebalance: "
                f"<span style='color:{CREAM}'>"
                f"{pd.Timestamp(sel_date).strftime('%B %Y')}"
                f"</span></div>",
                unsafe_allow_html=True,
            )
            latest_w_raw = pd.Series(w_hist[sel_date])
            latest_w     = latest_w_raw[latest_w_raw.abs() > 0.001].sort_values()
            longs        = (latest_w > 0).sum()
            shorts       = (latest_w < 0).sum()
            mode_info    = f"{longs} positions"
            invested_pct = latest_w_raw[latest_w_raw > 0].sum() * 100
            st.markdown(
                f"<div style='font-size:.75rem;color:{MUTED};"
                f"font-family:JetBrains Mono,monospace;margin-bottom:8px'>"
                f"{mode_info}  ·  Invested {invested_pct:.0f}%</div>",
                unsafe_allow_html=True,
            )
            fig_wt = go.Figure(go.Bar(
                x=latest_w.values * 100,
                y=[t.replace(".NS", "") for t in latest_w.index],
                orientation="h",
                marker_color=[GREEN if w > 0 else CORAL for w in latest_w.values],
                marker_line_color=BG, marker_line_width=0.5,
                text=[f"{w*100:+.1f}%" for w in latest_w.values],
                textposition="outside",
                textfont=dict(color=CREAM, size=9, family="JetBrains Mono"),
                hovertemplate="%{y}<br>Weight: %{x:.2f}%<extra></extra>"))
            fig_wt.add_vline(x=0, line_color=BORDER, line_width=0.8)
            fig_wt.update_layout(**plotly_layout(
                title=f"Portfolio Weights — {pd.Timestamp(sel_date).strftime('%B %Y')}",
                title_font=dict(size=12, color=CREAM),
                height=max(360, len(latest_w) * 22),
                xaxis_ticksuffix="%", showlegend=False))
            st.plotly_chart(fig_wt, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
            w_df = pd.DataFrame({
                "Ticker":   [t.replace(".NS", "") for t in latest_w.index],
                "Side":     ["Long" if w > 0 else "Short" for w in latest_w.values],
                "Weight %": [round(w * 100, 2) for w in latest_w.values],
                "Sector":   [SECTOR_MAP.get(t, "Other") for t in latest_w.index],
            }).sort_values("Weight %", ascending=False)
            st.dataframe(w_df.set_index("Ticker"), use_container_width=True, height=240)
            _weights_loaded = True
        else:
            _weights_loaded = False
            latest_w_raw    = pd.Series(dtype=float)

    with col_sec:
        if _weights_loaded:
            sec_exp = {}
            for tkr, w in latest_w_raw.items():
                sec = SECTOR_MAP.get(tkr, "Other")
                sec_exp[sec] = sec_exp.get(sec, 0) + w
            sec_s = pd.Series(sec_exp).sort_values()
            fig_sec = go.Figure(go.Bar(
                x=sec_s.values * 100, y=sec_s.index, orientation="h",
                marker_color=[GREEN if v > 0 else CORAL for v in sec_s.values],
                marker_line_color=BG, marker_line_width=0.5,
                text=[f"{v*100:+.1f}%" for v in sec_s.values],
                textposition="outside",
                textfont=dict(color=CREAM, size=10, family="JetBrains Mono"),
                hovertemplate="%{y}<br>Net: %{x:.1f}%<extra></extra>"))
            fig_sec.add_vline(x=0, line_color=BORDER, line_width=0.8)
            fig_sec.update_layout(**plotly_layout(
                title="Sector Net Exposure",
                title_font=dict(size=12, color=CREAM), height=320,
                xaxis_ticksuffix="%", showlegend=False))
            st.plotly_chart(fig_sec, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

            abs_sec = {k: abs(v) for k, v in sec_exp.items() if abs(v) > 0.003}
            if abs_sec:
                pie_clrs = [GOLD, CORAL, GREEN, ORANGE, "#9B8EA8",
                            "#6B9EC0","#C08060","#80A870","#A080B0"]
                fig_pie = go.Figure(go.Pie(
                    labels=list(abs_sec.keys()), values=list(abs_sec.values()),
                    marker=dict(colors=pie_clrs[:len(abs_sec)],
                                line=dict(color=BG, width=1.5)),
                    textfont=dict(color=WHITE, size=10, family="JetBrains Mono"),
                    hovertemplate="%{label}<br>%{percent}<extra></extra>", hole=0.38))
                _pl = dict(**_PLOTLY_BASE)
                _pl["legend"] = dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
                                     font=dict(color=CREAM, size=9))
                fig_pie.update_layout(**_pl, title="Sector Allocation (|Weight|)",
                                      title_font=dict(size=12, color=CREAM), height=300)
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

    # ── Holdings evolution over time ──────────────────────────────────────────
    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown("<div class='slabel'>How Holdings Changed Every Month</div>",
                unsafe_allow_html=True)
    st.caption(
        "Each column is one month. Each row is a stock. "
        "Green = long position, Red = short/zero. "
        "Darker = larger weight. Hover for exact value."
    )

    if w_hist:
        all_w_df = pd.DataFrame(w_hist).T
        all_w_df.index = pd.to_datetime(all_w_df.index)
        all_w_df.columns = [c.replace(".NS","") for c in all_w_df.columns]

        # Only show stocks that ever had a meaningful weight
        ever_held = all_w_df.columns[all_w_df.abs().max() > 0.005]
        plot_df   = all_w_df[ever_held] * 100  # convert to %

        # Sort stocks by average long weight descending
        sort_order = plot_df.clip(lower=0).mean().sort_values(ascending=False).index
        plot_df    = plot_df[sort_order]

        # Convert to rank-based score per month (0=lowest weight, 1=highest)
        # This makes the colour scale meaningful regardless of max_pos value
        # Every cell shows where the stock ranked that month, not its absolute weight
        rank_df = plot_df.apply(
            lambda row: row.rank(pct=True) if row.abs().sum() > 0.01 else row * 0,
            axis=1
        )
        # Stocks with zero/dust weight get rank=0 (shown as dark background)
        rank_df[plot_df.abs() < 0.15] = np.nan

        # customdata shape must match z shape (stocks × months after .T)
        # Each cell stores [stock_name, month_str, actual_weight, rank_str]
        n_months = len(plot_df)
        n_stocks = len(plot_df.columns)
        custom = np.empty((n_stocks, n_months), dtype=object)
        for si, stock in enumerate(plot_df.columns):
            for ti, dt in enumerate(plot_df.index):
                w_val = plot_df.iloc[ti, si]
                r_val = rank_df.iloc[ti, si]
                rank_str = (f"{int(r_val * n_stocks)}/{n_stocks}"
                            if not np.isnan(r_val) else "Not held")
                custom[si, ti] = (
                    f"<b>{stock}</b><br>"
                    f"{dt.strftime('%B %Y')}<br>"
                    f"Weight: {w_val:+.2f}%<br>"
                    f"Rank: {rank_str}"
                )

        fig_evo = go.Figure(go.Heatmap(
            z=rank_df.values.T,
            x=[d.strftime("%b %Y") for d in plot_df.index],
            y=list(plot_df.columns),
            colorscale=[
                [0.0, BG],
                [0.4, "#1a2e1a"],
                [0.7, "#2d6e2d"],
                [1.0, "#00A550"],
            ],
            zmin=0, zmax=1,
            customdata=custom,
            hovertemplate="%{customdata}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Rank within portfolio", font=dict(color=MUTED, size=9)),
                tickvals=[0, 0.5, 1],
                ticktext=["Not held", "Mid rank", "Top rank"],
                tickfont=dict(color=MUTED, size=9),
                outlinecolor=BORDER, outlinewidth=1,
            ),
        ))
        fig_evo.update_layout(**plotly_layout(
            title=(f"Monthly Holdings — {actual_start} to {actual_end}  ·  "
                   f"λ={primary_lam}  ·  {len(ever_held)} stocks ever held  ·  "
                   f"Colour = rank within portfolio (hover for exact weight)"),
            title_font=dict(size=11, color=CREAM),
            height=max(420, len(ever_held) * 18 + 120),
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                       tickfont=dict(color=MUTED, size=8), tickangle=-45),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                       tickfont=dict(color=MUTED, size=9), autorange="reversed"),
        ))
        st.plotly_chart(fig_evo, use_container_width=True,
                        config={'displayModeBar': True, 'responsive': True})
        st.caption("Darker green = higher rank within that month's portfolio. "
                   "White/empty = stock not held. Hover any cell for exact weight and rank.")

        # ── Top stocks by time-in-portfolio ──────────────────────────────────
        st.markdown("<div class='slabel' style='margin-top:8px'>Stock Appearance Summary</div>",
                    unsafe_allow_html=True)
        months_held = (all_w_df[ever_held].abs() > 0.005).sum()
        avg_wt      = all_w_df[ever_held].clip(lower=0).mean()
        peak_wt     = all_w_df[ever_held].clip(lower=0).max()
        sector_col  = pd.Series({c: SECTOR_MAP.get(c+".NS","Other") for c in ever_held})

        summary_df = pd.DataFrame({
            "Months Held":  months_held,
            "Avg Weight %": (avg_wt * 100).round(1),
            "Peak Weight %":(peak_wt * 100).round(1),
            "Sector":       sector_col,
        }).sort_values("Months Held", ascending=False)
        summary_df["Time in Portfolio"] = summary_df["Months Held"].apply(
            lambda x: f"{x}/{_total_months} months ({x/_total_months*100:.0f}%)")
        st.dataframe(
            summary_df[["Sector","Time in Portfolio","Avg Weight %","Peak Weight %"]],
            use_container_width=True, height=280,
        )

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown("<div class='slabel'>Return Correlation — Strategy vs Every Stock</div>",
                unsafe_allow_html=True)
    ret_m    = prices.resample("ME").last().pct_change().dropna()
    combined = ret_m.copy()
    combined["Strategy"] = primary_df["net_ret"].reindex(combined.index)
    corr_v = (combined[["Strategy"] + list(ret_m.columns)]
              .dropna()
              .corr().loc["Strategy", ret_m.columns].sort_values())
    fig_corr = go.Figure(go.Bar(
        x=[t.replace(".NS", "") for t in corr_v.index],
        y=corr_v.values,
        marker_color=[GREEN if v >= 0 else CORAL for v in corr_v.values],
        marker_line_color=BG, marker_line_width=0.5,
        text=[f"{v:.2f}" for v in corr_v.values],
        textposition="outside",
        textfont=dict(color=CREAM, size=8, family="JetBrains Mono"),
        hovertemplate="%{x}<br>Correlation: %{y:.3f}<extra></extra>"))
    fig_corr.add_hline(y=0, line_color=BORDER, line_width=0.8)
    fig_corr.update_layout(**plotly_layout(
        title=f"Strategy–Stock Return Correlation  (all {len(corr_v)} stocks)",
        title_font=dict(size=12, color=CREAM), height=320,
        xaxis_tickangle=-45, showlegend=False))
    st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
    st.caption(
        "Green = moves with strategy (strategy overweights these).  "
        "Red = moves against strategy (strategy underweights / avoids these).")

# ── TAB 4: Statistics ─────────────────────────────────────────────────────────
with tabs[4]:
    all_rows = []
    for lam, df in backtest_results.items():
        for label, ser in [("Gross", df["gross_ret"]), ("Net", df["net_ret"])]:
            s = compute_stats(ser)
            all_rows.append({
                "Strategy":   f"λ={lam} {label}",
                "Period":     f"{actual_start} – {actual_end}",
                "Ann Return": f"{s['ann_ret']*100:+.2f}%",
                "Ann Vol":    f"{s['ann_vol']*100:.2f}%",
                "Sharpe":     round(s["sharpe"],  3),
                "Sortino":    round(s["sortino"], 3),
                "Calmar":     round(s["calmar"],  3),
                "Max DD":     f"{s['max_dd']*100:.2f}%",
                "Hit Rate":   f"{s['hit_rate']*100:.1f}%",
            })
    s = compute_stats(naive_returns)
    all_rows.append({"Strategy": "Naïve Decile", "Period": f"{actual_start} – {actual_end}",
                     "Ann Return": f"{s['ann_ret']*100:+.2f}%", "Ann Vol": f"{s['ann_vol']*100:.2f}%",
                     "Sharpe": round(s["sharpe"],3), "Sortino": round(s["sortino"],3),
                     "Calmar": round(s["calmar"],3), "Max DD": f"{s['max_dd']*100:.2f}%",
                     "Hit Rate": f"{s['hit_rate']*100:.1f}%"})
    # Add Nifty 50 benchmark row
    if len(_bm_m_full) > 5:
        _bm_c = primary_df.index.intersection(_bm_m_full.index)
        if len(_bm_c) > 5:
            bm_s = compute_stats(_bm_m_full.loc[_bm_c])
            all_rows.append({"Strategy": "Nifty 50 B&H",
                             "Period": f"{actual_start} – {actual_end}",
                             "Ann Return": f"{bm_s['ann_ret']*100:+.2f}%",
                             "Ann Vol": f"{bm_s['ann_vol']*100:.2f}%",
                             "Sharpe": round(bm_s["sharpe"],3),
                             "Sortino": round(bm_s["sortino"],3),
                             "Calmar": round(bm_s["calmar"],3),
                             "Max DD": f"{bm_s['max_dd']*100:.2f}%",
                             "Hit Rate": f"{bm_s['hit_rate']*100:.1f}%"})
    st.dataframe(pd.DataFrame(all_rows).set_index("Strategy"), use_container_width=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown("<div class='slabel'>Signal Summary</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Signal": name, "Mean IC": round(ic_s.mean(),4), "IC Std": round(ic_s.std(),4),
         "ICIR": round(icir,3), "% Positive": f"{(ic_s>0).mean()*100:.1f}%",
         "Blend Weight": f"{meta['w_mom' if name=='Momentum' else 'w_qual']:.1%}",
         "Status": "✓ Tradeable" if abs(icir) > 0.5 else "○ Borderline"}
        for name, ic_s, icir in [
            ("Momentum", meta["ic_mom"],  meta["icir_mom"]),
            ("Quality",  meta["ic_qual"], meta["icir_qual"]),
        ]
    ]).set_index("Signal"), use_container_width=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)
st.markdown(
    f"<p style='font-size:.68rem;color:{BORDER};font-family:JetBrains Mono,"
    f"monospace;text-align:center'>"
    "NSE Factor Dashboard  ·  Momentum (Jegadeesh-Titman 1993)  ·  "
    "Quality (Novy-Marx 2013)  ·  IC² Signal Blend  ·  OSQP</p>",
    unsafe_allow_html=True,
)
