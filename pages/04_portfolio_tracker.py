import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from io import StringIO
from datetime import date, timedelta
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio vs Benchmark",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.stApp {
    background: #0d0d0d;
    color: #e8e0d0;
}

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}

/* Cards */
.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 500;
    color: #e8e0d0;
    font-family: 'DM Serif Display', serif;
}

.metric-value.positive { color: #7cfc7c; }
.metric-value.negative { color: #fc7c7c; }

/* Header strip */
.header-strip {
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}

.ticker-badge {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 2px;
    padding: 2px 8px;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #aaa;
    margin: 2px;
}

.skipped-list {
    font-size: 0.72rem;
    color: #555;
    line-height: 1.8;
}

/* Input overrides */
div[data-testid="stFileUploader"] {
    border: 1px dashed #333;
    border-radius: 4px;
    padding: 0.5rem;
}

.stButton > button {
    background: #e8e0d0 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 2rem !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    background: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ── Ticker resolution ─────────────────────────────────────────────────────────

def resolve_ticker(raw: str) -> str | None:
    return raw.upper()


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of Yahoo tickers."""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all")


def build_portfolio_index(
    weights: dict[str, float],
    prices: pd.DataFrame,
    alloc_date: pd.Timestamp,
) -> pd.Series:
    """
    Compute normalised portfolio value (base=100) from alloc_date.
    Weights that had no price data are excluded (re-normalised).
    """
    # Filter to tickers present in prices columns
    valid = {t: w for t, w in weights.items() if t in prices.columns}
    if not valid:
        return pd.Series(dtype=float)

    # Slice from alloc_date onward, drop columns that are entirely NaN
    px = prices.loc[alloc_date:, list(valid.keys())].copy()
    px = px.dropna(axis=1, how="all")   # drop tickers with zero data in range
    px = px.dropna(how="all")           # drop rows where everything is NaN

    if px.empty or len(px) == 0:
        return pd.Series(dtype=float)

    # Keep only tickers that have at least one valid price row
    valid = {t: w for t, w in valid.items() if t in px.columns}
    if not valid:
        return pd.Series(dtype=float)

    # Re-normalise weights after dropping bad tickers
    total_w = sum(valid.values())
    norm_w = {t: w / total_w for t, w in valid.items()}

    # Forward-fill within each column so intra-series gaps don't break things
    px = px[list(norm_w.keys())].ffill()

    # Drop any remaining rows where ALL tickers are still NaN (e.g. leading NaNs)
    px = px.dropna(how="all")
    if px.empty:
        return pd.Series(dtype=float)

    # Normalise each ticker to 1.0 on its first available price
    base = px.iloc[0]
    base = base.replace(0, float("nan"))   # avoid division by zero
    rel = px / base
    portfolio = sum(rel[t] * w for t, w in norm_w.items())
    return (portfolio * 100).rename("Portfolio")


# ── Parse TSV ────────────────────────────────────────────────────────────────

def parse_portfolio_tsv(text: str) -> tuple[dict[str, float], list[str]]:
    """
    Returns:
      weights   : { yahoo_ticker: weight_fraction }
      skipped   : list of raw tickers that were skipped
    """
    weights: dict[str, float] = {}
    skipped: list[str] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        raw_ticker = parts[0]
        pct_str = parts[-1].replace("%", "")
        try:
            pct = float(pct_str)
        except ValueError:
            continue

        if pct == 0:
            skipped.append(f"{raw_ticker} (0%)")
            continue

        yahoo = resolve_ticker(raw_ticker)
        if yahoo is None:
            skipped.append(f"{raw_ticker} (unresolvable)")
            continue

        weights[yahoo] = pct / 100.0

    return weights, skipped


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-strip">
<h1 style="margin:0; font-size:2.2rem; letter-spacing:-0.01em;">Portfolio vs Benchmark</h1>
<p style="color:#555; margin:0.3rem 0 0; font-size:0.8rem; letter-spacing:0.05em;">
  PERFORMANCE COMPARISON · NSE / BSE PORTFOLIOS
</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ① Portfolio")

    input_method = st.radio("Input method", ["Paste TSV", "Upload file"], label_visibility="collapsed")

    tsv_text = ""
    if input_method == "Paste TSV":
        tsv_text = st.text_area(
            "Paste your TSV here",
            height=280,
            value="""BAJFINANCE.NS	0.01%
ABBOTINDIA.NS	0.01%
BHARATFORG.NS	3.00%
NMDC.NS	2.00%
HINDCOPPER.NS	2.00%
DATAPATTNS.NS	2.00%
BEL.NS	5.00%
ASTRAMICRO.NS	2.00%
DYNAMATECH.NS	2.00%
KSB.NS	2.00%
NTPC.NS	3.00%
PFC.NS	2.00%
COALINDIA.NS	2.00%
ONGC.NS	2.00%
SBIN.NS	4.00%
ICICIBANK.NS	2.00%
LT.NS	5.00%
SYRMA.NS	3.00%
MOTHERSON.NS	3.00%
OBEROIRLTY.NS	3.00%
GLENMARK.NS	2.00%
SUNPHARMA.NS	4.00%
APOLLOHOSP.NS	3.00%
MAXHEALTH.NS	3.00%
3BBLACKBIO.BO	2.0%
BHARTIARTL.NS	4.00%
INDIGO.NS	2.00%
MARUTI.NS	4.00%
M&M.NS	3.00%
COROMANDEL.NS	5.00%
TATACONSUM.NS	4.0%""",
        )
    else:
        uploaded = st.file_uploader("Upload TSV / TXT", type=["tsv", "txt", "csv"])
        if uploaded:
            tsv_text = StringIO(uploaded.read().decode("utf-8")).read()

    st.markdown("### ② Dates")
    alloc_date = st.date_input(
        "Allocation / Start date",
        value=date.today() - timedelta(days=365),
        max_value=date.today() - timedelta(days=1),
    )
    end_date = st.date_input(
        "End date",
        value=date.today(),
        min_value=alloc_date + timedelta(days=1),
        max_value=date.today(),
    )

    st.markdown("### ③ Benchmark")
    benchmark_ticker = st.text_input(
        "Yahoo Finance ticker",
        value="^NSEI",
        help="e.g. ^NSEI (Nifty 50), ^BSESN (Sensex), AAPL, SPY …",
    )

    run = st.button("⟶  Run Comparison")

# ── Main panel ────────────────────────────────────────────────────────────────

if not run:
    st.markdown("""
    <div style="text-align:center; padding: 6rem 2rem; color:#333;">
      <div style="font-size:4rem; margin-bottom:1rem;">↖</div>
      <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#444;">
        Configure your inputs in the sidebar, then click Run Comparison.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not tsv_text.strip():
    st.error("Please paste or upload your portfolio TSV first.")
    st.stop()

# Parse portfolio
with st.spinner("Parsing portfolio…"):
    weights, skipped = parse_portfolio_tsv(tsv_text)

if not weights:
    st.error("No valid tickers found. Please check your TSV format.")
    st.stop()

all_tickers = list(weights.keys()) + ([benchmark_ticker] if benchmark_ticker else [])

# Fetch prices
with st.spinner(f"Fetching prices for {len(all_tickers)} instruments…"):
    prices = fetch_prices(
        all_tickers,
        start=alloc_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

if prices.empty:
    st.error("Could not fetch any price data. Check your date range or ticker symbols.")
    st.stop()

# Align to first common date after alloc_date
alloc_ts = pd.Timestamp(alloc_date)
available_dates = prices.index[prices.index >= alloc_ts]
if available_dates.empty:
    st.error("No price data available from the allocation date onwards.")
    st.stop()
first_date = available_dates[0]

# Build portfolio series
portfolio_series = build_portfolio_index(weights, prices, first_date)

if portfolio_series.empty:
    st.error("Could not construct portfolio — no matching price data.")
    st.stop()

# Build benchmark series
bench_series = None
missing_bench = False
if benchmark_ticker and benchmark_ticker in prices.columns:
    bp = prices.loc[first_date:, benchmark_ticker].dropna()
    if not bp.empty:
        bench_series = (bp / bp.iloc[0] * 100).rename(benchmark_ticker)
else:
    missing_bench = True

# ── Metrics ──────────────────────────────────────────────────────────────────

def pct_return(series: pd.Series) -> float:
    return (series.iloc[-1] / series.iloc[0] - 1) * 100

def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    return dd.min() * 100

def annualised_vol(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    return returns.std() * np.sqrt(252) * 100

port_ret = pct_return(portfolio_series)
port_dd = max_drawdown(portfolio_series)
port_vol = annualised_vol(portfolio_series)

col1, col2, col3, col4 = st.columns(4)

def metric_html(label, value, suffix="", positive_is_good=True):
    cls = ""
    if suffix == "%":
        if positive_is_good:
            cls = "positive" if value >= 0 else "negative"
        else:
            cls = "negative" if value < 0 else "positive"
    val_str = f"{value:+.2f}{suffix}" if suffix == "%" else f"{value:.2f}{suffix}"
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{val_str}</div>
    </div>"""

with col1:
    st.markdown(metric_html("Portfolio Return", port_ret, "%"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_html("Max Drawdown", port_dd, "%", positive_is_good=False), unsafe_allow_html=True)
with col3:
    st.markdown(metric_html("Ann. Volatility", port_vol, "%", positive_is_good=False), unsafe_allow_html=True)
with col4:
    if bench_series is not None:
        diff = port_ret - pct_return(bench_series)
        st.markdown(metric_html("Alpha vs Benchmark", diff, "%"), unsafe_allow_html=True)
    else:
        st.markdown(metric_html("Holdings", len(weights), ""), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────

fig = go.Figure()

# Portfolio line
fig.add_trace(go.Scatter(
    x=portfolio_series.index,
    y=portfolio_series.values,
    name="Portfolio",
    line=dict(color="#e8c96a", width=2.5),
    hovertemplate="%{x|%d %b %Y}<br>Portfolio: <b>%{y:.1f}</b><extra></extra>",
))

# Benchmark line
if bench_series is not None:
    fig.add_trace(go.Scatter(
        x=bench_series.index,
        y=bench_series.values,
        name=benchmark_ticker,
        line=dict(color="#6ab4e8", width=2, dash="dot"),
        hovertemplate=f"%{{x|%d %b %Y}}<br>{benchmark_ticker}: <b>%{{y:.1f}}</b><extra></extra>",
    ))

# Baseline
fig.add_hline(
    y=100,
    line_dash="dash",
    line_color="#333",
    line_width=1,
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#888", size=11),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
        x=0.01, y=0.99,
        bordercolor="#2a2a2a",
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor="#1a1a1a",
        showline=True,
        linecolor="#2a2a2a",
        tickformat="%b '%y",
    ),
    yaxis=dict(
        gridcolor="#1a1a1a",
        showline=True,
        linecolor="#2a2a2a",
        ticksuffix="",
        title="Indexed Value (base=100)",
        title_font=dict(size=10, color="#555"),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1a1a1a",
        bordercolor="#333",
        font=dict(color="#e8e0d0", family="DM Mono, monospace"),
    ),
    height=480,
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(fig, use_container_width=True)

# ── Missing / skipped notices ─────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    missing_tickers = [t for t in weights if t not in prices.columns or prices[t].dropna().empty]
    if missing_tickers:
        st.markdown("**Tickers with no price data** (excluded from portfolio):")
        st.markdown(
            "<div class='skipped-list'>" +
            " &nbsp;·&nbsp; ".join(f"<span class='ticker-badge'>{t}</span>" for t in missing_tickers) +
            "</div>",
            unsafe_allow_html=True
        )

with col_b:
    if skipped:
        st.markdown("**Skipped entries** (zero weight or unresolvable):")
        st.markdown(
            "<div class='skipped-list'>" +
            "<br>".join(skipped) +
            "</div>",
            unsafe_allow_html=True
        )
    if missing_bench:
        st.warning(f"Benchmark ticker **{benchmark_ticker}** not found. Try `^NSEI`, `^BSESN`, or another Yahoo symbol.")

# ── Weight table ──────────────────────────────────────────────────────────────

with st.expander("Portfolio composition", expanded=False):
    valid_weights = {t: w for t, w in weights.items() if t in prices.columns}
    total = sum(valid_weights.values())
    df = pd.DataFrame([
        {"Yahoo Ticker": t, "Weight": f"{w/total*100:.1f}%"}
        for t, w in sorted(valid_weights.items(), key=lambda x: -x[1])
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
