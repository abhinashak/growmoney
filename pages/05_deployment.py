"""
NSE Institutional Deployment Command  —  Streamlit App
Run:  streamlit run 05_deployment.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
import re

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Deployment Command",
    page_icon="🏛",
    layout="wide",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  div[data-testid="stMetric"] { background:#f8f9fb;
    border:1px solid #e4e7ec; border-radius:10px; padding:12px 16px; }
  .badge-bull    { color:#0a7c45; background:#d1fae5;
                   padding:2px 10px; border-radius:20px; font-size:12px; }
  .badge-bear    { color:#b91c1c; background:#fee2e2;
                   padding:2px 10px; border-radius:20px; font-size:12px; }
  .badge-neutral { color:#92400e; background:#fef3c7;
                   padding:2px 10px; border-radius:20px; font-size:12px; }
  .sort-bar { display:flex; gap:8px; flex-wrap:wrap; align-items:center;
              padding:10px 0 6px 0; }
  .stDataFrame { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Read target tickers and percent allocation
# ─────────────────────────────────────────────────────────────────────────────
absolute_path = os.path.abspath("targets.txt")
if not os.path.exists(absolute_path):
    error_msg = (
        f"❌ CRITICAL ERROR: Target file missing. "
        f"EXPECTED PATH: {absolute_path}"
    )
    st.error(error_msg)
    raise FileNotFoundError(error_msg)

with open(absolute_path, "r") as f:
    DEFAULT_TARGETS = f.read().replace('%', '')

SECTOR_MAP = {
    "ONGC.NS":"Oil & Gas","NTPC.NS":"Power","COALINDIA.NS":"Commodities",
    "HINDCOPPER.NS":"Commodities","NMDC.NS":"Commodities","JSWSTEEL.NS":"Metals",
    "BHARATFORG.NS":"Defense","BEL.NS":"Defense","LT.NS":"Industrials",
    "SBIN.NS":"Financials","ICICIBANK.NS":"Financials","MARUTI.NS":"Auto",
    "HEROMOTOCO.NS":"Auto","MOTHERSON.NS":"Auto Anc","INDIGO.NS":"Aviation",
    "BHARTIARTL.NS":"Telecom","SUNPHARMA.NS":"Pharma","GLENMARK.NS":"Pharma",
    "APOLLOHOSP.NS":"Healthcare","TATACONSUM.NS":"FMCG","TITAN.NS":"Consumer",
    "EIHOTEL.NS":"Hospitality","OBEROIRLTY.NS":"Real Estate",
    "COROMANDEL.NS":"Agrochem","KSB.NS":"Industrials","DYNAMATECH.NS":"Industrials",
    "SYRMA.NS":"Electronics","GOLDBEES.NS":"Gold ETF","SILVERBEES.NS":"Silver ETF",
    "0P0000YWL0.BO":"Mutual Fund","532067.BO":"Industrials",
}

# Sector geo sensitivity: (oil, inflation, war/vix, rate/fx)
SECTOR_SENS = {
    "Oil & Gas":   (0.30, 0.10, 0.10, 0.00),
    "Aviation":    (0.40, 0.10, 0.20, 0.00),
    "Auto":        (0.15, 0.10, 0.05, 0.10),
    "Auto Anc":    (0.15, 0.10, 0.05, 0.10),
    "Metals":      (0.10, 0.15, 0.20, 0.05),
    "Commodities": (0.10, 0.15, 0.20, 0.05),
    "Financials":  (0.05, 0.10, 0.10, 0.25),
    "Real Estate": (0.00, 0.05, 0.05, 0.25),
    "Defense":     (0.00, 0.05,-0.10, 0.00),
    "Industrials": (0.05, 0.10, 0.10, 0.10),
    "Pharma":      (0.00, 0.05, 0.00, 0.00),
    "Healthcare":  (0.00, 0.05, 0.00, 0.00),
    "FMCG":        (0.05, 0.10, 0.00, 0.05),
    "Consumer":    (0.05, 0.10, 0.05, 0.10),
    "Telecom":     (0.00, 0.05, 0.00, 0.10),
    "Power":       (0.10, 0.05, 0.05, 0.05),
    "Agrochem":    (0.05, 0.15, 0.05, 0.00),
    "Electronics": (0.05, 0.05, 0.10, 0.05),
    "Hospitality": (0.10, 0.10, 0.20, 0.10),
    "Gold ETF":    (-0.10,-0.10,-0.20, 0.00),
    "Silver ETF":  (-0.05,-0.05,-0.10, 0.00),
    "Mutual Fund": (0.05, 0.05, 0.05, 0.05),
}

MACRO_TICKERS = {
    "Oil (Brent)":  "BZ=F",
    "India VIX":    "^INDIAVIX",
    "US 10Y Yield": "^TNX",
    "USD/INR":      "USDINR=X",
    "SBIN":         "SBIN.NS",
}

# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_targets(text: str) -> pd.DataFrame:
    rows = []
    for line in text.strip().splitlines():
        parts = re.split(r'[,\t]+', line.strip())
        if len(parts) >= 2:
            try:
                rows.append({"Ticker": parts[1].strip(), "Target_%": float(parts[3].strip())})
            except (ValueError, IndexError):
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Ticker","Target_%"])


def parse_holdings(text: str) -> dict:
    out = {}
    if not text.strip():
        return out
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH  (cached 1 hour)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(tickers: tuple) -> pd.DataFrame:
    all_t = list(set(list(tickers) + list(MACRO_TICKERS.values())))
    raw   = yf.download(all_t, period="2y", interval="1d",
                        progress=False, auto_adjust=True, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]]
        close.columns = list(tickers)[:1]
    return close.ffill()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_volume(tickers: tuple) -> pd.Series:
    raw = yf.download(list(tickers), period="2mo", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        vol = raw.xs("Volume", axis=1, level=1)
    else:
        vol = raw[["Volume"]]
        vol.columns = list(tickers)[:1]
    return vol.fillna(0).tail(30).mean()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(tickers: tuple) -> dict:
    data = {}
    try:
        raw = yf.download(
            list(tickers), period="1d", interval="5m",
            progress=False, group_by="ticker"
        )
        for t in tickers:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[t].dropna()
            else:
                df = raw.dropna()
            if not df.empty:
                data[t] = df
    except Exception:
        pass
    return data


def calculate_portfolio_value(holdings_dict):
    if not holdings_dict:
        return 0
    tickers = list(holdings_dict.keys())
    data = yf.download(tickers, period="1d", interval="1m", progress=False)
    if len(tickers) > 1:
        latest_prices = data['Close'].iloc[-1]
    else:
        latest_prices = {tickers[0]: data['Close'].iloc[-1]}
    total_value_inr = 0.0
    for ticker, qty in holdings_dict.items():
        price = latest_prices.get(ticker, 0)
        if price > 0:
            total_value_inr += (price * qty)
    return total_value_inr


# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_intraday_signals(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 3:
        return {"signal": "NO DATA"}
    df = df.copy()
    df["cum_vol"] = df["Volume"].cumsum()
    df["cum_pv"]  = (df["Close"] * df["Volume"]).cumsum()
    df["vwap"]    = df["cum_pv"] / df["cum_vol"]
    price        = float(df["Close"].iloc[-1])
    vwap         = float(df["vwap"].iloc[-1])
    opening_high = df["High"].iloc[:3].max()
    opening_low  = df["Low"].iloc[:3].min()
    if price > opening_high and price > vwap:
        signal = "🚀 BREAKOUT"
    elif price < opening_low and price < vwap:
        signal = "🔻 BREAKDOWN"
    elif price > vwap:
        signal = "🟢 ABOVE VWAP"
    elif price < vwap:
        signal = "🔴 BELOW VWAP"
    else:
        signal = "⚪ NEUTRAL"
    return {"price": price, "vwap": vwap, "signal": signal}


def adjust_intraday_multiplier(base_mult: float, signal: str) -> tuple[float, str]:
    if "BREAKOUT" in signal:
        return base_mult * 1.25, "Momentum breakout"
    elif "ABOVE" in signal:
        return base_mult * 1.10, "Holding above VWAP"
    elif "BELOW" in signal:
        return base_mult * 0.75, "Weak below VWAP"
    elif "BREAKDOWN" in signal:
        return base_mult * 0.50, "Breakdown risk"
    return base_mult * 1.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# RISK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_sbin_cycle(close: pd.DataFrame) -> dict:
    ticker = "SBIN.NS"
    if ticker not in close.columns or len(close[ticker].dropna()) < 210:
        return {"regime":"UNKNOWN","score":50,"mult":0.65,"details":{}}
    s       = close[ticker].dropna()
    latest  = float(s.iloc[-1])
    dma200  = float(s.rolling(200).mean().iloc[-1])
    dma50   = float(s.rolling(50).mean().iloc[-1])
    ret_3y  = (latest / s.iloc[max(0,len(s)-756)] - 1) * 100
    ret_1y  = (latest / s.iloc[max(0,len(s)-252)] - 1) * 100
    ret_6m  = (latest / s.iloc[max(0,len(s)-126)] - 1) * 100
    golden  = dma50 > dma200
    score = 50
    if latest > dma200: score += 15
    if golden:          score += 15
    if ret_3y > 30:     score += 10
    if ret_1y > 10:     score += 10
    if ret_6m > 0:      score += 10
    if ret_6m < -10:    score -= 20
    if latest < dma200: score -= 20
    score  = int(np.clip(score, 0, 100))
    regime = "BULL" if score >= 65 else ("NEUTRAL" if score >= 40 else "BEAR")
    mult   = 1.00 if score>=65 else (0.85 if score>=55 else
             (0.65 if score>=45 else (0.45 if score>=35 else 0.30)))
    return {
        "regime": regime, "score": score, "mult": mult,
        "details": {
            "Price":         f"₹{latest:,.2f}",
            "200-DMA":       f"₹{dma200:,.2f}",
            "50-DMA":        f"₹{dma50:,.2f}",
            "Above 200-DMA": "Yes ✅" if latest > dma200 else "No ❌",
            "Golden Cross":  "Yes ✅" if golden else "No ❌",
            "3Y Return":     f"{ret_3y:+.1f}%",
            "1Y Return":     f"{ret_1y:+.1f}%",
            "6M Return":     f"{ret_6m:+.1f}%",
        }
    }


def compute_geo_risk(close: pd.DataFrame) -> dict:
    score = 0
    out   = {}

    # Oil
    ot = MACRO_TICKERS["Oil (Brent)"]
    if ot in close.columns:
        oil   = close[ot].dropna()
        px    = float(oil.iloc[-1])
        ma50  = float(oil.rolling(50).mean().iloc[-1])
        trend = px / ma50
        s = 25 if (px>100 or trend>1.15) else (18 if (px>85 or trend>1.08) else
            (10 if (px>70 or trend>1.02) else 5))
        out["oil"] = {"score":s,"value":f"${px:.1f}","trend":f"{trend:.2f}×",
                      "reason":"Price or trend spike — inflation risk" if s>=18 else "Contained"}
    else:
        s = 12
        out["oil"] = {"score":s,"value":"N/A","trend":"N/A","reason":"Data unavailable"}
    score += s

    # India VIX
    vt = MACRO_TICKERS["India VIX"]
    if vt in close.columns:
        vix = float(close[vt].dropna().iloc[-1])
        s = 25 if vix>25 else (18 if vix>22 else (10 if vix>18 else (5 if vix>15 else 2)))
        out["vix"] = {"score":s,"value":f"{vix:.1f}",
                      "reason":"Elevated fear / black-swan risk" if s>=18 else "Market calm"}
    else:
        s = 10
        out["vix"] = {"score":s,"value":"N/A","reason":"Data unavailable"}
    score += s

    # US 10Y Yield
    tt = MACRO_TICKERS["US 10Y Yield"]
    if tt in close.columns:
        y = float(close[tt].dropna().iloc[-1])
        s = 25 if y>5.0 else (20 if y>4.5 else (13 if y>4.0 else (8 if y>3.5 else 4)))
        out["yield"] = {"score":s,"value":f"{y:.2f}%",
                        "reason":"High yield = rate-hike pressure on equities" if s>=18 else "Manageable"}
    else:
        s = 12
        out["yield"] = {"score":s,"value":"N/A","reason":"Data unavailable"}
    score += s

    # USD/INR
    ft = MACRO_TICKERS["USD/INR"]
    if ft in close.columns:
        fx_s = close[ft].dropna()
        fx   = float(fx_s.iloc[-1])
        fx3m = float(fx_s.iloc[max(0,len(fx_s)-63)])
        chg  = (fx/fx3m - 1)*100
        s = 25 if (fx>87 or chg>3) else (15 if (fx>84 or chg>1) else (8 if fx>83 else 4))
        out["fx"] = {"score":s,"value":f"₹{fx:.2f}","chg":f"{chg:+.1f}% 3M",
                     "reason":"INR depreciation — imported inflation risk" if s>=15 else "Stable"}
    else:
        s = 12
        out["fx"] = {"score":s,"value":"N/A","chg":"","reason":"Data unavailable"}
    score += s

    score = int(np.clip(score, 0, 100))
    level = "HIGH" if score>=70 else ("MEDIUM" if score>=45 else "LOW")
    mult  = 0.60 if score>=70 else (0.80 if score>=45 else 1.00)
    out["total"] = score
    out["level"] = level
    out["mult"]  = mult
    return out


def geo_sector_mult(ticker: str, geo: dict) -> tuple[float, str]:
    sector = SECTOR_MAP.get(ticker, "Industrials")
    sens   = SECTOR_SENS.get(sector, (0.05,0.05,0.05,0.05))
    adj    = 1.0
    reasons = []
    if geo["oil"]["score"]   >= 18: adj -= sens[0]; reasons.append(f"Oil risk {'-' if sens[0]>0 else '+'}({abs(sens[0]):.0%})")
    if geo["yield"]["score"] >= 18: adj -= sens[1]; reasons.append(f"Inflation {'-' if sens[1]>0 else '+'}({abs(sens[1]):.0%})")
    if geo["vix"]["score"]   >= 18: adj -= sens[2]; reasons.append(f"War/VIX {'-' if sens[2]>0 else '+'}({abs(sens[2]):.0%})")
    if geo["fx"]["score"]    >= 18: adj -= sens[3]; reasons.append(f"Rate/FX {'-' if sens[3]>0 else '+'}({abs(sens[3]):.0%})")
    geo_m      = float(np.clip(adj, 0.50, 1.50))
    reason_str = ", ".join(reasons) if reasons else "No active geo adjustments"
    return geo_m, reason_str


def compute_stock_row(curr_portfolio_amt: float, ticker: str, weight: float,
                       close: pd.DataFrame, holding_qty: int, total_capital: float,
                       days_left: int, geo: dict, adv_series: pd.Series | None,
                       sbin_mult: float, geo_tranche_mult: float,
                       intraday_map: dict = None) -> dict:
    sector     = SECTOR_MAP.get(ticker, "Industrials")
    target_inr = total_capital * weight

    # ── Signal 1: 200-DMA ──────────────────────────────────────────────────
    price, dma200, pct_200, dma_signal, dma_mult, dma_reason = None, None, None, "NO DATA", 1.0, ""
    if ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) >= 210:
            price   = float(s.iloc[-1])
            dma200  = float(s.rolling(200).mean().iloc[-1])
            pct_200 = (price / dma200 - 1) * 100
            dma_mult = float(np.clip(1.0 - (price / dma200 - 1), 0.50, 1.50))
            if pct_200 <= -15:
                dma_signal = "STRONG BUY"
                dma_reason = (f"Price is {pct_200:.1f}% below 200-DMA (deeply discounted vs fair value). "
                              f"Accelerating buy — mult={dma_mult:.2f}×.")
            elif pct_200 <= -5:
                dma_signal = "BUY"
                dma_reason = (f"Price is {pct_200:.1f}% below 200-DMA — trading at a discount. "
                              f"Buying at above-normal pace (mult={dma_mult:.2f}×).")
            elif pct_200 <= 5:
                dma_signal = "BUY"
                dma_reason = (f"Price is {pct_200:+.1f}% vs 200-DMA — near fair value. "
                              f"Standard deployment pace (mult={dma_mult:.2f}×).")
            elif pct_200 <= 15:
                dma_signal = "SLOW"
                dma_reason = (f"Price is {pct_200:.1f}% above 200-DMA — moderately overextended. "
                              f"Slowing buy to avoid chasing (mult={dma_mult:.2f}×).")
            else:
                dma_signal = "WAIT"
                dma_reason = (f"Price is {pct_200:.1f}% above 200-DMA — significantly overextended. "
                              f"Minimal buying until price reverts toward DMA (mult={dma_mult:.2f}×).")
        elif len(s) >= 1:
            price      = float(s.iloc[-1])
            dma_reason = "Insufficient history for 200-DMA (< 210 days)"

    # ── Signal 2: Relative drawdown vs SBIN ────────────────────────────────
    rel_drop_mult, rel_drop_reason = 1.0, "SBIN data unavailable"
    stock_dd_pct, sbin_dd_pct, rel_drop_pct = None, None, None
    if "SBIN.NS" in close.columns and ticker in close.columns and price is not None:
        s_stk  = close[ticker].dropna()
        s_sbin = close["SBIN.NS"].dropna()
        window = min(252, len(s_stk), len(s_sbin))
        if window >= 60:
            stk_top      = float(s_stk.tail(window).max())
            sbin_top     = float(s_sbin.tail(window).max())
            sbin_px      = float(s_sbin.iloc[-1])
            stock_dd_pct = (price   / stk_top  - 1) * 100
            sbin_dd_pct  = (sbin_px / sbin_top - 1) * 100
            rel_drop_pct = stock_dd_pct - sbin_dd_pct
            rel_drop_mult = float(np.clip(1.0 - rel_drop_pct / 50, 0.60, 1.50))
            if rel_drop_pct < -15:
                rel_drop_reason = (
                    f"Stock is {abs(stock_dd_pct):.1f}% off its {window//21:.0f}-month high; "
                    f"SBIN is only {abs(sbin_dd_pct):.1f}% off its high. "
                    f"Excess relative drop = {rel_drop_pct:.1f}% → strong buy signal (mult={rel_drop_mult:.2f}×)."
                )
            elif rel_drop_pct < -5:
                rel_drop_reason = (
                    f"Stock down {abs(stock_dd_pct):.1f}% from top vs SBIN down {abs(sbin_dd_pct):.1f}%. "
                    f"Relative excess drop of {rel_drop_pct:.1f}% → buying above base rate (mult={rel_drop_mult:.2f}×)."
                )
            elif rel_drop_pct <= 5:
                rel_drop_reason = (
                    f"Stock and SBIN have corrected similarly ({stock_dd_pct:.1f}% vs {sbin_dd_pct:.1f}%). "
                    f"No relative edge → neutral pacing (mult={rel_drop_mult:.2f}×)."
                )
            else:
                rel_drop_reason = (
                    f"Stock is only {abs(stock_dd_pct):.1f}% off top while SBIN is {abs(sbin_dd_pct):.1f}% off. "
                    f"Stock has NOT corrected with the cycle (rel_drop=+{rel_drop_pct:.1f}%). "
                    f"Slowing deployment (mult={rel_drop_mult:.2f}×)."
                )

    # ── Signal 3: Geo/sector ───────────────────────────────────────────────
    geo_m, geo_reason = geo_sector_mult(ticker, geo)

    # ── Intraday adjustment ────────────────────────────────────────────────
    intraday_signal = "NO DATA"
    intraday_reason = ""
    if intraday_map and ticker in intraday_map:
        intraday_info   = compute_intraday_signals(intraday_map[ticker])
        intraday_signal = intraday_info["signal"]
        adj_mult, intraday_reason = adjust_intraday_multiplier(
            dma_mult * rel_drop_mult * geo_m, intraday_signal
        )
        final_mult = float(np.clip(adj_mult, 0.20, 2.00))
    else:
        final_mult = float(np.clip(dma_mult * rel_drop_mult * geo_m, 0.20, 1.80))

    # ── Portfolio gap & daily order ────────────────────────────────────────
    curr_inr = (holding_qty * price) if price else 0
    gap_inr  = max(0.0, target_inr - curr_inr)
    backlog  = None
    shares, outlay_inr, adv_flag, adv_pct = 0, 0.0, "", 0.0

    if gap_inr > 0 and price and price > 0 and days_left > 0:
        curr_inr     = (holding_qty * price) if price else 0
        target_inr   = total_capital * weight
        rebalance_inr = target_inr - curr_inr
        planned_daily = target_inr / (days_left + 1)
        backlog       = max(0.0, planned_daily - curr_inr)
        MAX_DAILY_MULT = 2.0
        raw_daily  = min(backlog if backlog > 0 else gap_inr / days_left,
                         planned_daily * MAX_DAILY_MULT)
        urgency_mult = 1.3 if backlog > planned_daily else 1.0
        adj_daily    = raw_daily * final_mult * sbin_mult * geo_tranche_mult * urgency_mult
        shares       = int(adj_daily / price)
        outlay_inr   = shares * price
        if adv_series is not None and ticker in adv_series.index:
            adv_val = float(adv_series[ticker])
            if adv_val > 0:
                adv_pct = (shares / adv_val) * 100
                if adv_pct > 5:   adv_flag = "⚠ ICEBERG"
                elif adv_pct > 2: adv_flag = "⚡ VWAP"

    is_complete = gap_inr <= 0
    if is_complete:
        shares     = 0
        outlay_inr = 0.0
        adv_flag   = ""
        status     = "FULL ✅"
    else:
        status = adv_flag if adv_flag else "BUY"

    rebalance_inr = target_inr - curr_inr
    target_cr  = 0.0
    current_cr = 0.0
    action     = "HOLD"
    if price and price > 0:
        target_cr  = (curr_portfolio_amt / 1e7) * weight
        current_cr = (holding_qty * price) / 1e7
        rebalance_cr = target_cr - current_cr
        action = "BUY" if rebalance_cr > 0 else "SELL"

    return {
        "Ticker":              ticker,
        "Sector":              sector,
        "Target %":            round(weight * 100, 2),
        "Target (Cr)":         round(target_cr, 2),
        "Current (Cr)":        round(current_cr, 2),
        "Target Value (₹L)":   round(target_cr * 100, 2),
        "Current Value (₹L)":  round(current_cr * 100, 2),
        "Rebalance (₹L)":      round((target_cr - current_cr) * 100, 2),
        "Gap (Cr)":            round(max(gap_inr, 0) / 1e7, 2),
        "Price (₹)":           round(price, 2) if price else None,
        "200-DMA Signal":      "FULL" if is_complete else dma_signal,
        "% vs 200-DMA":        round(pct_200, 1) if pct_200 is not None else None,
        "DMA Mult":            round(dma_mult, 3),
        "Stock DD %":          round(stock_dd_pct, 1) if stock_dd_pct is not None else None,
        "SBIN DD %":           round(sbin_dd_pct, 1) if sbin_dd_pct is not None else None,
        "Rel Drop %":          round(rel_drop_pct, 1) if rel_drop_pct is not None else None,
        "Rel Drop Mult":       round(rel_drop_mult, 3),
        "Geo Mult":            round(geo_m, 3),
        "Final Mult":          round(final_mult, 3),
        "Shares to Buy":       0 if is_complete else shares,
        "Today Outlay (₹L)":   0.0 if is_complete else round(outlay_inr / 1e5, 2),
        "Execution":           status,
        "_is_complete":        is_complete,
        "_dma_reason":         dma_reason,
        "_rel_drop_reason":    rel_drop_reason,
        "_geo_reason":         geo_reason,
        "_adv_pct":            round(adv_pct, 1),
        "Intraday Signal":     intraday_signal,
        "Intraday Reason":     intraday_reason,
        "Action":              action,
        "Backlog (₹L)":        round(backlog / 1e5, 2) if backlog is not None else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏛 ₹NSE Deployment Command")
st.caption("Risk-aware institutional portfolio builder · SBIN cycle · 200-DMA · Geo risk · ADV guard")
st.divider()

# ════════════════════════════════════════════════════
# INPUT SCREEN
# ════════════════════════════════════════════════════

st.subheader("Strategy inputs")
col_left, col_right = st.columns(2, gap="large")

with col_left:
    total_cap = st.number_input(
        "Total capital to deploy (INR)",
        value=100_00_000, step=1_00_000, format="%d",
        help="Enter total corpus in INR."
    )
    horizon = st.slider(
        "Deployment horizon (trading days)",
        min_value=10, max_value=452, value=20,
        help="Total days over which to fully build the portfolio"
    )
    elapsed = st.slider(
        "Days elapsed since deployment started",
        min_value=0, max_value=horizon, value=10,
        help="Set > 0 if you have already been deploying."
    )
    st.markdown("**Current portfolio (ticker [TAB] quantity)**")
    uploaded_file = st.file_uploader("Upload holdings file (TSV / CSV)", type=["tsv","csv"])
    holdings_text = ""
    if uploaded_file is not None:
        try:
            df_hold = pd.read_csv(uploaded_file, sep="\t", header=None)
            df_hold.columns = ["Ticker","Qty"]
            holdings_text = "\n".join([f"{r.Ticker}\t{int(r.Qty)}" for _, r in df_hold.iterrows()])
            st.success("✅ Holdings file loaded")
        except Exception:
            st.error("❌ Failed to parse file. Ensure format: Ticker[TAB]Qty")
    else:
        holdings_text = st.text_area(
            "holdings_input", value="", height=120,
            placeholder="COROMANDEL.NS\t500\nBEL.NS\t1200",
            label_visibility="collapsed",
        )

with col_right:
    st.markdown("**Target distribution (ticker [TAB] %)**")
    targets_text = st.text_area(
        "targets_input", value=DEFAULT_TARGETS, height=480,
        label_visibility="collapsed",
        help="Paste your target weights. Tab-separated: TICKER [TAB] WEIGHT%"
    )

st.divider()
run = st.button("🚀 Calculate today's tranche", width='stretch', type="primary")

# ════════════════════════════════════════════════════
# EXECUTION + OUTPUT SCREEN
# ════════════════════════════════════════════════════

if run:
    targets_df = parse_targets(targets_text)
    holdings   = parse_holdings(holdings_text)
    curr_portfolio_amt = calculate_portfolio_value(holdings)

    st.session_state["targets_df"] = targets_df
    st.session_state["holdings"] = holdings
    st.session_state["curr_portfolio_amt"] = curr_portfolio_amt

    if targets_df.empty:
        st.error("No valid targets found. Check the format: TICKER [TAB] WEIGHT%")
        st.stop()

    total_w = targets_df["Target_%"].sum()
    if abs(total_w - 100) > 1.5:
        st.warning(f"Weights sum to {total_w:.1f}% (not 100%). They will be normalised.")
    targets_df["weight"] = targets_df["Target_%"] / targets_df["Target_%"].sum()

    tickers_tuple = tuple(targets_df["Ticker"].tolist())
    days_left     = max(1, horizon - elapsed)
    st.session_state["days_left"] = days_left

    with st.spinner("Fetching market data from Yahoo Finance..."):
        try:
            close = fetch_all(tickers_tuple)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    with st.spinner("Computing SBIN cycle & geopolitical risk..."):
        sbin_info = compute_sbin_cycle(close)
        geo       = compute_geo_risk(close)
        st.session_state["sbin_info"] = sbin_info
        st.session_state["geo"] = geo

    try:
        adv = fetch_volume(tickers_tuple)
    except Exception:
        adv = None

    with st.spinner("Fetching intraday signals..."):
        intraday_map = fetch_intraday(tickers_tuple)
        st.session_state["intraday_map"] = intraday_map

    with st.spinner("Calculating today's orders..."):
        rows = []
        for _, r in targets_df.iterrows():
            qty = holdings.get(r["Ticker"], 0)
            row = compute_stock_row(
                curr_portfolio_amt,
                ticker=r["Ticker"], weight=r["weight"],
                close=close, holding_qty=qty,
                total_capital=total_cap, days_left=days_left,
                geo=geo, adv_series=adv,
                sbin_mult=sbin_info["mult"],
                geo_tranche_mult=geo["mult"],
                intraday_map=intraday_map,
            )
            rows.append(row)
        results_df = pd.DataFrame(rows)
        # STORE RESULTS
        st.session_state["results_df"] = results_df
        st.session_state["calculated"] = True
        st.session_state["sorted_df"] = None

if st.session_state.get("calculated", False):
    results_df = st.session_state["results_df"]
    sbin_info = st.session_state["sbin_info"]
    geo = st.session_state["geo"]
    targets_df = st.session_state["targets_df"]
    holdings = st.session_state["holdings"]
    curr_portfolio_amt = st.session_state["curr_portfolio_amt"]
    days_left = st.session_state["days_left"]
    intraday_map = st.session_state["intraday_map"]


    st.divider()

    # ════════════════════════════════════════════════════
    # SECTION 1 — RISK DASHBOARD
    # ════════════════════════════════════════════════════

    st.subheader("Risk dashboard")
    rc1, rc2 = st.columns(2, gap="large")

    with rc1:
        st.markdown("#### SBIN capex cycle (credit cycle proxy)")
        badge = (f'<span class="badge-bull">🟢 {sbin_info["regime"]}</span>'
                 if sbin_info["regime"] == "BULL" else
                 f'<span class="badge-bear">🔴 {sbin_info["regime"]}</span>'
                 if sbin_info["regime"] == "BEAR" else
                 f'<span class="badge-neutral">🟡 {sbin_info["regime"]}</span>')
        st.markdown(
            f"Regime: {badge} &nbsp;·&nbsp; Score: **{sbin_info['score']}/100** "
            f"&nbsp;·&nbsp; Portfolio gate mult: **{sbin_info['mult']:.2f}×**",
            unsafe_allow_html=True
        )
        regime_explain = {
            "BULL":    "SBIN above 200-DMA with golden cross. Full deployment — credit cycle is expansionary.",
            "NEUTRAL": "SBIN consolidating. Moderate deployment — cycle is neither expanding nor contracting.",
            "BEAR":    "SBIN below 200-DMA. Minimum deployment — bear credit cycle, preserve capital.",
        }.get(sbin_info["regime"], "")
        st.info(regime_explain)
        sbin_detail_df = pd.DataFrame(list(sbin_info["details"].items()), columns=["Metric","Value"])
        st.dataframe(sbin_detail_df, width='stretch', hide_index=True)

    with rc2:
        st.markdown("#### Geopolitical risk components")
        geo_level_color = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(geo["level"],"")
        st.markdown(
            f"Total: **{geo['total']}/100** &nbsp;·&nbsp; "
            f"Level: {geo_level_color} **{geo['level']}** &nbsp;·&nbsp; "
            f"Tranche mult: **{geo['mult']:.2f}×**"
        )
        geo_rows = [
            {"Component":"Oil (Brent)", "Value":geo["oil"]["value"],
             "Trend":geo["oil"]["trend"],"Score":geo["oil"]["score"],"Implication":geo["oil"]["reason"]},
            {"Component":"India VIX",   "Value":geo["vix"]["value"],
             "Trend":"—",               "Score":geo["vix"]["score"],"Implication":geo["vix"]["reason"]},
            {"Component":"US 10Y Yield","Value":geo["yield"]["value"],
             "Trend":"—",               "Score":geo["yield"]["score"],"Implication":geo["yield"]["reason"]},
            {"Component":"USD/INR",     "Value":geo["fx"]["value"],
             "Trend":geo["fx"].get("chg",""),"Score":geo["fx"]["score"],"Implication":geo["fx"]["reason"]},
        ]
        geo_df = pd.DataFrame(geo_rows)
        st.dataframe(
            geo_df.style.background_gradient(subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=25),
            width='stretch', hide_index=True
        )
        geo_explain = (
            "⚠️ HIGH macro risk — only 60% of each daily tranche deployed." if geo["level"]=="HIGH" else
            "🟡 MEDIUM macro risk — 80% of each daily tranche deployed."   if geo["level"]=="MEDIUM" else
            "✅ LOW macro risk — full tranche deployment."
        )
        st.info(geo_explain)

    st.divider()

    # ════════════════════════════════════════════════════
    # SECTION 2 — SUMMARY METRICS
    # ════════════════════════════════════════════════════

    st.subheader("Today's tranche summary")
    total_outlay   = results_df["Today Outlay (₹L)"].sum()
    total_gap_cr   = results_df["Gap (Cr)"].sum()
    avg_mult       = results_df.loc[~results_df["_is_complete"], "Final Mult"].mean()
    filled_pct     = (1 - total_gap_cr / (total_cap / 1e7)) * 100
    iceberg_count  = results_df["Execution"].str.startswith("⚠").sum()
    complete_count = results_df["_is_complete"].sum()

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Portfolio Value",        f"₹{curr_portfolio_amt/100000:.2f} L", delta="Excluding Gold")
    m2.metric("Today's total outlay",   f"₹{total_outlay:.2f} L",    delta=f"₹{total_outlay/100:.2f} Cr")
    m3.metric("Portfolio gap remaining",f"₹{total_gap_cr:.2f} Cr",   delta=f"{filled_pct:.1f}% filled", delta_color="inverse")
    m4.metric("Avg resilience mult",    f"{avg_mult:.2f}×" if not np.isnan(avg_mult) else "—", delta="Final Multiply")
    m5.metric("Days remaining",         f"{days_left}",               delta=f"Day {elapsed} of {horizon}")
    m6.metric("Fully invested",         f"{complete_count} / {len(results_df)}", delta="No action needed")
    m7.metric("Iceberg orders",         str(iceberg_count),
              delta="Need algo routing" if iceberg_count > 0 else "All market-able",
              delta_color="inverse" if iceberg_count > 0 else "normal")

    if iceberg_count > 0:
        st.warning(
            f"⚠️ **Institutional note:** {iceberg_count} ticker(s) require **Iceberg** or **VWAP algo** "
            f"orders — today's buy quantity exceeds 5% of the 30-day ADV. "
            f"Splitting into child orders over the session will avoid moving the market."
        )

    st.divider()

    # ════════════════════════════════════════════════════
    # SECTION 3 — ORDERS TABLE  (with sorting)
    # ════════════════════════════════════════════════════

    display_cols = [
        "Ticker","Sector","Price (₹)","Target %","Gap (Cr)",
        "200-DMA Signal","Final Mult","Shares to Buy","Today Outlay (₹L)","Execution",
        # keep internal cols for expander logic
        "DMA Mult","Rel Drop Mult","Geo Mult","% vs 200-DMA",
        "Target (Cr)","Current (Cr)","Target Value (₹L)","Current Value (₹L)",
        "Rebalance (₹L)","Action",
    ]

    # ── Colour helpers ─────────────────────────────────────────────────────
    def color_signal(val):
        if val == "STRONG BUY": return "color:#065f46;font-weight:bold;background:#d1fae5"
        if val == "BUY":        return "color:#0a7c45;font-weight:bold"
        if val == "SLOW":       return "color:#92400e;font-weight:bold"
        if val == "WAIT":       return "color:#b45309;font-weight:bold"
        if val == "FULL":       return "color:#1e40af;font-weight:bold"
        return ""

    def color_pct(val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return ""
        if val <= -15: return "color:#065f46;font-weight:bold"
        if val < -5:   return "color:#0a7c45"
        if val < 5:    return "color:#374151"
        if val < 15:   return "color:#92400e"
        return "color:#b91c1c"

    def color_mult(val):
        if isinstance(val, float):
            if val >= 1.2: return "background-color:#d1fae5"
            if val <= 0.6: return "background-color:#fee2e2"
        return ""

    shared_fmt = {
        "Target %":          "{:.2f}%",
        "Target (Cr)":       "₹{:.2f}",
        "Current (Cr)":      "₹{:.2f}",
        "Gap (Cr)":          "₹{:.2f}",
        "Price (₹)":         lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—",
        "% vs 200-DMA":      lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
        "DMA Mult":          "{:.3f}×",
        "Rel Drop Mult":     "{:.3f}×",
        "Geo Mult":          "{:.3f}×",
        "Final Mult":        "{:.3f}×",
        "Today Outlay (₹L)": "{:.2f}",
    }

    # ── Sort columns available to user ─────────────────────────────────────
    SORT_OPTIONS = {
        "Ticker · Sector":  ("Ticker",             True),
        "Price (₹)":        ("Price (₹)",           False),
        "Target %":         ("Target %",            False),
        "Gap (Cr)":         ("Gap (Cr)",            False),
        "200-DMA Signal":   ("200-DMA Signal",      True),
        "Final Mult":       ("Final Mult",          False),
        "Shares":           ("Shares to Buy",       False),
        "Outlay (₹L)":      ("Today Outlay (₹L)",   False),
        "Execution":        ("Execution",           True),
    }

    # ── TABLE A: Active buy orders ─────────────────────────────────────────
    st.subheader(f"📋 Active buy orders — {len(results_df[~results_df['_is_complete']])} stocks")
    st.caption("Each row is expandable — click to reveal multiplier reasoning, rebalancing details, and live intraday signal.")

    # Sort controls
    sa_col1, sa_col2, sa_col3 = st.columns([3, 1, 4])
    with sa_col1:
        active_sort_by  = st.selectbox(
            "Sort active orders by",
            options=list(SORT_OPTIONS.keys()),
            index=0,                        # default = Ticker · Sector
            key="active_sort_by",
            label_visibility="collapsed",
        )
    with sa_col2:
        active_sort_asc = st.checkbox("↑ Asc", value=True, key="active_sort_asc")

    active_sort_col, _ = SORT_OPTIONS[active_sort_by]

    active_base = results_df[~results_df["_is_complete"]].copy()
    if active_sort_col in active_base.columns:
        active_sorted = active_base.sort_values(active_sort_col, ascending=active_sort_asc)
    else:
        active_sorted = active_base

    if active_sorted.empty:
        st.success("🎉 All positions are fully invested! Nothing to buy today.")
    else:
        # Column header bar
        hcols = st.columns([2, 1.4, 1, 1.1, 1.5, 1.2, 1, 1.4, 1.2])
        for col, label in zip(hcols, list(SORT_OPTIONS.keys())):
            col.markdown(f"<small><b>{label}</b></small>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0 8px 0'>", unsafe_allow_html=True)

        for _, row in active_sorted.iterrows():
            ticker          = row["Ticker"]
            sector          = row["Sector"]
            price           = row["Price (₹)"]
            final_mult      = row["Final Mult"]
            shares          = row["Shares to Buy"]
            outlay          = row["Today Outlay (₹L)"]
            execution       = row["Execution"]
            dma_signal      = row["200-DMA Signal"]
            intraday_signal = row.get("Intraday Signal", "NO DATA")
            intraday_reason = row.get("Intraday Reason", "")

            mult_badge = ("🟢" if final_mult >= 1.2 else "🔴" if final_mult <= 0.6 else "🟡")
            expander_label = (
                f"{ticker}  ·  Sector : {sector}  ·  "
                f"Price : {'₹'+f'{price:,.2f}' if pd.notna(price) else '—'}  ·  "
                f"200-DMA Signal : {dma_signal}  ·  "
                f"Final Mult : {mult_badge} {final_mult:.3f}×  ·  "
                f"Shares : {int(shares):,} shares  ·  "
                f"Outlay (₹L) : ₹{outlay:.2f}L  ·  Execution : {execution}"
            )

            with st.expander(expander_label):
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Price (₹)",     f"₹{price:,.2f}" if pd.notna(price) else "—")
                sc2.metric("Target %",      f"{row['Target %']:.2f}%")
                sc3.metric("Gap (Cr)",      f"₹{row['Gap (Cr)']:.2f}")
                sc4.metric("Shares to Buy", f"{int(shares):,}")
                sc5.metric("Today Outlay",  f"₹{outlay:.2f}L")
                st.markdown("---")

                tab_mult, tab_rebal, tab_reason, tab_intraday = st.tabs([
                    "📊 Multiplier Reasoning","⚖️ Rebalancing","🧠 Full Detail","⚡ Intraday Signal",
                ])

                with tab_mult:
                    st.markdown(f"**Final Multiplier: `{final_mult:.3f}×`**")
                    mult_df = pd.DataFrame([{
                        "Layer":"200-DMA",            "Value":f"{row['DMA Mult']:.3f}×",      "Reasoning":row["_dma_reason"],
                    },{
                        "Layer":"Rel. Drawdown vs SBIN","Value":f"{row['Rel Drop Mult']:.3f}×","Reasoning":row["_rel_drop_reason"],
                    },{
                        "Layer":"Geo / Sector",        "Value":f"{row['Geo Mult']:.3f}×",     "Reasoning":row["_geo_reason"],
                    }])
                    st.dataframe(mult_df, width='stretch', hide_index=True)
                    if row["_adv_pct"] > 0:
                        st.caption(f"**ADV Impact:** {row['_adv_pct']:.1f}% of volume → {execution}")

                with tab_rebal:
                    rebal_val    = row["Rebalance (₹L)"]
                    action       = row["Action"]
                    action_color = "#d1fae5" if action == "BUY" else "#fee2e2"
                    st.markdown(
                        f"<span style='background:{action_color};padding:3px 10px;"
                        f"border-radius:12px;font-weight:bold'>{action}</span> &nbsp;"
                        f"Rebalance required: **₹{rebal_val:.2f}L**",
                        unsafe_allow_html=True,
                    )
                    rebal_df = pd.DataFrame([{
                        "Target %":           f"{row['Target %']:.2f}%",
                        "Target Value (₹L)":  f"₹{row['Target Value (₹L)']:.2f}",
                        "Current Value (₹L)": f"₹{row['Current Value (₹L)']:.2f}",
                        "Rebalance (₹L)":     f"₹{rebal_val:.2f}",
                        "Action":             action,
                        "Final Mult":         f"{final_mult:.2f}×",
                    }])
                    st.dataframe(rebal_df, width='stretch', hide_index=True)
                    st.caption("Rebalance = amount needed to reach target. BUY = under-allocated · SELL = over-allocated.")

                with tab_reason:
                    sig_color = {"STRONG BUY":"🟢","BUY":"🟢","SLOW":"🟡","WAIT":"🟠"}.get(dma_signal,"⚪")
                    st.markdown(f"#### {ticker} · {sector}  \nFinal Mult: `{final_mult:.3f}×`  {sig_color} {dma_signal}")
                    if "BREAKOUT" in intraday_signal and final_mult > 1.0:
                        st.success("🚀 STRONG BUY: Breakout + High Conviction. Enter quickly.")
                    elif "BREAKDOWN" in intraday_signal:
                        st.error("🔻 AVOID: Breakdown detected. Do not buy now.")
                    elif "BELOW" in intraday_signal and final_mult < 0.8:
                        st.warning("⚠️ WEAK: Below VWAP + Low Conviction. No action now.")
                    elif "ABOVE" in intraday_signal and final_mult > 1.0:
                        st.success("🟢 BUY ON DIP: Strong above VWAP. Split into 3–4 buys.")
                    else:
                        st.caption("⚪ Data unavailable.")
                    st.markdown(f"> **200-DMA:** {row['_dma_reason']}")
                    st.markdown(f"> **Rel. Drawdown vs SBIN:** {row['_rel_drop_reason']}")
                    st.markdown(f"> **Geo/Sector:** {row['_geo_reason']}")
                    if row["_adv_pct"] > 0:
                        st.markdown(f"> **ADV Impact:** {row['_adv_pct']:.1f}% of volume → {execution}")

                with tab_intraday:
                    if ticker in intraday_map:
                        sig_info = compute_intraday_signals(intraday_map[ticker])
                        if sig_info.get('price'):
                            ic1, ic2, ic3 = st.columns(3)
                            ic1.metric("Price",  f"₹{sig_info['price']:,.2f}")
                            ic2.metric("Vol Weighted Avg Price",   f"₹{sig_info['vwap']:,.2f}")
                            ic3.metric("Signal", sig_info["signal"])
                            if intraday_reason:
                                st.caption(f"Intraday adjustment: {intraday_reason}")
                        else:
                            st.info("No intraday data available for this ticker.")
                    else:
                        st.info("No intraday data available for this ticker.")

    st.divider()

    # ── TABLE B: Fully invested positions ─────────────────────────────────
    complete_base = results_df[results_df["_is_complete"]].copy()
    st.subheader(f"✅ Fully invested positions — {len(complete_base)} stocks")
    st.caption("These stocks have reached or exceeded their target allocation. No buying recommended.")

    if complete_base.empty:
        st.info("No positions are fully invested yet.")
    else:
        complete_show_cols = [
            "Ticker","Sector","Price (₹)","Target %","Gap (Cr)",
            "200-DMA Signal","Final Mult","Shares to Buy","Today Outlay (₹L)","Execution",
            "Target (Cr)","Current (Cr)",
        ]
        complete_sort_col = "Ticker.Sector"
        if complete_sort_col in complete_base.columns:
            complete_sorted = complete_base.sort_values(complete_sort_col, ascending=complete_sort_asc)
        else:
            complete_sorted = complete_base.sort_values("Ticker", ascending=True)

        complete_display = complete_sorted[complete_show_cols].reset_index(drop=True)

        complete_fmt = {
            **shared_fmt,
            "Shares to Buy":     lambda x: "FULL ✅",
            "Today Outlay (₹L)": lambda x: "—",
            "Gap (Cr)":          lambda x: "—",
        }
        def highlight_complete_row(row):
            return ["background-color: #f0fdf4"] * len(row)

        complete_styled = (
            complete_display.style
            .apply(highlight_complete_row, axis=1)
            .applymap(color_pct, subset=["% vs 200-DMA"] if "% vs 200-DMA" in complete_display.columns else [])
            .format(complete_fmt)
        )
        st.dataframe(complete_styled, width='stretch', hide_index=True,
                     height=min(500, 80 + len(complete_display) * 38))

    # ════════════════════════════════════════════════════
    # SECTION 4 — EXPORT
    # ════════════════════════════════════════════════════

    st.divider()
    st.subheader("Export")
    import io

    display_df = pd.concat([
        results_df[~results_df["_is_complete"]],
        results_df[results_df["_is_complete"]],
    ], ignore_index=True)

    reason_rows = results_df[~results_df["_is_complete"]][[
        "Ticker","Sector","Final Mult","_dma_reason","_rel_drop_reason","_geo_reason"
    ]].rename(columns={
        "_dma_reason":      "200-DMA Reasoning",
        "_rel_drop_reason": "Rel. Drawdown vs SBIN",
        "_geo_reason":      "Geo / Sector Reasoning",
    }).reset_index(drop=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        display_df.to_excel(writer, sheet_name="Today_Orders", index=False)
        reason_rows.to_excel(writer, sheet_name="Reasons", index=False)
        sbin_detail_df.to_excel(writer, sheet_name="SBIN_Cycle", index=False)
        geo_df.to_excel(writer, sheet_name="Geo_Risk", index=False)

    st.download_button(
        "⬇ Download full plan (.xlsx)",
        data=buf.getvalue(),
        file_name=f"deployment_plan_{datetime.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width='stretch',
    )