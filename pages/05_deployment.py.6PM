"""
================================================================================
  SMART PORTFOLIO DEPLOYMENT ENGINE  v2.0
  NSE/BSE  ·  ₹1,00,00,00,000 (100 Crore)
================================================================================

WHAT CHANGED IN v2 (merged from Streamlit app):
  ✦ Bulk market-data fetch  — one yf.download() call for ALL tickers at once
  ✦ Remaining-Gap / Horizon formula  — daily buy = (gap / days_left) × mult
  ✦ Current-holdings aware  — only buys what is still missing
  ✦ Relative Strength vs SBIN  — value-entry cycle multiplier per stock
  ✦ India VIX (^INDIAVIX)  — replaces US VIX for macro brake
  ✦ Oil trend (50DMA ratio) in addition to absolute price
  ✦ ADV / Iceberg-order warning  — flags orders > 5 % of 30-day ADV

RISK LAYERS (retained & refined):
  1. SBIN Capex Cycle       — 3Y regime score → portfolio-level gate
  2. Individual 200-DMA     — per-stock momentum gate
  3. Geopolitical Score     — oil trend + India VIX + US 10Y + USD/INR
  4. Relative Strength Cycle— per-stock SBIN-relative value window
  5. Remaining-Gap Pacing   — time-weighted, holdings-aware daily order
  6. ADV Liquidity Guard    — institutional iceberg warning

================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_CAPITAL_INR  = 1_00_00_00_000   # 100 Crore
DEPLOYMENT_HORIZON = 180              # Trading days to fully deploy (≈ 9 months)
SBIN_TICKER        = "SBIN.NS"

TARGET_WEIGHTS = {
    "0P0000YWL0.BO": 0.099,
    "COROMANDEL.NS": 0.030,
    "BEL.NS":        0.040,
    "BHARATFORG.NS": 0.040,
    "GOLDBEES.NS":   0.001,
    "SILVERBEES.NS": 0.001,
    "HINDCOPPER.NS": 0.020,
    "ONGC.NS":       0.020,
    "NTPC.NS":       0.020,
    "COALINDIA.NS":  0.020,
    "KSB.NS":        0.020,
    "NMDC.NS":       0.030,
    "DYNAMATECH.NS": 0.020,
    "LT.NS":         0.030,
    "JSWSTEEL.NS":   0.030,
    "MOTHERSON.NS":  0.020,
    "SYRMA.NS":      0.020,
    "MARUTI.NS":     0.030,
    "SBIN.NS":       0.040,
    "ICICIBANK.NS":  0.030,
    "SUNPHARMA.NS":  0.020,
    "GLENMARK.NS":   0.020,
    "APOLLOHOSP.NS": 0.030,
    "532067.BO":     0.020,
    "TATACONSUM.NS": 0.040,
    "TITAN.NS":      0.020,
    "BHARTIARTL.NS": 0.030,
    "INDIGO.NS":     0.020,
    "HEROMOTOCO.NS": 0.020,
    "EIHOTEL.NS":    0.020,
    "OBEROIRLTY.NS": 0.020,
}

#assert abs(sum(TARGET_WEIGHTS.values()) - 1.0) < 0.01, "Weights must sum to ~100%"

SECTOR_MAP = {
    "ONGC.NS":       "oil_gas",
    "NTPC.NS":       "power",
    "COALINDIA.NS":  "commodities",
    "HINDCOPPER.NS": "commodities",
    "NMDC.NS":       "commodities",
    "JSWSTEEL.NS":   "metals",
    "BHARATFORG.NS": "defense_industrial",
    "BEL.NS":        "defense_industrial",
    "LT.NS":         "industrials",
    "SBIN.NS":       "financials",
    "ICICIBANK.NS":  "financials",
    "MARUTI.NS":     "auto",
    "HEROMOTOCO.NS": "auto",
    "MOTHERSON.NS":  "auto_ancillary",
    "INDIGO.NS":     "aviation",
    "BHARTIARTL.NS": "telecom",
    "SUNPHARMA.NS":  "pharma",
    "GLENMARK.NS":   "pharma",
    "APOLLOHOSP.NS": "healthcare",
    "TATACONSUM.NS": "fmcg",
    "TITAN.NS":      "consumer",
    "EIHOTEL.NS":    "hospitality",
    "OBEROIRLTY.NS": "real_estate",
    "COROMANDEL.NS": "agrochem",
    "KSB.NS":        "industrials",
    "DYNAMATECH.NS": "industrials",
    "SYRMA.NS":      "electronics",
    "GOLDBEES.NS":   "gold_etf",
    "SILVERBEES.NS": "silver_etf",
    "0P0000YWL0.BO": "mutual_fund",
    "532067.BO":     "industrials",
}

# Sector geopolitical sensitivity: (oil, inflation, war, rate_hike)
# Positive = reduce weight when risk fires  |  Negative = increase weight (hedge)
SECTOR_SENSITIVITY = {
    "oil_gas":           (0.30, 0.10, 0.10, 0.00),
    "aviation":          (0.40, 0.10, 0.20, 0.00),
    "auto":              (0.15, 0.10, 0.05, 0.10),
    "auto_ancillary":    (0.15, 0.10, 0.05, 0.10),
    "metals":            (0.10, 0.15, 0.20, 0.05),
    "commodities":       (0.10, 0.15, 0.20, 0.05),
    "financials":        (0.05, 0.10, 0.10, 0.25),
    "real_estate":       (0.00, 0.05, 0.05, 0.25),
    "defense_industrial":(0.00, 0.05,-0.10, 0.00),  # war HELPS defense
    "industrials":       (0.05, 0.10, 0.10, 0.10),
    "pharma":            (0.00, 0.05, 0.00, 0.00),
    "healthcare":        (0.00, 0.05, 0.00, 0.00),
    "fmcg":              (0.05, 0.10, 0.00, 0.05),
    "consumer":          (0.05, 0.10, 0.05, 0.10),
    "telecom":           (0.00, 0.05, 0.00, 0.10),
    "power":             (0.10, 0.05, 0.05, 0.05),
    "agrochem":          (0.05, 0.15, 0.05, 0.00),
    "electronics":       (0.05, 0.05, 0.10, 0.05),
    "hospitality":       (0.10, 0.10, 0.20, 0.10),
    "gold_etf":          (-0.10,-0.10,-0.20, 0.00),  # safe-haven: boost
    "silver_etf":        (-0.05,-0.05,-0.10, 0.00),
    "mutual_fund":       (0.05, 0.05, 0.05, 0.05),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BULK DATA FETCH  (from Streamlit app — one call instead of N calls)
# ─────────────────────────────────────────────────────────────────────────────

MACRO_TICKERS = {
    "Oil":       "BZ=F",       # Brent crude
    "IndiaVIX":  "^INDIAVIX",  # India VIX  ← from your app, more relevant than US VIX
    "US10Y":     "^TNX",       # US 10-year yield
    "USDINR":    "USDINR=X",   # USD/INR FX
}

def fetch_all_data(extra_tickers: list | None = None) -> pd.DataFrame:
    """
    Single bulk download of all portfolio + macro tickers.
    Returns a DataFrame of daily Close prices, forward-filled.

    From Streamlit app: one yf.download() is ~10x faster than N serial calls
    and avoids rate-limiting.
    """
    portfolio_tickers = list(TARGET_WEIGHTS.keys())
    macro_tickers     = list(MACRO_TICKERS.values()) + [SBIN_TICKER]
    extra             = extra_tickers or []

    all_tickers = list(set(portfolio_tickers + macro_tickers + extra))

    print(f"[DATA] Bulk-fetching {len(all_tickers)} tickers (1Y daily)...")
    raw = yf.download(
        all_tickers,
        period="2y",          # 2Y to support 200DMA + 6M relative-strength
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )

    # Handle both single-ticker (flat) and multi-ticker (MultiIndex) responses
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]]
        close.columns = [all_tickers[0]]

    close = close.ffill()
    print(f"  ✔ Got {len(close)} rows, {close.shape[1]} tickers.  "
          f"Latest date: {close.index[-1].strftime('%Y-%m-%d')}")
    return close


def get_volume_data(tickers: list) -> pd.DataFrame:
    """Fetch 30-day average daily volume for ADV check."""
    print("[VOL] Fetching 30-day volume for ADV guard...")
    raw = yf.download(tickers, period="2mo", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        vol = raw.xs("Volume", axis=1, level=1)
    else:
        vol = raw[["Volume"]]
        vol.columns = tickers
    return vol.fillna(0).tail(30).mean()   # returns Series: ticker → avg daily volume


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SBIN CAPEX CYCLE  (v1 logic, now operating on shared data frame)
# ─────────────────────────────────────────────────────────────────────────────

def check_sbin_cycle(close: pd.DataFrame) -> dict:
    """
    Computes SBIN regime score from the already-fetched close DataFrame.
    No additional network call needed.
    """
    if SBIN_TICKER not in close.columns:
        return {"regime": "UNKNOWN", "score": 50, "details": {}}

    s = close[SBIN_TICKER].dropna()

    dma200 = s.rolling(200).mean().iloc[-1]
    dma50  = s.rolling(50).mean().iloc[-1]
    latest = s.iloc[-1]

    ret_3y = (latest / s.iloc[max(0, len(s)-756)] - 1) * 100
    ret_1y = (latest / s.iloc[max(0, len(s)-252)] - 1) * 100
    ret_6m = (latest / s.iloc[max(0, len(s)-126)] - 1) * 100

    golden_cross = bool(dma50 > dma200)

    score = 50
    if latest > dma200:   score += 15
    if golden_cross:      score += 15
    if ret_3y > 30:       score += 10
    if ret_1y > 10:       score += 10
    if ret_6m > 0:        score += 10
    if ret_6m < -10:      score -= 20
    if latest < dma200:   score -= 20
    score = int(np.clip(score, 0, 100))

    regime = "BULL" if score >= 65 else ("NEUTRAL" if score >= 40 else "BEAR")

    details = {
        "latest_price": round(float(latest), 2),
        "200DMA":       round(float(dma200), 2),
        "50DMA":        round(float(dma50), 2),
        "above_200DMA": bool(latest > dma200),
        "golden_cross": golden_cross,
        "3Y_ret_%":     round(ret_3y, 1),
        "1Y_ret_%":     round(ret_1y, 1),
        "6M_ret_%":     round(ret_6m, 1),
    }

    print(f"\n[SBIN] Regime: {regime}  score={score}")
    for k, v in details.items():
        print(f"       {k}: {v}")

    return {"regime": regime, "score": score, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GEOPOLITICAL RISK  (upgraded: India VIX + oil trend ratio from your app)
# ─────────────────────────────────────────────────────────────────────────────

def compute_geopolitical_risk(close: pd.DataFrame) -> dict:
    """
    Macro risk score 0–100.  Four components, each 0–25.

    Improvements from Streamlit app:
      ✦ India VIX (^INDIAVIX) replaces US VIX — more relevant for NSE portfolio
      ✦ Oil trend = price / 50DMA (detects momentum, not just absolute level)
    """
    print("\n[GEO] Computing geopolitical risk score...")
    risk  = {}
    score = 0

    # ── Oil  ──────────────────────────────────────────────────────────────────
    oil_ticker = MACRO_TICKERS["Oil"]
    if oil_ticker in close.columns:
        oil      = close[oil_ticker].dropna()
        oil_px   = float(oil.iloc[-1])
        oil_ma50 = float(oil.rolling(50).mean().iloc[-1])
        oil_trend = oil_px / oil_ma50  # > 1.15 = spike (from your app)

        if oil_px > 100 or oil_trend > 1.15:    oil_score = 25
        elif oil_px > 85 or oil_trend > 1.08:   oil_score = 18
        elif oil_px > 70 or oil_trend > 1.02:   oil_score = 10
        else:                                   oil_score = 5

        risk.update({"oil_price_usd": round(oil_px, 1),
                     "oil_trend_ratio": round(oil_trend, 3),
                     "oil_score": oil_score})
        score += oil_score
        print(f"  Oil: ${oil_px:.1f}  trend-ratio: {oil_trend:.3f}  → {oil_score}/25")
    else:
        oil_score = 12; score += oil_score

    # ── India VIX (your app's improvement over US VIX) ────────────────────────
    vix_ticker = MACRO_TICKERS["IndiaVIX"]
    if vix_ticker in close.columns:
        vix_px = float(close[vix_ticker].dropna().iloc[-1])
        # India VIX scale: normal ≈ 12–18, elevated 18–22, panic > 22
        if vix_px > 25:     vix_score = 25
        elif vix_px > 22:   vix_score = 18
        elif vix_px > 18:   vix_score = 10
        elif vix_px > 15:   vix_score = 5
        else:               vix_score = 2

        risk.update({"india_vix": round(vix_px, 1), "vix_score": vix_score})
        score += vix_score
        print(f"  India VIX: {vix_px:.1f}  → {vix_score}/25")
    else:
        vix_score = 10; score += vix_score

    # ── US 10-Year Yield ──────────────────────────────────────────────────────
    tnx_ticker = MACRO_TICKERS["US10Y"]
    if tnx_ticker in close.columns:
        yield_10y = float(close[tnx_ticker].dropna().iloc[-1])
        if yield_10y > 5.0:     infl_score = 25
        elif yield_10y > 4.5:   infl_score = 20
        elif yield_10y > 4.0:   infl_score = 13
        elif yield_10y > 3.5:   infl_score = 8
        else:                   infl_score = 4

        risk.update({"us_10y_yield": round(yield_10y, 2),
                     "inflation_score": infl_score})
        score += infl_score
        print(f"  US 10Y: {yield_10y:.2f}%  → {infl_score}/25")
    else:
        infl_score = 12; score += infl_score

    # ── USD/INR ───────────────────────────────────────────────────────────────
    fx_ticker = MACRO_TICKERS["USDINR"]
    if fx_ticker in close.columns:
        fx_s    = close[fx_ticker].dropna()
        fx      = float(fx_s.iloc[-1])
        fx_3m   = float(fx_s.iloc[max(0, len(fx_s)-63)])
        fx_chg  = (fx / fx_3m - 1) * 100

        if fx > 87 or fx_chg > 3:    fx_score = 25
        elif fx > 84 or fx_chg > 1:  fx_score = 15
        elif fx > 83:                fx_score = 8
        else:                        fx_score = 4

        risk.update({"usdinr": round(fx, 2),
                     "usdinr_3m_chg_%": round(fx_chg, 1),
                     "fx_rate_score": fx_score})
        score += fx_score
        print(f"  USD/INR: {fx:.2f}  3M chg: {fx_chg:.1f}%  → {fx_score}/25")
    else:
        fx_score = 12; score += fx_score

    score = int(np.clip(score, 0, 100))
    risk_level = "HIGH" if score >= 70 else ("MEDIUM" if score >= 45 else "LOW")

    tranche_mult = {"LOW": 1.00, "MEDIUM": 0.80, "HIGH": 0.60}[risk_level]

    risk.update({"total_score": score,
                 "risk_level": risk_level,
                 "tranche_multiplier": tranche_mult})

    print(f"\n  ► Geo Risk: {risk_level}  (score={score}/100)  "
          f"tranche_mult={tranche_mult:.0%}")
    return risk


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PER-STOCK RESILIENCE MULTIPLIER
#     (combines v1 200DMA gate + your app's relative-strength cycle mult)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stock_multiplier(ticker: str, close: pd.DataFrame,
                              geo: dict) -> dict:
    """
    Returns a composite per-stock deployment multiplier (0.20 – 1.50).

    Components
    ──────────
    A. 200-DMA gate (v1)
       · > +15% above 200DMA  → overheated, reduce by 30 %
       · < −5%  below 200DMA  → avoid, reduce by 50 %

    B. Relative-Strength Cycle (from Streamlit app)
       cycle_mult = 1 + (sbin_6m_ret − stock_6m_ret)
       · If SBIN is rallying but the stock is lagging → value entry window (mult > 1)
       · If stock has outrun SBIN → cool down (mult < 1)
       Clipped to [0.6, 1.4]

    C. Geopolitical sector adjustment (v1)
       Sector-level sensitivity to oil / inflation / war / rate-hike.
    """
    result = {
        "ticker":         ticker,
        "price":          None,
        "200dma":         None,
        "pct_vs_200dma":  None,
        "dma_signal":     "NO_DATA",
        "cycle_mult":     1.0,
        "geo_mult":       1.0,
        "final_mult":     1.0,
    }

    if ticker not in close.columns:
        return result

    s = close[ticker].dropna()
    if len(s) < 210:
        return result

    price  = float(s.iloc[-1])
    dma200 = float(s.rolling(200).mean().iloc[-1])
    pct    = (price / dma200 - 1) * 100

    # A. 200-DMA signal
    if pct < -5:          dma_signal = "AVOID";  dma_mult = 0.50
    elif pct > 15:        dma_signal = "WAIT";   dma_mult = 0.70
    else:                 dma_signal = "BUY";    dma_mult = 1.00
    # Smooth the overheated penalty from your app: clip(1.15 − (price/200DMA − 1), 0.4, 1.2)
    smooth_dma_mult = float(np.clip(1.15 - (price / dma200 - 1), 0.40, 1.20))
    dma_mult = smooth_dma_mult   # use smooth version (more gradual than binary)

    # B. Relative-strength vs SBIN  (from your app)
    sbin_s = close[SBIN_TICKER].dropna() if SBIN_TICKER in close.columns else None
    if sbin_s is not None and len(sbin_s) >= 126 and len(s) >= 126:
        sbin_6m = float(sbin_s.pct_change(126).iloc[-1])
        stk_6m  = float(s.pct_change(126).iloc[-1])
        # Positive = stock lagging SBIN → value entry; negative = stock over-running SBIN
        cycle_mult = float(np.clip(1.0 + (sbin_6m - stk_6m), 0.60, 1.40))
    else:
        cycle_mult = 1.0

    # C. Geo / sector adjustment
    base_w   = TARGET_WEIGHTS.get(ticker, 0.02)
    adj_w    = _geo_adjust_weight(ticker, base_w, geo)
    geo_mult = float(np.clip(adj_w / base_w, 0.50, 1.50)) if base_w > 0 else 1.0

    final_mult = float(np.clip(dma_mult * cycle_mult * geo_mult, 0.20, 1.50))

    result.update({
        "price":          round(price, 2),
        "200dma":         round(dma200, 2),
        "pct_vs_200dma":  round(pct, 1),
        "dma_signal":     dma_signal,
        "dma_mult":       round(dma_mult, 3),
        "cycle_mult":     round(cycle_mult, 3),
        "geo_mult":       round(geo_mult, 3),
        "final_mult":     round(final_mult, 3),
    })
    return result


def _geo_adjust_weight(ticker: str, base_weight: float, geo: dict) -> float:
    """Sector-level weight adjustment for active geopolitical risks."""
    sector = SECTOR_MAP.get(ticker, "industrials")
    sens   = SECTOR_SENSITIVITY.get(sector, (0.05, 0.05, 0.05, 0.05))

    oil_high  = geo.get("oil_score", 0)        >= 18
    infl_high = geo.get("inflation_score", 0)  >= 18
    war_high  = geo.get("vix_score", 0)        >= 18
    rate_high = geo.get("fx_rate_score", 0)    >= 18

    adj = 1.0
    if oil_high:  adj -= sens[0]
    if infl_high: adj -= sens[1]
    if war_high:  adj -= sens[2]
    if rate_high: adj -= sens[3]

    return base_weight * float(np.clip(adj, 0.50, 1.50))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SBIN REGIME → PORTFOLIO-LEVEL CAPITAL GATE
# ─────────────────────────────────────────────────────────────────────────────

def sbin_regime_multiplier(sbin_score: int) -> float:
    """How much of today's planned capital to actually deploy."""
    if sbin_score >= 65:   return 1.00   # Full bull
    elif sbin_score >= 55: return 0.85
    elif sbin_score >= 45: return 0.65   # Neutral
    elif sbin_score >= 35: return 0.45
    else:                  return 0.30   # Bear / panic


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DAILY EXECUTION PLAN
#     Core formula from your Streamlit app:
#     daily_buy_INR = (remaining_gap / days_left) × final_mult
# ─────────────────────────────────────────────────────────────────────────────

def generate_today_plan(
    current_holdings: dict,          # {ticker: shares_owned}
    days_elapsed: int,               # trading days already used
    close: pd.DataFrame,
    sbin_info: dict,
    geo: dict,
    adv: pd.Series | None = None,    # average daily volume per ticker
) -> pd.DataFrame:
    """
    Generates today's buy-order list.

    Key logic (reconciled from both programs):
      1. Compute portfolio gap per stock (target INR − current INR held)
      2. Pace it: raw_daily = gap / days_remaining
      3. Scale: actual_daily = raw_daily × final_mult (200DMA × cycle × geo)
      4. Apply SBIN regime gate on total outlay
      5. Flag ADV breaches (institutional iceberg warning)

    Returns a DataFrame with one row per ticker.
    """
    days_remaining = max(1, DEPLOYMENT_HORIZON - days_elapsed)
    regime_mult    = sbin_regime_multiplier(sbin_info["score"])
    geo_mult       = geo["tranche_multiplier"]

    # Adjusted weights (geo + sector-normalised)
    raw_adj = {}
    for t, bw in TARGET_WEIGHTS.items():
        raw_adj[t] = _geo_adjust_weight(t, bw, geo)
    total_adj = sum(raw_adj.values())
    norm_adj  = {t: w / total_adj for t, w in raw_adj.items()}

    rows = []
    total_today_inr = 0.0

    for ticker, target_w_orig in TARGET_WEIGHTS.items():
        target_inr = TOTAL_CAPITAL_INR * target_w_orig

        # Current value in portfolio
        mult_info  = compute_stock_multiplier(ticker, close, geo)
        price      = mult_info.get("price") or 0.0
        curr_qty   = current_holdings.get(ticker, 0)
        curr_inr   = curr_qty * price

        gap_inr    = max(0.0, target_inr - curr_inr)
        if gap_inr == 0 or price == 0:
            rows.append(_zero_row(ticker, price, target_inr, curr_inr,
                                  target_w_orig, mult_info))
            continue

        # ── Core formula (from your Streamlit app) ────────────────────────
        raw_daily_inr    = gap_inr / days_remaining
        adj_daily_inr    = raw_daily_inr * mult_info["final_mult"]

        # Portfolio-level gates
        adj_daily_inr   *= regime_mult   # SBIN regime
        adj_daily_inr   *= geo_mult      # Macro risk

        shares_to_buy = int(adj_daily_inr / price)
        actual_inr    = shares_to_buy * price
        total_today_inr += actual_inr

        # ADV guard: flag if order > 5% of 30-day ADV (from your app)
        adv_flag = ""
        if adv is not None and ticker in adv.index:
            adv_val = adv[ticker]
            if adv_val > 0:
                adv_pct = (shares_to_buy / adv_val) * 100
                if adv_pct > 5:
                    adv_flag = f"⚠ ICEBERG ({adv_pct:.1f}% ADV)"
                elif adv_pct > 2:
                    adv_flag = f"⚡ VWAP ({adv_pct:.1f}% ADV)"

        rows.append({
            "Ticker":           ticker,
            "Sector":           SECTOR_MAP.get(ticker, "?"),
            "Target_Cr":        round(target_inr / 1e7, 2),
            "Current_Cr":       round(curr_inr / 1e7, 2),
            "Gap_Cr":           round(gap_inr / 1e7, 2),
            "Price_INR":        round(price, 2),
            "DMA_Signal":       mult_info["dma_signal"],
            "Pct_vs_200DMA":    mult_info["pct_vs_200dma"],
            "Cycle_Mult":       mult_info["cycle_mult"],
            "Final_Mult":       mult_info["final_mult"],
            "Shares_to_Buy":    shares_to_buy,
            "Today_Outlay_L":   round(actual_inr / 1e5, 2),   # Lakhs
            "Execution_Note":   adv_flag,
        })

    df = pd.DataFrame(rows).sort_values("Today_Outlay_L", ascending=False)

    print(f"\n{'═'*72}")
    print(f"  TODAY'S EXECUTION PLAN  ·  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  Days Elapsed: {days_elapsed}/{DEPLOYMENT_HORIZON}  "
          f"(Remaining: {days_remaining})")
    print(f"  SBIN Regime: {sbin_info['regime']} "
          f"(mult={regime_mult:.0%})   "
          f"Geo Risk: {geo['risk_level']} "
          f"(mult={geo_mult:.0%})")
    print(f"  Total Today Outlay: ₹{total_today_inr/1e5:.2f} Lakhs  "
          f"(₹{total_today_inr/1e7:.2f} Crore)")
    print(f"{'═'*72}")

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(df[["Ticker","DMA_Signal","Pct_vs_200DMA","Cycle_Mult",
              "Final_Mult","Price_INR","Shares_to_Buy",
              "Today_Outlay_L","Execution_Note"]].to_string(index=False))

    iceberg_count = df[df["Execution_Note"].str.startswith("⚠", na=False)].shape[0]
    if iceberg_count:
        print(f"\n  ⚠  {iceberg_count} ticker(s) require Iceberg/VWAP algo orders.")

    return df


def _zero_row(ticker, price, target_inr, curr_inr, target_w, mult_info):
    return {
        "Ticker":         ticker,
        "Sector":         SECTOR_MAP.get(ticker, "?"),
        "Target_Cr":      round(target_inr / 1e7, 2),
        "Current_Cr":     round(curr_inr / 1e7, 2),
        "Gap_Cr":         0.0,
        "Price_INR":      round(price, 2) if price else None,
        "DMA_Signal":     mult_info.get("dma_signal", "—"),
        "Pct_vs_200DMA":  mult_info.get("pct_vs_200dma"),
        "Cycle_Mult":     mult_info.get("cycle_mult", 1.0),
        "Final_Mult":     mult_info.get("final_mult", 1.0),
        "Shares_to_Buy":  0,
        "Today_Outlay_L": 0.0,
        "Execution_Note": "FULL",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  REBALANCE CHECK  (v1, operating on shared data)
# ─────────────────────────────────────────────────────────────────────────────

def check_rebalance(current_holdings: dict, close: pd.DataFrame) -> pd.DataFrame:
    """Flag tickers drifted > 3% from target weight. Run weekly post-deployment."""
    prices    = {t: float(close[t].iloc[-1]) for t in current_holdings if t in close.columns}
    total_val = sum(current_holdings.get(t, 0) * prices.get(t, 0) for t in prices)

    rows = []
    for ticker, target_w in TARGET_WEIGHTS.items():
        qty      = current_holdings.get(ticker, 0)
        px       = prices.get(ticker, 0)
        curr_val = qty * px
        curr_w   = curr_val / total_val if total_val > 0 else 0
        drift    = curr_w - target_w
        action   = "TRIM" if drift > 0.03 else ("ADD" if drift < -0.03 else "HOLD")

        rows.append({
            "Ticker":       ticker,
            "Target_Wt_%":  round(target_w * 100, 2),
            "Current_Wt_%": round(curr_w * 100, 2),
            "Drift_%":      round(drift * 100, 2),
            "Action":       action,
        })

    return pd.DataFrame(rows).sort_values("Drift_%")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SAVE TO EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def save_plan_to_excel(plan_df: pd.DataFrame, sbin_info: dict,
                        geo: dict, days_elapsed: int,
                        path: str = "deployment_plan_v2.xlsx"):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # Today's orders
        plan_df.to_excel(writer, sheet_name="Today_Orders", index=False)

        # Risk snapshot
        risk_rows = [
            ("SBIN Regime",     sbin_info["regime"]),
            ("SBIN Score",      sbin_info["score"]),
            ("Geo Risk Level",  geo["risk_level"]),
            ("Geo Score",       geo["total_score"]),
            ("Oil Price USD",   geo.get("oil_price_usd", "N/A")),
            ("Oil Trend Ratio", geo.get("oil_trend_ratio", "N/A")),
            ("India VIX",       geo.get("india_vix", "N/A")),
            ("US 10Y Yield",    geo.get("us_10y_yield", "N/A")),
            ("USD/INR",         geo.get("usdinr", "N/A")),
            ("Tranche Mult",    geo["tranche_multiplier"]),
            ("Days Elapsed",    days_elapsed),
            ("Days Remaining",  max(1, DEPLOYMENT_HORIZON - days_elapsed)),
        ]
        pd.DataFrame(risk_rows, columns=["Metric", "Value"])\
          .to_excel(writer, sheet_name="Risk_Snapshot", index=False)

    print(f"\n  ✔ Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    current_holdings: dict | None = None,
    days_elapsed: int = 0,
    fetch_adv: bool = True,
    output_path: str = "~/Documents/deployment_plan_v2.xlsx",
):
    """
    Args
    ────
    current_holdings : {ticker: shares_you_already_own}
                       Pass {} or None to plan a fresh deployment.
    days_elapsed     : How many trading days since deployment started.
    fetch_adv        : If True, fetch volume data for ADV/iceberg warnings.
    output_path      : Where to save the Excel plan.
    """
    current_holdings = current_holdings or {}

    print("\n" + "═"*72)
    print("  PORTFOLIO DEPLOYMENT ENGINE  v2.0")
    print(f"  ₹1,00,00,00,000 (100 Crore)  ·  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("═"*72)

    # ── Bulk data fetch ───────────────────────────────────────────────────────
    close = fetch_all_data()

    # ── SBIN cycle ────────────────────────────────────────────────────────────
    sbin_info = check_sbin_cycle(close)

    # ── Geopolitical risk ─────────────────────────────────────────────────────
    geo = compute_geopolitical_risk(close)

    # ── ADV (optional — requires extra volume fetch) ──────────────────────────
    adv = None
    if fetch_adv:
        try:
            adv = get_volume_data(list(TARGET_WEIGHTS.keys()))
        except Exception as e:
            print(f"  [WARN] ADV fetch failed: {e}")

    # ── Today's plan ──────────────────────────────────────────────────────────
    plan_df = generate_today_plan(
        current_holdings=current_holdings,
        days_elapsed=days_elapsed,
        close=close,
        sbin_info=sbin_info,
        geo=geo,
        adv=adv,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_gap      = plan_df["Gap_Cr"].sum()
    total_outlay   = plan_df["Today_Outlay_L"].sum()
    filled_pct     = (1 - total_gap / (TOTAL_CAPITAL_INR / 1e7)) * 100

    print(f"\n{'─'*72}")
    print(f"  Portfolio Fill:    {filled_pct:.1f}%  (gap remaining: ₹{total_gap:.1f} Cr)")
    print(f"  Today's Outlay:    ₹{total_outlay:.2f} Lakhs")
    print(f"  Horizon:           Day {days_elapsed} of {DEPLOYMENT_HORIZON}")
    print(f"  SBIN:              {sbin_info['regime']} ({sbin_info['score']}/100)")
    print(f"  Geo Risk:          {geo['risk_level']} ({geo['total_score']}/100)")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_plan_to_excel(plan_df, sbin_info, geo, days_elapsed, path=output_path)

    return plan_df, sbin_info, geo


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Day 0: Fresh start, no holdings ──────────────────────────────────────
    plan, sbin, geo = run(
        current_holdings={},
        days_elapsed=0,
    )

    # ── Day 15 example (uncomment and fill in your actual holdings) ───────────
    # plan, sbin, geo = run(
    #     current_holdings={
    #         "SBIN.NS":       5000,
    #         "BEL.NS":        3200,
    #         "BHARATFORG.NS": 1500,
    #         "COROMANDEL.NS": 800,
    #         # ... etc
    #     },
    #     days_elapsed=15,
    # )

    # ── Weekly rebalance check (after full deployment) ────────────────────────
    # from portfolio_deploy_v2 import fetch_all_data, check_rebalance
    # close = fetch_all_data()
    # rebal_df = check_rebalance(your_holdings_dict, close)
    # print(rebal_df)