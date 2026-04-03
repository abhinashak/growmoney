import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Sector Rotation Strategy Engine")

BENCHMARK = "^NSEI"

# -------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------
for key in ("result", "config_snapshot", "blend_note"):
    if key not in st.session_state:
        st.session_state[key] = None


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

@st.cache_data
def get_price_series(ticker):
    try:
        ticker = str(ticker).upper().strip()
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return None
        series = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return series if len(series) >= 65 else None
    except Exception:
        return None


def compute_returns(series):
    try:
        if series is None or len(series) < 65:
            return None
        r1d = float(series.pct_change().iloc[-1]) * 100
        r1m = float(series.iloc[-1] / series.iloc[-21] - 1) * 100
        r3m = float(series.iloc[-1] / series.iloc[-63] - 1) * 100
        if any(not np.isfinite(v) for v in [r1d, r1m, r3m]):
            return None
        return r1d, r1m, r3m
    except Exception:
        return None


@st.cache_data
def get_stock_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or info.get("forwardPE")
        mcap = info.get("marketCap")
        return pe, mcap
    except Exception:
        return None, None


def fmt_mcap(val):
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"₹{val/1e12:.2f}T"
    if val >= 1e9:
        return f"₹{val/1e9:.2f}B"
    if val >= 1e7:
        return f"₹{val/1e7:.2f}Cr"
    return f"₹{val:,.0f}"


# -------------------------------------------------------
# BENCHMARK
# -------------------------------------------------------

bench_series  = get_price_series(BENCHMARK)
bench_returns = compute_returns(bench_series)

st.sidebar.header("Benchmark Returns")
if bench_returns:
    st.sidebar.write({
        "1D": round(bench_returns[0], 2),
        "1M": round(bench_returns[1], 2),
        "3M": round(bench_returns[2], 2),
    })
else:
    st.sidebar.warning("Benchmark data unavailable — signals will default to HOLD.")


# -------------------------------------------------------
# STRATEGY INPUT
# -------------------------------------------------------

st.header("Strategy Input")

default_df = pd.DataFrame({
    "Sector":            ["Defense",        "Metals"],
    "Weight":            [20,               20],
    "Logic & Selection": ["Defense exports", "Commodity cycle"],
    "Exit News":         ["Order slowdown",  "China slowdown"],
    "Stock1":            ["HAL.NS",          "HINDALCO.NS"],
    "Stock2":            ["BEL.NS",          "VEDL.NS"],
    "Stock3":            ["MAZDOCK.NS",      "NATIONALUM.NS"],
    "Stock4":            ["SOLARINDS.NS",    "NMDC.NS"],
    "Stock5":            ["BHARATFORG.NS",   "TATASTEEL.NS"],
})

uploaded = st.file_uploader("Upload strategy CSV (optional)")

if uploaded:
    raw = pd.read_csv(uploaded, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    raw = raw.rename(columns={"Sectors": "Sector", "Exit News to Watch": "Exit News"})
    st.info("CSV loaded — edit below before running if needed.")
    config = st.data_editor(raw, num_rows="dynamic", key="editor")
else:
    config = st.data_editor(default_df, num_rows="dynamic", key="editor")


# -------------------------------------------------------
# RUN STRATEGY — saves result into session_state
# -------------------------------------------------------

if st.button("Run Strategy Engine"):

    strategies = []
    stock_cols = ["Stock1", "Stock2", "Stock3", "Stock4", "Stock5"]

    for _, row in config.iterrows():
        stocks = [row.get(c) for c in stock_cols]
        returns_list = []

        for ticker in stocks:
            if pd.isna(ticker) or str(ticker).strip() == "":
                continue
            series = get_price_series(ticker)
            if series is None:
                continue
            r = compute_returns(series)
            if r is not None and len(r) == 3:
                returns_list.append({"1D": r[0], "1M": r[1], "3M": r[2]})

        if len(returns_list) < 2:
            continue

        df = pd.DataFrame(returns_list, columns=["1D", "1M", "3M"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            continue

        avg       = df.mean()
        score     = 0.5 * avg["3M"] + 0.3 * avg["1M"] + 0.2 * avg["1D"]
        exp_month = score / 3
        exp_year  = exp_month * 12
        vol       = max(df.std().mean(), 1e-6)
        sharpe    = exp_month / vol
        kelly     = max(exp_month / (vol ** 2), 0)

        signal = "HOLD"
        if bench_returns:
            if (avg["1D"] > bench_returns[0]
                    and avg["1M"] > bench_returns[1]
                    and avg["3M"] > bench_returns[2]):
                signal = "BUY"
            if avg["3M"] < bench_returns[2]:
                signal = "SELL"

        try:
            user_weight = float(row.get("Weight", 0))
            if not np.isfinite(user_weight) or user_weight < 0:
                user_weight = 0.0
        except (TypeError, ValueError):
            user_weight = 0.0

        strategies.append({
            "Sector":            row["Sector"],
            "User Weight %":     user_weight,
            "1D":                round(avg["1D"],  2),
            "1M":                round(avg["1M"],  2),
            "3M":                round(avg["3M"],  2),
            "Score":             round(score,       2),
            "Expected Annual %": round(exp_year,   2),
            "Sharpe":            round(sharpe,      2),
            "Kelly Raw":         kelly,
            "Signal":            signal,
        })

    if len(strategies) == 0:
        st.warning("No valid strategy data — check tickers and try again.")
        st.stop()

    result = pd.DataFrame(strategies)
    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    # --- Blended allocation ---
    total_user_w = result["User Weight %"].sum()
    total_kelly  = result["Kelly Raw"].sum()

    result["User Norm %"]  = (result["User Weight %"] / total_user_w * 100
                              if total_user_w > 0 else 100.0 / len(result))
    result["Kelly Norm %"] = (result["Kelly Raw"] / total_kelly * 100
                              if total_kelly > 0 else 100.0 / len(result))

    if total_user_w > 0:
        USER_W, KELLY_W = 0.70, 0.30
        blend_note = "70% your defined weights + 30% Kelly momentum tilt"
    else:
        USER_W, KELLY_W = 0.00, 1.00
        blend_note = "All user weights were 0 — using pure Kelly allocation"

    result["Blended %"] = USER_W * result["User Norm %"] + KELLY_W * result["Kelly Norm %"]
    result.loc[result["Signal"] == "SELL", "Blended %"] = 0.0

    total_blended = result["Blended %"].sum()
    result["Final Allocation %"] = (
        (result["Blended %"] / total_blended * 100).round(2)
        if total_blended > 0 else 0.0
    )

    # Persist to session state so other buttons don't wipe it
    st.session_state.result           = result
    st.session_state.config_snapshot  = config.copy()
    st.session_state.blend_note       = blend_note


# -------------------------------------------------------
# DISPLAY RESULTS  (reads from session_state — survives reruns)
# -------------------------------------------------------

if st.session_state.result is not None:
    result     = st.session_state.result
    config_ss  = st.session_state.config_snapshot
    blend_note = st.session_state.blend_note

    st.header("Strategy Ranking")
    st.dataframe(result, use_container_width=True)

    st.header("Capital Allocation")
    alloc = result[["Sector", "Signal", "User Weight %",
                    "Score", "Expected Annual %", "Final Allocation %"]]
    st.dataframe(alloc, use_container_width=True)
    st.caption(
        f"Final Allocation = {blend_note}. "
        "SELL-signal sectors are excluded and their weight redistributed proportionally."
    )

    buy = result[result["Signal"] == "BUY"]
    if not buy.empty:
        exp_return = np.average(buy["Expected Annual %"], weights=buy["Final Allocation %"])
        st.metric("Expected Portfolio Return %", round(exp_return, 2))
    else:
        st.info("No BUY signals — portfolio return not computed.")

    st.download_button(
        "Download Strategy Results",
        result.to_csv(index=False),
        file_name="strategy_results.csv",
    )

    # -------------------------------------------------------
    # INVESTMENT DISTRIBUTION
    # -------------------------------------------------------

    st.header("Investment Distribution")

    total_investment = st.number_input(
        "Enter Total Investment Amount (₹)",
        min_value=1000,
        value=100000,
        step=1000,
        format="%d",
        key="invest_amount",
    )

    if st.button("Calculate Investment Distribution"):

        active = result[
            (result["Signal"] != "SELL") & (result["Final Allocation %"] > 0)
        ].copy()

        if active.empty:
            st.warning("No active sectors to distribute investment.")
        else:
            stock_cols = ["Stock1", "Stock2", "Stock3", "Stock4", "Stock5"]
            rows = []

            for _, sec_row in active.iterrows():
                sector     = sec_row["Sector"]
                sec_amount = total_investment * sec_row["Final Allocation %"] / 100

                cfg_row = config_ss[config_ss["Sector"] == sector]
                if cfg_row.empty:
                    continue
                cfg_row = cfg_row.iloc[0]

                tickers = [
                    str(cfg_row.get(c, "")).strip()
                    for c in stock_cols
                    if not pd.isna(cfg_row.get(c))
                    and str(cfg_row.get(c, "")).strip() != ""
                ]
                if not tickers:
                    continue

                per_stock = sec_amount / len(tickers)
                for t in tickers:
                    rows.append({"Ticker": t, "Sector": sector, "Amount": per_stock})

            if not rows:
                st.warning("Could not build stock distribution.")
            else:
                dist_df = pd.DataFrame(rows)

                # Aggregate duplicate tickers (same stock in multiple sectors)
                dist_df = (
                    dist_df.groupby("Ticker", as_index=False)
                    .agg(
                        Sectors=("Sector", lambda x: ", ".join(sorted(set(x)))),
                        Amount =("Amount", "sum"),
                    )
                    .sort_values("Amount", ascending=False)
                    .reset_index(drop=True)
                )

                with st.spinner("Fetching P/E & Market Cap..."):
                    pe_list, mcap_list = [], []
                    for t in dist_df["Ticker"]:
                        pe, mc = get_stock_fundamentals(t)
                        pe_list.append(round(pe, 1) if pe and np.isfinite(pe) else None)
                        mcap_list.append(mc)

                dist_df["Invest (₹)"] = dist_df["Amount"].round(2)
                dist_df["P/E"]        = pe_list
                dist_df["Market Cap"] = [fmt_mcap(m) for m in mcap_list]

                st.dataframe(
                    dist_df[["Ticker", "Sectors", "Invest (₹)", "P/E", "Market Cap"]],
                    use_container_width=True,
                )

                # Summary metrics
                total_invested = dist_df["Invest (₹)"].sum()
                valid_pe = [
                    (pe, amt)
                    for pe, amt in zip(pe_list, dist_df["Invest (₹)"])
                    if pe is not None and np.isfinite(pe)
                ]
                avg_pe = round(np.average(*zip(*valid_pe)), 2) if valid_pe else "N/A"

                col1, col2 = st.columns(2)
                col1.metric("Total Invested (₹)", f"₹{total_invested:,.2f}")
                col2.metric("Weighted Avg P/E",   avg_pe)

                st.download_button(
                    "Download Distribution",
                    dist_df[["Ticker", "Sectors", "Invest (₹)", "P/E", "Market Cap"]].to_csv(index=False),
                    file_name="investment_distribution.csv",
                )