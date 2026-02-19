import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set page to wide mode for dashboard feel
st.set_page_config(layout="wide", page_title="Bloomberg-Style Asset Comparator")

# -----------------------------
# Default symbols 
# -----------------------------
SYMBOLS = {
    # 🇮🇳 Broad Market Indices
    "NIFTY 50": "^NSEI",
    "NIFTY 500": "0P0001IAU3.BO",
    "INDIA VIX": "^INDIAVIX",

    # 🇮🇳 Sector Indices (NSE)
    "NIFTY IT": "^CNXIT",
    "NIFTY Bank Index": "^NSEBANK",
    "NIFTY Financial Services": "^CNXFIN",
    "NIFTY Auto Index": "^CNXAUTO",
    "NIFTY Pharma Index": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Media": "^CNXMEDIA",
    "NIFTY Metal": "^CNXMETAL",
    "NIFTY Energy": "^CNXENERGY",
    "NIFTY Realty": "^CNXREALTY",
    "NIFTY PSU Bank": "^CNXPSUBANK",

    # 🇮🇳 ETFs
    "NIFTYBEES": "NIFTYBEES.NS",
    "Next 50": "JUNIORBEES.NS",
    "MID150BEES": "MID150BEES.NS",
    "NV20": "NV20BEES.NS",
    "BANKBEES": "BANKBEES.NS",
    "INFRABEES": "INFRABEES.NS",
    "PHARMABEES": "PHARMABEES.NS",
    "AUTOBEES": "AUTOBEES.NS",
    "Defence ETF": "MODEFENCE.NS",
    "GOLD": "GOLDBEES.NS",
    "SILVER": "SILVERBEES.NS",

    # 🇮🇳 Stocks
    "Airtel": "BHARTIARTL.NS",

    # 🇮🇳 Mutual Funds
    "PP Flexicap": "0P0000YWL1.BO",
    "SBI Hybrid": "0P0000XVJO.BO",
    "HDFC Balanced Adv": "0P0001EI15.BO",
    "Motilal Enhanced": "0P0001PHVM.BO",
    "Consumerables Fund": "0P0000XVT1.BO",

    # 🌍 Global Macro
    "DXY": "DX-Y.NYB",
}

# -----------------------------
# 🎨 Add Color Map 
# -----------------------------
COLOR_MAP = {
    "NIFTY 50": "#1f77b4", "NIFTY 500": "#0d3b66", "INDIA VIX": "#8b0000",
    "NIFTY IT": "#6f42c1", "NIFTY Bank Index": "#008080", "NIFTY Financial Services": "#20b2aa",
    "NIFTY Auto Index": "#ff7f0e", "NIFTY Pharma Index": "#d62728", "NIFTY FMCG": "#b8860b",
    "NIFTY Media": "#e83e8c", "NIFTY Metal": "#8c564b", "NIFTY Energy": "#2f4f4f",
    "NIFTY Realty": "#a0522d", "NIFTY PSU Bank": "#006400",
    "NIFTYBEES": "#1f77b4", "Next 50": "#17becf", "MID150BEES": "#2ca02c",
    "NV20": "#9467bd", "BANKBEES": "#008080", "INFRABEES": "#708090",
    "PHARMABEES": "#d62728", "AUTOBEES": "#ff7f0e", "Defence ETF": "#4b0082",
    "GOLD": "#d4af37", "SILVER": "#c0c0c0",
    "Airtel": "#dc143c",
    "PP Flexicap": "#4682b4", "SBI Hybrid": "#3cb371", "HDFC Balanced Adv": "#4169e1",
    "Motilal Enhanced": "#9370db", "Consumerables Fund": "#daa520",
    "DXY": "#000000",
}

# -----------------------------
# Sidebar controls & Logic
# -----------------------------
st.sidebar.header("📊 Select Assets")

selected_labels = []
for label in SYMBOLS:
    if st.sidebar.checkbox(label, value=(label in ["NIFTYBEES", "GOLD"])):
        selected_labels.append(label)

selected_symbols = [SYMBOLS[l] for l in selected_labels]

if len(selected_symbols) == 0:
    st.warning("Select at least one asset")
    st.stop()

# Moving Average Toggle
st.sidebar.markdown("---")
st.sidebar.header("📈 Chart Overlays")
ma_selection = st.sidebar.radio(
    "Moving Average", 
    ["None", "20-Day", "50-Day", "200-Day"], 
    index=0
)

# Parse MA days
ma_days = None
if ma_selection != "None":
    ma_days = int(ma_selection.split("-")[0])

st.title("📈 Market Dashboard")

# -----------------------------
# Date & Time Range Logic
# -----------------------------
selected_range = st.session_state.get("time_range", "1Y")
end_date = datetime.today().date() 

if selected_range == "1M":
    start_date = end_date - timedelta(days=30)
elif selected_range == "3M":
    start_date = end_date - timedelta(days=90)
elif selected_range == "6M":
    start_date = end_date - timedelta(days=180)
elif selected_range == "YTD":
    start_date = datetime(end_date.year, 1, 1).date()
elif selected_range == "1Y":
    start_date = end_date - timedelta(days=365)
elif selected_range == "3Y":
    start_date = end_date - timedelta(days=365*3)
elif selected_range == "5Y":
    start_date = end_date - timedelta(days=365*5)
elif selected_range == "10Y":
    start_date = end_date - timedelta(days=365*10)
elif selected_range == "Max":
    start_date = datetime(2000, 1, 1).date()
elif selected_range == "Custom":
    st.sidebar.markdown("---")
    start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1).date())
    end_date = st.sidebar.date_input("End Date", end_date)

# -----------------------------
# Data fetch & Cleaning 
# -----------------------------
@st.cache_data
def load_data(symbols, start, end):
    try:
        df = yf.download(symbols, start=start, end=end, progress=False)
        
        if df.empty or "Close" not in df:
            return pd.DataFrame()
            
        df = df["Close"]
        
        # Handle single symbol converting to a properly named DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(name=symbols[0])
        
        df = df.replace(0, np.nan).dropna()
        
        reverse_map = {v: k for k, v in SYMBOLS.items()}
        df.columns = [reverse_map.get(col, col) for col in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load data
prices = load_data(selected_symbols, start_date, end_date)

if prices.empty:
    st.error("No data available for the selected range.")
    st.stop()

# -----------------------------
# Bloomberg Header (Key Metrics)
# -----------------------------
st.markdown("### 🔀 Market Snapshot")
cols = st.columns(min(len(prices.columns), 6))

for idx, col_name in enumerate(prices.columns[:6]):
    latest_price = prices[col_name].iloc[-1]
    prev_price = prices[col_name].iloc[-2] if len(prices) > 1 else latest_price
    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100 if prev_price != 0 else 0
    
    with cols[idx]:
        st.metric(
            label=col_name,
            value=f"{latest_price:,.2f}",
            delta=f"{pct_change:.2f}%"
        )

st.markdown("---")

# -----------------------------
# Relative performance (Rebased) 
# -----------------------------
col_chart, col_stats = st.columns([3, 1])

with col_chart:
    st.subheader(f"Performance ({selected_range})")
    rebased = prices / prices.iloc[0] * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for column in rebased.columns:
        # Plot base line
        base_line, = ax.plot(rebased.index, rebased[column], label=column, 
            color=COLOR_MAP.get(column, None), linewidth=1.5)
        
        # Plot Moving Average if selected
        if ma_days is not None:
            # Calculate rolling average on the rebased data
            ma_series = rebased[column].rolling(window=ma_days).mean()
            ax.plot(rebased.index, ma_series, label=f"{column} ({ma_days}d MA)", 
                color=base_line.get_color(), linestyle='--', alpha=0.6, linewidth=1.2)
        
    ax.set_ylabel("Rebased (100 = Start)")
    ax.grid(True, linestyle=":", alpha=0.6, color='gray') 
    
    # Place legend outside to avoid clutter if multiple MAs are shown
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=True, fancybox=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

with col_stats:
    st.subheader("Return Stats")
    abs_return = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
    
    stats_df = pd.DataFrame({
        "Asset": abs_return.index,
        "Return %": abs_return.values
    }).sort_values("Return %", ascending=False)
    
    st.dataframe(
        stats_df.style.format({"Return %": "{:.2f}%"}).background_gradient(cmap="RdYlGn", subset=["Return %"]),
        hide_index=True,
        use_container_width=True
    )

# Time Range Buttons
ranges = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y", "Max", "Custom"]
st.radio("Select Period:", ranges, horizontal=True, key="time_range")


# -----------------------------
# Pairwise Analysis Section
# -----------------------------
st.markdown("---")
st.markdown("### 📋 Pairwise Analysis")

col1, col2 = st.columns(2)

with col1:
    base_ticker = st.text_input(
        "Base Asset (Type Any Yahoo Ticker)",
        placeholder="e.g. RELIANCE.NS",
        key="compare_left_input" 
    )

with col2:
    custom_ticker = st.text_input(
        "Compare With (Type Any Yahoo Ticker)",
        placeholder="e.g. TCS.NS",
        key="compare_right_input"
    )

pair_compare = None

if base_ticker and custom_ticker:
    try:
        custom_data_raw = yf.download(custom_ticker, start=start_date, end=end_date, progress=False)
        base_data_raw = yf.download(base_ticker, start=start_date, end=end_date, progress=False)

        if not custom_data_raw.empty and not base_data_raw.empty and "Close" in custom_data_raw and "Close" in base_data_raw:
            custom_data = custom_data_raw["Close"].replace(0, np.nan).dropna()
            custom_data.name = custom_ticker.upper()

            base_data = base_data_raw["Close"].replace(0, np.nan).dropna()
            base_data.name = base_ticker.upper()

            merged = pd.concat([base_data, custom_data], axis=1).dropna()
            
            if not merged.empty:
                rebased_pair = merged / merged.iloc[0] * 100
                pair_base = base_ticker.upper()
                pair_compare = custom_ticker.upper()
            else:
                st.warning("No overlapping data found for these dates.")
        else:
            st.warning("No valid data found for one or both tickers.")
    except Exception:
        st.error("Invalid ticker symbol or data fetch error.")

if pair_compare:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rebased_pair.index, rebased_pair[pair_base], label=pair_base, color='red', linewidth=2)
    ax.plot(rebased_pair.index, rebased_pair[pair_compare], label=pair_compare, linewidth=2, linestyle="--")
    ax.set_title(f"{pair_base} vs {pair_compare} (Relative)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
