import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import json
import math
import streamlit.components.v1 as components
from scipy.optimize import newton
import os

warnings.filterwarnings('ignore')

# -------------------------------
# Configuration & Setup
# -------------------------------
st.set_page_config(
    page_title="Professional Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state at the very beginning
if 'initial_portfolio' not in st.session_state:
    st.session_state.initial_portfolio = []
if 'stock_symbol' not in st.session_state:
    st.session_state.stock_symbol = ""
if 'num_stocks' not in st.session_state:
    st.session_state.num_stocks = 100
if 'entry_date' not in st.session_state:
    st.session_state.entry_date = datetime.now() - timedelta(days=365)


# Note: Some advanced features require additional libraries that may not be available in all environments
# If you encounter import errors, install them using: pip install talib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    st.warning("TA-Lib not available. Technical indicators will use simple implementations.")

# -------------------------------
# New Function to Load Tickers from a File
# -------------------------------
def load_tickers(file_path):
    """Loads ticker data from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Error: Ticker file '{file_path}' not found. Please create the file and add your tickers.")
        st.info("Using a small default list for demonstration.")
        return {
            "Infosys": {"symbol": "INFY.NS", "sector": "IT"},
            "HDFC Bank": {"symbol": "HDFCBANK.NS", "sector": "Financial Services"},
            "Reliance Industries": {"symbol": "RELIANCE.NS", "sector": "Energy"}
        }
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error: The file '{file_path}' is not a valid JSON file. Please check its content.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading tickers: {e}")
        return {}

# Load the tickers at the start of the script
tickers = load_tickers('tickers.json')

# -------------------------------
# Helper Functions
# -------------------------------

def find_nearest_date(date_to_find, date_index):
    """
    Finds the nearest date in a sorted DatetimeIndex that is less than or equal to the target date.
    This works as a fallback for get_loc(method='pad') in older pandas versions.
    """
    # Find the index of the first date that is greater than the date_to_find
    # If not found, all dates are <= date_to_find, so return the last date
    try:
        loc = date_index.searchsorted(date_to_find, side='right')
        # Ensure we don't go out of bounds and we find a date <= the target
        if loc == 0:
            # The target is before the first date in the index
            return date_index[0]
        return date_index[loc - 1]
    except IndexError:
        return date_index[-1]


@st.cache_data
def get_stock_data(symbols, start_date, end_date):
    """
    Fetches and caches stock data from Yahoo Finance.
    Using 'Close' prices.
    """
    return yf.download(symbols, start=start_date, end=end_date)['Close']

@st.cache_data
def get_fundamental_data(symbol):
    """
    Fetches fundamental data for a single stock from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        fundamentals = {
            'Symbol': symbol,
            'P/E Ratio': info.get('trailingPE') or info.get('forwardPE'),
            'P/B Ratio': info.get('priceToBook'),
            'Market Cap': info.get('marketCap'),
            'Dividend Yield': info.get('dividendYield'),
            'Beta': info.get('beta'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'DebtToEquity': info.get('debtToEquity'),
            'CurrentRatio': info.get('currentRatio'),
            'GrossMargin': info.get('grossMargins'),
            'OperatingMargin': info.get('operatingMargins'),
            'NetMargin': info.get('profitMargins'),
            'RevenueGrowth': info.get('revenueGrowth'),
            'EarningsGrowth': info.get('earningsGrowth'),
            'PayoutRatio': info.get('payoutRatio')
        }
        return fundamentals
    except Exception as e:
        st.warning(f"Error fetching fundamental data for {symbol}: {str(e)}")
        return None

def calculate_portfolio_value(prices, initial_portfolio):
    """
    Calculates the daily value of a static portfolio.
    Handles entry and optional exit dates.
    """
    if not initial_portfolio:
        return pd.Series(dtype=float)

    latest_date_in_data = prices.index.max()
    earliest_entry = min(pd.to_datetime(s['entry_date']) for s in initial_portfolio)

    total_value = pd.Series(0.0, index=prices.index)

    for stock_info in initial_portfolio:
        symbol = stock_info['symbol']
        quantity = stock_info['quantity']
        entry_date = pd.to_datetime(stock_info['entry_date'])
        exit_date = stock_info.get('exit_date')
        
        if symbol not in prices.columns:
            st.warning(f"Warning: Data for {symbol} not found. Skipping.")
            continue
            
        stock_value = prices[symbol].copy() * quantity
        stock_value[prices.index < entry_date] = 0
        
        if exit_date is not None:
            effective_exit_date = pd.to_datetime(exit_date)
            stock_value[prices.index > effective_exit_date] = stock_value[effective_exit_date]
        
        total_value += stock_value

    return total_value.loc[prices.index >= earliest_entry]

def calculate_risk_metrics(returns: pd.Series, benchmark_returns: pd.Series = None): 
    """Calculate comprehensive risk metrics""" 
    if returns.empty or len(returns) < 2: 
        return {} 
    # Basic metrics 
    annual_return = (1 + returns.mean()) ** 252 - 1 
    volatility = returns.std() * np.sqrt(252) 
    # Sharpe ratio (assuming 6% risk-free rate) 
    risk_free_rate = 0.06 
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0 
    # Maximum drawdown 
    cumulative = (1 + returns).cumprod() 
    rolling_max = cumulative.expanding().max() 
    drawdown = cumulative / rolling_max - 1 
    max_drawdown = drawdown.min() 
    # VaR and CVaR (95% confidence) 
    var_95 = np.percentile(returns, 5) 
    cvar_95 = returns[returns <= var_95].mean() if not returns[returns <= var_95].empty else var_95 
    # Calmar ratio 
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0 
    # Sortino ratio 
    downside_returns = returns[returns < 0] 
    downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0 
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0 
    metrics = { 
        'Annual_Return': annual_return, 
        'Volatility': volatility, 
        'Sharpe_Ratio': sharpe_ratio, 
        'Sortino_Ratio': sortino_ratio, 
        'Calmar_Ratio': calmar_ratio, 
        'Max_Drawdown': max_drawdown, 
        'VaR_95': var_95, 
        'CVaR_95': cvar_95, 
        'Skewness': returns.skew(), 
        'Kurtosis': returns.kurtosis() 
    } 
    # Beta and alpha if benchmark provided 
    if benchmark_returns is not None and len(benchmark_returns) == len(returns): 
        covariance = np.cov(returns, benchmark_returns)[0][1] 
        benchmark_variance = np.var(benchmark_returns) 
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1 
        alpha = annual_return - (risk_free_rate + beta * ((1 + benchmark_returns.mean()) ** 252 - 1 - risk_free_rate)) 
        metrics.update({ 
            'Beta': beta, 
            'Alpha': alpha, 
            'Correlation': returns.corr(benchmark_returns) 
        }) 
    return metrics


def run_monte_carlo_simulation(returns, num_simulations, forecast_days):
    """
    Runs a Monte Carlo simulation for portfolio returns.
    """
    mean_return = returns.mean()
    std_dev = returns.std()
    
    simulated_paths = np.zeros((forecast_days, num_simulations))
    initial_value = 1.0
    
    for i in range(num_simulations):
        daily_returns = np.random.normal(mean_return, std_dev, forecast_days)
        price_path = np.exp(np.log(1 + daily_returns).cumsum()) * initial_value
        simulated_paths[:, i] = price_path
        
    return simulated_paths

def run_portfolio_simulation(prices, strategy, start_date, lookback_months, top_n, initial_investment):
    """
    Simulates a portfolio strategy with monthly rebalancing.
    """
    simulation_prices = prices.loc[start_date:].copy()
    if simulation_prices.empty:
        return pd.Series(dtype=float)

    # Generate monthly rebalancing dates
    rebalancing_dates = pd.to_datetime(pd.date_range(start=start_date, end=prices.index.max(), freq='MS'))
    rebalancing_dates = rebalancing_dates[rebalancing_dates >= start_date].tolist()
    if not rebalancing_dates or rebalancing_dates[0] != start_date:
        rebalancing_dates.insert(0, start_date)

    portfolio_history = pd.DataFrame(index=simulation_prices.index, columns=['Value'])
    current_value = initial_investment
    current_holdings = {}

    for i in range(len(rebalancing_dates)):
        rebalance_date_dt = rebalancing_dates[i]
        
        # Find the actual trading date for rebalancing
        rebalance_date = find_nearest_date(rebalance_date_dt, prices.index)
        
        if rebalance_date not in prices.index:
            continue

        # Determine holdings based on strategy at rebalance date
        if strategy == "Equal Weight":
            equities = [s['symbol'] for s in st.session_state.initial_portfolio]
            num_equities = len(equities)
            if num_equities > 0:
                current_holdings = {symbol: 1.0 / num_equities for symbol in equities}
        elif strategy == "Market Cap":
            try:
                caps = {symbol: yf.Ticker(symbol).info.get('marketCap', 1e9) for symbol in [s['symbol'] for s in st.session_state.initial_portfolio]}
                total_cap = sum(caps.values())
                if total_cap > 0:
                    current_holdings = {symbol: cap / total_cap for symbol, cap in caps.items()}
            except:
                st.warning("Could not fetch market cap data. Defaulting to Equal Weight.")
                equities = [s['symbol'] for s in st.session_state.initial_portfolio]
                num_equities = len(equities)
                if num_equities > 0:
                    current_holdings = {symbol: 1.0 / num_equities for symbol in equities}
        elif strategy == "Momentum":
            lookback_end_date = rebalance_date
            lookback_start_date = lookback_end_date - timedelta(days=30 * lookback_months)
            
            lookback_prices = prices.loc[lookback_start_date:lookback_end_date].dropna(axis=1)
            
            if lookback_prices.empty or len(lookback_prices) < 2: continue
            
            returns = (lookback_prices.iloc[-1] / lookback_prices.iloc[0]) - 1
            momentum_ranking = returns.sort_values(ascending=False).head(top_n)
            
            new_holdings = {}
            for symbol in momentum_ranking.index:
                new_holdings[symbol] = 1.0 / top_n
            
            total_weight = sum(new_holdings.values())
            if total_weight > 0:
                current_holdings = {k: v / total_weight for k, v in new_holdings.items()}
            else:
                current_holdings = {}
        
        # Determine the period to simulate
        period_start_date = rebalance_date
        if i + 1 < len(rebalancing_dates):
            period_end_date = rebalancing_dates[i + 1]
        else:
            period_end_date = simulation_prices.index.max()
        
        period_data = simulation_prices.loc[period_start_date:period_end_date]
        
        if period_data.empty or not current_holdings:
            continue
        
        # Calculate the number of shares for each stock
        shares = {}
        # Get the price on the first trading day of the period, which is the first row of the DataFrame
        first_day_prices = period_data.iloc[0]
        for symbol, weight in current_holdings.items():
            if symbol in first_day_prices.index and not pd.isna(first_day_prices[symbol]):
                shares[symbol] = (current_value * weight) / first_day_prices[symbol]
        
        # Calculate the value of the portfolio for each day in the period
        period_values = pd.Series(index=period_data.index)
        for day in period_data.index:
            daily_value = 0
            for symbol, num_shares in shares.items():
                if symbol in period_data.columns and not pd.isna(period_data.loc[day, symbol]):
                    daily_value += num_shares * period_data.loc[day, symbol]
            period_values.loc[day] = daily_value
        
        # Update the master portfolio history with the new period's values
        portfolio_history.loc[period_values.index, 'Value'] = period_values.values
        
        # Update current_value for the next rebalancing period
        # Use the last value of the calculated period, as it's guaranteed to be in the series
        current_value = period_values.iloc[-1]
        
    return portfolio_history['Value'].dropna()

def calculate_period_returns(series, periods):
    """
    Calculates returns for a given series over specified periods.
    """
    returns = {}
    periods_map = {
        '1d': 1, '7d': 7, '1m': 30, '2m': 60, '3m': 90, '6m': 180, '1y': 365, '2y': 730
    }
    
    for period_label, days in periods_map.items():
        # Check if we have enough data points
        if len(series) > days:
            try:
                # For very short periods (1d, 7d), use a safer indexing approach
                if days == 1:
                    # For 1-day return, compare last two values
                    if len(series) >= 2:
                        start_price = series.iloc[-2]
                        end_price = series.iloc[-1]
                    else:
                        returns[period_label] = np.nan
                        continue
                else:
                    # For other periods, use the standard approach but with bounds checking
                    start_idx = max(0, len(series) - days - 1)
                    start_price = series.iloc[start_idx]
                    end_price = series.iloc[-1]
                
                # Calculate return with proper null checking
                if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
                    returns[period_label] = np.nan
                else:
                    returns[period_label] = (end_price / start_price) - 1
                    
            except (IndexError, KeyError):
                returns[period_label] = np.nan
        else:
            returns[period_label] = np.nan
    
    return returns

def get_dividend_history(symbol):
    """
    Fetches and returns dividend history for a stock.
    """
    try:
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
        return dividends
    except Exception as e:
        st.warning(f"Could not fetch dividend data for {symbol}: {e}")
        return pd.Series(dtype='float64')

def calculate_portfolio_composition(initial_portfolio, prices):
    """
    Calculates the current value and weight of each stock in the portfolio.
    """
    composition = []
    if not initial_portfolio:
        return pd.DataFrame()

    total_value = 0
    for stock in initial_portfolio:
        symbol = stock['symbol']
        quantity = stock['quantity']
        if symbol in prices.columns:
            last_price = prices[symbol].iloc[-1]
            current_value = last_price * quantity
            composition.append({'symbol': symbol, 'value': current_value})
            total_value += current_value
    
    composition_df = pd.DataFrame(composition)
    if not composition_df.empty and total_value > 0:
        composition_df['weight'] = composition_df['value'] / total_value
    return composition_df

def calculate_performance_attribution(initial_portfolio, prices):
    """
    Calculates each stock's contribution to the total portfolio return.
    """
    if not initial_portfolio:
        return pd.DataFrame()

    total_returns = []
    for stock in initial_portfolio:
        symbol = stock['symbol']
        quantity = stock['quantity']
        entry_date = pd.to_datetime(stock['entry_date'])
        
        if symbol in prices.columns:
            stock_prices = prices[symbol]
            valid_prices = stock_prices[stock_prices.index >= entry_date]
            if len(valid_prices) > 1:
                initial_value = valid_prices.iloc[0] * quantity
                current_value = valid_prices.iloc[-1] * quantity
                contribution = current_value - initial_value
                total_returns.append({'symbol': symbol, 'contribution': contribution})

    return pd.DataFrame(total_returns)


def calculate_sector_performance(all_prices, initial_portfolio, tickers):
    """
    Calculates the performance of each sector in the portfolio.
    """
    portfolio_symbols = [s['symbol'] for s in initial_portfolio]
    sectors = {v['symbol']: v['sector'] for k, v in tickers.items() if v['symbol'] in portfolio_symbols}
    
    # Calculate daily returns for all relevant stocks
    returns = all_prices.pct_change().dropna()
    
    sector_performance = {}
    for symbol, sector in sectors.items():
        if symbol in returns.columns:
            if sector not in sector_performance:
                sector_performance[sector] = []
            sector_performance[sector].append(returns[symbol])

    sector_returns = {}
    for sector, stock_returns in sector_performance.items():
        # Equal-weighted sector return for simplicity
        sector_returns_df = pd.concat(stock_returns, axis=1).mean(axis=1)
        annualized_return = (1 + sector_returns_df.mean()) ** 252 - 1
        sector_returns[sector] = annualized_return
    
    return pd.DataFrame(list(sector_returns.items()), columns=['Sector', 'Annualized Return'])

# New functions for XIRR and top/worst performers
def xirr(transactions):
    """Calculates XIRR for a series of cash flows."""
    dates = pd.to_datetime([t['date'] for t in transactions])
    amounts = np.array([t['amount'] for t in transactions])
    
    if len(dates) < 2:
        return np.nan

    def irr_func(rate):
        return np.sum(amounts / (1 + rate)**((dates - dates.min()).days / 365.25))

    try:
        from scipy.optimize import newton
        result = newton(irr_func, 0.1) # Initial guess of 10%
        return result
    except Exception:
        return np.nan

def calculate_portfolio_xirr(initial_portfolio, all_prices):
    """Calculates XIRR for the portfolio."""
    if not initial_portfolio or all_prices.empty:
        return np.nan
        
    transactions = []
    
    # Initial investments (negative cash flow)
    for stock in initial_portfolio:
        entry_date = pd.to_datetime(stock['entry_date'])
        quantity = stock['quantity']
        symbol = stock['symbol']
        
        if symbol in all_prices.columns:
            try:
                entry_price = all_prices.loc[entry_date, symbol]
                transactions.append({
                    'date': entry_date,
                    'amount': -entry_price * quantity
                })
            except KeyError:
                # Fallback to nearest date if exact date not available
                nearest_date = find_nearest_date(entry_date, all_prices.index)
                entry_price = all_prices.loc[nearest_date, symbol]
                transactions.append({
                    'date': nearest_date,
                    'amount': -entry_price * quantity
                })

    # Final valuation (positive cash flow)
    latest_date = all_prices.index.max()
    final_value = 0
    for stock in initial_portfolio:
        symbol = stock['symbol']
        quantity = stock['quantity']
        
        if symbol in all_prices.columns:
            final_value += all_prices[symbol].iloc[-1] * quantity
            
    transactions.append({
        'date': latest_date,
        'amount': final_value
    })
    
    return xirr(transactions)

def calculate_individual_stock_performance(initial_portfolio, all_prices):
    """Calculates performance for each stock in the portfolio."""
    if not initial_portfolio or all_prices.empty:
        return pd.DataFrame()

    stock_performance_list = []
    for stock in initial_portfolio:
        symbol = stock['symbol']
        quantity = stock['quantity']
        entry_date = pd.to_datetime(stock['entry_date'])
        
        if symbol in all_prices.columns:
            stock_prices = all_prices[symbol]
            valid_prices = stock_prices[stock_prices.index >= entry_date]
            
            if len(valid_prices) > 1:
                start_price = valid_prices.iloc[0]
                end_price = valid_prices.iloc[-1]
                
                total_return = (end_price * quantity) - (start_price * quantity)
                
                stock_performance_list.append({
                    'symbol': symbol,
                    'total_return_value': total_return,
                    'total_return_pct': (end_price/start_price) - 1
                })
    
    return pd.DataFrame(stock_performance_list)

def get_technical_kpis(hist):
    """
    Calculates and returns a dictionary of technical KPI signals.
    """
    kpis = {}
    if hist.empty or len(hist) < 200:
        return kpis # Not enough data for meaningful analysis

    latest_close = hist['Close'].iloc[-1]
    
    # RSI Signal
    rsi = hist['RSI'].iloc[-1]
    if rsi > 70:
        kpis['RSI Signal'] = f"Overbought ({rsi:.1f})"
    elif rsi < 30:
        kpis['RSI Signal'] = f"Oversold ({rsi:.1f})"
    else:
        kpis['RSI Signal'] = f"Neutral ({rsi:.1f})"

    # MACD Signal
    macd_val = hist['MACD'].iloc[-1]
    signal_val = hist['MACD_Signal'].iloc[-1]
    if macd_val > signal_val:
        kpis['MACD Signal'] = "Bullish"
    elif macd_val < signal_val:
        kpis['MACD Signal'] = "Bearish"
    else:
        kpis['MACD Signal'] = "Neutral"

    # Price vs SMA50
    sma50 = hist['SMA_50'].iloc[-1]
    if latest_close > sma50:
        kpis['Price vs SMA50'] = "Above"
    elif latest_close < sma50:
        kpis['Price vs SMA50'] = "Below"
    else:
        kpis['Price vs SMA50'] = "On Par"

    # Bollinger Position
    upper_bb = hist['BB_Upper'].iloc[-1]
    lower_bb = hist['BB_Lower'].iloc[-1]
    middle_bb = hist['BB_Middle'].iloc[-1]
    if latest_close > upper_bb:
        kpis['Bollinger Position'] = "Above Upper Band"
    elif latest_close < lower_bb:
        kpis['Bollinger Position'] = "Below Lower Band"
    elif latest_close > middle_bb:
        kpis['Bollinger Position'] = "Upper Half"
    else:
        kpis['Bollinger Position'] = "Lower Half"

    return kpis

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Professional Portfolio Analytics")
st.markdown("---")

# Simulation Parameters
st.sidebar.header("Simulation Parameters")
selected_strategy = st.sidebar.selectbox(
    "Select Portfolio Strategy",
    ("Momentum", "Equal Weight", "Market Cap")
)
lookback_months = st.sidebar.slider("Momentum Lookback Period (months)", 1, 12, 6) if selected_strategy == "Momentum" else None
top_n = st.sidebar.slider("Number of Top Stocks to Hold", 1, 10, 5) if selected_strategy == "Momentum" else None
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Start Date")
simulation_start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=90))

# Main content
if st.session_state.initial_portfolio:
    
    # --- Relative Stock Analysis ---
    st.subheader("Relative Stock Analysis")
    
    # Define mandatory options regardless of portfolio
    mandatory_options = ["Portfolio", "NIFTYBEES.BO", "GOLDBEES.NS", "0P0000YWL1.BO", "0P0001IAU9.BO"]
    
    # Get portfolio options and combine with mandatory ones, removing duplicates
    portfolio_symbols = list(set(s['symbol'] for s in st.session_state.initial_portfolio))
    all_options_to_display = sorted(list(set(mandatory_options + portfolio_symbols)))
    
    # Now use the combined list for options and set the mandatory ones as default
    selected_options = st.multiselect("Select stocks and portfolio to analyze", options=all_options_to_display, default=mandatory_options)
    
    # Combine symbols from the portfolio and the selected options to fetch all necessary data
    all_symbols_to_fetch = list(set(portfolio_symbols + [s for s in selected_options if s != 'Portfolio']))

    start_dates = [pd.to_datetime(s['entry_date']) for s in st.session_state.initial_portfolio]
    end_dates = [pd.to_datetime(s['exit_date']) if s.get('exit_date') is not None else datetime.now() for s in st.session_state.initial_portfolio]
    
    earliest_start_date = min(start_dates)
    if selected_strategy == "Momentum" and lookback_months:
        earliest_start_date -= timedelta(days=30 * lookback_months)
    
    latest_end_date = max(end_dates)
    
    st.info("Fetching all required stock data. This may take a moment...")
    all_prices = get_stock_data(all_symbols_to_fetch, earliest_start_date, datetime.now())
    
    if not all_prices.empty:
        as_is_portfolio = calculate_portfolio_value(all_prices, st.session_state.initial_portfolio)
        
        if not as_is_portfolio.empty and len(as_is_portfolio) > 1:
            
            # --- Top-Level Portfolio Metrics ---
            st.subheader("Current Portfolio Summary")
            col_xirr, col_vol, col_best, col_worst = st.columns(4)

            # XIRR
            xirr_val = calculate_portfolio_xirr(st.session_state.initial_portfolio, all_prices)
            col_xirr.metric("Portfolio XIRR", f"{xirr_val:.2%}" if not pd.isna(xirr_val) else "N/A")
            
            # Annualized Volatility
            returns = as_is_portfolio.pct_change().dropna()
            ann_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
            col_vol.metric("Annualized Volatility", f"{ann_volatility:.2%}" if not pd.isna(ann_volatility) else "N/A")
            
            # Best & Worst Performers
            individual_performance = calculate_individual_stock_performance(st.session_state.initial_portfolio, all_prices)
            
            if not individual_performance.empty:
                best_performer = individual_performance.loc[individual_performance['total_return_pct'].idxmax()]
                worst_performer = individual_performance.loc[individual_performance['total_return_pct'].idxmin()]
                
                col_best.metric("Best Performer", f"{best_performer['symbol']} ({best_performer['total_return_pct']:.2%})")
                col_worst.metric("Worst Performer", f"{worst_performer['symbol']} ({worst_performer['total_return_pct']:.2%})")
            else:
                col_best.metric("Best Performer", "N/A")
                col_worst.metric("Worst Performer", "N/A")

            st.markdown("---")

        # --- Portfolio Composition & Attribution ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Portfolio Composition")
            composition_df = calculate_portfolio_composition(st.session_state.initial_portfolio, all_prices)
            if not composition_df.empty:
                fig_comp = px.pie(composition_df, values='value', names='symbol', title='Current Portfolio Allocation',
                                  template="plotly_dark", hole=0.3)
                fig_comp.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No data to display portfolio composition.")
        
        with col2:
            st.subheader("Performance Attribution")
            attribution_df = calculate_performance_attribution(st.session_state.initial_portfolio, all_prices)
            if not attribution_df.empty:
                fig_attr = px.bar(attribution_df, x='symbol', y='contribution', title='Performance Contribution per Stock',
                                  template="plotly_dark")
                fig_attr.update_layout(xaxis_title="Stock Symbol", yaxis_title="Value Contribution (Rs.)")
                st.plotly_chart(fig_attr, use_container_width=True)
            else:
                st.info("No data to display performance attribution.")

        st.markdown("---")


        # --- Relative Stock Analysis ---
        if selected_options:
            # --- Individual Stock Returns ---
            st.markdown("#### Individual Stock Returns")
            
            returns_periods = {'1d' : 1, '7d': 7, '1m': 30, '2m': 60, '3m': 90, '6m': 180, '1y': 365, '2y': 730}
            
            stock_returns_df = pd.DataFrame(index=returns_periods.keys())
            
            for symbol in selected_options:
                if symbol == 'Portfolio':
                    if not as_is_portfolio.empty:
                        stock_returns_df['Portfolio'] = calculate_period_returns(as_is_portfolio, returns_periods).values()
                else:
                    stock_returns_df[symbol] = calculate_period_returns(all_prices[symbol], returns_periods).values()
            
            st.dataframe(stock_returns_df.T.style.format("{:.2%}"))

            # --- Individual Stock Price Movement (Normalized) ---
            st.markdown("#### Relative Stock Price Movement")
            
            time_period_map = {
                '1d' : 1, '7d': 7, '1m': 30, '2m': 60, '3m': 90, '6m': 180, '1y': 365, '2y': 730
            }
            selected_period = st.selectbox("Select time period for normalized chart", options=list(time_period_map.keys()))
            
            end_date = all_prices.index.max()
            start_date = end_date - timedelta(days=time_period_map[selected_period])
            
            prices_to_plot_df = all_prices.loc[start_date:end_date]
            
            # Normalize to 100
            fig_indiv = go.Figure()

            # Add stocks
            colors = px.colors.qualitative.Set1
            for i, col in enumerate(prices_to_plot_df.columns):
                if col in selected_options:
                    normalized_prices = (prices_to_plot_df[col] / prices_to_plot_df[col].iloc[0]) * 100
                    fig_indiv.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices, mode='lines', name=col, line=dict(color=colors[i % len(colors)], width=2)))

            # Add portfolio
            if 'Portfolio' in selected_options and not as_is_portfolio.empty:
                portfolio_slice = as_is_portfolio.loc[start_date:end_date]
                if not portfolio_slice.empty:
                    normalized_portfolio = (portfolio_slice / portfolio_slice.iloc[0]) * 100
                    fig_indiv.add_trace(go.Scatter(x=normalized_portfolio.index, y=normalized_portfolio, mode='lines', name='Portfolio', line=dict(color='black', width=4), hovertemplate='Portfolio<br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'))

            fig_indiv.update_layout(
                title_text=f"Normalized Stock Price Movement (Base 100) - {selected_period}",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                template="plotly_dark"
            )
            st.plotly_chart(fig_indiv, use_container_width=True)
            
        st.markdown("---")

        # --- Sector Performance Analysis ---
        st.subheader("Sector Performance Analysis")
        sector_performance_df = calculate_sector_performance(all_prices, st.session_state.initial_portfolio, tickers)
        if not sector_performance_df.empty:
            fig_sector = px.bar(sector_performance_df, x='Sector', y='Annualized Return', 
                                title='Annualized Return by Sector', template="plotly_dark")
            fig_sector.update_layout(yaxis_tickformat=".2%")
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No data to display sector performance.")
        
        st.markdown("---")
        
        # --- Asset Correlation Analysis ---
        st.subheader("Asset Correlation Analysis")
        stock_returns_df = all_prices.pct_change().dropna()
        if not stock_returns_df.empty and len(stock_returns_df.columns) > 1:
            correlation_matrix = stock_returns_df.corr()
            
            # Use the correlation matrix columns directly for both axes
            mapped_columns = correlation_matrix.columns.tolist()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=mapped_columns,
                y=mapped_columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Asset Correlation Matrix",
                width=720,
                height=680
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough assets in the portfolio to calculate correlation.")

        # --- Risk Analytics Dashboard ---
        st.subheader("Risk Analytics Dashboard")
        if not as_is_portfolio.empty:
            portfolio_returns = as_is_portfolio.pct_change().dropna()
            
            # Get the comprehensive risk metrics dictionary
            risk_metrics = calculate_risk_metrics(portfolio_returns)

            # Display all risk metrics in three columns
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            metrics_to_show = {
                "Annual Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Calmar Ratio": "{:.2f}",
                "VaR (95%)": "{:.2%}",
                "CVaR (95%)": "{:.2%}",
                "Skewness": "{:.2f}",
                "Kurtosis": "{:.2f}"
            }
            
            metric_list = list(metrics_to_show.keys())
            
            with risk_col1:
                st.metric(label=metric_list[0], value=metrics_to_show[metric_list[0]].format(risk_metrics.get('Annual_Return')) if not pd.isna(risk_metrics.get('Annual_Return')) else "N/A")
                st.metric(label=metric_list[1], value=metrics_to_show[metric_list[1]].format(risk_metrics.get('Volatility')) if not pd.isna(risk_metrics.get('Volatility')) else "N/A")
                st.metric(label=metric_list[2], value=metrics_to_show[metric_list[2]].format(risk_metrics.get('Sharpe_Ratio')) if not pd.isna(risk_metrics.get('Sharpe_Ratio')) else "N/A")
                
            with risk_col2:
                st.metric(label=metric_list[3], value=metrics_to_show[metric_list[3]].format(risk_metrics.get('Sortino_Ratio')) if not pd.isna(risk_metrics.get('Sortino_Ratio')) else "N/A")
                st.metric(label=metric_list[4], value=metrics_to_show[metric_list[4]].format(risk_metrics.get('Max_Drawdown')) if not pd.isna(risk_metrics.get('Max_Drawdown')) else "N/A")
                st.metric(label=metric_list[5], value=metrics_to_show[metric_list[5]].format(risk_metrics.get('Calmar_Ratio')) if not pd.isna(risk_metrics.get('Calmar_Ratio')) else "N/A")

            with risk_col3:
                st.metric(label=metric_list[6], value=metrics_to_show[metric_list[6]].format(risk_metrics.get('VaR_95')) if not pd.isna(risk_metrics.get('VaR_95')) else "N/A")
                st.metric(label=metric_list[7], value=metrics_to_show[metric_list[7]].format(risk_metrics.get('CVaR_95')) if not pd.isna(risk_metrics.get('CVaR_95')) else "N/A")
                st.metric(label=metric_list[8], value=metrics_to_show[metric_list[8]].format(risk_metrics.get('Skewness')) if not pd.isna(risk_metrics.get('Skewness')) else "N/A")
                st.metric(label=metric_list[9], value=metrics_to_show[metric_list[9]].format(risk_metrics.get('Kurtosis')) if not pd.isna(risk_metrics.get('Kurtosis')) else "N/A")
            
            st.markdown("---")
            st.subheader("Monte Carlo Simulation")
            st.write("Project your portfolio's future performance based on historical data.")
            
            # Monte Carlo Simulation parameters
            col1, col2 = st.columns(2)
            with col1:
                num_simulations = st.slider("Number of Simulations", 10, 1000, 100)
            with col2:
                forecast_days = st.slider("Forecast Period (days)", 30, 730, 365)
            
            if st.button("Run Monte Carlo Simulation"):
                with st.spinner('Running Monte Carlo simulation...'):
                    simulated_paths = run_monte_carlo_simulation(portfolio_returns, num_simulations, forecast_days)
                    
                    fig_mc = go.Figure()
                    for i in range(num_simulations):
                        fig_mc.add_trace(go.Scatter(x=list(range(forecast_days)), y=simulated_paths[:, i] * as_is_portfolio.iloc[-1],
                                                    mode='lines', line=dict(color='lightblue', width=0.5), showlegend=False, name=f'Simulation {i+1}'))
                    
                    percentiles = [5, 25, 50, 75, 95]
                    colors = ['red', 'orange', 'green', 'orange', 'red']
                    names = ['5th Percentile', '25th Percentile', 'Median', '75th Percentile', '95th Percentile']
                    
                    for p, color, name in zip(percentiles, colors, names):
                        percentile_values = np.percentile(simulated_paths, p, axis=1) * as_is_portfolio.iloc[-1]
                        fig_mc.add_trace(go.Scatter(
                            x=list(range(forecast_days)),
                            y=percentile_values,
                            mode='lines',
                            line=dict(color=color, width=3),
                            name=name
                        ))

                    fig_mc.update_layout(
                        title=f"Monte Carlo Simulation - {forecast_days} Days, {num_simulations} Simulations",
                        xaxis_title="Days",
                        yaxis_title="Projected Portfolio Value (Rs.)",
                        height=500
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.warning("Cannot calculate risk metrics or run Monte Carlo simulation without a valid portfolio.")
            
        st.markdown("---")


        # --- Simulated Portfolio Performance ---
        st.subheader("Simulated Portfolio Performance")
        
        if as_is_portfolio.empty:
            st.warning("The 'As-Is' portfolio calculation resulted in an empty series. Please check your inputs.")
        else:
            valid_start_date = as_is_portfolio.index[as_is_portfolio.index < pd.to_datetime(simulation_start_date)].max()
            
            if not pd.isna(valid_start_date):
                initial_rebalance_investment = as_is_portfolio.loc[valid_start_date]
                rebalanced_portfolio = run_portfolio_simulation(all_prices, selected_strategy, pd.to_datetime(simulation_start_date), lookback_months, top_n, initial_rebalance_investment)
            else:
                st.warning("Cannot find a valid 'as-is' portfolio value to start the rebalanced simulation on the selected date. Please choose a later date for the simulation.")
                rebalanced_portfolio = pd.Series(dtype=float)

            combined_df = pd.DataFrame({
                'As-Is Portfolio': as_is_portfolio,
                f'{selected_strategy} Rebalanced Portfolio': rebalanced_portfolio
            }).dropna()
            
            if combined_df.empty:
                st.warning("Not enough data to run the simulation. Please ensure your dates and symbols are correct.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['As-Is Portfolio'], mode='lines', name='As-Is Portfolio', line=dict(color='black', width=4), hovertemplate='Portfolio<br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'))
                fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[f'{selected_strategy} Rebalanced Portfolio'], mode='lines', name=f'{selected_strategy} Rebalanced', line=dict(color=px.colors.qualitative.Set1[0], width=2), hovertemplate='Rebalanced<br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'))
                fig.update_layout(
                    title_text=f"As-Is vs. {selected_strategy} Rebalanced Portfolio Performance",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (Rs.)",
                    hovermode="x unified",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")


        # --- Detailed Single Stock Analysis ---
        st.subheader("Detailed Single Stock Analysis")
        
        # Combine default tickers and portfolio symbols for the dropdown
        portfolio_symbols_for_analysis = list(set(s['symbol'] for s in st.session_state.initial_portfolio))
        all_symbols_for_dropdown = sorted(list(set(list(tickers.keys()) + portfolio_symbols_for_analysis)))

        selected_stock_for_analysis = st.selectbox("Select a stock for detailed analysis", options=all_symbols_for_dropdown)
        
        # ------------------------------- 
        # Enhanced Data Functions 
        # ------------------------------- 
        def calculate_sma(series, window):
            return series.rolling(window=window).mean()

        def calculate_rsi(series, window=14):
            delta = series.diff().dropna()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
            ema_fast = series.ewm(span=fast_period, adjust=False).mean()
            ema_slow = series.ewm(span=slow_period, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            hist = macd - signal
            return macd, signal, hist

        def calculate_bollinger_bands(series, window=20, num_std=2):
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band
            
        @st.cache_data(ttl=300) 
        def get_comprehensive_data(symbol: str, period: str = "1y"): 
            """Get comprehensive stock data with technical indicators""" 
            try:
                ticker = yf.Ticker(symbol) 
                # Get historical data 
                hist = ticker.history(period=period) 
                if hist.empty: 
                    return None 
                # Technical indicators - use TA-Lib if available, otherwise fallback 
                if TALIB_AVAILABLE: 
                    hist['SMA_20'] = talib.SMA(hist['Close'].values, timeperiod=20) 
                    hist['SMA_50'] = talib.SMA(hist['Close'].values, timeperiod=50) 
                    hist['SMA_200'] = talib.SMA(hist['Close'].values, timeperiod=200) 
                    hist['RSI'] = talib.RSI(hist['Close'].values, timeperiod=14) 
                    hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = talib.MACD(hist['Close'].values) 
                    hist['BB_Upper'], hist['BB_Middle'], hist['BB_Lower'] = talib.BBANDS(hist['Close'].values) 
                    hist['Volume_SMA'] = talib.SMA(hist['Volume'].values, timeperiod=20) 
                else: 
                    # Fallback implementations 
                    hist['SMA_20'] = calculate_sma(hist['Close'], 20) 
                    hist['SMA_50'] = calculate_sma(hist['Close'], 50) 
                    hist['SMA_200'] = calculate_sma(hist['Close'], 200) 
                    hist['RSI'] = calculate_rsi(hist['Close']) 
                    hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = calculate_macd(hist['Close']) 
                    hist['BB_Upper'], hist['BB_Middle'], hist['BB_Lower'] = calculate_bollinger_bands(hist['Close']) 
                    hist['Volume_SMA'] = calculate_sma(hist['Volume'], 20) 
                # Get fundamental data 
                info = ticker.info 
                fundamentals = { 
                    'PE': info.get('trailingPE') or info.get('forwardPE'), 
                    'PB': info.get('priceToBook'), 
                    'ROE': info.get('returnOnEquity'), 
                    'ROA': info.get('returnOnAssets'), 
                    'DebtToEquity': info.get('debtToEquity'), 
                    'CurrentRatio': info.get('currentRatio'), 
                    'GrossMargin': info.get('grossMargins'), 
                    'OperatingMargin': info.get('operatingMargins'), 
                    'NetMargin': info.get('profitMargins'), 
                    'RevenueGrowth': info.get('revenueGrowth'), 
                    'EarningsGrowth': info.get('earningsGrowth'), 
                    'DividendYield': info.get('dividendYield'), 
                    'PayoutRatio': info.get('payoutRatio'), 
                    'MarketCap': info.get('marketCap'), 
                    'Beta': info.get('beta'), 
                    'AverageVolume': info.get('averageVolume'), 
                    'ESGScore': info.get('esgScores', {}).get('totalEsg') if info.get('esgScores') else None 
                } 
                return {'history': hist, 'fundamentals': fundamentals, 'info': info} 
            except Exception as e: 
                st.warning(f"Error fetching data for {symbol}: {str(e)}") 
                return None
        
        if selected_stock_for_analysis:
            
            # --- Technical Analysis ---
            st.markdown("#### Technical Analysis")

            # Button to open the new pop-up window
            components.html( """ <button onclick="window.open('https://chatgpt.com/share/68a9af5a-8e60-8011-b219-941c19506608', 'RSI_PopUp', 'width=780,height=400');"> What is RSI?  </button> 
                                 <button onclick="window.open('https://chatgpt.com/share/68a9b14d-a840-8011-986a-f0ddab8df75b', 'MACD_PopUp', 'width=750,height=400');"> What is MACD?  </button>""", height=34,)

            # Updated to use the tickers dictionary correctly to get the symbol
            selected_symbol = tickers[selected_stock_for_analysis]['symbol'] if selected_stock_for_analysis in tickers else selected_stock_for_analysis
            stock_data = get_comprehensive_data(selected_symbol)
            
            if stock_data:
                hist = stock_data['history']

                # --- Technical KPIs ---
                st.markdown("#### Technical KPIs")
                kpis = get_technical_kpis(hist)
                if kpis:
                    kpi_cols = st.columns(len(kpis))
                    for i, (kpi_name, kpi_value) in enumerate(kpis.items()):
                        kpi_cols[i].metric(kpi_name, kpi_value)
                else:
                    st.info("Not enough data to calculate technical KPIs.")
                st.markdown("---")

                fig_ta = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                    vertical_spacing=0.1,
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Price, SMAs and Bollinger Bands
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='yellow')), row=1, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='Upper BB', line=dict(color='gray', dash='dash')), row=1, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['BB_Middle'], name='Middle BB', line=dict(color='blue', dash='dash')), row=1, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='Lower BB', line=dict(color='gray', dash='dash')), row=1, col=1)

                # Add SMAs
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='green', width=1)), row=1, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='red', width=1)), row=1, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], name='SMA 200', line=dict(color='orange', width=2)), row=1, col=1)
                
                # RSI
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='cyan')), row=2, col=1)
                fig_ta.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig_ta.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig_ta.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='MACD Hist', marker_color='red'), row=3, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='yellow')), row=3, col=1)
                fig_ta.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal', line=dict(color='magenta')), row=3, col=1)
                
                fig_ta.update_layout(title=f"Technical Indicators for {selected_symbol}", height=800, template="plotly_dark", xaxis=dict(tickformat='%Y-%m-%d'))
                st.plotly_chart(fig_ta, use_container_width=True)

            # --- Fundamental Analysis ---
            st.markdown("#### Fundamental Analysis")
            fundamentals_data = get_fundamental_data(selected_symbol)
            if fundamentals_data:
                
                # Financial Health Metrics
                st.markdown("##### Financial Health Metrics")
                financial_health = {
                    'P/E Ratio': fundamentals_data.get('P/E Ratio'),
                    'P/B Ratio': fundamentals_data.get('P/B Ratio'),
                    'ROE': fundamentals_data.get('ROE'),
                    'ROA': fundamentals_data.get('ROA'),
                    'DebtToEquity': fundamentals_data.get('DebtToEquity'),
                    'Current Ratio': fundamentals_data.get('CurrentRatio'),
                }
                st.table(pd.DataFrame([financial_health]).T.rename(columns={0: 'Value'}).dropna().style.format({
                    'Value': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                }))
                
                # Profitability Metrics
                st.markdown("##### Profitability Metrics")
                profitability = {
                    'Operating Margin': fundamentals_data.get('OperatingMargin'),
                    'Profit Margins': fundamentals_data.get('NetMargin'),
                    'Revenue Growth': fundamentals_data.get('RevenueGrowth'),
                    'Earnings Growth': fundamentals_data.get('EarningsGrowth'),
                }
                profitability_df = pd.DataFrame([profitability]).T.rename(columns={0: 'Value'})
                # Add computed column
                profitability_df.loc['Operating Margin + Revenue Growth'] = (
                    profitability_df.loc['Operating Margin'].iloc[0] + profitability_df.loc['Revenue Growth'].iloc[0]
                ) if not pd.isna(profitability_df.loc['Operating Margin'].iloc[0]) and not pd.isna(profitability_df.loc['Revenue Growth'].iloc[0]) else None
                st.table(profitability_df.dropna().style.format({
                    'Value': lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and x >= 0 else f"{x:.2f}"
                }))
                
                # Dividend Analysis
                st.markdown("##### Dividend Analysis")
                dividend_analysis = {
                    'Dividend Yield': fundamentals_data.get('Dividend Yield'),
                    'Payout Ratio': fundamentals_data.get('PayoutRatio'),
                }
                st.table(pd.DataFrame([dividend_analysis]).T.rename(columns={0: 'Value'}).dropna().style.format({
                    'Value': "{:.2%}"
                }))
            
        st.markdown("---")
        
    else:
        st.error("Failed to download stock data. Please check the symbols and dates.")

st.sidebar.header("Portfolio Inputs")


with st.sidebar.expander("Define Initial Portfolio"):
    st.markdown("**Enter each stock's details manually:**")
    st.session_state.stock_symbol = st.selectbox("Stock Name", options=list(tickers.keys()))
    st.session_state.num_stocks = st.number_input("Number of Stocks", min_value=1, value=100)
    st.session_state.entry_date = st.date_input("Date of Entry", value=datetime.now() - timedelta(days=365))
    
    is_still_holding = st.checkbox("Still Holding", True)
    exit_date = None
    if not is_still_holding:
        exit_date = st.date_input("Date of Exit", value=datetime.now())

    if st.button("Add Stock to Portfolio", key="add_stock_btn"):
        # Ensure the selected stock is a valid key in the tickers dictionary
        if st.session_state.stock_symbol in tickers:
            st.session_state.initial_portfolio.append({
                'symbol': tickers[st.session_state.stock_symbol]['symbol'],
                'quantity': st.session_state.num_stocks,
                'entry_date': st.session_state.entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d') if exit_date else None
            })
            st.success(f"Added {st.session_state.num_stocks} stocks of {st.session_state.stock_symbol}.")
        else:
            st.error(f"Selected stock {st.session_state.stock_symbol} not found in tickers. Please check the `tickers.json` file.")
    
    st.markdown("---")
    st.markdown("**Or, upload a JSON file to load your portfolio:**")
    uploaded_file = st.file_uploader("Upload Portfolio JSON", type=["json"])
    if uploaded_file is not None:
        try:
            portfolio_data = json.load(uploaded_file)
            if isinstance(portfolio_data, list):
                st.session_state.initial_portfolio = portfolio_data
                st.success("Portfolio loaded successfully from file!")
            else:
                st.error("Invalid JSON format. Please upload a JSON file containing a list of portfolio items.")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please ensure the file is correctly formatted.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display current portfolio
st.sidebar.subheader("Current Portfolio")
if st.session_state.initial_portfolio:
    portfolio_df = pd.DataFrame(st.session_state.initial_portfolio)
    
    # Calculate total quantity for each unique symbol
    summary_df = portfolio_df.groupby('symbol')['quantity'].sum().reset_index()

    for index, row in summary_df.iterrows():
        symbol = row['symbol']
        total_quantity = row['quantity']
        
        # Create an expander for each stock symbol
        with st.sidebar.expander(f"{symbol} (Total: {total_quantity})"):
            # Filter the original dataframe for the current symbol
            stock_details = portfolio_df[portfolio_df['symbol'] == symbol]
            
            # Display the details for this stock
            st.dataframe(stock_details[['entry_date', 'exit_date', 'quantity']].reset_index(drop=True), hide_index=True)
            
    if st.sidebar.button("Clear Portfolio"):
        st.session_state.initial_portfolio = []
        st.rerun()
else:
    st.sidebar.info("Your portfolio is currently empty. Add stocks using the form above or by uploading a file.")
