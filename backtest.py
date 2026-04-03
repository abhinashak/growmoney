import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def backtest_accumulation_strategy(ticker_symbol, start_date, end_date, total_allocation=800000):
    print(f"Downloading data for {ticker_symbol}...")
    # Fetch historical data
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        print("No data found for the given dates.")
        return
    
    # Calculate 200-Day Moving Average
    df['200_DMA'] = df['Close'].rolling(window=200).mean()
    
    # Portfolio State Variables
    cash = total_allocation
    shares = 0
    base_price = 0
    average_price = 0
    tranche_size = total_allocation * 0.10 # 10% per trade
    
    # Scaling Trackers
    highest_up_level = 0
    lowest_down_level = 0
    
    portfolio_values = []
    
    for date, row in df.iterrows():
        price = row['Close']
        dma_200 = row['200_DMA']
        
        # Skip days until we have enough data for the 200 DMA
        if pd.isna(dma_200):
            portfolio_values.append(cash)
            continue
            
        # STATE 1: Not in a position, looking for an entry
        if shares == 0:
            if price < dma_200:
                shares_to_buy = tranche_size / price
                shares += shares_to_buy
                cash -= tranche_size
                base_price = price
                average_price = price
                highest_up_level = 0
                lowest_down_level = 0
                print(f"[{date.date()}] BASE ENTRY: Bought {shares_to_buy:.2f} shares at ${price:.2f}")
                
        # STATE 2: In a position, managing trades
        else:
            # Check Stop-Loss (20% below base price)
            if price <= base_price * 0.80:
                print(f"[{date.date()}] STOP LOSS HIT: Sold all at ${price:.2f}. Base was ${base_price:.2f}")
                cash += shares * price
                shares = 0
                
            # Check Take Profit (50% above average price)
            elif price >= average_price * 1.50:
                print(f"[{date.date()}] TAKE PROFIT HIT: Sold all at ${price:.2f}. Avg Cost was ${average_price:.2f}")
                cash += shares * price
                shares = 0
                
            # Check Scaling Conditions
            else:
                # Upside Scaling (Every 5% rise from base)
                current_up_level = int((price - base_price) / (base_price * 0.05))
                if current_up_level > highest_up_level and cash >= tranche_size:
                    shares_to_buy = tranche_size / price
                    total_cost = (shares * average_price) + tranche_size
                    shares += shares_to_buy
                    cash -= tranche_size
                    average_price = total_cost / shares
                    highest_up_level = current_up_level
                    print(f"[{date.date()}] UPSIDE SCALING: Bought {shares_to_buy:.2f} shares at ${price:.2f}")

                # Downside Scaling (Every 5% fall from base)
                current_down_level = int((base_price - price) / (base_price * 0.05))
                if current_down_level > lowest_down_level and cash >= tranche_size:
                    shares_to_buy = tranche_size / price
                    total_cost = (shares * average_price) + tranche_size
                    shares += shares_to_buy
                    cash -= tranche_size
                    average_price = total_cost / shares
                    lowest_down_level = current_down_level
                    print(f"[{date.date()}] DOWNSIDE SCALING: Bought {shares_to_buy:.2f} shares at ${price:.2f}")
                    
        # Record daily portfolio value (Cash + Value of open shares)
        portfolio_values.append(cash + (shares * price))
        
    df['Portfolio_Value'] = portfolio_values
    
    # Calculate Final Results
    final_value = df['Portfolio_Value'].iloc[-1]
    profit = final_value - total_allocation
    print("\n--- BACKTEST RESULTS ---")
    print(f"Ticker: {ticker_symbol}")
    print(f"Initial Capital: ${total_allocation:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Profit/Loss: ${profit:,.2f} ({(profit/total_allocation)*100:.2f}%)")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='blue')
    plt.axhline(y=total_allocation, color='red', linestyle='--', label='Initial Capital')
    plt.title(f'Accumulation Strategy Backtest: {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the backtest on Apple (AAPL) over the last 5 years
backtest_accumulation_strategy("PGEL.NS", "2024-03-01", "2026-02-23")
