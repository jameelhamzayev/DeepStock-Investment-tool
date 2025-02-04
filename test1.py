
"""

historycal
opponents
latest pce
interest rate 
outlook
unemployment rate
ummumi cash flov
revenue,ebitda, fcf growth yoy qoq roic 






"""




import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  
import datetime
import sys
import warnings
warnings.filterwarnings('ignore')
# --- FUNCTIONS FOR DATA RETRIEVAL & CALCULATION ---

def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.
    :param ticker: Stock ticker symbol (e.g., 'AAPL')
    :param period: Time period (e.g., '1y' for one year)
    :param interval: Data interval (e.g., '1d' for daily data)
    :return: DataFrame with historical stock data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        print(f"No data found for ticker {ticker}.")
        sys.exit(1)
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate several technical indicators and add them as columns to the DataFrame.
    Indicators include:
        - Simple Moving Averages (SMA) 20-day and 50-day
        - Exponential Moving Average (EMA) 20-day
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD)
        - Bollinger Bands
    :param df: DataFrame with at least 'Close' column
    :return: DataFrame with additional columns for each indicator
    """
    # Ensure we have enough data
    if len(df) < 50:
        print("Not enough data to calculate all indicators (need at least 50 data points).")
        sys.exit(1)

    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # MACD
    macd_indicator = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Diff'] = macd_indicator.macd_diff()

    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    df['BB_Mid'] = bb_indicator.bollinger_mavg()

    return df

def print_fundamental_info(ticker: str):
    """
    Print basic fundamental information about the stock using yfinance.
    :param ticker: Stock ticker symbol.
    """
    stock = yf.Ticker(ticker)
    info = stock.info  # dictionary containing fundamentals
    print("=== Fundamental Information ===")
    print(f"Ticker: {ticker.upper()}")
    print(f"Name: {info.get('longName', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Market Cap: {info.get('marketCap', 'N/A')}")
    print(f"Trailing P/E: {info.get('trailingPE', 'N/A')}")
    print(f"Forward P/E: {info.get('forwardPE', 'N/A')}")
    print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    print(f"52 Week Change: {info.get('52WeekChange', 'N/A')}")
    print("===============================")
    print()

def print_technical_summary(df: pd.DataFrame):
    """
    Print a summary of technical indicators based on the latest available data.
    :param df: DataFrame with technical indicators.
    """
    latest = df.iloc[-1]
    print("=== Technical Summary (Latest Data) ===")
    print(f"Date: {latest.name.date()}")
    print(f"Close Price: {latest['Close']:.2f}")
    print(f"SMA20: {latest['SMA20']:.2f}")
    print(f"SMA50: {latest['SMA50']:.2f}")
    print(f"EMA20: {latest['EMA20']:.2f}")
    print(f"RSI: {latest['RSI']:.2f}")
    print(f"MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_Signal']:.2f} | Diff: {latest['MACD_Diff']:.2f}")
    print(f"Bollinger Bands: Low {latest['BB_Low']:.2f} | Mid {latest['BB_Mid']:.2f} | High {latest['BB_High']:.2f}")
    print("=======================================")
    print()

def plot_stock_analysis(df: pd.DataFrame, ticker: str):
    """
    Plot stock price along with SMA and Bollinger Bands.
    Also creates a subplot for RSI.
    """
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Price Chart with SMA and Bollinger Bands
    

    # RSI Chart
    ax2.plot(df.index, df['RSI'], label='RSI', color='green')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
    ax2.axhline(30, linestyle='--', color='blue', alpha=0.5)
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.legend()

   

# --- MAIN PROGRAM ---

def main():
    # User input for ticker and period (you can expand this to include more options)
    ticker = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
    period = input("Enter the period for historical data (e.g., 1y, 6mo, 2y): ").strip() or "1y"
    interval = "1d"  # daily data

    # Fetch and calculate
    print("\nFetching historical data...")
    df = fetch_stock_data(ticker, period=period, interval=interval)
    df = calculate_technical_indicators(df)

    # Print Fundamental and Technical Summaries
    print_fundamental_info(ticker)
    print_technical_summary(df)

   
if __name__ == "__main__":
    main()
