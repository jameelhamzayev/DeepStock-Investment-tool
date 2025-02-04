import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            raise ValueError(f"No data available for {ticker}")
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_technical_indicators(data):

    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

   
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

 
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

   
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std().squeeze()  
    data['BB_upper'] = data['BB_middle'] + 2 * std_dev
    data['BB_lower'] = data['BB_middle'] - 2 * std_dev

    return data

def analyze_trend(data):
  
    close_prices = data['Close'].dropna().values.reshape(-1, 1)
    days = np.arange(len(close_prices)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(days, close_prices)
    trend = "Upward" if model.coef_[0] > 0 else "Downward"

    return trend, float(model.coef_[0])

def calculate_risk_metrics(data):

    returns = data['Close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
    
    # Beta calculation
    market_data = yf.download('^GSPC', start=data.index[0], end=data.index[-1], progress=False)['Close']
    market_returns = market_data.pct_change().dropna()
    
    # Align the returns data
    combined_returns = pd.concat([returns, market_returns], axis=1).dropna()
    covariance = combined_returns.cov().iloc[0, 1]
    market_variance = combined_returns.iloc[:, 1].var()
    beta = covariance / market_variance

    # Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    sharpe = float(np.sqrt(252) * excess_returns.mean() / returns.std())

    return {
        'Volatility': volatility,
        'Beta': float(beta),
        'Sharpe_Ratio': sharpe,
        'Value_at_Risk': float(np.percentile(returns, 5) * 100)
    }

def plot_stock_analysis(data, ticker):
    
    plt.figure(figsize=(15, 10))
    

    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.6)
    plt.plot(data.index, data['SMA_50'], label='SMA 50', color='orange', alpha=0.7)
    plt.plot(data.index, data['SMA_200'], label='SMA 200', color='green', alpha=0.7)
    plt.plot(data.index, data['BB_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.6)
    plt.plot(data.index, data['BB_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.6)
    plt.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1, color='gray')
    plt.title(f"{ticker} Technical Analysis")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)

   
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['RSI'], label='RSI', color='purple', alpha=0.7)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.fill_between(data.index, data['RSI'], 70, where=(data['RSI'] >= 70), color='r', alpha=0.3)
    plt.fill_between(data.index, data['RSI'], 30, where=(data['RSI'] <= 30), color='g', alpha=0.3)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def generate_analysis_report(data, trend, slope, risk_metrics):

    latest_price = float(data['Close'].iloc[-1])
    rsi_value = float(data['RSI'].iloc[-1])
    macd = float(data['MACD'].iloc[-1])
    signal_line = float(data['Signal_Line'].iloc[-1])
    daily_change = float(data['Close'].pct_change().iloc[-1] * 100)
    
    report = [
        f"\nPrice Analysis:",
        f"Current Price: ${latest_price:.2f} ({daily_change:+.2f}%)",
        f"52-Week Range: ${float(data['Close'].min()):.2f} - ${float(data['Close'].max()):.2f}",
        
        f"\nTechnical Indicators:",
        f"RSI: {rsi_value:.1f}",
        f"MACD: {macd:.2f}",
        f"Signal Line: {signal_line:.2f}",
        f"50-day MA: ${float(data['SMA_50'].iloc[-1]):.2f}",
        f"200-day MA: ${float(data['SMA_200'].iloc[-1]):.2f}",
        
        f"\nTrend Analysis:",
        f"Overall Trend: {trend}",
        f"Trend Strength: {abs(slope):.6f}",
        
        f"\nRisk Metrics:",
        f"Volatility (Annual): {risk_metrics['Volatility']:.1f}%",
        f"Beta: {risk_metrics['Beta']:.2f}",
        f"Sharpe Ratio: {risk_metrics['Sharpe_Ratio']:.2f}",
        f"Value at Risk (95%): {risk_metrics['Value_at_Risk']:.1f}%",
        
        f"\nTrading Signals:"
    ]


    if rsi_value < 30:
        report.append("- RSI indicates oversold conditions (Potential Buy)")
    elif rsi_value > 70:
        report.append("- RSI indicates overbought conditions (Potential Sell)")
    
    if macd > signal_line:
        report.append("- MACD shows bullish signal")
    else:
        report.append("- MACD shows bearish signal")
    
    if latest_price > float(data['SMA_200'].iloc[-1]):
        report.append("- Price above 200-day MA (Bullish)")
    else:
        report.append("- Price below 200-day MA (Bearish)")

    return report

def main():
    ticker = input("Enter the stock ticker symbol (e.g., AAPL): ").upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"Analyzing {ticker}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    if stock_data is not None:
        stock_data = calculate_technical_indicators(stock_data)
        trend, slope = analyze_trend(stock_data)
        risk_metrics = calculate_risk_metrics(stock_data)
        

        report = generate_analysis_report(stock_data, trend, slope, risk_metrics)
        for line in report:
            print(line)
            

        plot_stock_analysis(stock_data, ticker)

if __name__ == "__main__":
    main()
