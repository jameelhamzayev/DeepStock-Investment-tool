import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data = None
        self.fundamentals = None
        self.load_data()
        
    def load_data(self, period="2y"):

        self.data = self.stock.history(period=period)
        self.fundamentals = {
            'info': self.stock.info,
            'financials': self.stock.financials,
            'balance_sheet': self.stock.balance_sheet,
            'cash_flow': self.stock.cashflow
        }
    
    def technical_analysis(self):
        df = self.data.copy()
        
       
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
      
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
       
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
        
        return df
    
    def fundamental_analysis(self):
        try:
            info = self.fundamentals['info']
            
            
            analysis = {
                'Market Cap': info.get('marketCap'),
                'P/E Ratio': info.get('trailingPE'),
                'Forward P/E': info.get('forwardPE'),
                'PEG Ratio': info.get('pegRatio'),
                'Price/Book': info.get('priceToBook'),
                'Dividend Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'Debt/Equity': info.get('debtToEquity'),
                'Return on Equity': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'Profit Margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                'Operating Margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
            }
            
            return analysis
        except Exception as e:
            return f"Error in fundamental analysis: {str(e)}"
    
    def risk_analysis(self):
      
        returns = self.data['Close'].pct_change().dropna()
        
        risk_metrics = {
            'Volatility (Annual)': returns.std() * np.sqrt(252) * 100,
            'Beta': self.calculate_beta(),
            'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
            'Value at Risk (95%)': np.percentile(returns, 5) * 100
        }
        
        return risk_metrics
    
    def calculate_beta(self):
        
        try:
            spy = yf.download('^GSPC', start=self.data.index[0], end=self.data.index[-1])['Close']
            stock_returns = self.data['Close'].pct_change().dropna()
            market_returns = spy.pct_change().dropna()
            
            
            combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
            covariance = combined.cov().iloc[0,1]
            market_variance = combined.iloc[:,1].var()
            
            return covariance / market_variance
        except Exception as e:
            return 0.0
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
      
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def generate_report(self):
        
        print(f"\n=== Stock Analysis Report for {self.ticker} ===\n")
        
        current_price = self.data['Close'][-1]
        print(f"Current Price: ${current_price:.2f}")
        print(f"52-Week Range: ${self.data['Low'].min():.2f} - ${self.data['High'].max():.2f}")
        
       
        tech_data = self.technical_analysis()
        print("\n--- Technical Indicators (Latest) ---")
        print(f"RSI: {tech_data['RSI'][-1]:.2f}")
        print(f"MACD: {tech_data['MACD'][-1]:.2f}")
        print(f"Signal Line: {tech_data['Signal_Line'][-1]:.2f}")
        
        print("\n--- Moving Averages ---")
        print(f"20-day SMA: ${tech_data['SMA_20'][-1]:.2f}")
        print(f"50-day SMA: ${tech_data['SMA_50'][-1]:.2f}")
        print(f"200-day SMA: ${tech_data['SMA_200'][-1]:.2f}")
        
     
        print("\n--- Fundamental Analysis ---")
        fundamentals = self.fundamental_analysis()
        for key, value in fundamentals.items():
            if isinstance(value, (int, float)):
                if 'Ratio' in key or 'Margin' in key or 'Yield' in key:
                    print(f"{key}: {value:.2f}%")
                elif 'Market Cap' in key:
                    print(f"{key}: ${value:,.0f}")
                else:
                    print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
      
        print("\n--- Risk Analysis ---")
        risk_metrics = self.risk_analysis()
        for key, value in risk_metrics.items():
            print(f"{key}: {value:.2f}")
        
       
        print("\n--- Trading Signals ---")
        self.generate_trading_signals()
    
    def generate_trading_signals(self):
        
        tech_data = self.technical_analysis()
        latest = tech_data.iloc[-1]
        
        signals = []
        
        
        if latest['RSI'] < 30:
            signals.append("RSI indicates oversold conditions (Bullish)")
        elif latest['RSI'] > 70:
            signals.append("RSI indicates overbought conditions (Bearish)")
            
   
        if latest['MACD'] > latest['Signal_Line']:
            signals.append("MACD above signal line (Bullish)")
        else:
            signals.append("MACD below signal line (Bearish)")
            
        
        if latest['Close'] > latest['SMA_200']:
            signals.append("Price above 200-day SMA (Bullish trend)")
        else:
            signals.append("Price below 200-day SMA (Bearish trend)")
            
        
        if latest['Close'] < latest['BB_lower']:
            signals.append("Price below lower Bollinger Band (Potential bounce)")
        elif latest['Close'] > latest['BB_upper']:
            signals.append("Price above upper Bollinger Band (Potential reversal)")
            
        for signal in signals:
            print(signal)


if __name__ == "__main__":
   
    analyzer = StockAnalyzer(input("Enter stock ticker (e.g., TSLA): ").strip().upper())
    analyzer.generate_report()
