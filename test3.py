import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas_ta as ta
import matplotlib.pyplot as plt
from google.generativeai import configure, GenerativeModel
from concurrent.futures import ThreadPoolExecutor
import requests_cache

# Configuration
configure(api_key='YOUR_API_KEY')
gemini_model = GenerativeModel('gemini-2.0-flash-exp')
requests_cache.install_cache('stock_cache', expire_after=3600)

def get_fundamental_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        balance_sheet = stock.balance_sheet.iloc[:,0].squeeze() if not stock.balance_sheet.empty else pd.Series()
        income_stmt = stock.income_stmt.iloc[:,0].squeeze() if not stock.income_stmt.empty else pd.Series()
        cash_flow = stock.cashflow.iloc[:,0].squeeze() if not stock.cashflow.empty else pd.Series()

        ratios = {
            'PE Ratio': info.get('trailingPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Debt/Equity': info.get('debtToEquity'),
            'ROE': info.get('returnOnEquity'),
            'Beta': info.get('beta'),
            'Dividend Yield': info.get('dividendYield'),
            'Profit Margin': (income_stmt.get('Net Income', 0) / 
                             income_stmt.get('Total Revenue', 1)) if income_stmt.get('Net Income') else 0
        }
        
        return {
            'ratios': ratios,
            'cash_flow': cash_flow.get('Free Cash Flow', 0),
            'info': info
        }
    except Exception as e:
        print(f"Fundamental analysis error: {e}")
        return {}

def get_technical_analysis(ticker):
    try:
        data = yf.download(ticker, period='6mo', progress=False)
        if data.empty:
            return {}

        # Clean column names
        data.columns = [str(col).replace(' ', '_') for col in data.columns]

        # Calculate indicators
        data.ta.rsi(length=14, append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=200, append=True)
        
        return {
            'RSI': data.get('RSI_14', pd.Series([0]))[-1],
            'EMA_50': data.get('EMA_50', pd.Series([0]))[-1],
            'EMA_200': data.get('EMA_200', pd.Series([0]))[-1],
            'Close': data['Close'][-1] if 'Close' in data else 0,
            'MACD': data.ta.macd()['MACD_12_26_9'][-1] if 'Close' in data else 0
        }
    except Exception as e:
        print(f"Technical analysis error: {e}")
        return {}

def get_macro_economic_data():
    """Reliable macro data scraping with CSS selectors"""
    try:
        url = 'https://tradingeconomics.com/united-states/indicators'
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        macro = {}
        for row in soup.select('tr.datatable-row'):
            cells = row.select('td')
            if len(cells) >= 2:
                name = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                macro[name] = value
                
        return {
            'Interest Rate': macro.get('Federal Funds Rate', 'N/A'),
            'Inflation': macro.get('Inflation Rate', 'N/A'),
            'GDP Growth': macro.get('GDP Growth Rate', 'N/A')
        }
    except Exception as e:
        print(f"Macro data error: {e}")
        return {}

def get_news_sentiment(ticker):
    """Robust sentiment analysis with error fallback"""
    try:
        news_url = f'https://news.google.com/rss/search?q={ticker}+stock'
        response = requests.get(news_url, timeout=5)
        soup = BeautifulSoup(response.text, 'xml')
        
        headlines = [item.title.text for item in soup.find_all('item')[:5]]
        prompt = f"""Analyze sentiment from these headlines: {headlines}
        Return ONLY one word: positive/neutral/negative"""
        
        response = gemini_model.generate_content(prompt, request_options={'timeout': 7})
        return response.text.strip().lower()
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 'neutral'

def generate_report(ticker):
    """Error-resistant report generation"""
    try:
        with ThreadPoolExecutor() as executor:
            fundamental_future = executor.submit(get_fundamental_analysis, ticker)
            technical_future = executor.submit(get_technical_analysis, ticker)
            macro_future = executor.submit(get_macro_economic_data)
            
            fundamental = fundamental_future.result()
            technical = technical_future.result()
            macro = macro_future.result()
            sentiment = get_news_sentiment(ticker)

        # Format numerical values safely
        def safe_format(value, fmt):
            try: return fmt.format(value)
            except: return str(value)

        report = f"""
        === {ticker.upper()} STOCK ANALYSIS REPORT ===
        
        [FUNDAMENTAL ANALYSIS]
        • PE Ratio: {safe_format(fundamental.get('ratios', {}).get('PE Ratio'), '{:.2f}')}
        • Debt/Equity: {safe_format(fundamental.get('ratios', {}).get('Debt/Equity'), '{:.2f}')}
        • Profit Margin: {safe_format(fundamental.get('ratios', {}).get('Profit Margin'), '{:.2%}')}
        • Cash Flow: ${safe_format(fundamental.get('cash_flow', 0), '{:,.2f}')}
        
        [TECHNICAL ANALYSIS]
        • Price: ${safe_format(technical.get('Close', 0), '{:.2f}')}
        • RSI(14): {safe_format(technical.get('RSI', 50), '{:.1f}')} {'(Overbought)' if technical.get('RSI', 0) > 70 else '(Oversold)' if technical.get('RSI', 0) < 30 else ''}
        • Trend: {'Bullish' if technical.get('EMA_50', 0) > technical.get('EMA_200', 0) else 'Bearish'}
        
        [MACRO ENVIRONMENT]
        • Interest Rate: {macro.get('Interest Rate', 'N/A')}
        • Inflation: {macro.get('Inflation', 'N/A')}
        
        [MARKET SENTIMENT]
        • News Sentiment: {sentiment.capitalize()}
        
        === RECOMMENDATION ===
        {'Consider Buying' if (
            fundamental.get('ratios', {}).get('PE Ratio', 100) < 25 and
            technical.get('RSI', 70) < 70 and
            sentiment == 'positive'
        ) else 'Neutral/Hold'}
        """

        # Plotting with validation
        if technical.get('Close'):
            plt.figure(figsize=(10,5))
            plt.title(f'{ticker} Price Movement')
            if 'EMA_50' in technical:
                plt.plot(technical['EMA_50'], label='50 EMA')
            if 'EMA_200' in technical:
                plt.plot(technical['EMA_200'], label='200 EMA')
            plt.legend()
            plt.show()

        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    print(generate_report(ticker))


