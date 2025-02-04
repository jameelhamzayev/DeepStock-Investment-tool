from phi.playground import Playground, serve_playground_app
from phi.agent import Agent, RunResponse
from phi.model.google import Gemini
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')


os.environ["GOOGLE_API_KEY"] = 'YOUR API KEY'
ticker = input("Enter the company symbols (e.g. MSFT): ").strip().upper()

'''agents:
industry analyze (where they are going, what they are doing)
country economy analyze (how the country is doing)
competitor analyze (how strong they are, differentiate, compare)
company analyze (how they are doing, incomes, main products, what they are doing, what they are planning, analyze of their financials)
stock analyze (how the stock is doing, what is the prediction)
news analyze (what is happening in the world, how it is affecting the stock)
important events analyze (how they are affecting the stock)
important news analyze (how they are affecting the stock)


'''
sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Search and interpret news articles.",
    model=Gemini(id="gemini-2.0-flash-exp"),  
    tools=[GoogleSearch()],
    instructions=[
        "Find relevant news articles for each company and analyze the sentiment.",
        "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources.",
        "Cite your sources. Be specific and provide links."
    ],
    show_tool_calls=True,
    markdown=True,
)


industry_agent = Agent(
    name="Industry Agent",
    role="Research the industry of company. Expert in analyzing industries and market trends. Responsible for identifying growth opportunities, understanding emerging technologies, and assessing challenges within specific industries. Delivers insights into the competitive landscape and future market trajectory.",
    model=Gemini(id="gemini-2.0-flash-exp"),  
    tools=[GoogleSearch()],
    instructions=[
        "Provide a comprehensive analysis of the industry.",
        "Include: Current trends and emerging technologies or innovations.",
        "Major challenges and risks affecting the industry. Key players, market share distribution, and recent developments.",
        "Expected future trajectory, growth potential, and evolving consumer demands.",
        "How these trends impact the competitive landscape and financial opportunities within the industry"
    ],
    show_tool_calls=True,
    markdown=True,
)

countryeconomy_agent = Agent(
    name="Country Economy Agent",
    role=" Specialist in evaluating the economic health of countries. Research the economy of company's country. Focuses on key economic indicators, government policies, and global or regional events affecting economic performance. Provides detailed forecasts and risk assessments for economic trends.",
    model=Gemini(id="gemini-2.0-flash-exp"),  
    tools=[GoogleSearch()],
    instructions=[
        "Analyze key economic indicators like GDP growth, unemployment rates, inflation, and consumer confidence.",
        "Provide insights on recent fiscal or monetary policy decisions and their potential impacts.",
        "Compare the country’s economic performance to regional or global benchmarks.",
        "Highlight how the country's economic health affects the stock market or specific industries.",
        "Include key events (e.g., elections, trade agreements, geopolitical tensions) influencing the economy.",
        "Offer forecasts and risk assessments for future economic trends."
    ],
    show_tool_calls=True,
    markdown=True,
)

competitor_agent = Agent(
    name="Competitor Agent",
    role="Analyze the company's competitors. Expert in competitive analysis and benchmarking. Responsible for evaluating the strengths and weaknesses of competitors, identifying market opportunities, and assessing competitive threats. Delivers insights into competitor strategies, product offerings, and market positioning.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[GoogleSearch()],
    instructions=[
        "Identify the company's main competitors and their market share.",
        "Analyze the strengths and weaknesses of each competitor, including product offerings and market positioning.",
        "Evaluate competitor strategies, recent developments, and potential threats to the company.",
        "Compare their products, pricing, and innovation levels.",
        "Highlight what makes each competitor unique or competitive (e.g., technology, brand reputation).",
        "Discuss how their activities and strategies may impact the target company or market.",
    ],
    show_tool_calls=True,
    markdown=True,
)

company_agent = Agent(
    name="Company Agent",
    role="Analyze the company's financials and operations. Expert in financial analysis and corporate performance evaluation. Responsible for assessing the company's financial health, revenue streams, profitability, and growth prospects. Delivers insights into the company's key metrics, competitive advantages, and strategic direction. Read reports of company and analyze the company's financials.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[GoogleSearch()],
    instructions=[
        "Analyze the company's financial statements, revenue streams, and profitability.",
        "Analyze recent financial performance (revenue, profit, expenses, etc.).",
        "Summarize the company's main products, services, and markets.",
        "Discuss the company's competitive advantages and market positioning.",
        "Highlight any major developments, such as new product launches or strategic partnerships.",
        "Provide insights into the company's strategic direction and future growth prospects,  initiatives, or challenges (e.g., expansion, R&D, sustainability).",
        "Provide insights into how the company compares to competitors or aligns with industry trends.",
    ],
    show_tool_calls=True,   
    markdown=True,
)

stock_agent = Agent(
    name="Stock Agent",
    role="Analyze stock performance and predict future trends. Expert in financial metrics, market analysis, and stock predictions. Responsible for evaluating stock trends and providing actionable insights for investors.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[GoogleSearch()],
    instructions=[
        "Provide an overview of the stock’s recent performance (e.g., price changes, volume, volatility).",
        "Analyze key financial ratios and metrics (e.g., P/E ratio, dividend yield, EPS growth).",
        "Use historical trends and patterns to make short-term and long-term predictions.",
        "Include insights from analyst recommendations and investor sentiment.",
        "Highlight any risks or opportunities for investors.",
    ],
    show_tool_calls=True,   
    markdown=True,
)

important_news_agent = Agent(
    name="Important News Agent",
    role="Examine impactful news and its market implications. Expert in evaluating the significance of recent news and its influence on industries, markets, and stocks.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[GoogleSearch()],
    instructions=[
        "Identify and summarize recent news stories that are likely to influence the stock market.",
        "Analyze the key drivers behind the news and its expected impact on specific sectors or companies.",
        "Assess market sentiment and reactions to the news.",
        "Provide actionable insights on how investors or companies should respond.",
    ],
    show_tool_calls=True,   
    markdown=True,
)

events_agent = Agent(
    name="Events Agent",
    role="Analyze major events and their influence on markets. Expert in assessing the impact of significant events on stock markets and industries. Responsible for identifying risks, opportunities, and timelines of key actions.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[GoogleSearch()],
    instructions=[
        "Identify significant recent or upcoming events (e.g., earnings reports, product launches, government decisions).",
        "Discuss how these events may impact the stock market, specific industries, or companies.",
        "Highlight potential risks and opportunities associated with the events.",
        "Provide a timeline of key actions or outcomes to watch.",
    ],
    show_tool_calls=True,   
    markdown=True,
)


finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data and interpret trends.",
    model=Gemini(id="gemini-2.0-flash-exp"),  
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=[
        "Retrieve stock prices, analyst recommendations, and key financial data.",
        "Focus on trends and present the data in tables with key insights."
    ],
    show_tool_calls=True,
    markdown=True,
)

analyst_agent = Agent(
    name="Analyst Agent",
    role="Ensure thoroughness and draw conclusions.",
    model=Gemini(id="gemini-2.0-flash-exp"),  
    instructions=[
        "Check outputs for accuracy and completeness.",
        "Synthesize data to provide a final sentiment score (1-10) with justification."
    ],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),  
    team=[sentiment_agent, finance_agent, analyst_agent, industry_agent, countryeconomy_agent, competitor_agent, company_agent, stock_agent, important_news_agent, events_agent],
    instructions=[
        "Combine the expertise of all agents to provide a cohesive, well-supported response.",
        "Always include references and dates for all data points and sources.",
        "Present all data in structured tables for clarity.",
        "Explain the methodology used to arrive at the sentiment scores."
        "Provide a summary of the most important findings and insights so users can quickly grasp the key points."
        "Provide findings in a clear and concise manner, with actionable insights for investors."
    ],
    show_tool_calls=True,
    markdown=True,
)



analysis_prompt = (
    f"Analyze the sentiment for the following company during the time period of {input('Enter the analysis time period (e.g., January 21st-26th, 2025): ')}: {ticker}. \n\n"
    "1. **Industry Analysis**: Evaluate the overall industry trends, key players, challenges, and opportunities. Discuss the direction the industry is heading and how it aligns with or diverges from market expectations.\n\n"
    
    "2. **Country Economy Analysis**: Analyze the economic health of the relevant country during this period. Highlight key indicators such as GDP growth, inflation, unemployment rates, and fiscal or monetary policies. Discuss how the economic environment impacts industries or markets.\n\n"
    
    "3. **Competitor Analysis**: Identify major competitors, their market positioning, strengths, and weaknesses. Compare their strategies, products, and performance to understand their relative standing in the industry.\n\n"
    
    "4. **Company Analysis**: Dive into the company's financial performance, revenue streams, main products/services, and key developments. Analyze its strategic direction, competitive advantages, and financial health.\n\n"
    
    "5. **Stock Analysis**: Evaluate the stock's performance, price trends, volume, and volatility. Include analyst recommendations, key financial ratios, and any predictions based on historical patterns.\n\n"
    
    "6. **Important Events Analysis**: Highlight significant events (e.g., geopolitical changes, product launches, earnings reports) and their effect on the market or specific companies/stocks.\n\n"
    
    "7. **Important News Analysis**: Focus on the most critical news items and explain their direct or indirect impact on the stock, company, or industry. Provide sentiment scores where applicable.\n\n"

    "8. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n"
  
    "9. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.\n\n"
    
    "10. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings.\n\n"
    "Ensure your response is accurate, structured, thorough, and includes data-backed insights with proper references and publication dates."
)

agent_team.print_response(analysis_prompt, stream=True)


def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            raise ValueError(f"No data available for {ticker}")
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_technical_indicators(data):

    # Simple Moving Averages (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Exponential Moving Averages (EMA)
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD and Signal Line
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std().squeeze()  # ensure Series output
    data['BB_upper'] = data['BB_middle'] + 2 * std_dev
    data['BB_lower'] = data['BB_middle'] - 2 * std_dev

    return data

def analyze_trend(data):
    
    #Analyze the overall trend of the stock price using linear regression.
    
    close_prices = data['Close'].dropna().values.reshape(-1, 1)
    days = np.arange(len(close_prices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, close_prices)
    trend = "Upward" if model.coef_[0] > 0 else "Downward"
    return trend, float(model.coef_[0])

def calculate_risk_metrics(data):
  
    #risk-related metrics: volatility, beta, Sharpe ratio, and Value at Risk.
    
    returns = data['Close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252) * 100)  # Annual volatility
    market_data = yf.download('^GSPC', start=data.index[0], end=data.index[-1], progress=False)['Close']
    market_returns = market_data.pct_change().dropna()
    combined_returns = pd.concat([returns, market_returns], axis=1).dropna()
    covariance = combined_returns.cov().iloc[0, 1]
    market_variance = combined_returns.iloc[:, 1].var()
    beta = covariance / market_variance if market_variance != 0 else np.nan
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    sharpe = float(np.sqrt(252) * excess_returns.mean() / returns.std())
    var = float(np.percentile(returns, 5) * 100)

    return {
        'Volatility': volatility,
        'Beta': float(beta),
        'Sharpe_Ratio': sharpe,
        'Value_at_Risk': var
    }

def plot_stock_analysis(data, ticker):

    plt.figure(figsize=(15, 10))
    
    # Price and Moving Averages, Bollinger Bands
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

    # RSI
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['RSI'], label='RSI', color='purple', alpha=0.7)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.fill_between(data.index, data['RSI'], 70, where=(data['RSI']>=70), color='r', alpha=0.3)
    plt.fill_between(data.index, data['RSI'], 30, where=(data['RSI']<=30), color='g', alpha=0.3)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Macroeconomic & Financial Data
def get_macro_data():

    macro_data = {
        'PCE': 3.2, 
        'PCE_Outlook': 'Stable',
        'Interest_Rate': 4.5, 
        'Interest_Rate_Outlook': 'Increasing gradually',
        'Unemployment_Rate': 3.7
    }
    return macro_data

def get_financial_metrics(ticker):
    tkr = yf.Ticker(ticker)
    info = tkr.info

    revenue = info.get('totalRevenue', None)
    ebitda = info.get('ebitda', None)
    total_cash_flow = info.get('totalCashflow', None)
    #There are some problems in ROIC. It will be fixed. 
    roic = info.get('returnOnInvestedCapital', None)
    if roic is None:
        operating_income = info.get('operatingIncome', None)
        total_assets = info.get('totalAssets', None)
        current_liabilities = info.get('currentLiabilities', None)
        if operating_income and total_assets and current_liabilities and (total_assets - current_liabilities) != 0:
            tax_rate = 0.21  # Default
            roic_calculated = (operating_income * (1 - tax_rate)) / (total_assets - current_liabilities)
            roic = roic_calculated * 100  #percentage
        else:
            roic = "N/A"


    fcf_growth_qoq = None
    revenue_growth_qoq = None
    ebitda_growth_qoq = None


    q_cash = tkr.quarterly_cashflow
    if not q_cash.empty and 'Free Cash Flow' in q_cash.index:
        fcf_series = q_cash.loc['Free Cash Flow']
        if len(fcf_series) >= 2:
            latest, previous = fcf_series.iloc[0], fcf_series.iloc[1]
            if previous:
                fcf_growth_qoq = ((latest - previous) / abs(previous)) * 100

 
    q_fin = tkr.quarterly_financials
    if not q_fin.empty:
        if 'Total Revenue' in q_fin.index:
            rev_series = q_fin.loc['Total Revenue']
            if len(rev_series) >= 2:
                latest, previous = rev_series.iloc[0], rev_series.iloc[1]
                if previous:
                    revenue_growth_qoq = ((latest - previous) / abs(previous)) * 100
        if 'EBITDA' in q_fin.index:
            ebitda_series = q_fin.loc['EBITDA']
            if len(ebitda_series) >= 2:
                latest, previous = ebitda_series.iloc[0], ebitda_series.iloc[1]
                if previous:
                    ebitda_growth_qoq = ((latest - previous) / abs(previous)) * 100

    metrics = {
        'Revenue': revenue,
        'EBITDA': ebitda,
        'Total_Cash_Flow': total_cash_flow,
        'FCF_Growth_QoQ': fcf_growth_qoq,
        'Revenue_Growth_QoQ': revenue_growth_qoq,
        'EBITDA_Growth_QoQ': ebitda_growth_qoq,
        'ROIC': roic
    }
    return metrics

def get_competitors(ticker):

    competitors_map = {
        'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'HPQ', 'DELL'],
        'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'IBM', 'SAP'],
        'GOOGL': ['AAPL', 'MSFT', 'FB', 'AMZN', 'TWTR'],
        'AMZN': ['EBAY', 'WMT', 'SHOP', 'BABA', 'GOOGL'],
        'FB':   ['SNAP', 'TWTR', 'PINTEREST', 'GOOGL'],
        'TSLA': ['GM', 'F', 'NIO', 'BYD'],
        'NFLX': ['DIS', 'HULU', 'AMZN', 'AAPL'],
        'NVDA': ['AMD', 'INTC', 'QCOM', 'MSFT'],
        'IBM':  ['MSFT', 'ORCL', 'HPQ', 'DELL'],
        'ORCL': ['IBM', 'SAP', 'MSFT', 'ADP'],
        # You need to add competitors manually for other companies for now but it will be updated with a more robust method in next version.
    }
    return competitors_map.get(ticker, [])


# Report Generation


def generate_analysis_report(ticker, data, trend, slope, risk_metrics, macro_data, financial_metrics):
    
   #combining technical, macro, and financial analyses.
   
    latest_price = float(data['Close'].iloc[-1])
    rsi_value = float(data['RSI'].iloc[-1])
    macd = float(data['MACD'].iloc[-1])
    signal_line = float(data['Signal_Line'].iloc[-1])
    daily_change = float(data['Close'].pct_change().iloc[-1] * 100)
    
    report = [
        f"\nStock Analysis Report for {ticker}:",
        f"Current Price: ${latest_price:.2f} ({daily_change:+.2f}%)",
        f"Trend: {trend} (Slope: {slope:.6f})",
        f"Volatility: {risk_metrics['Volatility']:.2f}%",
        f"Beta: {risk_metrics['Beta']:.2f}",
        f"Sharpe Ratio: {risk_metrics['Sharpe_Ratio']:.2f}",
        f"Value at Risk (95%): {risk_metrics['Value_at_Risk']:.2f}%",
        "\nTechnical Indicators:",
        f"RSI: {rsi_value:.1f}",
        f"MACD: {macd:.2f}",
        f"Signal Line: {signal_line:.2f}",
        f"50-day MA: ${float(data['SMA_50'].iloc[-1]):.2f}",
        f"200-day MA: ${float(data['SMA_200'].iloc[-1]):.2f}",
        "\nMacroeconomic Data:",
        f"PCE: {macro_data['PCE']}% (Outlook: {macro_data['PCE_Outlook']})",
        f"Interest Rate: {macro_data['Interest_Rate']}% (Outlook: {macro_data['Interest_Rate_Outlook']})",
        f"Unemployment Rate: {macro_data['Unemployment_Rate']}%",
        "\nFinancial Metrics:",
        f"Revenue: {financial_metrics['Revenue']}",
        f"EBITDA: {financial_metrics['EBITDA']}",
        f"Total Cash Flow: {financial_metrics['Total_Cash_Flow']}",
        f"FCF Growth QoQ: {financial_metrics['FCF_Growth_QoQ'] if financial_metrics['FCF_Growth_QoQ'] is not None else 'N/A'}%",
        f"Revenue Growth QoQ: {financial_metrics['Revenue_Growth_QoQ'] if financial_metrics['Revenue_Growth_QoQ'] is not None else 'N/A'}%",
        f"EBITDA Growth QoQ: {financial_metrics['EBITDA_Growth_QoQ'] if financial_metrics['EBITDA_Growth_QoQ'] is not None else 'N/A'}%",
        f"ROIC: {financial_metrics['ROIC']}",
        "\nTrading Signals:"
    ]

    #trading signals based on RSI and MACD
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


# Main Program


def main():
    competitors = get_competitors(ticker) #do not forget to check competitor map in line 416
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"\nAnalyzing {ticker}...\n")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data is None:
        return
    
    # Process main stock data
    stock_data = calculate_technical_indicators(stock_data)
    trend, slope = analyze_trend(stock_data)
    risk_metrics = calculate_risk_metrics(stock_data)
    macro_data = get_macro_data()  # Using dummy macro data here
    financial_metrics = get_financial_metrics(ticker)
    report = generate_analysis_report(ticker, stock_data, trend, slope, risk_metrics, macro_data, financial_metrics)
    
    for line in report:
        print(line)
    plot_stock_analysis(stock_data, ticker)
    
    # Process competitors
    if competitors:
        print("\n\n--- Competitor Analysis ---")
        for comp in competitors:
            print(f"\nAnalyzing competitor: {comp}")
            comp_data = fetch_stock_data(comp, start_date, end_date)
            if comp_data is None:
                print(f"Could not fetch data for {comp}.")
                continue
            comp_data = calculate_technical_indicators(comp_data)
            comp_trend, comp_slope = analyze_trend(comp_data)
            comp_risk_metrics = calculate_risk_metrics(comp_data)
            comp_macro_data = get_macro_data()  
            comp_financial_metrics = get_financial_metrics(comp)
            comp_report = generate_analysis_report(comp, comp_data, comp_trend, comp_slope, comp_risk_metrics, comp_macro_data, comp_financial_metrics)
            for line in comp_report:
                print(line)
            plot_stock_analysis(comp_data, comp)

if __name__ == "__main__":
    main()
