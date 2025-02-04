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


os.environ["GOOGLE_API_KEY"] = 'AIzaSyAd3oTLCZAZIaAJmH0KdkbRlHezap9lsxE'
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