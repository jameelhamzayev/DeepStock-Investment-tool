# DeepStock:Comprehensive Stock Analysis Tool

## Overview
This is a powerful stock analysis tool that combines technical analysis, fundamental analysis, and sentiment analysis using multiple AI agents. The tool provides in-depth analysis of stocks, their competitors, industry trends, and economic factors affecting market performance.
The main file is main-stock-ai-agent.py. You can check out other files. Especially you might want to check can_be_usefull1.py and can_be_usefull2.py files.
## Features
- **Multi-Agent Analysis System**
  - Sentiment Analysis Agent
  - Industry Analysis Agent
  - Economic Analysis Agent
  - Competitor Analysis Agent
  - Company Analysis Agent
  - Stock Analysis Agent
  - News Analysis Agent
  - Events Analysis Agent
  - Finance Agent
  - Analyst Agent

- **Technical Analysis**
  - Price trends and patterns
  - Moving averages (SMA & EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Risk metrics (Volatility, Beta, Sharpe Ratio, Value at Risk)

- **Fundamental Analysis**
  - Revenue analysis
  - EBITDA metrics
  - Cash flow analysis
  - ROIC calculations (main code has some problems but you can use other files to calculate ROIC untill I fix that)
  - Quarter-over-Quarter growth rates

- **Visualization**
  - Interactive price charts
  - Technical indicator plots
  - Trend analysis visualizations

## Requirements
```python
pip install -r requirements.txt
```

Required packages:
- phi
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn
- warnings

## Environment Setup
1. Set up your Google API key in the environment variables:
```python
os.environ["GOOGLE_API_KEY"] = 'your_google_api_key'
```

## Usage
1. Run the main script:
```python
python main-stock-ai-agent.py
```

2. Enter the required information when prompted:
   - Company symbol (e.g., MSFT)
   - Analysis time period (e.g., January 21st-26th, 2025)

3. The program will generate:
   - Comprehensive analysis report
   - Technical analysis charts
   - Competitor analysis (if available)

## Analysis Components

### 1. Industry Analysis
- Overall industry trends
- Key players analysis
- Challenges and opportunities
- Future industry direction

### 2. Economic Analysis
- Country-specific economic indicators
- GDP growth
- Inflation rates
- Unemployment data
- Fiscal/monetary policies

### 3. Competitor Analysis
- Market positioning
- Competitive advantages
- Product comparisons
- Strategy evaluation

### 4. Company Analysis
- Financial performance
- Product/service analysis
- Strategic initiatives
- Growth prospects

### 5. Technical Analysis
- Price trend analysis
- Technical indicators
- Risk metrics
- Trading signals

### 6. News and Events Analysis
- Market-moving news
- Important events
- Sentiment analysis
- Impact assessment

## Competitor Mapping
The tool includes pre-defined competitor mappings for major companies. Current mappings include:
- AAPL: MSFT, GOOGL, AMZN, HPQ, DELL
- MSFT: AAPL, GOOGL, ORCL, IBM, SAP
- (More companies listed in the code)

To add new competitor mappings, update the `competitors_map` dictionary in the code. 

## Future Improvements will be:
- [ ] Add more robust method instead of competitor mapping system
- [ ] Implement machine learning-based prediction models
- [ ] Add more technical indicators
- [ ] Enhance visualization capabilities
- [ ] Add real-time data streaming
- [ ] Expand economic indicators database
- [ ] Improve prompts and agents
- [ ] and more...

## This is just a demo. Future improvements will make it good enough for analysis and making investment decision.
## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Disclaimer
This tool is for educational and research purposes only. Always do your own due diligence before making investment decisions.
