import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class StockAnalyzer:
    def __init__(self):
        self.economic_data = {
            'PCE': {'current': 2.4, 'outlook': 'Expected to moderate to 2.2% by end of 2025'},
            'Interest_Rate': {'current': 5.25, 'outlook': 'Expected 3-4 cuts in 2025'},
            'Unemployment': {'current': 3.7, 'outlook': 'Expected to remain stable around 3.8%'}
        }

    def get_competitors(self, ticker):

        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', '')
            industry = stock.info.get('industry', '')
            
       
            competitor_mapping = {
                'AAPL': ['MSFT', 'GOOGL', 'SAMSUNG'],
                'TSLA': ['GM', 'F', 'TM'],
                # Add more mappings as needed
            }
            
            return competitor_mapping.get(ticker, [])
        except Exception as e:
            print(f"Error getting competitors for {ticker}: {str(e)}")
            return []

    def get_historical_data(self, ticker, period='2y'):

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def calculate_metrics(self, ticker):
  
        try:
            stock = yf.Ticker(ticker)
            
    
            income_stmt = stock.income_stmt if hasattr(stock, 'income_stmt') else stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

  
            print(f"Available columns in income statement: {income_stmt.index.tolist()}")
            

            metrics = {
                'Revenue': {
                    'Current': 0,
                    'YoY_Growth': 0,
                    'QoQ_Growth': 0
                },
                'EBITDA': {
                    'Current': 0,
                    'YoY_Growth': 0
                },
                'Free_Cash_Flow': {
                    'Current': 0,
                    'YoY_Growth': 0
                },
                'ROIC': 0
            }


            revenue_cols = ['TotalRevenue', 'Total Revenue', 'Revenue']
            for col in revenue_cols:
                if col in income_stmt.index:
                    metrics['Revenue']['Current'] = income_stmt.loc[col].iloc[0]
                    metrics['Revenue']['YoY_Growth'] = self._calculate_growth(income_stmt.loc[col])
                    metrics['Revenue']['QoQ_Growth'] = self._calculate_growth(income_stmt.loc[col], quarterly=True)
                    break

  
            ebitda_cols = ['EBITDA', 'Ebitda']
            for col in ebitda_cols:
                if col in income_stmt.index:
                    metrics['EBITDA']['Current'] = income_stmt.loc[col].iloc[0]
                    metrics['EBITDA']['YoY_Growth'] = self._calculate_growth(income_stmt.loc[col])
                    break


            fcf_cols = ['FreeCashFlow', 'Free Cash Flow']
            for col in fcf_cols:
                if col in cash_flow.index:
                    metrics['Free_Cash_Flow']['Current'] = cash_flow.loc[col].iloc[0]
                    metrics['Free_Cash_Flow']['YoY_Growth'] = self._calculate_growth(cash_flow.loc[col])
                    break

            # Calculate ROIC
            metrics['ROIC'] = self._calculate_roic(income_stmt, balance_sheet)
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics for {ticker}: {str(e)}")
            return None

    def _calculate_growth(self, series, quarterly=False):
        """Calculate YoY or QoQ growth."""
        try:
            if len(series) < 2:
                return 0
            
            if quarterly:
                period = 1
            else:
                period = 4
                
            if len(series) <= period:
                return 0
                
            return ((series.iloc[0] - series.iloc[period]) / series.iloc[period]) * 100
        except Exception as e:
            print(f"Error calculating growth: {str(e)}")
            return 0

    def _calculate_roic(self, income_stmt, balance_sheet):
        """Calculate Return on Invested Capital."""
        try:
            # Try different possible column names for Net Income
            net_income_cols = ['NetIncome', 'Net Income']
            net_income = 0
            for col in net_income_cols:
                if col in income_stmt.index:
                    net_income = income_stmt.loc[col].iloc[0]
                    break

            # Try different possible column names for Total Assets and Current Liabilities
            assets_cols = ['TotalAssets', 'Total Assets']
            liabilities_cols = ['TotalCurrentLiabilities', 'Total Current Liabilities']
            
            total_assets = 0
            current_liabilities = 0
            
            for col in assets_cols:
                if col in balance_sheet.index:
                    total_assets = balance_sheet.loc[col].iloc[0]
                    break
                    
            for col in liabilities_cols:
                if col in balance_sheet.index:
                    current_liabilities = balance_sheet.loc[col].iloc[0]
                    break

            nopat = net_income * (1 - 0.21)  # Assuming 21% tax rate
            invested_capital = total_assets - current_liabilities
            
            if invested_capital == 0:
                return 0
                
            return (nopat / invested_capital) * 100
        except Exception as e:
            print(f"Error calculating ROIC: {str(e)}")
            return 0

    def generate_analysis_report(self, ticker):
    
        competitors = self.get_competitors(ticker)
        main_analysis = self.calculate_metrics(ticker)
        
        if main_analysis is None:
            return None
            
        competitor_analyses = {}
        for comp in competitors:
            comp_analysis = self.calculate_metrics(comp)
            if comp_analysis is not None:
                competitor_analyses[comp] = comp_analysis
        
        report = {
            'Company': ticker,
            'Economic_Indicators': self.economic_data,
            'Financial_Metrics': main_analysis,
            'Competitor_Analysis': competitor_analyses,
            'Historical_Performance': self.get_historical_data(ticker)
        }
        
        return report

    def format_report(self, report):

        if report is None:
            return "Error generating report. Please check the ticker symbol and try again."
            
        output = f"""
Stock Analysis Report for {report['Company']}
==========================================

Economic Indicators:
------------------
PCE: {report['Economic_Indicators']['PCE']['current']}%
PCE Outlook: {report['Economic_Indicators']['PCE']['outlook']}
Interest Rate: {report['Economic_Indicators']['Interest_Rate']['current']}%
Interest Rate Outlook: {report['Economic_Indicators']['Interest_Rate']['outlook']}
Unemployment: {report['Economic_Indicators']['Unemployment']['current']}%
Unemployment Outlook: {report['Economic_Indicators']['Unemployment']['outlook']}

Company Metrics:
--------------
Revenue: ${report['Financial_Metrics']['Revenue']['Current']:,.2f}
Revenue YoY Growth: {report['Financial_Metrics']['Revenue']['YoY_Growth']:.2f}%
Revenue QoQ Growth: {report['Financial_Metrics']['Revenue']['QoQ_Growth']:.2f}%

EBITDA: ${report['Financial_Metrics']['EBITDA']['Current']:,.2f}
EBITDA YoY Growth: {report['Financial_Metrics']['EBITDA']['YoY_Growth']:.2f}%

Free Cash Flow: ${report['Financial_Metrics']['Free_Cash_Flow']['Current']:,.2f}
FCF YoY Growth: {report['Financial_Metrics']['Free_Cash_Flow']['YoY_Growth']:.2f}%

ROIC: {report['Financial_Metrics']['ROIC']:.2f}%

Competitor Analysis:
------------------
"""
        for comp, metrics in report['Competitor_Analysis'].items():
            output += f"\n{comp}:\n"
            output += f"Revenue: ${metrics['Revenue']['Current']:,.2f} (YoY: {metrics['Revenue']['YoY_Growth']:.2f}%)\n"
            output += f"EBITDA: ${metrics['EBITDA']['Current']:,.2f} (YoY: {metrics['EBITDA']['YoY_Growth']:.2f}%)\n"
            output += f"FCF: ${metrics['Free_Cash_Flow']['Current']:,.2f} (YoY: {metrics['Free_Cash_Flow']['YoY_Growth']:.2f}%)\n"
            output += f"ROIC: {metrics['ROIC']:.2f}%\n"
        
        return output

def main():
    analyzer = StockAnalyzer()
    
    while True:
        ticker = input("Enter stock ticker (or 'quit' to exit): ").upper()
        if ticker.lower() == 'quit':
            break
            
        report = analyzer.generate_analysis_report(ticker)
        formatted_report = analyzer.format_report(report)
        print(formatted_report)

if __name__ == "__main__":
    main()