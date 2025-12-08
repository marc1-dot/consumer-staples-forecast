"""
data_loader.py
----------------
Downloads market data (daily prices) and financial statement data (income, EPS, etc.)
for Consumer Staples companies using yfinance.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import yfinance as yf
import pandas as pd
import os

# ============================
# Configuration
# ============================
TICKERS = {
    'Nestle': 'NESN.SW',
    'Procter & Gamble': 'PG',
    'Unilever': 'UL',
    'Coca-Cola': 'KO',
    'PepsiCo': 'PEP'
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================
# Download stock price data
# ============================
def download_stock_data(ticker: str, start='2015-01-01', end='2025-11-01') -> pd.DataFrame:
    """Download historical stock prices."""
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    data['Ticker'] = ticker
    data.to_csv(os.path.join(DATA_DIR, f"{ticker}_prices.csv"), index=False)
    print(f"✅ Saved price data for {ticker}")
    return data

# ============================
# Download financial data
# ============================
def get_financials(ticker: str) -> pd.DataFrame:
    """Retrieve revenues, EPS, and net income from Yahoo Finance."""
    company = yf.Ticker(ticker)
    
    # Try to extract key fundamentals
    income = company.income_stmt.T if company.income_stmt is not None else pd.DataFrame()
    balance = company.balance_sheet.T if company.balance_sheet is not None else pd.DataFrame()
    cashflow = company.cashflow.T if company.cashflow is not None else pd.DataFrame()

    # Combine relevant fields
    df = pd.DataFrame()
    if not income.empty:
        df['Total Revenue'] = income.get('Total Revenue', pd.Series(dtype='float'))
        df['Gross Profit'] = income.get('Gross Profit', pd.Series(dtype='float'))
        df['Net Income'] = income.get('Net Income', pd.Series(dtype='float'))
        df['EPS'] = income.get('Basic EPS', pd.Series(dtype='float'))
        df['Operating Income'] = income.get('Operating Income', pd.Series(dtype='float'))

    df['Year'] = df.index.year
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print(f"⚠️ No financial data available for {ticker}.")
        return pd.DataFrame()

    df.to_csv(os.path.join(DATA_DIR, f"{ticker}_financials.csv"), index=False)
    print(f"✅ Saved financial data for {ticker}")
    return df

# ============================
# Fetch all data
# ============================
def fetch_all_data():
    for name, ticker in TICKERS.items():
        print(f"\n📊 Fetching data for {name} ({ticker})...")
        download_stock_data(ticker)
        get_financials(ticker)

if __name__ == "__main__":
    print("\n🚀 Collecting financial & market data...\n")
    fetch_all_data()
    print("\n🎯 All data downloaded in /data/")
