"""
data_loader.py
----------------
Downloads market data (daily prices) and financial statement data (income, EPS, etc.)
for Consumer Staples companies using yfinance.
Also retrieves key macroeconomic indicators (US 10-Year Treasury yield and CPI inflation).

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

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
# Download macroeconomic data
# ============================
def get_macro_data(start='2020-01-01', end=None) -> pd.DataFrame:
    """
    Downloads key macroeconomic indicators:
    - US 10-Year Treasury Yield (^TNX)
    - US Inflation Rate (CPI)
    """

    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    macro_series = {
        'US10Y_Yield': '^TNX',      # 10-Year Treasury Yield
        'US_CPI': 'CPI'             # Consumer Price Index (Inflation proxy)
    }

    macro_df = pd.DataFrame()

    for name, ticker in macro_series.items():
        print(f"📈 Fetching {name} ({ticker})...")
        data = yf.download(ticker, start=start, end=end)[['Close']]
        data.rename(columns={'Close': name}, inplace=True)
        if macro_df.empty:
            macro_df = data
        else:
            macro_df = macro_df.join(data, how='outer')

    macro_df.reset_index(inplace=True)
    macro_df.to_csv(os.path.join(DATA_DIR, 'macro_data.csv'), index=False)
    print(f"✅ Saved macroeconomic data to macro_data.csv")

    return macro_df


# ============================
# Fetch all data
# ============================
def fetch_all_data():
    for name, ticker in TICKERS.items():
        print(f"\n📊 Fetching data for {name} ({ticker})...")
        download_stock_data(ticker)
        get_financials(ticker)

    # Fetch macro data (5 latest years)
    print("\n🌍 Fetching macroeconomic indicators (US10Y & CPI)...")
    get_macro_data(start='2020-01-01')
    print("✅ Macroeconomic data collected successfully.")


# ============================
# Main
# ============================
if __name__ == "__main__":
    print("\n🚀 Collecting financial & market data...\n")
    fetch_all_data()
    print("\n🎯 All data downloaded in /data/")
