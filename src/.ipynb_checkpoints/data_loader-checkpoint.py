"""
data_loader.py
----------------
This script handles automatic data collection for the Consumer Staples Forecasting project.
It retrieves historical stock price data and key financial metrics (revenues, EPS, etc.)
from Yahoo Finance using the yfinance library.


Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""


# ============================
# Import required libraries
# ============================

import yfinance as yf # Yahoo Finance API for market data
import pandas as pd # Data manipulation
import os # File management

# ============================
# Configuration
# ============================

# Define the tickers for major Consumer Staples companies

TICKERS = {
'Nestle': 'NESN.SW',
'Procter & Gamble': 'PG',
'Unilever': 'UL',
'Coca-Cola': 'KO',
'PepsiCo': 'PEP'
}

# Create a folder for raw data if it doesn’t exist
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================
# Function: download_stock_data
# ============================
def download_stock_data(ticker: str, start: str = '2015-01-01', end: str = '2025-11-01') -> pd.DataFrame:
"""
Downloads daily stock price data for a given ticker using yfinance.


Parameters
----------
ticker : str
The stock ticker symbol (e.g., 'AAPL', 'KO', 'PG').
start : str
Start date for data retrieval in 'YYYY-MM-DD' format.
end : str
End date for data retrieval in 'YYYY-MM-DD' format.


Returns
-------
pd.DataFrame
DataFrame containing the historical price data with Date as index.
"""
try:
data = yf.download(ticker, start=start, end=end)
data.reset_index(inplace=True)
data.to_csv(os.path.join(DATA_DIR, f"{ticker}_prices.csv"), index=False)
print(f"✅ Data for {ticker} saved successfully.")
return data
except Exception as e:
print(f"❌ Error downloading data for {ticker}: {e}")
return pd.DataFrame()

# ============================
# Function: get_financials
# ============================
def get_financials(ticker: str) -> pd.DataFrame:
"""
Retrieves key financial statement data (revenues, EPS, net income, etc.)
using yfinance's financials and earnings attributes.


Parameters
----------
ticker : str
The stock ticker symbol.


Returns
-------
pd.DataFrame
Combined DataFrame with key metrics (revenue, net income, EPS).
"""
try:
company = yf.Ticker(ticker)
financials = company.financials.T # Income statement (transposed)
earnings = company.earnings # Annual earnings summary


# Merge data and clean
df = financials.merge(earnings, left_index=True, right_index=True, how='outer')
df.reset_index(inplace=True)
df.rename(columns={'index': 'Year'}, inplace=True)


df.to_csv(os.path.join(DATA_DIR, f"{ticker}_financials.csv"), index=False)
print(f"✅ Financial data for {ticker} saved successfully.")
return df
except Exception as e:
print(f"❌ Error retrieving financial data for {ticker}: {e}")
return pd.DataFrame()

# ============================
# Function: fetch_all_data
# ============================
def fetch_all_data():
"""
Iterates through all company tickers, downloads both market and financial data,
and stores them in the data/ folder.
"""
for name, ticker in TICKERS.items():
print(f"\n📊 Fetching data for {name} ({ticker})...")
price_data = download_stock_data(ticker)
fin_data = get_financials(ticker)


if not price_data.empty and not fin_data.empty:
print(f"✅ Completed data retrieval for {name}.")
else:
print(f"⚠️ Partial data for {name}.")