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