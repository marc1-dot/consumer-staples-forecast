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