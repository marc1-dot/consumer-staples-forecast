"""
preprocessing.py
----------------
This script prepares and cleans the data collected by `data_loader.py`.
It merges price data and financial statement data, handles missing values
(including EPS gaps before 2019), and creates derived features for modeling.


Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""


# ============================
# Import required libraries
# ============================
import os
import pandas as pd
import numpy as np

# ============================
# Configuration
# ============================
# Define directories
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================
# Function: load_company_data
# ============================
def load_company_data(ticker: str) -> pd.DataFrame:
    """
    Loads price and financial data for a specific ticker from CSV files,
    merges them on date/year, and aligns financial indicators with prices.
    
    
    Parameters
    ----------
    ticker : str
    The stock ticker symbol.
    
    
    Returns
    -------
    pd.DataFrame
    Combined DataFrame containing both market and financial data.
    """
    price_path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
    fin_path = os.path.join(DATA_DIR, f"{ticker}_financials.csv")
    
    
    if not os.path.exists(price_path) or not os.path.exists(fin_path):
        print(f"⚠️ Missing data for {ticker}, skipping...")
        return pd.DataFrame()


    price_df = pd.read_csv(price_path)
    fin_df = pd.read_csv(fin_path)


    # Convert date columns
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df['Year'] = price_df['Date'].dt.year


    # Merge financial data by year
    merged = pd.merge(price_df, fin_df, left_on='Year', right_on='Year', how='left')


    return merged