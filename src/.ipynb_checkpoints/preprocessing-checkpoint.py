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
    # Ensure 'Year' column exists in financials
if 'Year' not in fin_df.columns:
    if 'index' in fin_df.columns:
        fin_df.rename(columns={'index': 'Year'}, inplace=True)
    else:
        fin_df.reset_index(inplace=True)
        fin_df.rename(columns={'index': 'Year'}, inplace=True)

# Merge financial data by year
merged = pd.merge(price_df, fin_df, on='Year', how='left')


return merged

# ============================
# Function: handle_missing_values
# ============================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing financial values and fills gaps logically.
    
    
    Strategy:
    - EPS, Revenue, Net Income: forward-fill (to propagate last known values).
    - Remaining NaNs: replaced by median (robust imputation).
    
    
    Parameters
    ----------
    df : pd.DataFrame
    The combined dataset with missing values.
    
    
    Returns
    -------
    pd.DataFrame
    Cleaned dataset ready for feature engineering.
    """
    fill_cols = ['Earnings', 'Revenue', 'Net Income', 'EPS']


    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            df[col] = df[col].fillna(df[col].median())


        df = df.dropna(subset=['Close']) # Ensure target variable is complete
    return df

# ============================
# Function: create_features
# ============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from price and financial data.
    
    
    Generated features include:
    - Daily returns
    - Rolling volatility (30 days)
    - Year-over-year revenue and EPS growth
    
    
    Parameters
    ----------
    df : pd.DataFrame
    Clean dataset with both price and fundamental variables.
    
    
    Returns
    -------
    pd.DataFrame
    Dataset enriched with engineered features.
    """
    df = df.sort_values('Date')
    
    
    # Market-based features
    df['Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Return'].rolling(window=30).std()
    
    
    # Financial growth features
    if 'Total Revenue' in df.columns:
        df['Revenue_Growth'] = df['Total Revenue'].pct_change()
    if 'Earnings' in df.columns:
        df['Earnings_Growth'] = df['Earnings'].pct_change()
    
    
    # Drop early NaNs
    df = df.dropna().reset_index(drop=True)
    return df

# ============================
# Function: preprocess_all
# ============================
def preprocess_all(tickers: list):
    """
    Processes all tickers: load, clean, feature engineer, and save.
    
    
    Parameters
    ----------
    tickers : list
    List of tickers to preprocess.
    """
    combined_data = []
    
    
    for ticker in tickers:
        print(f"\n⚙️ Preprocessing {ticker}...")
        df = load_company_data(ticker)
        if df.empty:
            continue
    
    
    df = handle_missing_values(df)
    df = create_features(df)
    df['Ticker'] = ticker
    
    
    combined_data.append(df)
    
    
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        save_path = os.path.join(PROCESSED_DIR, 'combined_data.csv')
        final_df.to_csv(save_path, index=False)
        print(f"✅ Preprocessing complete. Data saved to {save_path}")
    else:
        print("❌ No valid data to process.")
# ============================
# Main script execution
# ============================
if __name__ == "__main__":
    from data_loader import TICKERS


    print("\n🚀 Starting preprocessing pipeline...\n")
    preprocess_all(list(TICKERS.values()))
    print("\n🎯 All company data cleaned and saved in the 'processed/' folder.")