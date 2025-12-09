"""
preprocessing.py
----------------
Preprocesses and cleans the data collected by `data_loader.py`.
It merges price data, financial statement data, and macroeconomic indicators.
Handles missing values (including EPS gaps before 2019)
and creates engineered features for modeling.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np

# ============================
# Configuration
# ============================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================
# Load and merge price + financial data
# ============================
def load_company_data(ticker: str) -> pd.DataFrame:
    """Loads and merges stock price and financial data for a company."""
    price_path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
    fin_path = os.path.join(DATA_DIR, f"{ticker}_financials.csv")

    if not os.path.exists(price_path) or not os.path.exists(fin_path):
        print(f"⚠️ Missing data for {ticker}. Skipping.")
        return pd.DataFrame()

    # Load data
    price_df = pd.read_csv(price_path)
    fin_df = pd.read_csv(fin_path)

    # Convert Date to datetime and extract Year
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df['Year'] = price_df['Date'].dt.year

    # ✅ Convert numeric columns to float (fix for the 'str' error)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_cols:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

    # Ensure Year column exists in financials
    if 'Year' not in fin_df.columns:
        fin_df.reset_index(inplace=True)
        fin_df.rename(columns={'index': 'Year'}, inplace=True)

    # Merge on Year
    merged = pd.merge(price_df, fin_df, on='Year', how='left')

    # Drop any rows missing 'Close' price (useless for analysis)
    merged = merged.dropna(subset=['Close'])

    return merged
# ============================
# Handle missing values
# ============================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing financial data and cleans dataset."""
    fill_cols = ['Earnings', 'Revenue', 'Net Income', 'EPS']

    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median())

    # Drop rows where the main price column is missing
    df = df.dropna(subset=['Close'])
    return df


# ============================
# Create derived features
# ============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates engineered features like returns and growth rates."""
    if 'Date' not in df.columns:
        print("⚠️ Skipping feature creation: missing 'Date' column.")
        return df

    df = df.sort_values('Date')

    # Market-based features
    df['Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Return'].rolling(window=30).std()

    # Financial growth features
    if 'Total Revenue' in df.columns:
        df['Revenue_Growth'] = df['Total Revenue'].pct_change()
    if 'Earnings' in df.columns:
        df['Earnings_Growth'] = df['Earnings'].pct_change()
    if 'EPS' in df.columns:
        df['EPS_Growth'] = df['EPS'].pct_change()
    if 'Net Income' in df.columns:
        df['Net_Income_Growth'] = df['Net Income'].pct_change()

    # Clean
    df = df.dropna().reset_index(drop=True)
    return df


# ============================
# Add macroeconomic indicators
# ============================
def merge_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges company data with macroeconomic indicators (US10Y, CPI)
    using nearest date matching.
    Handles null or invalid dates in the macro dataset.
    """
    macro_path = os.path.join(DATA_DIR, "macro_data.csv")

    if not os.path.exists(macro_path):
        print("⚠️ No macroeconomic data found. Skipping merge.")
        return df

    # Load and clean macro data
    macro_df = pd.read_csv(macro_path)
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')

    # ✅ Remove invalid or missing dates
    macro_df = macro_df.dropna(subset=['Date'])

    # ✅ Ensure numeric types
    for col in ['US10Y_Yield', 'US_CPI']:
        if col in macro_df.columns:
            macro_df[col] = pd.to_numeric(macro_df[col], errors='coerce')

    # Sort both datasets
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    macro_df = macro_df.sort_values('Date')

    # ✅ Use merge_asof for time-based alignment
    merged_df = pd.merge_asof(
        df,
        macro_df,
        on='Date',
        direction='backward',  # Use most recent macro data available
        tolerance=pd.Timedelta("60D")  # Allow up to 60 days gap
    )

    # Fill missing macro values
    merged_df[['US10Y_Yield', 'US_CPI']] = merged_df[['US10Y_Yield', 'US_CPI']].ffill().bfill()

    print("✅ Macroeconomic data merged successfully.")
    return merged_df

# ============================
# Full preprocessing pipeline
# ============================
def preprocess_all(tickers: list):
    all_data = []

    for ticker in tickers:
        print(f"\n⚙️ Preprocessing {ticker}...")

        df = load_company_data(ticker)
        if df.empty:
            continue

        df = handle_missing_values(df)
        df = create_features(df)
        df = merge_macro_data(df)

        all_data.append(df)

    # Combine all company data
    if not all_data:
        print("❌ No data processed.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(os.path.join(PROCESSED_DIR, "combined_data.csv"), index=False)

    print(f"✅ Preprocessing complete. Data saved to {PROCESSED_DIR}/combined_data.csv")


# ============================
# Main
# ============================
if __name__ == "__main__":
    from data_loader import TICKERS

    print("\n🚀 Starting preprocessing pipeline...\n")
    preprocess_all(list(TICKERS.values()))
    print("\n🎯 All company + macro data cleaned and saved in 'processed/' folder.")
