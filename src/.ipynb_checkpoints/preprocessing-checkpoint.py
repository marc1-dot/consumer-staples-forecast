"""
preprocessing.py
----------------
This script prepares and cleans the data collected by `data_loader.py`.
It merges price data, financial statement data, and macroeconomic indicators
(US 10-Year Yield and CPI), handles missing values, and creates derived features.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

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
# Load & merge company data
# ============================
def load_company_data(ticker: str) -> pd.DataFrame:
    price_path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
    fin_path = os.path.join(DATA_DIR, f"{ticker}_financials.csv")

    if not os.path.exists(price_path) or not os.path.exists(fin_path):
        print(f"⚠️ Missing data files for {ticker}. Skipping.")
        return pd.DataFrame()

    price_df = pd.read_csv(price_path)
    fin_df = pd.read_csv(fin_path)

    # Ensure correct date format
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df['Year'] = price_df['Date'].dt.year

    if 'Year' not in fin_df.columns:
        fin_df.reset_index(inplace=True)
        fin_df.rename(columns={'index': 'Year'}, inplace=True)

    merged = pd.merge(price_df, fin_df, on='Year', how='left')
    return merged


# ============================
# Handle missing values
# ============================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    fill_cols = ['Earnings', 'Revenue', 'Net Income', 'EPS', 'Total Revenue', 'Gross Profit']
    for col in fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].ffill().bfill().fillna(df[col].median())

    df = df.dropna(subset=['Close'])
    return df


# ============================
# Feature creation
# ============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from price and financial data.
    """

    # Vérifie que la colonne Date existe
    if 'Date' not in df.columns:
        print("⚠️ Skipping: 'Date' column not found.")
        return df

    df = df.sort_values('Date')

    # ✅ Convert numeric columns that might be strings
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Market-based features
    df['Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Return'].rolling(window=30).std()

    # Financial growth features
    if 'Total Revenue' in df.columns:
        df['Revenue_Growth'] = pd.to_numeric(df['Total Revenue'], errors='coerce').pct_change(fill_method=None)
    if 'Earnings' in df.columns:
        df['Earnings_Growth'] = pd.to_numeric(df['Earnings'], errors='coerce').pct_change(fill_method=None)

    df = df.dropna().reset_index(drop=True)
    return df

# ============================
# Merge with macroeconomic data
# ============================
def merge_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    macro_path = os.path.join(DATA_DIR, "macro_data.csv")

    if not os.path.exists(macro_path):
        print("⚠️ No macroeconomic data found. Skipping merge.")
        return df

    macro_df = pd.read_csv(macro_path)
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
    macro_df = macro_df.dropna(subset=['Date'])
    macro_df = macro_df.sort_values('Date')

    # Ensure numeric types
    for col in ['US10Y_Yield', 'US_CPI']:
        if col in macro_df.columns:
            macro_df[col] = pd.to_numeric(macro_df[col], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')

    merged_df = pd.merge_asof(
        df,
        macro_df,
        on='Date',
        direction='backward',
        tolerance=pd.Timedelta("60D")
    )

    merged_df[['US10Y_Yield', 'US_CPI']] = merged_df[['US10Y_Yield', 'US_CPI']].ffill().bfill()

    print("✅ Macroeconomic data merged successfully.")
    print("📊 Macro sample:\n", merged_df[['Date', 'US10Y_Yield', 'US_CPI']].tail(3))
    return merged_df


# ============================
# Main preprocessing pipeline
# ============================
def preprocess_all(ticker_list):
    all_data = []

    for ticker in ticker_list:
        print(f"\n⚙️ Preprocessing {ticker}...")
        df = load_company_data(ticker)
        if df.empty:
            continue

        df = handle_missing_values(df)
        df = create_features(df)
        df = merge_macro_data(df)

        all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(PROCESSED_DIR, "combined_data.csv"), index=False)
        print("✅ Combined dataset saved successfully.")
        print("📊 Columns now include:", [c for c in combined.columns if 'US' in c])
        print("🔍 Preview of last 3 rows:\n", combined[['Date', 'US10Y_Yield', 'US_CPI']].tail(3))
    else:
        print("❌ No valid data found.")


# ============================
# Run pipeline
# ============================
if __name__ == "__main__":
    from data_loader import TICKERS
    print("\n🚀 Starting preprocessing pipeline...\n")
    preprocess_all(list(TICKERS.values()))
    print("\n🎯 All company + macro data cleaned and saved in the 'processed/' folder.")
