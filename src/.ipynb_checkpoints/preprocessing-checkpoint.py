"""
preprocessing.py
----------------
Full preprocessing pipeline for the Consumer Staples Forecasting project.
Converts all market and financial data to weekly frequency, merges macro data,
handles missing values, and creates derived features.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
from data_loader import TICKERS

# ============================
# Configuration
# ============================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================
# Helper functions
# ============================

def load_company_data(ticker: str) -> pd.DataFrame:
    """Loads and merges price and financial data for a company."""
    price_path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
    fin_path = os.path.join(DATA_DIR, f"{ticker}_financials.csv")

    if not os.path.exists(price_path) or not os.path.exists(fin_path):
        print(f"⚠️ Missing data for {ticker}. Skipping.")
        return pd.DataFrame()

    price_df = pd.read_csv(price_path)
    fin_df = pd.read_csv(fin_path)

    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df['Year'] = price_df['Date'].dt.year

    if 'Year' not in fin_df.columns:
        fin_df.reset_index(inplace=True)
        fin_df.rename(columns={'index': 'Year'}, inplace=True)

    merged = pd.merge(price_df, fin_df, on='Year', how='left')
    merged['Ticker'] = ticker
    return merged


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values for key columns."""
    for col in ['EPS', 'Net Income', 'Total Revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
    df = df.dropna(subset=['Close'])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates daily return and volatility features."""
    # Convert numeric columns safely
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('Date')
    df['Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Return'].rolling(window=30).std()

    return df


def merge_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merges macroeconomic data (US yields and inflation) with company data."""
    macro_path = os.path.join(DATA_DIR, "macro_data_clean.csv")
    if not os.path.exists(macro_path):
        print("⚠️ No macro data file found.")
        return df

    macro_df = pd.read_csv(macro_path)
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
    macro_df = macro_df.dropna(subset=['Date']).sort_values('Date')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')

    merged = pd.merge_asof(df, macro_df, on='Date', direction='backward')

    print(f"✅ Macro data merged for {merged['Date'].dt.year.min()}–{merged['Date'].dt.year.max()} ({len(merged)} rows)")
    return merged


def convert_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates data to weekly frequency (Friday close), keeping only relevant features."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')

    # 🧠 On retire Open, High, Low — on garde seulement les infos utiles
    weekly = df.resample('W-FRI').agg({
        'Close': 'last',
        'Volume': 'sum',
        'Return': 'mean',
        'Volatility_30d': 'last',
        'US10Y_Yield': 'last' if 'US10Y_Yield' in df.columns else 'first',
        'US_CPI': 'last' if 'US_CPI' in df.columns else 'first',
        'EPS': 'last' if 'EPS' in df.columns else 'first',
        'Net Income': 'last' if 'Net Income' in df.columns else 'first',
        'Total Revenue': 'last' if 'Total Revenue' in df.columns else 'first'
    }).reset_index()

    weekly['Weekly_Return'] = weekly['Close'].pct_change()
    weekly = weekly.dropna(subset=['Close'])

    print(f"📅 Converted to weekly data ({len(weekly)} rows, ending on {weekly['Date'].max().date()})")
    return weekly


# ============================
# Main pipeline
# ============================
def preprocess_all(tickers):
    all_data = []

    for ticker in tickers:
        print(f"\n⚙️ Preprocessing {ticker}...")
        df = load_company_data(ticker)
        if df.empty:
            continue

        df = handle_missing_values(df)
        df = create_features(df)
        df = merge_macro_data(df)
        df = convert_to_weekly(df)
        df['Ticker'] = ticker

        all_data.append(df)

    if not all_data:
        print("❌ No valid data found.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['Ticker', 'Date'])

    # ✅ Décalage par entreprise
    combined['Target_Weekly_Return'] = combined.groupby('Ticker')['Weekly_Return'].shift(-1)

    # ✅ Suppression des colonnes inutiles
    drop_cols = ['Open', 'High', 'Low']
    combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns], errors='ignore')

    combined.to_csv(os.path.join(PROCESSED_DIR, "combined_data.csv"), index=False)
    print(f"✅ All tickers processed into weekly data. Final shape: {combined.shape}")


# ============================
# Execution
# ============================
if __name__ == "__main__":
    print("\n🚀 Starting full weekly preprocessing pipeline...\n")
    preprocess_all(list(TICKERS.values()))
    print("\n🎯 All data converted to weekly and saved in 'processed/combined_data.csv'\n")
