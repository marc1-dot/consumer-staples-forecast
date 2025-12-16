"""
preprocessing.py
----------------
Enhanced preprocessing pipeline for the Consumer Staples Forecasting project.
Converts all market and financial data to weekly frequency, merges macro data,
handles missing values, creates advanced financial and technical features,
and outputs cleaned data ready for model training.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

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
# Helper Functions - Data Loading
# ============================

def load_company_data(ticker: str) -> pd.DataFrame:
    """
    Loads and merges price and financial data for a company.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'PG', 'KO')
    
    Returns:
        pd.DataFrame: Merged dataset with prices and financials
    """
    price_path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
    fin_path = os.path.join(DATA_DIR, f"{ticker}_financials.csv")

    if not os.path.exists(price_path) or not os.path.exists(fin_path):
        print(f"⚠️ Missing data for {ticker}. Skipping.")
        return pd.DataFrame()

    price_df = pd.read_csv(price_path)
    fin_df = pd.read_csv(fin_path)

    # Convert 'Date' column to datetime safely
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df['Year'] = price_df['Date'].dt.year

    # Ensure financials have a 'Year' column
    if 'Year' not in fin_df.columns:
        fin_df.reset_index(inplace=True)
        fin_df.rename(columns={'index': 'Year'}, inplace=True)

    # Merge by year
    merged = pd.merge(price_df, fin_df, on='Year', how='left')
    merged['Ticker'] = ticker
    return merged


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values for financial columns using forward/backward fill.
    
    Args:
        df (pd.DataFrame): Input dataframe with potential missing values
    
    Returns:
        pd.DataFrame: Cleaned dataframe with filled values
    """
    for col in ['EPS', 'Net Income', 'Total Revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
    return df.dropna(subset=['Close'])


# ============================
# Feature Engineering - Basic
# ============================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates basic return and volatility features using Close prices.
    
    Args:
        df (pd.DataFrame): Input dataframe with price data
    
    Returns:
        pd.DataFrame: Dataframe with basic features added
    """
    df = df.sort_values('Date')

    # Ensure numeric columns
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Remove rows where Close is missing
    df = df.dropna(subset=['Close'])

    # Compute daily return and rolling volatility
    df['Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Return'].rolling(window=30, min_periods=5).std()

    return df


# ============================
# Feature Engineering - Advanced
# ============================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    
    Args:
        series (pd.Series): Price series (typically Close prices)
        period (int): Lookback period for RSI calculation (default: 14)
    
    Returns:
        pd.Series: RSI values (0-100 scale)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates advanced financial and technical features.
    Includes: financial ratios, technical indicators, volatility metrics, and macro derivatives.
    
    Args:
        df (pd.DataFrame): Input dataframe with basic features
    
    Returns:
        pd.DataFrame: Dataframe with advanced features added
    """
    df = df.sort_values('Date').copy()
    
    # ==================
    # 1. FINANCIAL RATIOS
    # ==================
    
    # Price-to-Earnings Ratio
    if 'EPS' in df.columns and 'Close' in df.columns:
        df['PE_Ratio'] = df['Close'] / (df['EPS'] + 1e-6)  # Avoid division by zero
        df['PE_Ratio'] = df['PE_Ratio'].replace([np.inf, -np.inf], np.nan).ffill()
    
    # Revenue Growth (Year-over-Year approximation)
    if 'Total Revenue' in df.columns:
        df['Revenue_Growth_YoY'] = df['Total Revenue'].pct_change(4)  # ~1 month lag
        df['Revenue_Momentum'] = df['Total Revenue'].pct_change(12)   # ~3 months lag
    
    # EPS Growth
    if 'EPS' in df.columns:
        df['EPS_Growth'] = df['EPS'].pct_change(4)
    
    # Profit Margin
    if 'Net Income' in df.columns and 'Total Revenue' in df.columns:
        df['Profit_Margin'] = df['Net Income'] / (df['Total Revenue'] + 1e-6)
        df['Profit_Margin'] = df['Profit_Margin'].replace([np.inf, -np.inf], np.nan).ffill()
    
    # ==================
    # 2. TECHNICAL INDICATORS
    # ==================
    
    if 'Close' in df.columns:
        # Moving Averages
        df['MA_20'] = df['Close'].rolling(window=20, min_periods=5).mean()
        df['MA_50'] = df['Close'].rolling(window=50, min_periods=10).mean()
        
        # Price distance from moving averages (normalized)
        df['Price_to_MA20'] = (df['Close'] - df['MA_20']) / (df['MA_20'] + 1e-6)
        df['Price_to_MA50'] = (df['Close'] - df['MA_50']) / (df['MA_50'] + 1e-6)
        
        # Relative Strength Index
        df['RSI'] = calculate_rsi(df['Close'], period=14)
        
        # Bollinger Bands
        rolling_std = df['Close'].rolling(window=20, min_periods=5).std()
        df['BB_Upper'] = df['MA_20'] + (2 * rolling_std)
        df['BB_Lower'] = df['MA_20'] - (2 * rolling_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-6)
        df['BB_Position'] = df['BB_Position'].replace([np.inf, -np.inf], np.nan).ffill()
    
    # ==================
    # 3. VOLATILITY METRICS
    # ==================
    
    if 'Return' in df.columns:
        # Short-term and long-term volatility
        df['Vol_20'] = df['Return'].rolling(window=20, min_periods=5).std()
        df['Vol_50'] = df['Return'].rolling(window=50, min_periods=10).std()
        df['Vol_Ratio'] = df['Vol_20'] / (df['Vol_50'] + 1e-6)
        
        # Return distribution characteristics
        df['Return_Skew'] = df['Return'].rolling(window=20, min_periods=5).skew()
        df['Return_Kurt'] = df['Return'].rolling(window=20, min_periods=5).kurt()
    
    # ==================
    # 4. MACRO-DERIVED FEATURES
    # ==================
    
    # Interest rate changes
    if 'US10Y_Yield' in df.columns:
        df['Yield_Change'] = df['US10Y_Yield'].diff()
        df['Yield_MA'] = df['US10Y_Yield'].rolling(window=12, min_periods=3).mean()
    
    # Inflation rate (YoY CPI change)
    if 'US_CPI' in df.columns:
        df['Inflation_Rate'] = df['US_CPI'].pct_change(12)  # Year-over-year inflation
    
    # ==================
    # 5. VOLUME FEATURES
    # ==================
    
    if 'Volume' in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=5).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-6)
    
    print(f"✅ Advanced features created. New shape: {df.shape}")
    return df


# ============================
# Macroeconomic Data Integration
# ============================

def merge_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges macroeconomic data (US yields and inflation) with company data.
    Uses backward fill to align macro indicators with stock data.
    
    Args:
        df (pd.DataFrame): Company dataframe with Date column
    
    Returns:
        pd.DataFrame: Dataframe with macro indicators merged
    """
    macro_path = os.path.join(DATA_DIR, "macro_data.csv")
    if not os.path.exists(macro_path):
        print("⚠️ No macro data file found.")
        return df

    macro_df = pd.read_csv(macro_path)
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
    macro_df = macro_df.dropna(subset=['Date']).sort_values('Date')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')

    merged = pd.merge_asof(df, macro_df, on='Date', direction='backward')
    print(f"✅ Macro data merged ({len(merged)} rows).")
    return merged


# ============================
# Temporal Aggregation
# ============================

def convert_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates all data to weekly frequency (Friday close).
    Uses appropriate aggregation methods for different feature types.
    
    Args:
        df (pd.DataFrame): Daily dataframe with Date index
    
    Returns:
        pd.DataFrame: Weekly aggregated dataframe
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')

    # Define aggregation rules for each column type
    agg_dict = {
        'Close': 'last',
        'Volume': 'sum',
        'Return': 'mean',
        'Volatility_30d': 'last',
        'US10Y_Yield': 'last',
        'US_CPI': 'last',
        'EPS': 'last',
        'Net Income': 'last',
        'Total Revenue': 'last',
        # Advanced features
        'PE_Ratio': 'last',
        'Revenue_Growth_YoY': 'last',
        'Revenue_Momentum': 'last',
        'EPS_Growth': 'last',
        'Profit_Margin': 'last',
        'MA_20': 'last',
        'MA_50': 'last',
        'Price_to_MA20': 'last',
        'Price_to_MA50': 'last',
        'RSI': 'last',
        'BB_Upper': 'last',
        'BB_Lower': 'last',
        'BB_Position': 'last',
        'Vol_20': 'last',
        'Vol_50': 'last',
        'Vol_Ratio': 'last',
        'Return_Skew': 'last',
        'Return_Kurt': 'last',
        'Yield_Change': 'last',
        'Yield_MA': 'last',
        'Inflation_Rate': 'last',
        'Volume_MA': 'last',
        'Volume_Ratio': 'last'
    }
    
    # Only aggregate columns that exist in the dataframe
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}

    weekly = df.resample('W-FRI').agg(agg_dict_filtered).reset_index()

    # Compute weekly return from weekly close prices
    weekly['Weekly_Return'] = weekly['Close'].pct_change()
    weekly = weekly.dropna(subset=['Close'])

    print(f"📅 Converted to weekly ({len(weekly)} rows, ending {weekly['Date'].max().date()})")
    return weekly


# ============================
# Main Preprocessing Pipeline
# ============================

def preprocess_all(tickers):
    """
    Main preprocessing pipeline that processes all tickers.
    Steps:
    1. Load company data (prices + financials)
    2. Handle missing values
    3. Create basic features (returns, volatility)
    4. Create advanced features (ratios, indicators)
    5. Merge macroeconomic data
    6. Convert to weekly frequency
    7. Combine all tickers into single dataset
    
    Args:
        tickers (list): List of ticker symbols to process
    """
    all_data = []

    for ticker in tickers:
        print(f"\n⚙️ Preprocessing {ticker}...")
        df = load_company_data(ticker)
        if df.empty:
            continue

        # Step-by-step feature engineering
        df = handle_missing_values(df)
        df = create_features(df)                    # Basic features
        df = create_advanced_features(df)           # Advanced features
        df = merge_macro_data(df)                   # Add macro indicators
        df = convert_to_weekly(df)                  # Weekly aggregation
        df['Ticker'] = ticker

        # Select final columns for modeling
        keep_cols = [
            'Date', 'Close', 'Volume', 'Return', 'Volatility_30d',
            'US10Y_Yield', 'US_CPI', 'EPS', 'Net Income', 'Total Revenue',
            'Weekly_Return', 'Ticker',
            # Advanced features
            'PE_Ratio', 'Revenue_Growth_YoY', 'Revenue_Momentum',
            'EPS_Growth', 'Profit_Margin',
            'MA_20', 'MA_50', 'Price_to_MA20', 'Price_to_MA50',
            'RSI', 'BB_Position',
            'Vol_20', 'Vol_50', 'Vol_Ratio',
            'Return_Skew', 'Return_Kurt',
            'Yield_Change', 'Yield_MA', 'Inflation_Rate',
            'Volume_MA', 'Volume_Ratio'
        ]
        
        # Keep only columns that exist
        df = df[[c for c in keep_cols if c in df.columns]]
        all_data.append(df)

    if not all_data:
        print("❌ No valid data found.")
        return

    # Combine all tickers
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(os.path.join(PROCESSED_DIR, "combined_data.csv"), index=False)
    print(f"\n✅ All tickers processed with advanced features.")
    print(f"📊 Final dataset shape: {combined.shape}")
    print(f"📋 Features included: {list(combined.columns)}")


# ============================
# Execution
# ============================

if __name__ == "__main__":
    print("\n🚀 Starting enhanced preprocessing pipeline...\n")
    print("=" * 60)
    print("PREPROCESSING STEPS:")
    print("1. Load price and financial data")
    print("2. Handle missing values")
    print("3. Create basic features (returns, volatility)")
    print("4. Create advanced features (ratios, indicators)")
    print("5. Merge macroeconomic indicators")
    print("6. Convert to weekly frequency")
    print("7. Combine all tickers")
    print("=" * 60)
    
    preprocess_all(list(TICKERS.values()))
    
    print("\n" + "=" * 60)
    print("🎯 PREPROCESSING COMPLETE!")
    print(f"📁 Output saved: {os.path.join(PROCESSED_DIR, 'combined_data.csv')}")
    print("=" * 60 + "\n")