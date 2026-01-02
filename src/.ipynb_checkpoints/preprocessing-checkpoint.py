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

# ============================
# Configuration
# ============================

# ✅ 5 tickers uniquement
TICKERS = {
    "NESN.SW": "Nestlé",
    "PG": "Procter & Gamble",
    "UL": "Unilever",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo"
}

# ✅ CHEMINS ABSOLUS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================
# Helper Functions - Data Loading
# ============================

def load_raw_stock_data() -> pd.DataFrame:
    """
    ✅ Charge les 5 fichiers CSV individuels et les combine.
    """
    print(f"\n🔍 DEBUG - CHEMINS:")
    print(f"   BASE_DIR = {BASE_DIR}")
    print(f"   DATA_DIR = {DATA_DIR}")
    print(f"   RAW_DIR  = {RAW_DIR}")
    print()

    all_ticker_data = []

    for ticker in TICKERS.keys():
        ticker_file = os.path.join(RAW_DIR, f"{ticker}_data.csv")

        print(f"   📂 Checking {ticker}_data.csv...", end=" ")

        if not os.path.exists(ticker_file):
            print("❌ NOT FOUND")
            print(f"      Expected path: {ticker_file}")
            continue

        df = pd.read_csv(ticker_file)

        # Ensure Ticker column exists
        if "Ticker" not in df.columns:
            df["Ticker"] = ticker

        print(f"✅ {len(df):,} rows")
        all_ticker_data.append(df)

    if not all_ticker_data:
        raise FileNotFoundError(
            f"❌ No ticker data files found in: {RAW_DIR}\n"
            f"   Expected files: NESN.SW_data.csv, PG_data.csv, UL_data.csv, KO_data.csv, PEP_data.csv\n"
            f"   Please run 'python src/data_loader.py' first!"
        )

    combined_df = pd.concat(all_ticker_data, ignore_index=True)

    # ✅ DEBUG : Affiche les données chargées (robuste aux NaN dans Ticker)
    print(f"\n🔍 DEBUG - DONNÉES COMBINÉES:")
    print(f"   Total lignes: {len(combined_df):,}")
    print(f"   Colonnes: {list(combined_df.columns)}")
    if "Ticker" in combined_df.columns:
        tickers_unique = combined_df["Ticker"].dropna().astype(str).unique()
        print(f"   Tickers uniques: {sorted(tickers_unique)}")
        print(f"   Nombre de tickers: {len(tickers_unique)}")
        print(f"\n   📊 Répartition par ticker:")
        for t, count in combined_df["Ticker"].dropna().astype(str).value_counts().sort_index().items():
            print(f"      {t}: {count:,} lignes")
    print()

    combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
    combined_df = combined_df.dropna(subset=["Date", "Close"])

    print(f"✅ Loaded raw data: {len(combined_df):,} rows, {combined_df['Ticker'].nunique()} tickers")
    print(f"   Date range: {combined_df['Date'].min().date()} to {combined_df['Date'].max().date()}")

    return combined_df


def load_company_financials(ticker: str) -> pd.DataFrame:
    fin_path = os.path.join(DATA_DIR, f"{ticker}_financials.csv")

    if not os.path.exists(fin_path):
        print(f"   ⚠️  No financials found for {ticker}, using price data only")
        return pd.DataFrame()

    fin_df = pd.read_csv(fin_path)

    if "Year" not in fin_df.columns:
        fin_df.reset_index(inplace=True)
        fin_df.rename(columns={"index": "Year"}, inplace=True)

    return fin_df


def merge_financials_with_prices(price_df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
    if fin_df.empty:
        return price_df

    price_df = price_df.copy()
    price_df["Year"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.year

    merged = pd.merge(price_df, fin_df, on="Year", how="left")

    for col in ["EPS", "Net Income", "Total Revenue"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill().bfill()

    return merged


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["EPS", "Net Income", "Total Revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()

    return df.dropna(subset=["Close"])


# ============================
# Feature Engineering - Basic
# ============================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=["Close"])

    df["Return"] = df["Close"].pct_change()
    df["Volatility_30d"] = df["Return"].rolling(window=30, min_periods=5).std()

    return df


# ============================
# Feature Engineering - Advanced
# ============================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features financières, techniques, volatilité, et macro-dérivées.
    """
    df = df.sort_values("Date").copy()

    # ==================
    # 1. FINANCIAL RATIOS
    # ==================
    if "EPS" in df.columns and "Close" in df.columns:
        df["PE_Ratio"] = df["Close"] / (df["EPS"] + 1e-6)
        df["PE_Ratio"] = df["PE_Ratio"].replace([np.inf, -np.inf], np.nan).ffill()

    if "Total Revenue" in df.columns:
        df["Revenue_Growth_YoY"] = df["Total Revenue"].pct_change(4)
        df["Revenue_Momentum"] = df["Total Revenue"].pct_change(12)

    if "EPS" in df.columns:
        df["EPS_Growth"] = df["EPS"].pct_change(4)

    if "Net Income" in df.columns and "Total Revenue" in df.columns:
        df["Profit_Margin"] = df["Net Income"] / (df["Total Revenue"] + 1e-6)
        df["Profit_Margin"] = df["Profit_Margin"].replace([np.inf, -np.inf], np.nan).ffill()

    # ==================
    # 2. TECHNICAL INDICATORS
    # ==================
    if "Close" in df.columns:
        df["MA_20"] = df["Close"].rolling(window=20, min_periods=5).mean()
        df["MA_50"] = df["Close"].rolling(window=50, min_periods=10).mean()

        df["Price_to_MA20"] = (df["Close"] - df["MA_20"]) / (df["MA_20"] + 1e-6)
        df["Price_to_MA50"] = (df["Close"] - df["MA_50"]) / (df["MA_50"] + 1e-6)

        df["RSI"] = calculate_rsi(df["Close"], period=14)

        rolling_std = df["Close"].rolling(window=20, min_periods=5).std()
        df["BB_Upper"] = df["MA_20"] + (2 * rolling_std)
        df["BB_Lower"] = df["MA_20"] - (2 * rolling_std)
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-6)
        df["BB_Position"] = df["BB_Position"].replace([np.inf, -np.inf], np.nan).ffill()

    # ==================
    # 3. VOLATILITY METRICS
    # ==================
    if "Return" in df.columns:
        df["Vol_20"] = df["Return"].rolling(window=20, min_periods=5).std()
        df["Vol_50"] = df["Return"].rolling(window=50, min_periods=10).std()
        df["Vol_Ratio"] = df["Vol_20"] / (df["Vol_50"] + 1e-6)

        df["Return_Skew"] = df["Return"].rolling(window=20, min_periods=5).skew()
        df["Return_Kurt"] = df["Return"].rolling(window=20, min_periods=5).kurt()

    # ==================
    # 4. MACRO-DERIVED FEATURES
    # ==================

    # Interest rate changes
    if "US_10Y_Yield" in df.columns:
        df["Yield_10Y_Change"] = df["US_10Y_Yield"].diff()
        df["Yield_10Y_MA"] = df["US_10Y_Yield"].rolling(window=12, min_periods=3).mean()

    if "US_5Y_Yield" in df.columns:
        df["Yield_5Y_Change"] = df["US_5Y_Yield"].diff()
        df["Yield_5Y_MA"] = df["US_5Y_Yield"].rolling(window=12, min_periods=3).mean()

    if "US_3M_TBill" in df.columns:
        df["TBill_3M_Change"] = df["US_3M_TBill"].diff()
        df["TBill_3M_MA"] = df["US_3M_TBill"].rolling(window=12, min_periods=3).mean()

    # Yield curves
    if "US_10Y_Yield" in df.columns and "US_3M_TBill" in df.columns:
        df["Yield_Curve_10Y_3M"] = df["US_10Y_Yield"] - df["US_3M_TBill"]

    if "US_10Y_Yield" in df.columns and "US_5Y_Yield" in df.columns:
        df["Yield_Curve_10Y_5Y"] = df["US_10Y_Yield"] - df["US_5Y_Yield"]

    # Inflation rate (YoY CPI change)
    if "US_CPI" in df.columns:
        df["Inflation_Rate"] = df["US_CPI"].pct_change(12)
        df["Inflation_Rate_QoQ"] = df["US_CPI"].pct_change(3)

    # USD
    if "USD_Index" in df.columns:
        df["USD_Change"] = df["USD_Index"].pct_change()
        df["USD_MA"] = df["USD_Index"].rolling(window=12, min_periods=3).mean()
        df["USD_Volatility"] = df["USD_Change"].rolling(window=12, min_periods=3).std()

    # Oil
    if "Crude_Oil" in df.columns:
        df["Oil_Change"] = df["Crude_Oil"].pct_change()
        df["Oil_MA"] = df["Crude_Oil"].rolling(window=12, min_periods=3).mean()
        df["Oil_Volatility"] = df["Oil_Change"].rolling(window=12, min_periods=3).std()

    # Gold
    if "Gold" in df.columns:
        df["Gold_Change"] = df["Gold"].pct_change()
        df["Gold_MA"] = df["Gold"].rolling(window=12, min_periods=3).mean()
        df["Gold_Volatility"] = df["Gold_Change"].rolling(window=12, min_periods=3).std()

    # ==================
    # 5. VOLUME FEATURES
    # ==================
    if "Volume" in df.columns:
        df["Volume_MA"] = df["Volume"].rolling(window=20, min_periods=5).mean()
        df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-6)

    print(f"   ✅ Advanced features created. Shape: {df.shape}")
    return df


# ============================
# Macroeconomic Data Integration
# ============================

def merge_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro data (downloaded by data_loader.py) into daily ticker data.
    Expects data/raw/macro_data.csv
    """
    macro_path = os.path.join(RAW_DIR, "macro_data.csv")

    if not os.path.exists(macro_path):
        print("   ⚠️  No macro data file found, skipping macro features")
        return df

    macro_df = pd.read_csv(macro_path)
    macro_df["Date"] = pd.to_datetime(macro_df["Date"], errors="coerce")
    macro_df = macro_df.dropna(subset=["Date"]).sort_values("Date")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    merged = pd.merge_asof(df, macro_df, on="Date", direction="backward")
    print(f"   ✅ Macro data merged ({len(merged)} rows)")
    return merged


# ============================
# Temporal Aggregation
# ============================

def convert_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates all data to weekly frequency (Friday close).
    Uses appropriate aggregation methods for different feature types.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

    agg_dict = {
        "Close": "last",
        "Volume": "sum",
        "Return": "mean",
        "Weekly_Return": "last",  # (si existe déjà, sinon ignoré)
        "Volatility_30d": "last",

        # ✅ Macro variables (raw)
        "US_3M_TBill": "last",
        "US_10Y_Yield": "last",
        "US_5Y_Yield": "last",
        "USD_Index": "last",
        "Crude_Oil": "last",
        "Gold": "last",
        "US_CPI": "last",
        "US_CPI_Proxy": "last",

        # Financial ratios
        "EPS": "last",
        "Net Income": "last",
        "Total Revenue": "last",
        "PE_Ratio": "last",
        "Revenue_Growth_YoY": "last",
        "Revenue_Momentum": "last",
        "EPS_Growth": "last",
        "Profit_Margin": "last",

        # Technical indicators
        "MA_20": "last",
        "MA_50": "last",
        "Price_to_MA20": "last",
        "Price_to_MA50": "last",
        "RSI": "last",
        "BB_Upper": "last",
        "BB_Lower": "last",
        "BB_Position": "last",

        # Volatility
        "Vol_20": "last",
        "Vol_50": "last",
        "Vol_Ratio": "last",
        "Return_Skew": "last",
        "Return_Kurt": "last",

        # ✅ Macro-derived features
        "Yield_10Y_Change": "last",
        "Yield_10Y_MA": "last",
        "Yield_5Y_Change": "last",
        "Yield_5Y_MA": "last",
        "TBill_3M_Change": "last",
        "TBill_3M_MA": "last",
        "Yield_Curve_10Y_3M": "last",
        "Yield_Curve_10Y_5Y": "last",
        "Inflation_Rate": "last",
        "Inflation_Rate_QoQ": "last",
        "USD_Change": "last",
        "USD_MA": "last",
        "USD_Volatility": "last",
        "Oil_Change": "last",
        "Oil_MA": "last",
        "Oil_Volatility": "last",
        "Gold_Change": "last",
        "Gold_MA": "last",
        "Gold_Volatility": "last",

        # Volume
        "Volume_MA": "last",
        "Volume_Ratio": "last",
    }

    # Only keep columns that exist in the dataframe
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}

    weekly = df.resample("W-FRI").agg(agg_dict_filtered).reset_index()

    # Compute weekly return from weekly close prices
    if "Close" in weekly.columns:
        weekly["Weekly_Return"] = weekly["Close"].pct_change()
        weekly = weekly.dropna(subset=["Close"])

    print(f"   📅 Converted to weekly: {len(weekly)} rows (ending {weekly['Date'].max().date()})")
    return weekly


# ============================
# Main Preprocessing Pipeline
# ============================

def preprocess_all():
    print("\n🚀 Starting enhanced preprocessing pipeline...\n")
    print("=" * 80)
    print("PREPROCESSING STEPS:")
    print("1. Load raw stock data from individual CSV files")
    print("2. Load macroeconomic data (7 indicators)")
    print("3. Process each ticker:")
    print("   - Load financials (if available)")
    print("   - Handle missing values")
    print("   - Create basic features (returns, volatility)")
    print("   - Merge macroeconomic indicators")
    print("   - Create advanced features (incl. macro-derived)")
    print("   - Convert to weekly frequency")
    print("4. Combine all tickers")
    print("=" * 80 + "\n")

    raw_df = load_raw_stock_data()
    all_data = []

    for ticker in TICKERS.keys():
        print(f"\n⚙️  Processing {ticker} ({TICKERS[ticker]})...")

        ticker_df = raw_df[raw_df["Ticker"] == ticker].copy()

        if ticker_df.empty:
            print(f"   ❌ No data found for {ticker}, skipping")
            continue

        print(f"   📊 Loaded {len(ticker_df)} daily rows")

        fin_df = load_company_financials(ticker)
        if not fin_df.empty:
            ticker_df = merge_financials_with_prices(ticker_df, fin_df)
            print("   ✅ Financials merged")

        ticker_df = handle_missing_values(ticker_df)
        ticker_df = create_features(ticker_df)
        ticker_df = merge_macro_data(ticker_df)          # ✅ merge macro AVANT macro-derived
        ticker_df = create_advanced_features(ticker_df)  # ✅ macro-derived
        ticker_df = convert_to_weekly(ticker_df)
        ticker_df["Ticker"] = ticker

        # ✅ Keep all available columns
        keep_cols = [
            "Date", "Ticker", "Close", "Volume", "Return", "Weekly_Return", "Volatility_30d",

            # Macro (raw)
            "US_3M_TBill", "US_10Y_Yield", "US_5Y_Yield", "USD_Index", "Crude_Oil", "Gold", "US_CPI",

            # Financial
            "EPS", "Net Income", "Total Revenue", "PE_Ratio", "Revenue_Growth_YoY",
            "Revenue_Momentum", "EPS_Growth", "Profit_Margin",

            # Technical
            "MA_20", "MA_50", "Price_to_MA20", "Price_to_MA50", "RSI", "BB_Position",

            # Volatility
            "Vol_20", "Vol_50", "Vol_Ratio", "Return_Skew", "Return_Kurt",

            # Macro-derived
            "Yield_10Y_Change", "Yield_10Y_MA", "Yield_5Y_Change", "Yield_5Y_MA",
            "TBill_3M_Change", "TBill_3M_MA", "Yield_Curve_10Y_3M", "Yield_Curve_10Y_5Y",
            "Inflation_Rate", "Inflation_Rate_QoQ",
            "USD_Change", "USD_MA", "USD_Volatility",
            "Oil_Change", "Oil_MA", "Oil_Volatility",
            "Gold_Change", "Gold_MA", "Gold_Volatility",

            # Volume
            "Volume_MA", "Volume_Ratio",
        ]

        ticker_df = ticker_df[[c for c in keep_cols if c in ticker_df.columns]]
        all_data.append(ticker_df)

        # ✅ FIX JS -> PYTHON
        print(f"   ✅ {ticker} complete: {len(ticker_df)} weekly rows, {ticker_df.shape[1]} features")

    if not all_data:
        print("\n❌ No valid data found for any ticker!")
        return

    combined = pd.concat(all_data, ignore_index=True)

    output_path = os.path.join(PROCESSED_DIR, "combined_data.csv")
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("🎯 PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("\n📊 FINAL DATASET STATISTICS:")
    print(f"   - Total rows: {len(combined):,}")

    # ✅ FIX JS -> PYTHON
    print(f"   - Total features: {combined.shape[1]}")

    tickers_unique = combined["Ticker"].dropna().astype(str).unique()
    print(f"   - Tickers: {len(tickers_unique)} ({', '.join(sorted(tickers_unique))})")

    print(
        f"   - Date range: {pd.to_datetime(combined['Date']).min().date()} "
        f"to {pd.to_datetime(combined['Date']).max().date()}"
    )
    print(f"   - Avg rows per ticker: {len(combined) / combined['Ticker'].nunique():.0f}")
    print(f"\n📁 Output saved to: {output_path}")

    # ✅ FIX JS -> PYTHON
    print(f"\n📋 Features included ({combined.shape[1]} total):")

    # Group features by category
    feature_groups = {
        "Basic": ["Date", "Ticker", "Close", "Volume", "Return", "Weekly_Return", "Volatility_30d"],
        "Macro (Raw)": ["US_3M_TBill", "US_10Y_Yield", "US_5Y_Yield", "USD_Index", "Crude_Oil", "Gold", "US_CPI"],
        "Financial": ["EPS", "Net Income", "Total Revenue", "PE_Ratio", "Revenue_Growth_YoY",
                      "Revenue_Momentum", "EPS_Growth", "Profit_Margin"],
        "Technical": ["MA_20", "MA_50", "Price_to_MA20", "Price_to_MA50", "RSI", "BB_Position"],
        "Volatility": ["Vol_20", "Vol_50", "Vol_Ratio", "Return_Skew", "Return_Kurt"],
        "Macro-Derived": ["Yield_10Y_Change", "Yield_10Y_MA", "Yield_5Y_Change", "Yield_5Y_MA",
                          "TBill_3M_Change", "TBill_3M_MA", "Yield_Curve_10Y_3M", "Yield_Curve_10Y_5Y",
                          "Inflation_Rate", "Inflation_Rate_QoQ", "USD_Change", "USD_MA", "USD_Volatility",
                          "Oil_Change", "Oil_MA", "Oil_Volatility", "Gold_Change", "Gold_MA", "Gold_Volatility"],
        "Volume": ["Volume_MA", "Volume_Ratio"],
    }

    for group_name, features in feature_groups.items():
        present_features = [f for f in features if f in combined.columns]
        if present_features:
            print(f"\n   {group_name} ({len(present_features)}):")
            for feat in present_features:
                print(f"      • {feat}")

    print("\n" + "=" * 80 + "\n")


# ============================
# Execution
# ============================

if __name__ == "__main__":
    preprocess_all()
