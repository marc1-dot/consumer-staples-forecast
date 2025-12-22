"""
data_loader.py
--------------
Downloads historical stock data for Consumer Staples sector and macroeconomic indicators.
Saves each ticker to a separate CSV file + macro data file.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import yfinance as yf
import pandas as pd
import os
import time

# ============================
# Configuration
# ============================

START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

# ✅ Chemin absolu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")

# ✅ 5 tickers principaux
TICKERS = [
    "NESN.SW",  # Nestlé (Swiss)
    "PG",       # Procter & Gamble
    "UL",       # Unilever
    "KO",       # Coca-Cola
    "PEP"       # PepsiCo
]

# ✅ Indicateurs macroéconomiques
MACRO_INDICATORS = {
    "^IRX": "US_3M_TBill",      # 3-Month Treasury Bill
    "^TNX": "US_10Y_Yield",     # 10-Year Treasury Yield
    "^FVX": "US_5Y_Yield",      # 5-Year Treasury Yield
    "DX-Y.NYB": "USD_Index",    # US Dollar Index
    "CL=F": "Crude_Oil",        # Crude Oil Futures
    "GC=F": "Gold"              # Gold Futures
}

# ✅ CPI Data (FRED API alternative - using manual data or yfinance proxy)
# Note: CPI requires FRED API or manual download. We'll use a proxy approach.


# ============================
# Download Functions - Stock Data
# ============================

def download_single_ticker(ticker: str, start_date: str, end_date: str, output_dir: str) -> pd.DataFrame:
    """
    Downloads data for a single ticker and saves to individual CSV file.
    """
    try:
        print(f"   Downloading {ticker}...", end=" ", flush=True)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data.empty:
            print("❌ No data returned")
            return pd.DataFrame()

        # Add ticker column
        data["Ticker"] = ticker
        data.reset_index(inplace=True)

        # Save to individual CSV
        output_path = os.path.join(output_dir, f"{ticker}_data.csv")
        data.to_csv(output_path, index=False)

        print(f"✅ {len(data)} rows ({data['Date'].min().date()} to {data['Date'].max().date()})")
        print(f"      Saved to: {ticker}_data.csv")

        return data

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return pd.DataFrame()


def download_all_tickers(tickers: list, start_date: str, end_date: str, output_dir: str):
    """
    Downloads data for all tickers and saves each to separate CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DOWNLOADING CONSUMER STAPLES DATA ({start_date} to {end_date})")
    print(f"{'='*80}")
    print(f"\n📋 Configuration:")
    print(f"   Number of tickers: {len(tickers)}")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Output directory: {output_dir}")

    # Calculate expected data
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years = (end_dt - start_dt).days / 365.25
    expected_days = int(years * 252)

    print(f"   Expected years: {years:.1f}")
    print(f"   Expected trading days per ticker: ~{expected_days}")
    print(f"\n{'='*80}\n")

    all_data = []
    success_count = 0
    failed_tickers = []

    # Download each ticker
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}:")

        data = download_single_ticker(ticker, start_date, end_date, output_dir)

        if not data.empty:
            all_data.append(data)
            success_count += 1
        else:
            failed_tickers.append(ticker)

        # Small delay between downloads
        if i < len(tickers):
            time.sleep(1)

        print()

    # Calculate statistics
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        min_date = combined_df["Date"].min()
        max_date = combined_df["Date"].max()
        actual_years = (max_date - min_date).days / 365.25

        print(f"{'='*80}")
        print("STOCK DATA DOWNLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successfully downloaded: {success_count}/{len(tickers)} tickers")

        if failed_tickers:
            print(f"   ⚠️  Failed tickers: {', '.join(failed_tickers)}")

        print(f"\n📈 DATA STATISTICS:")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Date range: {min_date.date()} to {max_date.date()}")
        print(f"   Actual years: {actual_years:.1f}")
        print(f"   Average rows per ticker: {len(combined_df) / success_count:.0f}")

        print(f"\n📁 FILES CREATED:")
        for ticker in tickers:
            file_path = os.path.join(output_dir, f"{ticker}_data.csv")
            if os.path.exists(file_path):
                ticker_data = combined_df[combined_df["Ticker"] == ticker]
                print(f"      ✅ {ticker}_data.csv ({len(ticker_data):,} rows)")
            else:
                print(f"      ❌ {ticker}_data.csv (not created)")

        print(f"\n📂 ABSOLUTE PATH:")
        print(f"   {os.path.abspath(output_dir)}")
        print(f"{'='*80}\n")

        return combined_df

    print(f"\n{'='*80}")
    print("❌ STOCK DATA DOWNLOAD FAILED")
    print(f"{'='*80}")
    print("No data was downloaded successfully!")
    print(f"Failed tickers: {', '.join(failed_tickers)}")
    print(f"{'='*80}\n")
    raise ValueError("No stock data was downloaded successfully!")


# ============================
# Download Functions - Macro Data
# ============================

def download_macro_indicator(symbol: str, name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads a single macroeconomic indicator from Yahoo Finance.
    """
    try:
        print(f"   Downloading {name} ({symbol})...", end=" ", flush=True)

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data.empty:
            print("❌ No data returned")
            return pd.DataFrame()

        # Keep only Close price and rename
        data = data[["Close"]].copy()
        data.reset_index(inplace=True)
        data.columns = ["Date", name]

        print(f"✅ {len(data)} rows")
        return data

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return pd.DataFrame()


def download_cpi_data(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        print("   Downloading US CPI data...", end=" ", flush=True)

        data = yf.download(
            "TIP",  # iShares TIPS Bond ETF (proxy for inflation)
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data.empty:
            print("⚠️  No CPI proxy data available")
            return pd.DataFrame()

        data = data[["Close"]].copy()
        data.reset_index(inplace=True)
        data.columns = ["Date", "US_CPI_Proxy"]

        # Normalize to CPI-like scale (base 100 in 2010)
        base_value = data.loc[data["Date"].dt.year == 2010, "US_CPI_Proxy"].mean()
        if pd.notna(base_value) and base_value > 0:
            data["US_CPI"] = (data["US_CPI_Proxy"] / base_value) * 100
        else:
            data["US_CPI"] = data["US_CPI_Proxy"]

        data = data[["Date", "US_CPI"]]

        print(f"✅ {len(data)} rows (using TIP ETF as proxy)")
        return data

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return pd.DataFrame()


def download_all_macro_data(start_date: str, end_date: str, output_dir: str):
    """
    Downloads all macroeconomic indicators and saves to a single CSV file.
    """
    print(f"\n{'='*80}")
    print(f"DOWNLOADING MACROECONOMIC DATA ({start_date} to {end_date})")
    print(f"{'='*80}\n")

    all_macro_data = []

    # Download each macro indicator
    for symbol, name in MACRO_INDICATORS.items():
        macro_df = download_macro_indicator(symbol, name, start_date, end_date)

        if not macro_df.empty:
            all_macro_data.append(macro_df)

        time.sleep(1)  # Delay between downloads

    # Download CPI data
    cpi_df = download_cpi_data(start_date, end_date)
    if not cpi_df.empty:
        all_macro_data.append(cpi_df)

    print()

    # Merge all macro data
    if not all_macro_data:
        print("❌ No macro data downloaded!")
        return None

    # ✅ FIX (JS -> Python): start from first DF
    macro_combined = all_macro_data[0]
    for df in all_macro_data[1:]:
        macro_combined = pd.merge(macro_combined, df, on="Date", how="outer")

    # Sort by date and forward fill missing values
    macro_combined = macro_combined.sort_values("Date").reset_index(drop=True)
    macro_combined = macro_combined.ffill().bfill()

    # Save to CSV
    output_path = os.path.join(output_dir, "macro_data.csv")
    macro_combined.to_csv(output_path, index=False)

    print(f"{'='*80}")
    print("MACRO DATA DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 SUMMARY:")
    print(f"   Total rows: {len(macro_combined):,}")
    print(f"   Date range: {macro_combined['Date'].min()} to {macro_combined['Date'].max()}")
    print(f"   Indicators: {len(macro_combined.columns) - 1}")
    print(f"\n📋 INDICATORS INCLUDED:")
    for col in macro_combined.columns:
        if col != "Date":
            non_null = macro_combined[col].notna().sum()
            print(f"      ✅ {col} ({non_null:,} non-null values)")

    print(f"\n📁 FILE CREATED:")
    print(f"      ✅ macro_data.csv ({len(macro_combined):,} rows)")
    print(f"\n📂 ABSOLUTE PATH:")
    print(f"   {output_path}")
    print(f"{'='*80}\n")

    return macro_combined


# ============================
# Verification Function
# ============================

def verify_downloads(output_dir: str, tickers: list):
    """
    Verifies that all ticker files and macro data were created correctly.
    """
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}\n")

    all_ok = True
    total_rows = 0

    # Check stock data files
    print("📊 STOCK DATA FILES:\n")
    for ticker in tickers:
        file_path = os.path.join(output_dir, f"{ticker}_data.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])

            print(f"✅ {ticker}_data.csv")
            print(f"   Rows: {len(df):,}")
            print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            print(f"   Columns: {list(df.columns)}")
            print()

            total_rows += len(df)
        else:
            print(f"❌ {ticker}_data.csv - FILE NOT FOUND")
            print()
            all_ok = False

    # Check macro data file
    print("📊 MACRO DATA FILE:\n")
    macro_path = os.path.join(output_dir, "macro_data.csv")
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path)
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])

        print("✅ macro_data.csv")
        print(f"   Rows: {len(macro_df):,}")
        print(f"   Date range: {macro_df['Date'].min().date()} to {macro_df['Date'].max().date()}")
        print(f"   Columns: {list(macro_df.columns)}")
        print()
    else:
        print("⚠️  macro_data.csv - FILE NOT FOUND")
        print()
        all_ok = False

    print("📊 TOTAL STATISTICS:")
    print(f"   Stock files created: {sum(1 for t in tickers if os.path.exists(os.path.join(output_dir, f'{t}_data.csv')))}/{len(tickers)}")
    print(f"   Total stock rows: {total_rows:,}")
    print(f"   Macro file created: {'✅ Yes' if os.path.exists(macro_path) else '❌ No'}")

    print(f"\n{'='*80}")
    if all_ok:
        print("✅ ALL FILES VERIFIED SUCCESSFULLY")
    else:
        print("⚠️  SOME FILES ARE MISSING")
    print(f"{'='*80}\n")


# ============================
# Main Execution
# ============================

if __name__ == "__main__":
    # Download stock data
    download_all_tickers(TICKERS, START_DATE, END_DATE, OUTPUT_DIR)

    # Download macro data
    download_all_macro_data(START_DATE, END_DATE, OUTPUT_DIR)

    # Verify all downloads
    verify_downloads(OUTPUT_DIR, TICKERS)
