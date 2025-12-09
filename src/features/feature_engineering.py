"""
feature_engineering.py
----------------------
Creates additional derived features to enhance predictive performance.

Includes:
- Lagged returns
- Moving averages and rolling volatility
- Macro deltas (Δ Inflation, Δ 10Y Yield)

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds time-based features such as lagged returns and rolling statistics."""
    df = df.copy()

    # --- Create lagged returns ---
    if "Return" in df.columns:
        for lag in [1, 5, 10]:
            df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    # --- Rolling volatility and moving averages ---
    if "Return" in df.columns:
        df["Rolling_Vol_7d"] = df["Return"].rolling(7).std()
        df["Rolling_Vol_30d"] = df["Return"].rolling(30).std()
        df["Rolling_Mean_7d"] = df["Return"].rolling(7).mean()

    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds macro change features (delta Inflation and Yield)."""
    df = df.copy()

    if "US10Y_Yield" in df.columns:
        df["Delta_Yield"] = df["US10Y_Yield"].diff()
        df["Delta_Yield_5d"] = df["US10Y_Yield"].diff(5)

    if "US_CPI" in df.columns:
        df["Delta_CPI"] = df["US_CPI"].diff()
        df["Delta_CPI_5d"] = df["US_CPI"].diff(5)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_macro_features(df)

    # Drop early NaNs (due to rolling/lags)
    df = df.dropna().reset_index(drop=True)

    print(f"✅ Feature engineering complete: {df.shape[1]} features")
    return df
