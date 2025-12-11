"""
feature_engineering.py
----------------------
Basic feature engineering module for Consumer Staples Forecast project.

This version is intentionally simple — it avoids overfitting and keeps
interpretable, fundamental features suitable for forecasting weekly returns.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic, interpretable features to the dataset.

    Features include:
    - Rolling volatility (4-week window)
    - Rolling average return (4-week window)
    - Volume change rate
    - Interaction terms between macro & financial indicators
    """

    print("✅ Starting feature engineering (basic)...")

    # Ensure correct sorting for rolling features
    df = df.sort_values(["Ticker", "Date"])

    # --- Rolling volatility (4 weeks)
    df["Volatility_4w"] = (
        df.groupby("Ticker")["Weekly_Return"]
        .rolling(window=4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    # --- Rolling average return (trend)
    df["Mean_Return_4w"] = (
        df.groupby("Ticker")["Weekly_Return"]
        .rolling(window=4, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Volume change rate
    if "Volume" in df.columns:
        df["Volume_Change"] = df.groupby("Ticker")["Volume"].pct_change()

    # --- Macro interactions (optional)
    if "US10Y_Yield" in df.columns and "US_CPI" in df.columns:
        df["Yield_CPI_Ratio"] = df["US10Y_Yield"] / (df["US_CPI"] + 1e-6)

    # --- Financial ratios (simple combinations)
    if all(col in df.columns for col in ["EPS", "Total Revenue", "Net Income"]):
        df["Profit_Margin"] = df["Net Income"] / df["Total Revenue"]
        df["Revenue_per_EPS"] = df["Total Revenue"] / (df["EPS"] + 1e-6)

    # Drop rows with missing core features
    df = df.dropna(subset=["Weekly_Return"])

    print(f"✅ Feature engineering complete: {df.shape[1]} columns.")
    return df
