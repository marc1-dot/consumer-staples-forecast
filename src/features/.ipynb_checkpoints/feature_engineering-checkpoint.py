"""
feature_engineering.py
----------------------
Feature engineering module for the Consumer Staples Forecasting project.
Adds lagged features and derived metrics to enhance predictive performance.
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance dataset with lagged and derived features."""
    print("✅ Starting feature engineering (lags)...")

    # Ensure data is sorted by date
    df = df.sort_values("Date").reset_index(drop=True)

    # === Base numerical features ===
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # === 1️⃣ Lag Features ===
    for lag in [1, 2, 3]:
        for col in ["Return", "Volatility_30d", "Volume"]:
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # === 2️⃣ Rolling Statistics ===
    if "Return" in df.columns:
        df["Return_rolling_mean_4"] = df["Return"].rolling(window=4).mean()
        df["Return_rolling_std_4"] = df["Return"].rolling(window=4).std()

    # === Drop early NaNs introduced by shifting ===
    df = df.dropna().reset_index(drop=True)

    print(f"✅ Feature engineering complete: {df.shape[1]} columns.")
    return df
