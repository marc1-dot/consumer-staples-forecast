"""
create_train_validation_test_split.py
-------------------------------------
Creates train/validation/test split from the combined dataset.
CRITICAL: Handles look-ahead bias by shifting the target variable (t+1).
Saves X_train, X_val, X_test, y_train, y_val, y_test as separate CSV files.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
import numpy as np

# ============================
# Configuration
# ============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# Split Ratios (Course Requirement: 70% Train / 20% Val / 10% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def create_train_val_test_split():
    """Create and save train/validation/test splits with chronological integrity."""
    print("\n" + "=" * 80)
    print("CREATING TRAIN/VALIDATION/TEST SPLIT (70% / 20% / 10%)")
    print("=" * 80)
    
    # 1. Load Data
    print(f"\nLoading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Combined data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ==============================================================================
    # 🚨 CRITICAL STEP: PREVENT LOOK-AHEAD BIAS
    # We must predict the NEXT week's return using CURRENT week's features.
    # We shift the target variable (Weekly_Return) UP by 1 period per Ticker.
    # ==============================================================================
    print("⚠️  APPLYING TARGET LAG: Predicting Next Week's Return (t+1) using Current Features (t)")
    
    # Create the prediction target (t+1)
    df["Target_Next_Week"] = df.groupby("Ticker")["Weekly_Return"].shift(-1)
    
    # Drop the last row of each ticker (which becomes NaN because there is no 'next week')
    initial_len = len(df)
    df = df.dropna(subset=["Target_Next_Week"])
    print(f"   Rows dropped due to lag (end of series): {initial_len - len(df)}")
    
    # 2. Define Features (X) and Target (y)
    # We must exclude the Target and any feature that contains 'current' return info to prevent leakage
    exclude_cols = [
        "Target_Next_Week", # The Target itself
        "Weekly_Return",    # Current week return (Leaky)
        "Return",           # Current daily return (Leaky)
        "Date",             # Metadata
        "Ticker",           # Metadata
        "Close",            # Current Price level (Leaky for stationary models)
        
        # Highly correlated technicals derived from current Close (Risk of leakage)
        "Price_to_MA20", 
        "Price_to_MA50",    
        "BB_Position"
    ]
    
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df["Target_Next_Week"]
    
    print(f"\nFeatures selected: {len(X.columns)}")
    # print(f"Features list: {X.columns.tolist()}")
    
    # 3. Handle Missing Values in Features
    # Forward fill (propagate last known value) then Backward fill (for initial NaNs)
    nan_counts = X.isna().sum().sum()
    if nan_counts > 0:
        print(f"\n   ⚠️  Handling {nan_counts} missing values in features (Fill Forward/Backward)...")
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # If any NaNs remain (rare), drop those rows
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    
    # 4. Chronological Split (No Shuffling!)
    # We split strictly by time to respect the temporal order of financial data.
    train_idx = int(len(X) * TRAIN_RATIO)
    val_idx = int(len(X) * (TRAIN_RATIO + VAL_RATIO))
    
    X_train = X.iloc[:train_idx]
    X_val = X.iloc[train_idx:val_idx]
    X_test = X.iloc[val_idx:]
    
    y_train = y.iloc[:train_idx]
    y_val = y.iloc[train_idx:val_idx]
    y_test = y.iloc[val_idx:]
    
    print("\n✅ Split summary:")
    print(f"   Train set:      {len(X_train):,} samples ({TRAIN_RATIO * 100:.0f}%)")
    print(f"   Validation set: {len(X_val):,} samples ({VAL_RATIO * 100:.0f}%)")
    print(f"   Test set:       {len(X_test):,} samples ({TEST_RATIO * 100:.0f}%)")
    
    # Verify no overlap or leaks
    # (Optional check: Ensure indices are sequential)
    
    # 5. Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    
    print("\n✅ Files saved successfully to 'data/processed/':")
    print(f"   X_train.csv, X_val.csv, X_test.csv")
    print(f"   y_train.csv, y_val.csv, y_test.csv")
    
    print("\n" + "=" * 80)
    print("SPLIT COMPLETE - READY FOR TRAINING")
    print("=" * 80)


if __name__ == "__main__":
    create_train_val_test_split()
