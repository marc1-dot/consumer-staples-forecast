"""
create_train_validation_test_split.py
--------------------------
Creates train/validation/test split from the combined dataset.
Saves X_train, X_val, X_test, y_train, y_val, y_test as separate CSV files.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_data.csv")  # ✅ FIX: ajout de "data"
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")  # ✅ FIX: ajout de "data"

# ✅ RATIO DU COURS : 70% train / 20% validation / 10% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def create_train_val_test_split():
    """Create and save train/validation/test splits."""
    print("\n" + "=" * 80)
    print("CREATING TRAIN/VALIDATION/TEST SPLIT (70% / 20% / 10%)")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Combined data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Remove rows with missing target
    df = df.dropna(subset=["Weekly_Return"])
    print(f"After removing missing targets: {len(df):,} rows")
    
    # ✅ FIX: Exclude columns that cause data leakage (COMPLET)
    exclude_cols = [
        "Weekly_Return",    # Target variable
        "Date",             # Date column
        "Ticker",           # Stock ticker
        "Return",           # 🔴 LEAKAGE: Daily return (99.57% correlation)
        "Close",            # 🔴 LEAKAGE: Current price
        "Price_to_MA20",    # 🔴 LEAKAGE: Calculated from Close (73.21% correlation)
        "Price_to_MA50",    # 🔴 LEAKAGE: Calculated from Close (AJOUTÉ!)
        "BB_Position",      # 🔴 LEAKAGE: Calculated from Close (69.48% correlation)
        "MA_20",            # ⚠️ LEAKAGE: Moving average of Close (AJOUTÉ!)
        "MA_50",            # ⚠️ LEAKAGE: Moving average of Close (AJOUTÉ!)
    ]
    
    # Separate features and target
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df["Weekly_Return"]
    
    print(f"\nFeatures after exclusion: {X.columns.tolist()}")
    print(f"Number of features: {len(X.columns)}")
    
    # ✅ NEW: Vérifier et gérer les NaN
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print("\n⚠️  WARNING: Missing values detected in features:")
        nan_features = nan_counts[nan_counts > 0]
        for feat, count in nan_features.items():
            print(f"   {feat}: {count} NaN ({count/len(X)*100:.2f}%)")
        
        print("\n   Filling NaN with forward fill + backward fill...")
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        remaining_nan = X.isna().sum().sum()
        if remaining_nan > 0:
            print(f"   ⚠️  Still {remaining_nan} NaN remaining - dropping those rows")
            valid_idx = X.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            print(f"   Final dataset: {len(X):,} rows")
    else:
        print("\n✅ No missing values in features")
    
    # ✅ Time-series split (70% / 20% / 10%)
    train_idx = int(len(X) * TRAIN_RATIO)
    val_idx = int(len(X) * (TRAIN_RATIO + VAL_RATIO))
    
    X_train = X.iloc[:train_idx]
    X_val = X.iloc[train_idx:val_idx]
    X_test = X.iloc[val_idx:]
    
    y_train = y.iloc[:train_idx]
    y_val = y.iloc[train_idx:val_idx]
    y_test = y.iloc[val_idx:]
    
    print("\n✅ Split created (as per course recommendations):")
    print(f"   Train set:      {len(X_train):,} samples ({TRAIN_RATIO * 100:.0f}%)")
    print(f"   Validation set: {len(X_val):,} samples ({VAL_RATIO * 100:.0f}%)")
    print(f"   Test set:       {len(X_test):,} samples ({TEST_RATIO * 100:.0f}%)")
    
    # ✅ Vérifier qu'il n'y a pas de NaN dans les splits
    for name, data in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"   ⚠️  WARNING: {name} has {nan_count} NaN values!")
        else:
            print(f"   ✅ {name}: No NaN values")
    
    # ✅ Save to CSV (including validation set)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    
    print("\n✅ Files saved:")
    print(f"   X_train.csv: {len(X_train):,} rows × {len(X_train.columns)} columns")
    print(f"   X_val.csv:   {len(X_val):,} rows × {len(X_val.columns)} columns")
    print(f"   X_test.csv:  {len(X_test):,} rows × {len(X_test.columns)} columns")
    print(f"   y_train.csv: {len(y_train):,} rows")
    print(f"   y_val.csv:   {len(y_val):,} rows")
    print(f"   y_test.csv:  {len(y_test):,} rows")
    
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    print("\n" + "=" * 80)
    print("TRAIN/VALIDATION/TEST SPLIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    create_train_val_test_split()