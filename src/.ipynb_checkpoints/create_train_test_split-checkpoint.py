"""
create_train_test_split.py
--------------------------
Creates train/test split from the combined dataset.
Saves X_train, X_test, y_train, y_test as separate CSV files.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

# Train/Test split ratio
TRAIN_RATIO = 0.8


def create_train_test_split():
    """Create and save train/test splits."""
    print("\n" + "=" * 80)
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 80)
    
    # Load data
    print("\nLoading data from: {}".format(DATA_PATH))
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Combined data not found at {}".format(DATA_PATH))
    
    df = pd.read_csv(DATA_PATH)
    print("Loaded {} rows, {} columns".format(len(df), len(df.columns)))
    
    # Remove rows with missing target
    df = df.dropna(subset=["Weekly_Return"])
    print("After removing missing targets: {} rows".format(len(df)))
    
    # Exclude columns that cause data leakage
    exclude_cols = [
        "Weekly_Return",    # Target variable
        "Date",             # Date column
        "Ticker",           # Stock ticker
        "Return",           # 🔴 LEAKAGE: Daily return (99.57% correlation)
        "Close",            # 🔴 LEAKAGE: Current price
        "Price_to_MA20",    # ⚠️ Calculated from Close (73.21% correlation)
        "BB_Position",      # ⚠️ Calculated from Close (69.48% correlation)
    ]
    
    # Separate features and target
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df["Weekly_Return"]
    
    print("\nFeatures after exclusion: {}".format(X.columns.tolist()))
    print("Number of features: {}".format(len(X.columns)))
    
    # Time-series split (80/20)
    split_idx = int(len(X) * TRAIN_RATIO)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print("\nSplit created:")
    print("   Train set: {} samples ({:.1f}%)".format(len(X_train), TRAIN_RATIO * 100))
    print("   Test set:  {} samples ({:.1f}%)".format(len(X_test), (1 - TRAIN_RATIO) * 100))
    
    # Save to CSV
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    
    print("\nFiles saved:")
    print("   X_train.csv: {} rows × {} columns".format(len(X_train), len(X_train.columns)))
    print("   X_test.csv:  {} rows × {} columns".format(len(X_test), len(X_test.columns)))
    print("   y_train.csv: {} rows".format(len(y_train)))
    print("   y_test.csv:  {} rows".format(len(y_test)))
    
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    create_train_test_split()