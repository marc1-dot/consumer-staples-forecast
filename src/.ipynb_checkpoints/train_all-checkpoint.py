"""
train_all.py
-------------
Trains all models (LinearRegression, RandomForest, XGBoost, NeuralNetwork)
on the Consumer Staples dataset with advanced features.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
import numpy as np
import joblib

from models.linear_model import train_linear_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.neural_network import train_neural_network


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "trained")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_data():
    """Load the preprocessed dataset with advanced features."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def prepare_data(df):
    """Prepare features and target for model training."""
    print("\n⚙️ Preparing data for training...")
    print(f"Initial columns: {list(df.columns)}")
    
    df = df.dropna(subset=["Weekly_Return"])
    print(f"Removed rows with missing target. Remaining: {len(df)} rows")
    
    exclude_cols = ["Weekly_Return", "Close", "Return", "Date", "Ticker"]
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df["Weekly_Return"]
    
    print(f"\nFeatures selected for training: {list(X.columns)}")
    
    # Convert all features to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing and infinite values
    print(f"🔍 Checking for missing values...")
    missing_before = X.isnull().sum().sum()
    print(f"   Missing before cleaning: {missing_before}")
    
    X = X.ffill().bfill().fillna(0)
    
    missing_after = X.isnull().sum().sum()
    print(f"   Missing after cleaning: {missing_after}")
    
    if np.isinf(X).any().any():
        print("⚠️ Found infinite values — replacing with 0.")
        X = X.replace([np.inf, -np.inf], 0)
    
    # Split chronologically (no shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n✅ Data preparation complete.")
    print(f"   Training set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"   Test set: {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test


def display_feature_statistics(X_train):
    """Display basic statistics of the training features."""
    print("\n📊 Feature Statistics (Training Set):")
    print("=" * 80)
    stats = X_train.describe().T[['mean', 'std', 'min', 'max']]
    print(stats.to_string())
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING PIPELINE FOR CONSUMER STAPLES")
    print("=" * 80)

    print("\nSTEP 1️⃣: Loading data...")
    df = load_data()

    print("\nSTEP 2️⃣: Preparing features and target...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    display_feature_statistics(X_train)

    print("\n" + "=" * 80)
    print("STEP 3️⃣: Training models...")
    print("=" * 80)
    
    models = {}

    print("\n[1/4] Training Linear Regression...")
    models["LinearRegression"] = train_linear_regression(X_train, y_train)
    
    print("\n[2/4] Training Random Forest...")
    models["RandomForest"] = train_random_forest(X_train, y_train)
    
    print("\n[3/4] Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)
    
    print("\n[4/4] Training Neural Network...")
    models["NeuralNetwork"] = train_neural_network(X_train, y_train)

    print("\n" + "=" * 80)
    print("STEP 4️⃣: Saving trained models...")
    print("=" * 80)
    
    for name, model in models.items():
        model_path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Saved: {name:20s} → {model_path}")

    print("\n" + "=" * 80)
    print("🎯 ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📂 Models saved in: {MODEL_SAVE_DIR}")
    print(f"📈 Training samples: {len(X_train)}")
    print(f"🧩 Features used: {X_train.shape[1]}")
    print("\n➡️ Next step: Run `evaluation.py` to test model performance.\n")
