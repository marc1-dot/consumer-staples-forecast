"""
train_all.py
-------------
Trains all models (LinearRegression, RandomForest, XGBoost, NeuralNetwork)
on the Consumer Staples dataset (5 core companies).

⚠️ This script only performs training on 80% of the data (no evaluation here).
Evaluation is done separately in `evaluation.py`.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Import model training functions
from models.linear_model import train_linear_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.neural_network import train_neural_network


# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "trained")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ============================
# Helper functions
# ============================
def load_data():
    """Load the preprocessed dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def prepare_data(df):
    """
    Prepare features and target for model training.
    Removes potential leakage (Close, Return, Weekly_Return)
    and ensures all features are numeric and complete.
    """
    print("⚙️ Preparing data for training...")

    # Keep relevant columns only
    keep_cols = [
        "Close", "Volume", "Return", "Volatility_30d",
        "US10Y_Yield", "US_CPI", "EPS", "Net Income",
        "Total Revenue", "Weekly_Return"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Drop missing target
    df = df.dropna(subset=["Weekly_Return"])

    # Remove leakage columns
    X = df.drop(columns=["Close", "Return", "Weekly_Return"], errors="ignore")
    y = df["Weekly_Return"]

    # Ensure numeric types and handle NaNs
    X = X.apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0)

    # Time-based split (no shuffle to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"✅ Training data ready: {X_train.shape[0]} rows, {X.shape[1]} features.")
    print(f"🧩 Features used: {list(X.columns)}")

    return X_train, y_train


# ============================
# Main training routine
# ============================
if __name__ == "__main__":
    print("\n🚀 Starting training pipeline for Consumer Staples...\n")

    # Step 1: Load preprocessed data
    df = load_data()

    # Step 2: Prepare data for model training
    X_train, y_train = prepare_data(df)

    # Step 3: Train all models
    print("\n🧠 Training models...\n")
    models = {
        "LinearRegression": train_linear_regression(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
        "NeuralNetwork": train_neural_network(X_train, y_train)
    }

    # Step 4: Save each trained model
    for name, model in models.items():
        model_path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"💾 {name} saved at: {model_path}")

    print("\n🎯 All models trained and saved successfully!\n")
