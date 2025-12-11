"""
train_all.py
-------------
Main script to train and evaluate baseline models (Linear, RandomForest, XGBoost)
to predict weekly returns for Consumer Staples stocks, avoiding data leakage.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Import model trainers
from models.linear_model import train_linear_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost


# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================
# Helper Functions
# ============================
def load_data():
    """Load preprocessed dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data ready: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def prepare_data(df):
    """Prepare features and target for training, avoiding leakage."""
    print("⚙️ Preparing data for training...")

    # ✅ Keep only relevant columns
    keep_cols = [
        "Close", "Volume", "Return", "Volatility_30d",
        "US10Y_Yield", "US_CPI", "EPS", "Net Income",
        "Total Revenue", "Weekly_Return"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Drop missing target
    df = df.dropna(subset=["Weekly_Return"])

    # ✅ Remove columns that cause data leakage
    leak_cols = ["Close", "Return", "Weekly_Return"]
    X = df.drop(columns=[c for c in leak_cols if c in df.columns])
    y = df["Weekly_Return"]

    # Fill missing values
    X = X.ffill().bfill().fillna(0)

    # Time-based split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"✅ Data split complete: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, {X.shape[1]} features.")
    print(f"🧩 Features used: {list(X.columns)}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n📊 {model_name} Performance:")
    print(f"   - MAE  : {mae:.5f}")
    print(f"   - RMSE : {rmse:.5f}")
    print(f"   - R²   : {r2:.5f}")

    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


def save_model(model, name):
    """Save trained model."""
    path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"💾 {name} saved at: {path}")


# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    print("\n🚀 Loading preprocessed data...\n")

    # Step 1: Load Data
    df = load_data()

    # Step 2: Prepare Data (safe version)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 3: Train Models
    print("\n🧠 Training models...\n")
    models = {
        "LinearRegression": train_linear_regression(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train)
    }

    # Step 4: Evaluate Models
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        save_model(model, name)

    # Step 5: Summary Table
    results_df = pd.DataFrame(results)
    print("\n🏁 Summary of Model Performance:")
    print(results_df.to_string(index=False))

    # Step 6: Save Summary
    summary_path = os.path.join(RESULTS_DIR, "model_performance_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n📄 Results saved to: {summary_path}")

    print("\n🎯 All models trained, evaluated, and saved successfully!\n")
