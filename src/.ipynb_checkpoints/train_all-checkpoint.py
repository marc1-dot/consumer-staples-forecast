"""
train_all.py
-------------
Main script to train and evaluate all models (Linear, RandomForest, XGBoost)
on both raw and feature-engineered data, with proper temporal handling.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Import model trainers
from models.linear_model import train_linear_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost

# ✅ Feature engineering module
from features.feature_engineering import engineer_features


# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "trained")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


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
    """Prepare training and testing data for time series forecasting."""
    # === Step 1: Ensure correct target shift ===
    if "Target_Weekly_Return" not in df.columns:
        if "Weekly_Return" in df.columns:
            df = df.sort_values(["Ticker", "Date"])
            df["Target_Weekly_Return"] = df.groupby("Ticker")["Weekly_Return"].shift(-1)
            print("📈 Created Target_Weekly_Return (1-week forward shift).")
        else:
            raise ValueError("❌ No target found: Weekly_Return or Target_Weekly_Return missing.")

    # === Step 2: Remove rows without a valid target ===
    df = df.dropna(subset=["Target_Weekly_Return"])

    # === Step 3: Remove features that leak information ===
    exclude_cols = ["Weekly_Return", "Return", "Close", "Target_Weekly_Return"]
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors="ignore")

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    y = df["Target_Weekly_Return"].astype(float)

    # === Step 4: Time-based split (no shuffle) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print(f"✅ Data split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, {X.shape[1]} features.")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate and print model metrics."""
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

    # Step 2: Feature Engineering
    df = engineer_features(df)
    print(f"✅ Feature engineering complete: {df.shape[1]} columns.")

    # Step 3: Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 4: Train Models
    models = {
        "LinearRegression": train_linear_regression(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train)
    }

    # Step 5: Evaluate Models
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        save_model(model, name)

    # Step 6: Summary
    print("\n🏁 Summary of Model Performance:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n🎯 All models trained, evaluated, and saved successfully!\n")
