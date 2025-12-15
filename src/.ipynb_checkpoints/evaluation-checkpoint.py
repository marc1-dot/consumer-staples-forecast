"""
evaluation.py
--------------
Evaluate previously trained models (LinearRegression, RandomForest, XGBoost)
on the held-out 20% test set.

This script does not retrain any models — it only loads the saved `.pkl` models
and assesses their predictive performance.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================
# Helper functions
# ============================
def load_data():
    """Load and prepare the dataset (only for testing)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ File not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Keep relevant columns
    keep_cols = [
        "Close", "Volume", "Return", "Volatility_30d",
        "US10Y_Yield", "US_CPI", "EPS", "Net Income",
        "Total Revenue", "Weekly_Return"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Drop missing target
    df = df.dropna(subset=["Weekly_Return"])

    # Split features and target
    X = df.drop(columns=["Close", "Return", "Weekly_Return"], errors="ignore")
    y = df["Weekly_Return"]

    # Handle missing values
    X = X.ffill().bfill().fillna(0)

    # Use last 20% as test set (time-based)
    split_index = int(len(X) * 0.8)
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    print(f"🧩 Test data: {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}% of total)")
    return X_test, y_test, X.columns


def load_model(name):
    """Load a trained model by name."""
    path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found: {path}")
    model = joblib.load(path)
    print(f"✅ Loaded model: {name}")
    return model


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2, preds


def plot_predictions(y_test, preds, model_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(preds, label="Predicted", linestyle="--")
    plt.title(f"{model_name} — Weekly Return Prediction (Test Set)")
    plt.xlabel("Time Index")
    plt.ylabel("Weekly Return")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{model_name}_test_predictions.png")
    plt.savefig(path)
    plt.close()
    print(f"📈 Saved test plot → {path}")


# ============================
# Main evaluation
# ============================
if __name__ == "__main__":
    print("\n🚀 Evaluating trained models on held-out data...\n")

    # Step 1: Load data
    X_test, y_test, feature_names = load_data()

    # Step 2: Load models
    model_names = ["LinearRegression", "RandomForest", "XGBoost"]
    results = []

    # Step 3: Evaluate each model
    for name in model_names:
        model = load_model(name)
        mae, rmse, r2, preds = evaluate_model(model, X_test, y_test)

        print(f"\n📊 {name} — Test Results:")
        print(f"   - MAE  : {mae:.5f}")
        print(f"   - RMSE : {rmse:.5f}")
        print(f"   - R²   : {r2:.5f}")

        plot_predictions(y_test, preds, name)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    # Step 4: Save summary results
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(RESULTS_DIR, "model_test_results.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n🏁 Summary of Model Test Performance:")
    print(results_df.to_string(index=False))
    print(f"\n📄 Saved to: {summary_path}\n")
