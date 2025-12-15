"""
evaluation.py
--------------
Evaluates all trained models (LinearRegression, RandomForest, XGBoost, NeuralNetwork)
on the held-out 20% test set.

Generates:
- Performance metrics (MAE, RMSE, R²)
- Prediction vs. Actual plots
- Summary CSV in /results/

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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================
# Helper functions
# ============================
def load_data():
    """Load data and prepare test split identical to training."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Weekly_Return"])

    # Keep relevant columns
    keep_cols = [
        "Close", "Volume", "Return", "Volatility_30d",
        "US10Y_Yield", "US_CPI", "EPS", "Net Income",
        "Total Revenue", "Weekly_Return"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Prepare features and target
    X = df.drop(columns=["Close", "Return", "Weekly_Return"], errors="ignore")
    y = df["Weekly_Return"]

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0)

    # Split (same 80/20 time-based)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"✅ Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features.")
    return X_test, y_test


def evaluate_model(model, X_test, y_test, name):
    """Compute and print MAE, RMSE, and R² metrics for a given model."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n📊 {name} — Test Results:")
    print(f"   - MAE  : {mae:.6f}")
    print(f"   - RMSE : {rmse:.6f}")
    print(f"   - R²   : {r2:.6f}")

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}, preds


def plot_predictions(y_test, preds, model_name):
    """Plot predicted vs actual Weekly_Return for a given model."""
    plt.figure(figsize=(8, 4))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(preds, label="Predicted", linewidth=2, linestyle="--")
    plt.title(f"{model_name} — Weekly Return Prediction")
    plt.xlabel("Weeks")
    plt.ylabel("Weekly Return")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"{model_name}_test_predictions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved test plot → {save_path}")


# ============================
# Main evaluation
# ============================
if __name__ == "__main__":
    print("\n🚀 Evaluating trained models on held-out data...\n")

    # Step 1 — Load test data
    X_test, y_test = load_data()

    # Step 2 — List models
    model_names = ["LinearRegression", "RandomForest", "XGBoost", "NeuralNetwork"]

    results = []

    # Step 3 — Evaluate each model
    for name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ Model file missing for {name}, skipping.")
            continue

        model = joblib.load(model_path)
        metrics, preds = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        plot_predictions(y_test, preds, name)

    # Step 4 — Summary table
    results_df = pd.DataFrame(results)
    print("\n🏁 Summary of Model Test Performance:")
    print(results_df.to_string(index=False))

    # Step 5 — Save results
    summary_path = os.path.join(RESULTS_DIR, "model_test_results.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n📄 Saved to: {summary_path}")

    print("\n🎯 All model evaluations completed successfully!\n")
