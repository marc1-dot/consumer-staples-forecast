"""
models.py
---------
Machine Learning modeling for the Consumer Staples Forecasting project.

This version includes both baseline (micro-only) and macro-enriched models.
It compares their performance and saves both models for future evaluation.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

# ============================
# Import required libraries
# ============================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Function: load_data
# ============================
def load_data() -> pd.DataFrame:
    """Loads and cleans the preprocessed dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Convert all numeric columns and drop missing
    df = df.select_dtypes(include=[np.number]).dropna()

    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

# ============================
# Function: prepare_data
# ============================
def prepare_data(df: pd.DataFrame, include_macro: bool = False):
    """
    Prepares features (X) and target (y) for modeling.

    Parameters
    ----------
    include_macro : bool
        If True, includes macroeconomic variables (US10Y_Yield, US_CPI)
    """
    target_col = "Return" if "Return" in df.columns else "Close"

    # Select features
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    if not include_macro:
        X = X.drop(columns=[col for col in X.columns if "US" in col], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    mode = "with macros" if include_macro else "micro-only"
    print(f"✅ Data prepared ({mode}): {X_train.shape[1]} features, {X_train.shape[0]} train rows")
    return X_train, X_test, y_train, y_test

# ============================
# Model training & evaluation
# ============================
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label="Model"):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n📊 {label} Evaluation:")
    print(f"   - MAE  : {mae:.4f}")
    print(f"   - RMSE : {rmse:.4f}")
    print(f"   - R²   : {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_model(model, filename):
    model_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    print("\n🚀 Starting model training pipeline...\n")

    df = load_data()

    # --- Model 1: Baseline (micro variables only)
    X_train, X_test, y_train, y_test = prepare_data(df, include_macro=False)
    model_micro = train_linear_regression(X_train, y_train)
    metrics_micro = evaluate_model(model_micro, X_test, y_test, label="Baseline (Micro only)")
    save_model(model_micro, "linear_regression_micro.pkl")

    # --- Model 2: Macro-enriched
    X_train_m, X_test_m, y_train_m, y_test_m = prepare_data(df, include_macro=True)
    model_macro = train_linear_regression(X_train_m, y_train_m)
    metrics_macro = evaluate_model(model_macro, X_test_m, y_test_m, label="Macro-Enriched")
    save_model(model_macro, "linear_regression_macro.pkl")

    print("\n🎯 Model training complete. Both versions saved successfully.\n")
