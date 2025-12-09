"""
baseline_model.py
-----------------
This script trains and evaluates baseline regression models
for the Consumer Staples Forecasting project using the weekly preprocessed data.

It focuses on predicting next-week returns using financial, macro, and historical features.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import random

# ============================
# Reproducibility
# ============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Function: load_data
# ============================
def load_data() -> pd.DataFrame:
    """Loads the preprocessed dataset and ensures it’s clean."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.select_dtypes(include=[np.number]).dropna()

    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

# ============================
# Function: prepare_data
# ============================
def prepare_data(df: pd.DataFrame, include_macro: bool = True):
    """
    Prepares features (X) and target (y) for modeling.
    Removes columns that could cause data leakage or redundancy.
    """
    target_col = "Target_Weekly_Return"

    # Définir les colonnes à exclure
    excluded_cols = ["Weekly_Return", "Return", "Close", target_col]

    # Exclure les features macro si demandé
    if not include_macro:
        excluded_cols += [col for col in df.columns if col.startswith("US")]

    # Créer la matrice de features
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].astype(np.float64)
    y = df[target_col].astype(np.float64)

    # Split temporel (pas de shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=False
    )

    scope = "macro" if include_macro else "micro"
    print(f"✅ Data prepared ({scope}): {X.shape[1]} features, {X_train.shape[0]} train rows")
    return X_train, X_test, y_train, y_test

# ============================
# Function: train_linear_regression
# ============================
def train_linear_regression(X_train, y_train):
    """Trains a simple Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Linear Regression model trained successfully.")
    return model

# ============================
# Function: evaluate_model
# ============================
def evaluate_model(model, X_test, y_test, model_name="LinearRegression"):
    """Evaluates the model and prints key performance metrics."""
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n📊 {model_name} Performance:")
    print(f"   - MAE  : {mae:.6f}")
    print(f"   - RMSE : {rmse:.6f}")
    print(f"   - R²   : {r2:.6f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ============================
# Function: save_model
# ============================
def save_model(model, filename):
    """Saves a trained model."""
    model_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, model_path)
    print(f"💾 {filename} saved at: {model_path}")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    print("\n🚀 Starting baseline model training pipeline...\n")

    # 1️⃣ Load data
    df = load_data()

    # 2️⃣ Train micro-only model (sans macro)
    X_train, X_test, y_train, y_test = prepare_data(df, include_macro=False)
    model_micro = train_linear_regression(X_train, y_train)
    metrics_micro = evaluate_model(model_micro, X_test, y_test, "LinearRegression (Micro)")
    save_model(model_micro, "linear_regression_micro.pkl")

    # 3️⃣ Train macro-enriched model (avec macro)
    if "US10Y_Yield" in df.columns and "US_CPI" in df.columns:
        print("\n💡 Training macro-enriched model...\n")
        X_train_m, X_test_m, y_train_m, y_test_m = prepare_data(df, include_macro=True)
        model_macro = train_linear_regression(X_train_m, y_train_m)
        metrics_macro = evaluate_model(model_macro, X_test_m, y_test_m, "LinearRegression (Macro)")
        save_model(model_macro, "linear_regression_macro.pkl")

    print("\n🎯 Baseline models trained and saved successfully.\n")
