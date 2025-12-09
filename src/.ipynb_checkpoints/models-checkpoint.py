"""
models.py
---------
This module handles the machine learning modeling for the
Consumer Staples Forecasting project.

It loads the preprocessed data, splits it into training and testing sets,
trains baseline models (starting with Linear Regression), and saves them
for future evaluation and comparison.

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
import joblib  # to save models

# ============================
# Configuration
# ============================
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Function: load_data
# ============================
def load_data() -> pd.DataFrame:
    """Loads the preprocessed dataset for modeling."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Ensure numerical columns
    df = df.select_dtypes(include=[np.number]).dropna()

    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

# ============================
# Function: prepare_data
# ============================
def prepare_data(df: pd.DataFrame):
    """Prepares features (X) and target (y) for modeling."""

    # Target variable: we predict next-day return (or price if missing)
    target_col = "Return" if "Return" in df.columns else "Close"

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"✅ Data split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")
    return X_train, X_test, y_train, y_test

# ============================
# Function: train_linear_regression
# ============================
def train_linear_regression(X_train, y_train):
    """Trains a baseline Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Linear Regression model trained successfully.")
    return model

# ============================
# Function: evaluate_model
# ============================
def evaluate_model(model, X_test, y_test):
    """Evaluates a trained model and prints performance metrics."""
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 Model Evaluation Results:")
    print(f"   - MAE  : {mae:.4f}")
    print(f"   - RMSE : {rmse:.4f}")
    print(f"   - R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ============================
# Function: save_model
# ============================
def save_model(model, filename="linear_regression.pkl"):
    """Saves a trained model to the models directory."""
    model_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    print("\n🚀 Starting model training pipeline...\n")

    # Step 1: Load data
    df = load_data()

    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 3: Train model
    model = train_linear_regression(X_train, y_train)

    # Step 4: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Save model
    save_model(model)

    print("\n🎯 Baseline model training complete and saved successfully.\n")

