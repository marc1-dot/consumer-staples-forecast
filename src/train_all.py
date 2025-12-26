"""
train_all.py
------------
Main training script.
RESPONSIBILITY: Data Cleaning, Training, Validation, Saving Artifacts.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import model training functions
from models.linear_model import train_linear_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.neural_network import train_neural_network

# Import evaluation function
from model_evaluate import evaluate_model, compare_models


def load_data():
    """Load raw split data."""
    print("\n📂 STEP 1: Loading raw split data...")
    data_dir = BASE_DIR / "data" / "processed"

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")

    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_and_save_data(X_train, X_val, X_test):
    """
    Handle missing values consistently using Training statistics.
    CRITICAL: Saves the cleaned data so test_all.py uses the SAME values.
    """
    print("\n⚙️  STEP 2: Preparing and Cleaning Data...")
    
    # 1. Compute statistics on TRAIN set only (to avoid leakage)
    train_means = X_train.mean(numeric_only=True)
    
    # 2. Apply to all sets
    # We use the SAME mean for Val and Test to maintain consistency
    X_train_clean = X_train.fillna(train_means)
    X_val_clean = X_val.fillna(train_means)
    X_test_clean = X_test.fillna(train_means)
    
    # 3. Save CLEANED data for the Test Script
    # We prefix with 'clean_' to distinguish from raw split
    save_dir = BASE_DIR / "data" / "processed"
    print(f"   💾 Saving cleaned datasets to {save_dir}...")
    
    X_train_clean.to_csv(save_dir / "X_train_clean.csv", index=False)
    X_val_clean.to_csv(save_dir / "X_val_clean.csv", index=False)
    X_test_clean.to_csv(save_dir / "X_test_clean.csv", index=False)
    
    print("   ✅ Data cleaning complete and saved.")
    return X_train_clean, X_val_clean, X_test_clean


def train_models(X_train, y_train, X_val, y_val):
    """Train all models."""
    print("\n🤖 STEP 3: Training models...")
    models = {}

    print("\n[1/4] Linear Regression...")
    models["Linear Regression"] = train_linear_regression(X_train, y_train)

    print("\n[2/4] Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train)

    print("\n[3/4] XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train, X_val, y_val)

    print("\n[4/4] Neural Network...")
    models["NeuralNetwork"] = train_neural_network(X_train, y_train)

    return models


def evaluate_on_validation(models, X_val, y_val):
    """Evaluate on Validation set."""
    print("\n📊 STEP 4: Validation Set Evaluation...")
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        results[name] = metrics

    best_model = compare_models(results)
    return best_model


def save_models(models):
    """Save trained models."""
    print("\n💾 STEP 5: Saving models...")
    models_dir = BASE_DIR / "trained_models"
    models_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        filename = name.replace(" ", "_").lower() + ".pkl"
        filepath = models_dir / filename
        joblib.dump(model, filepath)
        print(f"   ✅ Saved {name}")


def main():
    print("\n" + "="*80)
    print("🚀 TRAINING PIPELINE (Data Prep -> Train -> Validate -> Save)")
    print("="*80)

    # 1. Load
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Prepare & SAVE CLEAN DATA (Fixes the consistency bug)
    X_train, X_val, X_test = prepare_and_save_data(X_train, X_val, X_test)

    # 3. Train
    models = train_models(X_train, y_train, X_val, y_val)

    # 4. Validate
    best_model = evaluate_on_validation(models, X_val, y_val)

    # 5. Save
    save_models(models)

    print("\n" + "="*80)
    print(f"✅ TRAINING COMPLETE. Best Model (Validation): {best_model}")
    print("👉 Now run 'test_all.py' to evaluate on the Test Set.")
    print("="*80)


if __name__ == "__main__":
    main()
