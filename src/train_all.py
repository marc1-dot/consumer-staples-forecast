"""
train_all.py
------------
Main training script that orchestrates the training of all models.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
Date: December 2024
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
    """
    Load preprocessed training, validation, and test data.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("\n📂 STEP 1: Loading data...")

    data_dir = BASE_DIR / "data" / "processed"

    # Debug: Print paths
    print(f"\n🔍 DEBUG - Chemins des fichiers:")
    print(f"   DATA_DIR: {data_dir}")
    print(f"   X_train:  {data_dir / 'X_train.csv'}")
    print(f"   Exists:   {(data_dir / 'X_train.csv').exists()}")

    # Load data
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")

    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    print(f"\n✅ Data loaded successfully:")
    print(f"   X_train: {X_train.shape[0]:,} rows, {X_train.shape[1]} columns")
    print(f"   X_val:   {X_val.shape[0]:,} rows, {X_val.shape[1]} columns")
    print(f"   X_test:  {X_test.shape[0]:,} rows, {X_test.shape[1]} columns")
    print(f"   y_train: {len(y_train):,} samples")
    print(f"   y_val:   {len(y_val):,} samples")
    print(f"   y_test:  {len(y_test):,} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_data(X_train, X_val, X_test):
    """
    Prepare features for training (handle missing values, etc.).

    Returns:
        Cleaned X_train, X_val, X_test
    """
    print("\n⚙️  STEP 2: Preparing features and target...")
    print("\n⚙️  Preparing data for training...")
    print(f"Initial columns ({len(X_train.columns)}): {X_train.columns.tolist()}")

    # Missing values
    print(f"\n🔍 Checking for missing values...")
    missing_train = X_train.isnull().sum().sum()
    missing_val = X_val.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    print(f"   Missing before cleaning: {missing_train + missing_val + missing_test}")

    if (missing_train > 0) or (missing_val > 0) or (missing_test > 0):
        train_means = X_train.mean(numeric_only=True)
        X_train = X_train.fillna(train_means)
        X_val = X_val.fillna(train_means)
        X_test = X_test.fillna(train_means)

    missing_after = X_train.isnull().sum().sum() + X_val.isnull().sum().sum() + X_test.isnull().sum().sum()
    print(f"   Missing after cleaning: {missing_after}")

    print(f"\n✅ Data preparation complete:")
    print(f"   Training set:   {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"   Validation set: {X_val.shape[0]:,} rows, {X_val.shape[1]} features")
    print(f"   Test set:       {X_test.shape[0]:,} rows, {X_test.shape[1]} features")

    return X_train, X_val, X_test


def print_feature_statistics(X_train):
    """Print detailed statistics about features."""
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS (Training Set)")
    print("=" * 80)
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of samples:  {X_train.shape[0]:,}")

    print(f"\n📋 Feature list ({len(X_train.columns)} features):")
    for i, col in enumerate(X_train.columns, 1):
        print(f"   [{i:2d}] {col}")

    print(f"\n📊 Basic statistics:")
    stats = X_train.describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())
    print("=" * 80)


def train_models(X_train, y_train, X_val, y_val):
    """
    Train all models (Linear Regression, Random Forest, XGBoost, Neural Network).

    Returns:
        Dictionary of trained models
    """
    print("\n" + "=" * 80)
    print("🤖 STEP 3: Training models...")
    print("=" * 80)

    models = {}

    # 1) Linear Regression
    print("\n[1/4] Training Linear Regression...")
    print("   Note: Linear models don't use validation during training")
    models["Linear Regression"] = train_linear_regression(X_train, y_train)

    # 2) Random Forest
    print("\n[2/4] Training Random Forest...")
    print("   Note: Random Forest uses OOB (Out-of-Bag) error internally")
    models["Random Forest"] = train_random_forest(X_train, y_train)

    # 3) XGBoost
    print("\n[3/4] Training XGBoost...")
    print("   Using validation set for early stopping...")
    models["XGBoost"] = train_xgboost(X_train, y_train, X_val, y_val)

    # 4) Neural Network
    print("\n[4/4] Training Neural Network...")
    print("   Using validation set for early stopping...")
    models["NeuralNetwork"] = train_neural_network(X_train, y_train)

    print("\n" + "=" * 80)
    print("✅ All models trained successfully!")
    print("=" * 80)

    return models


def evaluate_on_validation(models, X_val, y_val):
    """
    Evaluate all models on validation set.

    Returns:
        (results_dict, best_model_name)
    """
    print("\n" + "=" * 80)
    print("📊 STEP 4: Evaluating models on VALIDATION set...")
    print("=" * 80)

    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        results[name] = metrics

    best_model = compare_models(results)
    return results, best_model


def evaluate_on_test(models, X_test, y_test, best_model_name):
    """Evaluate all models on test set."""
    print("\n" + "=" * 80)
    print("🎯 STEP 5: Evaluating models on TEST set...")
    print("=" * 80)

    test_results = {}
    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Testing {name}...")
        print(f"{'=' * 80}")
        metrics = evaluate_model(model, X_test, y_test, name)
        test_results[name] = metrics

    print("\n" + "=" * 80)
    print("TEST SET COMPARISON")
    print("=" * 80)
    compare_models(test_results)

    print(f"\n💡 Best model from validation: {best_model_name}")
    print(f"   Test R²: {test_results[best_model_name]['r2']:.4f}")

    return test_results


def save_models(models):
    """Save trained models to disk."""
    print("\n" + "=" * 80)
    print("💾 STEP 6: Saving models...")
    print("=" * 80)

    models_dir = BASE_DIR / "trained_models"
    models_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        filename = name.replace(" ", "_").lower() + ".pkl"
        filepath = models_dir / filename
        joblib.dump(model, filepath)
        print(f"   ✅ Saved {name} to {filepath}")

    print(f"\n✅ All models saved to {models_dir}")
    print("=" * 80)


def main():
    """Main function to orchestrate the entire training pipeline."""
    print("\n" + "=" * 80)
    print("STARTING TRAINING PIPELINE FOR CONSUMER STAPLES")
    print("=" * 80)
    print(f"BASE_DIR: {BASE_DIR}")

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Prepare data
    X_train, X_val, X_test = prepare_data(X_train, X_val, X_test)

    # Print feature statistics
    print_feature_statistics(X_train)

    # 3. Train models
    models = train_models(X_train, y_train, X_val, y_val)

    # 4. Evaluate on validation set
    val_results, best_model = evaluate_on_validation(models, X_val, y_val)

    # 5. Evaluate on test set
    test_results = evaluate_on_test(models, X_test, y_test, best_model)

    # 6. Save models
    save_models(models)

    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n🏆 Best model: {best_model}")
    print(f"   Validation R²: {val_results[best_model]['r2']:.4f}")
    print(f"   Test R²:       {test_results[best_model]['r2']:.4f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
