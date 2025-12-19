"""
train_all.py
-------------
Trains all models (LinearRegression, RandomForest, XGBoost, NeuralNetwork)
on the Consumer Staples dataset with advanced features.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
import numpy as np
import joblib

from models.linear_model import train_linear_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.neural_network import train_neural_network


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "trained")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_data():
    """Load the preprocessed and already split datasets."""
    X_TRAIN_PATH = os.path.join(BASE_DIR, "processed", "X_train.csv")
    X_TEST_PATH = os.path.join(BASE_DIR, "processed", "X_test.csv")
    Y_TRAIN_PATH = os.path.join(BASE_DIR, "processed", "y_train.csv")
    Y_TEST_PATH = os.path.join(BASE_DIR, "processed", "y_test.csv")

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(f"Data not found at {X_TRAIN_PATH}")
    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(f"Data not found at {X_TEST_PATH}")
    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(f"Data not found at {Y_TRAIN_PATH}")
    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError(f"Data not found at {Y_TEST_PATH}")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print("Data loaded:")
    print(f"   X_train: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
    print(f"   X_test:  {X_test.shape[0]} rows, {X_test.shape[1]} columns")
    # y_train / y_test are Series -> shape is (n_samples,)
    print(f"   y_train: {y_train.shape[0]} samples")
    print(f"   y_test:  {y_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def prepare_data(X_train, X_test, y_train, y_test):
    """Prepare features and target for model training."""
    print("\nPreparing data for training...")
    print("Initial columns: {}".format(list(X_train.columns)))

    # Ensure numeric
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    print("Checking for missing values...")
    missing_before = int(X_train.isnull().sum().sum())
    print("   Missing before cleaning: {}".format(missing_before))

    # Fill missing values
    X_train = X_train.ffill().bfill().fillna(0)
    X_test = X_test.ffill().bfill().fillna(0)

    missing_after = int(X_train.isnull().sum().sum())
    print("   Missing after cleaning: {}".format(missing_after))

    # Replace inf values
    if np.isinf(X_train).any().any():
        print("Found infinite values in X_train - replacing with 0.")
        X_train = X_train.replace([np.inf, -np.inf], 0)

    if np.isinf(X_test).any().any():
        print("Found infinite values in X_test - replacing with 0.")
        X_test = X_test.replace([np.inf, -np.inf], 0)

    print("\nData preparation complete.")
    print(f"   Training set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"   Test set:     {X_test.shape[0]} rows")

    return X_train, X_test, y_train, y_test


def display_feature_statistics(X_train):
    """Display basic statistics of the training features."""
    print("\nFeature Statistics (Training Set):")
    print("=" * 80)
    stats = X_train.describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING TRAINING PIPELINE FOR CONSUMER STAPLES")
    print("=" * 80)

    print("\nSTEP 1: Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("\nSTEP 2: Preparing features and target...")
    X_train, X_test, y_train, y_test = prepare_data(X_train, X_test, y_train, y_test)

    display_feature_statistics(X_train)

    print("\n" + "=" * 80)
    print("STEP 3: Training models...")
    print("=" * 80)

    models = {}

    print("\n[1/4] Training Linear Regression...")
    models["LinearRegression"] = train_linear_regression(X_train, y_train)

    print("\n[2/4] Training Random Forest...")
    models["RandomForest"] = train_random_forest(X_train, y_train)

    print("\n[3/4] Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    print("\n[4/4] Training Neural Network...")
    models["NeuralNetwork"] = train_neural_network(X_train, y_train)

    print("\n" + "=" * 80)
    print("STEP 4: Saving trained models...")
    print("=" * 80)

    for name, model in models.items():
        model_path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print("Saved: {:20s} -> {}".format(name, model_path))

    print("\n" + "=" * 80)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 80)
    print("\nModels saved in: {}".format(MODEL_SAVE_DIR))
    print("Training samples: {}".format(len(X_train)))
    print(f"Features used: {X_train.shape[1]}")
    print("\nNext step: Run evaluation.py to test model performance.\n")
