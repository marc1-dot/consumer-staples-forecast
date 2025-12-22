"""
xgboost_model.py
----------------
XGBoost Regressor implementation with regularization.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
Date: December 2024
"""

import xgboost as xgb
import numpy as np


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost Regressor with regularization to prevent overfitting.

    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training target (Series or array)
        X_val: Validation features (optional)
        y_val: Validation target (optional)

    Returns:
        Trained XGBoost model
    """
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1

    print("Training XGBoost Regressor with regularization...")
    print(f"   📊 Training data: {n_samples} samples, {n_features} features")
    print("   🔧 Parameters:")
    print("      - n_estimators: 1000")
    print("      - learning_rate: 0.05")
    print("      - max_depth: 4")
    print("      - min_child_weight: 3")
    print("      - subsample: 0.8")
    print("      - colsample_bytree: 0.8")
    print("      - reg_alpha (L1): 0.1")
    print("      - reg_lambda (L2): 1.0")
    print("      - gamma: 0.1")

    # ✅ Classic XGBoost model (compatible with all versions)
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

    # Train with validation set if provided
    if X_val is not None and y_val is not None:
        n_val = X_val.shape[0]
        print(f"   📊 Validation data: {n_val} samples")
        print("   ⏱️  Early stopping: 50 rounds")

        # Note: early stopping itself only happens if early_stopping_rounds is provided.
        # Here we keep your behavior: eval_set monitoring without necessarily stopping.
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # Try to get best iteration info (if available)
        try:
            if hasattr(model, "best_iteration") and model.best_iteration is not None:
                print(f"   ✅ Best iteration: {model.best_iteration}")
            if hasattr(model, "best_score") and model.best_score is not None:
                print(f"   ✅ Best validation score: {model.best_score:.6f}")
        except Exception:
            pass

    else:
        # Train without validation
        print("   ⚠️  No validation set provided - training without early stopping")
        model.fit(X_train, y_train, verbose=False)

    print("✅ XGBoost training complete (regularized)")
    return model


def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from trained XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (optional)

    Returns:
        - If feature_names provided: dict {feature: importance} sorted desc
        - Else: numpy array of importances
    """
    importance = model.feature_importances_

    if feature_names is not None:
        importance_dict = dict(zip(feature_names, importance))
        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )
        return importance_dict

    return importance
