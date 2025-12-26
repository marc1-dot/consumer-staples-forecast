"""
xgboost_model.py
----------------
XGBoost Regressor implementation with regularization and early stopping.
FIX: Compatible with newer XGBoost versions (early_stopping_rounds in constructor).

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import xgboost as xgb
import numpy as np


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost Regressor with regularization to prevent overfitting.
    Includes Early Stopping.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (Required for early stopping)
        y_val: Validation target (Required for early stopping)

    Returns:
        Trained XGBoost model
    """
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1

    print("Training XGBoost Regressor with regularization...")
    print(f"   📊 Training data: {n_samples} samples, {n_features} features")

    # ==========================================================================
    # FIX FOR NEWER XGBOOST VERSIONS:
    # 'early_stopping_rounds' is now defined here, not in .fit()
    # ==========================================================================
    early_stop = 50 if (X_val is not None and y_val is not None) else None

    # Model Configuration
    model = xgb.XGBRegressor(
        n_estimators=1000,        # Maximum number of trees
        learning_rate=0.05,       # Slower learning for better generalization
        max_depth=4,              # Shallow trees to prevent overfitting
        min_child_weight=3,       # High weight to avoid isolating noise
        subsample=0.8,            # Use 80% of data per tree
        colsample_bytree=0.8,     # Use 80% of features per tree
        reg_alpha=0.1,            # L1 Regularization (Lasso)
        reg_lambda=1.0,           # L2 Regularization (Ridge)
        gamma=0.1,                # Minimum loss reduction required
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stop  # ✅ MOVED HERE (Constructor)
    )

    # Train with validation set if provided
    if X_val is not None and y_val is not None:
        n_val = X_val.shape[0]
        print(f"   📊 Validation data: {n_val} samples")
        print("   ⏱️  Early stopping enabled: Stops if no improvement for 50 rounds")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
            # ❌ REMOVED: early_stopping_rounds=50 (caused the error)
        )

        # Retrieve performance metrics
        try:
            if hasattr(model, "best_iteration") and model.best_iteration is not None:
                print(f"   ✅ Best iteration: {model.best_iteration}")
            if hasattr(model, "best_score") and model.best_score is not None:
                print(f"   ✅ Best validation score: {model.best_score:.6f}")
        except Exception:
            pass

    else:
        # Train without validation (Not recommended for this project)
        print("   ⚠️  No validation set provided - training without early stopping")
        model.fit(X_train, y_train, verbose=False)

    print("✅ XGBoost training complete (regularized)")
    return model


def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from trained XGBoost model.
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
