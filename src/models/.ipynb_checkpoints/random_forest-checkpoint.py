"""
random_forest.py
----------------
Random Forest regression model with regularization for better generalization.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np


def train_random_forest(X_train, y_train):
    """
    Train Random Forest regression model with strong regularization.

    Regularization techniques applied:
    - max_depth: Limits tree depth to prevent overfitting
    - min_samples_split: Requires more samples to split nodes
    - min_samples_leaf: Requires more samples in leaf nodes
    - max_features: Uses only sqrt(n_features) per split
    - max_samples: Bootstrap samples on 70% of data

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained RandomForestRegressor model
    """
    print("Training Random Forest Regressor with regularization...")

    # ✅ FIX: correct Python indexing for shape
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    # Adaptive parameters
    min_samples_split = max(20, int(n_samples * 0.01))   # At least 1% of data
    min_samples_leaf = max(10, int(n_samples * 0.005))   # At least 0.5% of data

    print(f"   📊 Training data: {n_samples} samples, {n_features} features")
    print("   🔧 Parameters:")
    print("      - n_estimators: 150")
    print("      - max_depth: 6")
    print(f"      - min_samples_split: {min_samples_split}")
    print(f"      - min_samples_leaf: {min_samples_leaf}")
    print(f"      - max_features: sqrt ({int(np.sqrt(n_features))} features per split)")
    print("      - max_samples: 0.7 (70% bootstrap)")

    model = RandomForestRegressor(
        n_estimators=150,                 # ✅ More trees for stability
        max_depth=6,                      # ✅ Limit depth to prevent overfitting
        min_samples_split=min_samples_split,  # ✅ Adaptive regularization
        min_samples_leaf=min_samples_leaf,    # ✅ Adaptive regularization
        max_features="sqrt",              # ✅ Use sqrt(n_features) per split
        max_samples=0.7,                  # ✅ Bootstrap on 70% of data
        random_state=42,
        n_jobs=-1,
        verbose=0,
        oob_score=True                    # ✅ Out-of-bag score for validation
    )

    model.fit(X_train, y_train)

    # Display OOB score
    if hasattr(model, "oob_score_"):
        print(f"   ✅ Out-of-Bag R² score: {model.oob_score_:.4f}")

    print("✅ Random Forest training complete (regularized)")

    return model
