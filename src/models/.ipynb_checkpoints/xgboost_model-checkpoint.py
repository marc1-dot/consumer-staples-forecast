"""
XGBoost Model
--------------
Trains a tuned XGBoost Regressor for weekly return prediction.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

from xgboost import XGBRegressor

def train_xgboost(X_train, y_train):
    """
    Train a tuned XGBoost model with conservative parameters
    for better temporal generalization.
    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,     # Lower learning rate for smoother training
        max_depth=5,             # Moderate tree depth
        subsample=0.8,           # Random subsampling of rows
        colsample_bytree=0.8,    # Random subsampling of features
        reg_lambda=1.0,          # L2 regularization
        gamma=0.1,               # Minimum loss reduction for splits
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("✅ XGBoost model trained successfully.")
    return model
