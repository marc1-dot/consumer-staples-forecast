"""
Random Forest Model
-------------------
Trains a tuned Random Forest Regressor for weekly return prediction.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model with moderate regularization
    to prevent overfitting and improve stability.
    """
    model = RandomForestRegressor(
        n_estimators=400,        # Slightly higher number of trees for stability
        max_depth=10,            # Limit tree depth to prevent overfitting
        min_samples_split=5,     # Require more samples to split nodes
        min_samples_leaf=3,      # Enforce a minimum leaf size
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )

    model.fit(X_train, y_train)
    print("✅ Random Forest model trained successfully.")
    return model
