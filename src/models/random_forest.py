from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train):
    """Trains a Random Forest regression model."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest model trained successfully.")
    return model
