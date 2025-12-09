from xgboost import XGBRegressor

def train_xgboost(X_train, y_train):
    """Trains an XGBoost regression model."""
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost model trained successfully.")
    return model
