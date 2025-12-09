import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, name="Model"):
    """Evaluates model performance and prints results."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n📊 {name} Performance:")
    print(f"   - MAE  : {mae:.4f}")
    print(f"   - RMSE : {rmse:.4f}")
    print(f"   - R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_model(model, model_dir, name):
    """Saves trained model."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"💾 {name} saved at: {path}")
