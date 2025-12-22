"""
model_evaluate.py
-----------------
Evaluation functions for trained models.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
Date: December 2025
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(model, X, y, model_name="Model"):
    """
    Evaluate a trained model on given data.

    Returns:
        dict: {'mse': ..., 'mae': ..., 'rmse': ..., 'r2': ...}
    """
    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Display results
    print(f"\n{model_name}:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R²:   {r2:.4f}")

    # Return metrics as dictionary
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def compare_models(results):
    """
    Compare multiple models based on their evaluation results.

    Args:
        results: dict like {'LinearRegression': {'r2': 0.85, 'rmse': ...}, ...}

    Returns:
        str: Name of the best model (based on R² score)
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Sort models by R² score (descending)
    sorted_models = sorted(results.items(), key=lambda item: item[1]["r2"], reverse=True)

    # Print header
    print(f"\n{'Model':<25} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'MSE':>12}")
    print("-" * 80)

    # Print each model's metrics
    for model_name, metrics in sorted_models:
        print(
            f"{model_name:<25} "
            f"{metrics['r2']:>10.4f} "
            f"{metrics['rmse']:>12.6f} "
            f"{metrics['mae']:>12.6f} "
            f"{metrics['mse']:>12.6f}"
        )

    print("-" * 80)

    # Identify best model
    best_model_name = sorted_models[0][0]
    best_r2 = sorted_models[0][1]["r2"]

    print(f"\n🏆 Best model: {best_model_name} (R² = {best_r2:.4f})")
    print("=" * 80)

    return best_model_name


def evaluate_on_test_set(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model on test set with detailed output.

    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION - {model_name}")
    print(f"{'='*80}")

    metrics = evaluate_model(model, X_test, y_test, model_name)

    print(f"{'='*80}\n")
    return metrics
