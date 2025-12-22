"""
test_all.py
-----------
Evaluates all trained models on the test set and generates comparison visualizations.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
Date: December 2024
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Configuration
MODEL_DIR = BASE_DIR / "trained_models"
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_test_data():
    """
    Load the test set (10% of data - never seen during training/validation).

    Returns:
        X_test: Test features
        y_test: Test target
    """
    print("\n📂 Loading test data...")

    X_TEST_PATH = DATA_DIR / "X_test.csv"
    Y_TEST_PATH = DATA_DIR / "y_test.csv"

    # Check if files exist
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"❌ Test data not found at {X_TEST_PATH}")
    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(f"❌ Test data not found at {Y_TEST_PATH}")

    # Load data
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    # Clean data
    print("   🧹 Cleaning test data...")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.ffill().bfill().fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # ✅ FIX: proper Python shape indexing
    print(f"   ✅ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print("   📌 This represents 10% of total data - held out for final evaluation")

    return X_test, y_test


def load_models():
    """
    Load all trained models from disk.

    Returns:
        Dictionary of {model_name: model}
    """
    # Model filenames (as saved by train_all.py)
    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "Neural Network": "neuralnetwork.pkl",
    }

    models = {}

    print("\n📦 Loading trained models...")
    for display_name, filename in model_files.items():
        model_path = MODEL_DIR / filename
        if model_path.exists():
            try:
                models[display_name] = joblib.load(model_path)
                print(f"   ✅ Loaded: {display_name}")
            except Exception as e:
                print(f"   ⚠️  Failed to load {display_name}: {e}")
        else:
            print(f"   ⚠️  Not found: {display_name} ({model_path})")

    if not models:
        raise FileNotFoundError(f"❌ No trained models found in {MODEL_DIR}")

    print(f"\n   📊 Total models loaded: {len(models)}")
    return models


def evaluate_one_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model on test set.

    Returns:
        results: Dictionary of metrics
        y_pred: Predictions
    """
    print(f"\n   Evaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results = {
        "Model": model_name,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
    }

    return results, y_pred


def plot_predictions(y_test, predictions, model_names):
    """Plot actual vs predicted values for all models."""
    n_models = len(model_names)

    # Determine grid size
    if n_models == 4:
        nrows, ncols = 2, 2
    elif n_models == 3:
        nrows, ncols = 2, 2
    elif n_models == 2:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 1, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))

    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, name in enumerate(model_names):
        if idx >= len(axes):
            break

        ax = axes[idx]
        y_pred = predictions[name]

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors="k", linewidths=0.5)

        min_val = min(y_test.min(), np.min(y_pred))
        max_val = max(y_test.max(), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

        r2 = r2_score(y_test, y_pred)

        ax.set_xlabel("Actual Weekly Return", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted Weekly Return", fontsize=12, fontweight="bold")
        ax.set_title(f"{name}\n(Test Set R² = {r2:.4f})", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "predictions_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_residuals(y_test, predictions, model_names):
    """Plot residual distributions for all models."""
    n_models = len(model_names)

    # Determine grid size
    if n_models == 4:
        nrows, ncols = 2, 2
    elif n_models == 3:
        nrows, ncols = 2, 2
    elif n_models == 2:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 1, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))

    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, name in enumerate(model_names):
        if idx >= len(axes):
            break

        ax = axes[idx]
        y_pred = predictions[name]

        residuals = y_test - y_pred

        ax.hist(residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")

        mean_residual = residuals.mean()
        std_residual = residuals.std()

        ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{name}\n(Mean: {mean_residual:.6f}, Std: {std_residual:.6f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "residuals_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(results_df):
    """Plot comparison of all metrics across models."""
    metrics = ["MSE", "MAE", "RMSE", "R²"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        sorted_df = results_df.sort_values(by=metric, ascending=(metric != "R²"))

        bars = ax.bar(
            sorted_df["Model"],
            sorted_df[metric],
            color="steelblue",
            edgecolor="black",
            linewidth=1.5,
        )

        if metric == "R²":
            best_idx = sorted_df[metric].idxmax()
        else:
            best_idx = sorted_df[metric].idxmin()

        bars[list(sorted_df.index).index(best_idx)].set_color("gold")

        ax.set_ylabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(f"{metric} Comparison (Test Set)", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION ON TEST SET (10% of data)")
    print("=" * 80)
    print(f"BASE_DIR: {BASE_DIR}")

    # Load test data
    print("\n" + "=" * 80)
    print("📊 STEP 1: Loading test data")
    print("=" * 80)
    X_test, y_test = load_test_data()

    # Load models
    print("\n" + "=" * 80)
    print("📦 STEP 2: Loading trained models")
    print("=" * 80)
    models = load_models()

    # Evaluate all models
    print("\n" + "=" * 80)
    print("🔍 STEP 3: Evaluating models on test set")
    print("=" * 80)

    results = []
    predictions = {}

    for name, model in models.items():
        result, y_pred = evaluate_one_model(model, X_test, y_test, name)
        results.append(result)
        predictions[name] = y_pred

        print(f"\n   {name}:")
        print(f"      MSE:  {result['MSE']:.6f}")
        print(f"      MAE:  {result['MAE']:.6f}")
        print(f"      RMSE: {result['RMSE']:.6f}")
        print(f"      R²:   {result['R²']:.4f}")

    results_df = pd.DataFrame(results)

    # Find best model
    print("\n" + "=" * 80)
    best_model = results_df.loc[results_df["R²"].idxmax(), "Model"]
    best_r2 = results_df["R²"].max()
    print(f"🏆 BEST MODEL ON TEST SET: {best_model}")
    print(f"   R² Score: {best_r2:.4f}")
    print("=" * 80)

    # Save results
    print("\n" + "=" * 80)
    print("💾 STEP 4: Saving results")
    print("=" * 80)
    results_path = OUTPUT_DIR / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   ✅ Saved: {results_path}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("📈 STEP 5: Generating visualizations")
    print("=" * 80)
    plot_predictions(y_test, predictions, list(models.keys()))
    plot_residuals(y_test, predictions, list(models.keys()))
    plot_metrics_comparison(results_df)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📊 Summary:")
    print(f"   - Test samples:     {len(X_test):,} (10% of total data)")
    print(f"   - Models evaluated: {len(models)}")
    print(f"   - Best model:       {best_model}")
    print(f"   - Best R² score:    {best_r2:.4f}")
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")

    print("\n📋 Complete Results Table:")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
