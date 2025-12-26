"""
test_all.py
-----------
Evaluation & Visualization Script.
RESPONSIBILITY: Load trained models, Load CLEANED test data, Report results.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

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


def load_clean_test_data():
    """
    Load the CLEANED test set saved by train_all.py.
    This ensures we use the exact same feature processing (means, etc.).
    """
    print("\n📂 Loading CLEANED test data...")

    # 👇 C'EST ICI LA MODIFICATION IMPORTANTE : on charge le _clean.csv
    X_TEST_PATH = DATA_DIR / "X_test_clean.csv"
    Y_TEST_PATH = DATA_DIR / "y_test.csv"

    # Vérification de sécurité
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(
            f"❌ Cleaned test data not found at {X_TEST_PATH}.\n"
            "👉 Please run 'train_all.py' first to generate cleaned datasets."
        )

    # Chargement direct (plus besoin de fillna ici, c'est déjà fait !)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print(f"   ✅ Loaded {len(X_test)} samples (Pre-cleaned by training script)")
    return X_test, y_test


def load_models():
    """Load all trained models from disk."""
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
            print(f"   ⚠️  Not found: {display_name}")

    if not models:
        raise FileNotFoundError(f"❌ No trained models found in {MODEL_DIR}")

    return models


def evaluate_one_model(model, X_test, y_test, model_name):
    """Evaluate a single model on test set."""
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
    
    # Grid setup
    if n_models == 4: nrows, ncols = 2, 2
    elif n_models == 3: nrows, ncols = 2, 2
    elif n_models == 2: nrows, ncols = 1, 2
    else: nrows, ncols = 1, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    if n_models == 1: axes = [axes]
    else: axes = axes.flatten()

    for idx, name in enumerate(model_names):
        if idx >= len(axes): break
        ax = axes[idx]
        y_pred = predictions[name]

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors="k", linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), np.min(y_pred))
        max_val = max(y_test.max(), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

        r2 = r2_score(y_test, y_pred)
        ax.set_title(f"{name}\n(Test Set R² = {r2:.4f})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "predictions_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved plot: {output_path}")
    plt.close()


def plot_residuals(y_test, predictions, model_names):
    """Plot residual distributions."""
    n_models = len(model_names)
    
    if n_models == 4: nrows, ncols = 2, 2
    elif n_models == 3: nrows, ncols = 2, 2
    elif n_models == 2: nrows, ncols = 1, 2
    else: nrows, ncols = 1, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    if n_models == 1: axes = [axes]
    else: axes = axes.flatten()

    for idx, name in enumerate(model_names):
        if idx >= len(axes): break
        ax = axes[idx]
        residuals = y_test - predictions[name]

        ax.hist(residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
        
        ax.set_title(f"{name} Residuals", fontsize=14, fontweight="bold")
        ax.set_xlabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "residuals_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved plot: {output_path}")
    plt.close()


def plot_metrics_comparison(results_df):
    """Bar chart comparison of R2 scores."""
    plt.figure(figsize=(10, 6))
    
    # Sort by R2 descending
    sorted_df = results_df.sort_values("R²", ascending=False)
    
    sns.barplot(data=sorted_df, x="Model", y="R²", palette="viridis", edgecolor="black")
    
    plt.title("Model Comparison - R² Score (Test Set)", fontsize=16, fontweight="bold")
    plt.ylabel("R² Score")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    output_path = OUTPUT_DIR / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved plot: {output_path}")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("🎯 TEST SET EVALUATION REPORT")
    print("=" * 80)

    # 1. Load CLEAN test data
    try:
        X_test, y_test = load_clean_test_data()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Load Models
    try:
        models = load_models()
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Evaluate models
    print("\n" + "=" * 80)
    print("🔍 STEP 3: Computing Metrics on Test Set")
    print("=" * 80)

    results = []
    predictions = {}

    for name, model in models.items():
        res, y_pred = evaluate_one_model(model, X_test, y_test, name)
        results.append(res)
        predictions[name] = y_pred

        print(f"\n   {name}:")
        print(f"      R²:   {res['R²']:.4f}")
        print(f"      RMSE: {res['RMSE']:.6f}")

    results_df = pd.DataFrame(results)

    # 4. Save CSV Results
    csv_path = OUTPUT_DIR / "test_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved results table to: {csv_path}")

    # 5. Generate Plots
    print("\n📈 Generating visualizations...")
    plot_predictions(y_test, predictions, list(models.keys()))
    plot_residuals(y_test, predictions, list(models.keys()))
    plot_metrics_comparison(results_df)

    # Final Summary
    best_model_row = results_df.loc[results_df['R²'].idxmax()]
    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL: {best_model_row['Model']}")
    print(f"   R² Score: {best_model_row['R²']:.4f}")
    print("=" * 80)
    print(f"\n✅ Evaluation complete. Check {OUTPUT_DIR} for graphs.")


if __name__ == "__main__":
    main()
