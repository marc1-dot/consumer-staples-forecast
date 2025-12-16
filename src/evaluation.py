"""
evaluation.py
-------------
Evaluates all trained models on the test set and generates performance reports.
Computes metrics (MSE, MAE, R²) and creates visualizations.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Load preprocessed data and prepare train/test split."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Weekly_Return"])
    
    # Exclude same columns as in training
    exclude_cols = ["Weekly_Return", "Close", "Return", "Date", "Ticker"]
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df["Weekly_Return"]
    
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.ffill().bfill().fillna(0)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    
    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test


def load_models():
    """Load all trained models from disk."""
    models = {}
    model_files = ["LinearRegression.pkl", "RandomForest.pkl", "XGBoost.pkl", "NeuralNetwork.pkl"]
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace(".pkl", "")
            models[model_name] = joblib.load(model_path)
            print(f"✅ Loaded: {model_name}")
        else:
            print(f"⚠️ Missing: {model_file}")
    
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "model": model_name,
        "predictions": y_pred,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def create_comparison_table(results):
    """Create a formatted DataFrame of all results."""
    return pd.DataFrame([
        {
            "Model": r["model"],
            "MSE": round(r["MSE"], 6),
            "RMSE": round(r["RMSE"], 6),
            "MAE": round(r["MAE"], 6),
            "R²": round(r["R2"], 4)
        }
        for r in results
    ])


def plot_predictions(y_test, results):
    """Plot Actual vs Predicted returns for each model."""
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        y_pred = result["predictions"]
        model_name = result["model"]

        ax.scatter(y_test, y_pred, alpha=0.5, s=30)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f'{model_name}\nR²={result["R2"]:.4f}, RMSE={result["RMSE"]:.5f}', fontweight='bold')
        ax.set_xlabel('Actual Weekly Return')
        ax.set_ylabel('Predicted Weekly Return')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "predictions_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"📊 Saved plot: {path}")
    plt.close()


def plot_residuals(y_test, results):
    """Plot residuals (errors) for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        y_pred = result["predictions"]
        residuals = y_test.values - y_pred
        model_name = result["model"]

        ax.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        ax.set_title(f'{model_name} Residuals', fontweight='bold')
        ax.set_xlabel('Predicted Return')
        ax.set_ylabel('Residuals')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "residuals_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"📉 Saved plot: {path}")
    plt.close()


def plot_metrics_comparison(results):
    """Compare metrics (RMSE, MAE, R²) across models."""
    models = [r["model"] for r in results]
    rmse_values = [r["RMSE"] for r in results]
    mae_values = [r["MAE"] for r in results]
    r2_values = [r["R2"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE
    axes[0].bar(models, rmse_values, color='steelblue', alpha=0.7)
    axes[0].set_title("RMSE Comparison", fontweight='bold')
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis='x', rotation=45)

    # MAE
    axes[1].bar(models, mae_values, color='coral', alpha=0.7)
    axes[1].set_title("MAE Comparison", fontweight='bold')
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis='x', rotation=45)

    # R²
    axes[2].bar(models, r2_values, color='seagreen', alpha=0.7)
    axes[2].set_title("R² Score Comparison", fontweight='bold')
    axes[2].set_ylabel("R² Score")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"📈 Saved plot: {path}")
    plt.close()


def save_results_summary(df_metrics, results):
    """Save metrics and summary text report."""
    csv_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"💾 Saved metrics CSV: {csv_path}")

    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\nMODEL EVALUATION REPORT\n" + "=" * 80 + "\n\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n\nBest models:\n")
        f.write(f"  ➤ Best RMSE: {min(results, key=lambda r: r['RMSE'])['model']}\n")
        f.write(f"  ➤ Best MAE:  {min(results, key=lambda r: r['MAE'])['model']}\n")
        f.write(f"  ➤ Best R²:   {max(results, key=lambda r: r['R2'])['model']}\n")
    print(f"🧾 Saved text report: {report_path}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 STARTING MODEL EVALUATION")
    print("=" * 80)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Load models
    print("\nSTEP 2️⃣: Loading trained models...")
    models = load_models()
    if not models:
        print("❌ No models found. Run train_all.py first.")
        exit(1)

    # Evaluate
    print("\nSTEP 3️⃣: Evaluating models...\n")
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        print(f"{name:15s} → RMSE: {result['RMSE']:.5f} | MAE: {result['MAE']:.5f} | R²: {result['R2']:.4f}")

    # Summary
    df_metrics = create_comparison_table(results)
    print("\n" + "=" * 80)
    print("📊 Model Performance Summary")
    print("=" * 80)
    print(df_metrics.to_string(index=False))

    # Visuals
    print("\nSTEP 4️⃣: Generating visualizations...")
    plot_predictions(y_test, results)
    plot_residuals(y_test, results)
    plot_metrics_comparison(results)

    # Save
    print("\nSTEP 5️⃣: Saving results...")
    save_results_summary(df_metrics, results)

    print("\n🎯 Evaluation complete. Results saved in:", RESULTS_DIR)
    print("=" * 80)
