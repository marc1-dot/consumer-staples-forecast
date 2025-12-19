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
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Load the preprocessed and already split datasets."""
    X_TRAIN_PATH = os.path.join(BASE_DIR, "processed", "X_train.csv")
    X_TEST_PATH = os.path.join(BASE_DIR, "processed", "X_test.csv")
    Y_TRAIN_PATH = os.path.join(BASE_DIR, "processed", "y_train.csv")
    Y_TEST_PATH = os.path.join(BASE_DIR, "processed", "y_test.csv")
    
    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError("Data not found at {}".format(X_TRAIN_PATH))
    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError("Data not found at {}".format(X_TEST_PATH))
    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError("Data not found at {}".format(Y_TRAIN_PATH))
    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError("Data not found at {}".format(Y_TEST_PATH))
    
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    
    print("Data loaded: {} train, {} test samples.".format(len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test


def clean_data(X_train, X_test, y_train, y_test):
    """Clean data: handle missing values and infinities."""
    print("\nCleaning data...")
    
    # Convert to numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    
    # Check missing values
    missing_train_before = X_train.isnull().sum().sum()
    missing_test_before = X_test.isnull().sum().sum()
    print("   Missing values before cleaning:")
    print("      X_train: {}".format(missing_train_before))
    print("      X_test: {}".format(missing_test_before))
    
    # Fill missing values
    X_train = X_train.ffill().bfill().fillna(0)
    X_test = X_test.ffill().bfill().fillna(0)
    
    # Check missing values after
    missing_train_after = X_train.isnull().sum().sum()
    missing_test_after = X_test.isnull().sum().sum()
    print("   Missing values after cleaning:")
    print("      X_train: {}".format(missing_train_after))
    print("      X_test: {}".format(missing_test_after))
    
    # Handle infinities
    if np.isinf(X_train.values).any():
        print("   Found infinite values in X_train - replacing with 0.")
        X_train = X_train.replace([np.inf, -np.inf], 0)
    
    if np.isinf(X_test.values).any():
        print("   Found infinite values in X_test - replacing with 0.")
        X_test = X_test.replace([np.inf, -np.inf], 0)
    
    print("Data cleaning complete.\n")
    
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
            print("Loaded: {}".format(model_name))
        else:
            print("Missing: {}".format(model_file))
    
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
        
        ax.set_title('{}\nR²={:.4f}, RMSE={:.5f}'.format(model_name, result["R2"], result["RMSE"]), fontweight='bold')
        ax.set_xlabel('Actual Weekly Return')
        ax.set_ylabel('Predicted Weekly Return')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "predictions_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved plot: {}".format(path))
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
        ax.set_title('{} Residuals'.format(model_name), fontweight='bold')
        ax.set_xlabel('Predicted Return')
        ax.set_ylabel('Residuals')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "residuals_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved plot: {}".format(path))
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
    print("Saved plot: {}".format(path))
    plt.close()


def save_results_summary(df_metrics, results):
    """Save metrics and summary text report."""
    csv_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print("Saved metrics CSV: {}".format(csv_path))

    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\nMODEL EVALUATION REPORT\n" + "=" * 80 + "\n\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n\nBest models:\n")
        f.write("  Best RMSE: {}\n".format(min(results, key=lambda r: r['RMSE'])['model']))
        f.write("  Best MAE:  {}\n".format(min(results, key=lambda r: r['MAE'])['model']))
        f.write("  Best R²:   {}\n".format(max(results, key=lambda r: r['R2'])['model']))
    print("Saved text report: {}".format(report_path))


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING MODEL EVALUATION")
    print("=" * 80)

    # Load data
    print("\nSTEP 1: Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Clean data
    print("\nSTEP 2: Cleaning data...")
    X_train, X_test, y_train, y_test = clean_data(X_train, X_test, y_train, y_test)

    # Load models
    print("\nSTEP 3: Loading trained models...")
    models = load_models()
    if not models:
        print("No models found. Run train_all.py first.")
        exit(1)

    # Evaluate
    print("\nSTEP 4: Evaluating models...\n")
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        print("{:15s} -> RMSE: {:.5f} | MAE: {:.5f} | R²: {:.4f}".format(name, result['RMSE'], result['MAE'], result['R2']))

    # Summary
    df_metrics = create_comparison_table(results)
    print("\n" + "=" * 80)
    print("Model Performance Summary")
    print("=" * 80)
    print(df_metrics.to_string(index=False))

    # Visuals
    print("\nSTEP 5: Generating visualizations...")
    plot_predictions(y_test, results)
    plot_residuals(y_test, results)
    plot_metrics_comparison(results)

    # Save
    print("\nSTEP 6: Saving results...")
    save_results_summary(df_metrics, results)

    print("\nEvaluation complete. Results saved in: {}".format(RESULTS_DIR))
    print("=" * 80)
