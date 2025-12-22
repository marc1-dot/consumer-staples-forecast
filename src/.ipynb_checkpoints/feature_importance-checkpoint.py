"""
feature_importance_analysis.py
------------------------------
Analyzes which features contribute most to model predictions.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
Date: December 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Configuration
MODEL_DIR = BASE_DIR / "trained_models"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "feature_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def analyze_feature_importance():
    """
    Analyze feature importance for tree-based models (Random Forest & XGBoost).

    Returns:
        Dictionary of {model_name: importance_dataframe}
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Load training data to get feature names
    X_train_path = DATA_DIR / "X_train.csv"

    if not X_train_path.exists():
        print(f"❌ ERROR: Training data not found at {X_train_path}")
        return None

    X_train = pd.read_csv(X_train_path)
    feature_names = X_train.columns.tolist()

    print(f"\n📊 Loaded {len(feature_names)} features from training data")

    # Models with feature importance
    models_info = {
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    # Check which models exist
    available_models = {}
    for model_name, filename in models_info.items():
        model_path = MODEL_DIR / filename
        if model_path.exists():
            available_models[model_name] = model_path
            print(f"   ✅ Found: {model_name}")
        else:
            print(f"   ⚠️  Not found: {model_name}")

    if not available_models:
        print("❌ ERROR: No tree-based models found!")
        return None

    # Create subplots
    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

    # Normalize axes to a list for consistent indexing
    if n_models == 1:
        axes = [axes]
    else:
        axes = list(axes)

    importance_results = {}

    for idx, (model_name, model_path) in enumerate(available_models.items()):
        print("\n" + "-" * 80)
        print(f"📈 Analyzing {model_name}")
        print("-" * 80)

        try:
            # Load model
            model = joblib.load(model_path)
            print("   ✅ Model loaded successfully")

            # Get feature importance
            importances = model.feature_importances_

            # Create DataFrame
            importance_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

            importance_results[model_name] = importance_df

            # Print top 10
            print("\n   🏆 Top 10 Most Important Features:")
            for i, row in enumerate(importance_df.head(10).itertuples(index=False), 1):
                print(f"      {i:2d}. {row.Feature:30s} {row.Importance:.6f}")

            # Plot
            ax = axes[idx]
            top_features = importance_df.head(15)

            bars = ax.barh(
                range(len(top_features)),
                top_features["Importance"].values,
                edgecolor="black",
                linewidth=0.5,
            )

            # Highlight top 3
            for j in range(min(3, len(bars))):
                bars[j].set_color("gold")

            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features["Feature"].values, fontsize=10)
            ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
            ax.set_title(f"{model_name}\nTop 15 Features", fontsize=14, fontweight="bold")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

            # Save to CSV
            csv_path = RESULTS_DIR / f"{model_name.replace(' ', '_')}_feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)
            print(f"   💾 Saved: {csv_path}")

        except Exception as e:
            print(f"   ❌ ERROR analyzing {model_name}: {e}")
            continue

    # Save plot
    plt.tight_layout()
    plot_path = RESULTS_DIR / "feature_importance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Feature importance plot saved: {plot_path}")
    plt.close()

    return importance_results


def analyze_correlation_with_target():
    """
    Analyze correlation between features and target variable.

    Returns:
        Series of correlations sorted by absolute value
    """
    print("\n" + "=" * 80)
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print("=" * 80)

    # Try different possible locations for the data
    possible_paths = [
        DATA_DIR / "features_with_target.csv",
        BASE_DIR / "data" / "processed" / "full_dataset.csv",
        BASE_DIR / "data" / "processed" / "combined_data.csv",
    ]

    df = None
    for path in possible_paths:
        if path.exists():
            print(f"\n📂 Loading data from: {path}")
            df = pd.read_csv(path)
            break

    # If not found, combine train and validation data
    if df is None:
        print("\n📂 Combining training and validation data...")
        X_train = pd.read_csv(DATA_DIR / "X_train.csv")
        y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
        X_val = pd.read_csv(DATA_DIR / "X_val.csv")
        y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

        X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_val], axis=0, ignore_index=True)

        df = X_combined.copy()
        df["Target"] = y_combined.values

    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"   Total columns: {len(df.columns)}")

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()

    print(f"   Numeric columns: {len(numeric_cols)}")

    # Check for non-numeric columns
    non_numeric = set(df.columns) - set(numeric_cols)
    if non_numeric:
        print(f"   ⚠️  Non-numeric columns excluded: {non_numeric}")

    # Ensure Target exists
    if "Target" not in df_numeric.columns:
        target_candidates = [
            col for col in df_numeric.columns
            if ("target" in col.lower()) or ("return" in col.lower())
        ]
        if target_candidates:
            chosen = target_candidates[0]
            print(f"   ⚠️  'Target' not found, using '{chosen}' instead")
            df_numeric = df_numeric.rename(columns={chosen: "Target"})
        else:
            print("❌ ERROR: Target column not found in dataset")
            return None

    # Calculate correlations
    print("\n🔍 Calculating correlations...")
    correlations = df_numeric.corr(numeric_only=True)["Target"].sort_values(ascending=False)

    # Remove Target itself + NaNs
    correlations = correlations.drop("Target", errors="ignore").dropna()

    print(f"   ✅ Calculated correlations for {len(correlations)} features")

    # Print top positive correlations
    print("\n🔼 Top 10 Positive Correlations with Target:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"   {i:2d}. {feature:40s} {corr:>8.4f}")

    # Print top negative correlations
    print("\n🔽 Top 10 Negative Correlations with Target:")
    for i, (feature, corr) in enumerate(correlations.tail(10).items(), 1):
        print(f"   {i:2d}. {feature:40s} {corr:>8.4f}")

    # Plot
    plt.figure(figsize=(12, 10))

    top_positive = correlations.head(10)
    top_negative = correlations.tail(10)
    top_corr = pd.concat([top_positive, top_negative])

    colors = ["green" if x > 0 else "red" for x in top_corr.values]

    plt.barh(
        range(len(top_corr)),
        top_corr.values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    plt.yticks(range(len(top_corr)), top_corr.index, fontsize=10)
    plt.xlabel("Correlation with Target", fontsize=12, fontweight="bold")
    plt.title(
        "Features Most Correlated with Target Returns\n(Top 10 Positive & Negative)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "feature_correlation_with_target.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Correlation plot saved: {plot_path}")
    plt.close()

    csv_path = RESULTS_DIR / "feature_correlations.csv"
    correlations.to_csv(csv_path, header=["Correlation"])
    print(f"✅ Correlation data saved: {csv_path}")

    return correlations


def plot_correlation_heatmap():
    """
    Create a heatmap of correlations between top features.
    """
    print("\n" + "=" * 80)
    print("CORRELATION HEATMAP (TOP FEATURES)")
    print("=" * 80)

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

    df = X_train.copy()
    df["Target"] = y_train.values

    df_numeric = df.select_dtypes(include=[np.number])

    correlations = df_numeric.corr(numeric_only=True)["Target"].abs().sort_values(ascending=False)
    top_features = correlations.head(21).index.tolist()  # 20 + Target

    corr_matrix = df_numeric[top_features].corr(numeric_only=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Correlation Heatmap - Top 20 Features + Target",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    plot_path = RESULTS_DIR / "correlation_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Heatmap saved: {plot_path}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")

    # 1. Feature importance
    importance_results = analyze_feature_importance()

    # 2. Correlation analysis
    correlations = analyze_correlation_with_target()

    # 3. Correlation heatmap
    try:
        plot_correlation_heatmap()
    except Exception as e:
        print(f"⚠️  Could not create heatmap: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("✅✅✅ FEATURE ANALYSIS COMPLETE ✅✅✅")
    print("=" * 80)

    if importance_results:
        print("\n📊 Feature Importance Analysis:")
        print(f"   - Models analyzed: {len(importance_results)}")
        for model_name in importance_results.keys():
            print(f"   - {model_name}: ✅")

    if correlations is not None and len(correlations) > 0:
        print("\n📊 Correlation Analysis:")
        print(f"   - Features analyzed: {len(correlations)}")
        print(f"   - Highest correlation: {correlations.iloc[0]:.4f} ({correlations.index[0]})")
        print(f"   - Lowest correlation: {correlations.iloc[-1]:.4f} ({correlations.index[-1]})")

    print(f"\n📁 All results saved to: {RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
