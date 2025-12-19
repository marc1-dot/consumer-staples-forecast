"""
Feature Importance Analysis
---------------------------
Analyzes which features contribute most to model predictions.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import pandas as pd
import numpy as np
import joblib  # ← Changé de pickle à joblib
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance():
    """Analyze feature importance for tree-based models."""
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load data
    X_train = pd.read_csv('/files/consumer-staples-forecast/processed/X_train.csv')
    feature_names = X_train.columns.tolist()
    
    # Models with feature importance
    models = {
        'RandomForest': '/files/consumer-staples-forecast/models/trained/RandomForest.pkl',
        'XGBoost': '/files/consumer-staples-forecast/models/trained/XGBoost.pkl'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (model_name, model_path) in enumerate(models.items()):
        print(f"\n{model_name} Feature Importance:")
        print("-" * 50)
        
        # Load model avec joblib
        model = joblib.load(model_path)
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Print top 10
        print("\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']:30s} {row['Importance']:.4f}")
        
        # Plot
        ax = axes[idx]
        top_features = importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['Importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top 15 Features')
        ax.invert_yaxis()
        
        # Save to CSV
        importance_df.to_csv(
            f'/files/consumer-staples-forecast/results/{model_name}_feature_importance.csv',
            index=False
        )
    
    plt.tight_layout()
    plt.savefig('/files/consumer-staples-forecast/results/feature_importance.png', 
                dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved")
    
    return importance_df


def analyze_correlation_with_target():
    """Analyze correlation between features and target."""
    
    print("\n" + "="*70)
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print("="*70)
    
    # Load full dataset
    df = pd.read_csv('/files/consumer-staples-forecast/processed/features_with_target.csv')
    
    # ✅ CORRECTION : Exclure les colonnes non-numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols]
    
    print(f"\nTotal features: {len(df.columns)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Non-numeric columns excluded: {set(df.columns) - set(numeric_cols)}")
    
    # Vérifier que Target existe
    if 'Target' not in df_numeric.columns:
        print("❌ ERROR: 'Target' column not found in numeric columns")
        return None
    
    # Calculate correlations
    correlations = df_numeric.corr()['Target'].sort_values(ascending=False)
    
    # Enlever Target lui-même de la liste
    correlations = correlations.drop('Target')
    
    print("\nTop 10 Positive Correlations:")
    for feature, corr in correlations.head(10).items():
        print(f"  {feature:30s} {corr:>8.4f}")
    
    print("\nTop 10 Negative Correlations:")
    for feature, corr in correlations.tail(10).items():
        print(f"  {feature:30s} {corr:>8.4f}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_corr = pd.concat([correlations.head(10), correlations.tail(10)])
    colors = ['green' if x > 0 else 'red' for x in top_corr.values]
    plt.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_corr)), top_corr.index)
    plt.xlabel('Correlation with Target')
    plt.title('Features Most Correlated with Target Returns')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/files/consumer-staples-forecast/results/feature_correlation.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Correlation plot saved")
    
    # Save to CSV
    correlations.to_csv(
        '/files/consumer-staples-forecast/results/feature_correlations.csv',
        header=['Correlation']
    )
    print("✓ Correlation data saved to CSV")
    
    return correlations


if __name__ == "__main__":
    analyze_feature_importance()
    analyze_correlation_with_target()
    print("\n" + "="*70)
    print("✓✓✓ FEATURE ANALYSIS COMPLETE ✓✓✓")
    print("="*70 + "\n")