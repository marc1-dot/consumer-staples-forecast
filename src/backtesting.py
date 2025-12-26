"""
backtesting.py
--------------
Financial Simulation Script.
RESPONSIBILITY: Simulate trading strategies based on model predictions.
SCENARIOS: 0.0%, 0.1%, and 0.5% transaction costs.
OUTPUTS: Console logs, Equity Curve plot (PNG), Summary Tables (PNG).

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

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Configuration
MODEL_DIR = BASE_DIR / "trained_models"
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "backtesting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")

def load_data_and_models():
    """Load cleaned test data and trained models."""
    print("📂 Loading data and models...")
    
    # 1. Load Data
    try:
        X_test = pd.read_csv(DATA_DIR / "X_test_clean.csv")
        y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze() # Return series
    except FileNotFoundError:
        print("❌ Error: Cleaned data not found. Run test_all.py first.")
        sys.exit(1)

    # 2. Load Models
    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "Neural Network": "neuralnetwork.pkl",
    }
    
    models = {}
    for name, filename in model_files.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    
    return X_test, y_test, models

def calculate_metrics(equity_curve):
    """Calculate financial metrics from an equity curve."""
    # Convert equity curve to daily returns for metrics
    daily_returns = equity_curve.pct_change().dropna()
    
    total_return = (equity_curve.iloc[-1] - 1.0) * 100
    
    # Sharpe Ratio (Assuming 252 trading days, risk-free rate = 0 for simplicity)
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret != 0 else 0
    
    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    return total_return, sharpe, max_drawdown

def run_strategy(y_true, y_pred, transaction_cost=0.0):
    """
    Simulate a Long/Short Strategy.
    - If pred > 0: Long (+1)
    - If pred < 0: Short (-1)
    """
    # 1. Generate Signals (-1, 0, 1)
    signals = np.sign(y_pred)
    signals[signals == 0] = 0 # Neutral if exact 0
    
    # 2. Calculate Gross Returns (Strategy matches direction of market?)
    gross_returns = signals * y_true
    
    # 3. Calculate Transaction Costs
    previous_signals = pd.Series(signals).shift(1).fillna(0)
    turnover = np.abs(signals - previous_signals)
    costs = turnover * transaction_cost
    
    # 4. Net Returns
    net_returns = gross_returns - costs
    
    # 5. Build Equity Curve (Start at 1.0)
    equity_curve = (1 + net_returns).cumprod()
    
    return equity_curve

def print_scenario_table(results, cost_label):
    """Pretty print a table for a specific cost scenario and return DF."""
    df = pd.DataFrame(results).T
    df = df.sort_values("Total Return (%)", ascending=False)
    
    print(f"\n💰 SCENARIO: {cost_label}")
    print("-" * 85)
    print(f"{'Strategy':<20} | {'Total Return':<15} | {'Sharpe Ratio':<12} | {'Max Drawdown':<12}")
    print("-" * 85)
    
    for strategy, row in df.iterrows():
        print(f"{strategy:<20} | {row['Total Return (%)']:>11.2f}%    | {row['Sharpe Ratio']:>10.2f}   | {row['Max Drawdown (%)']:>10.2f}%")
    print("-" * 85)
    return df

def save_tables_as_image(all_scenarios_data):
    """
    Generates a single PNG image containing the tables for all cost scenarios.
    
    Args:
        all_scenarios_data (list): List of tuples (cost_label, dataframe)
    """
    print(f"\n🖼️  Generating summary tables image...")
    
    num_scenarios = len(all_scenarios_data)
    # Figure setup: vertical layout
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(10, 3 * num_scenarios))
    
    if num_scenarios == 1: axes = [axes] # Handle single case
    
    for i, (cost_label, df) in enumerate(all_scenarios_data):
        ax = axes[i]
        ax.axis('off') # Hide axis
        
        # Add Title
        ax.set_title(f"SCENARIO: {cost_label}", fontsize=12, fontweight='bold', loc='center', pad=10)
        
        # Format Data for Display (add % signs and round)
        display_df = df.copy()
        display_df['Total Return (%)'] = display_df['Total Return (%)'].apply(lambda x: f"{x:.2f}%")
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].apply(lambda x: f"{x:.2f}%")
        
        # Reset index to make 'Strategy' a column
        display_df = display_df.reset_index().rename(columns={'index': 'Strategy'})
        
        # Create Table
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc='center',
            loc='center',
            colColours=["#f2f2f2"] * len(display_df.columns) # Light gray header
        )
        
        # Styling
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5) # Increase row height
        
        # Make header bold (optional manual tweak, hard in matplotlib simple table)
        
    plt.tight_layout()
    output_path = OUTPUT_DIR / "backtest_summary_tables.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved tables to: {output_path}")

def main():
    print("\n" + "="*85)
    print("🚀 BACKTESTING SIMULATION")
    print("="*85)

    X_test, y_test, models = load_data_and_models()
    
    # Define Buy & Hold Baseline
    bnh_equity = (1 + y_test).cumprod()
    bnh_tot, bnh_sharpe, bnh_dd = calculate_metrics(bnh_equity)
    
    baseline_metrics = {
        "Total Return (%)": bnh_tot,
        "Sharpe Ratio": bnh_sharpe,
        "Max Drawdown (%)": bnh_dd
    }

    costs = [0.0, 0.001, 0.005] # 0%, 0.1%, 0.5%
    equity_curves_mid_cost = {} # To store 0.1% curves for plotting
    
    # Store dataframes for the image generation
    all_scenarios_data = []

    for cost in costs:
        cost_label = f"Transaction Cost: {cost*100:.1f}%"
        scenario_results = {}
        
        # Add Buy & Hold
        scenario_results["Buy & Hold"] = baseline_metrics

        for name, model in models.items():
            y_pred = model.predict(X_test)
            equity = run_strategy(y_test, y_pred, transaction_cost=cost)
            
            tot, sharpe, dd = calculate_metrics(equity)
            scenario_results[name] = {
                "Total Return (%)": tot,
                "Sharpe Ratio": sharpe,
                "Max Drawdown (%)": dd
            }
            
            # Save for plotting (only for the realistic 0.1% case)
            if cost == 0.001:
                equity_curves_mid_cost[name] = equity

        # Print Table and collect DF
        df_results = print_scenario_table(scenario_results, cost_label)
        all_scenarios_data.append((cost_label, df_results))

    # --- SAVE TABLES IMAGE ---
    save_tables_as_image(all_scenarios_data)

    # --- PLOTTING (Based on 0.1% Cost) ---
    print(f"\n📈 Generating equity curve plot (Cost = 0.1%)...")
    plt.figure(figsize=(12, 7))
    
    # Plot Buy & Hold
    plt.plot(bnh_equity.values, label="Buy & Hold", color="black", linestyle="--", linewidth=2, alpha=0.7)
    
    # Plot Models
    for name, equity in equity_curves_mid_cost.items():
        plt.plot(equity.values, label=name, linewidth=1.5)

    plt.title("Cumulative Performance (Transaction Cost = 0.1%)", fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Value (Start = 1.0)")
    plt.xlabel("Test Set Period (Days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = OUTPUT_DIR / "equity_curves_0.1pct_cost.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
