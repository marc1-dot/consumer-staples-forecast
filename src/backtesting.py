"""
backtesting.py (NEURAL NETWORK VERSION)
---------------------------------------
Backtesting with Neural Network model
Tests 3 transaction cost scenarios: 0.0%, 0.1%, 0.5%

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "trained_models"
RESULTS_DIR = BASE_DIR / "results" / "backtest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Transaction costs to test
TRANSACTION_COSTS = [0.0, 0.001, 0.005]  # 0%, 0.1%, 0.5%
INITIAL_CAPITAL = 10000

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("BACKTESTING WITH NEURAL NETWORK - ML STRATEGY VS BUY & HOLD")
print("=" * 80)
print(f"BASE_DIR: {BASE_DIR}")

print("\n[1/6] Loading test data...")

X_test_path = DATA_DIR / "X_test.csv"
y_test_path = DATA_DIR / "y_test.csv"

if not X_test_path.exists() or not y_test_path.exists():
    print("❌ Error: Test data not found!")
    sys.exit(1)

# Load X_test
X_test = pd.read_csv(X_test_path)

if "Date" in X_test.columns:
    X_test = X_test.set_index("Date")
else:
    # ✅ FIX: replace X_test.columns,[object Object],lower()
    first_col_lower = str(X_test.columns[0]).lower()
    if first_col_lower in ["date", "datetime", "timestamp"]:
        # ✅ FIX: replace X_test.columns,[object Object],
        X_test = X_test.set_index(X_test.columns[0])

print(f"   ✅ X_test loaded: {X_test.shape}")

# Load y_test
y_test = pd.read_csv(y_test_path)

if "Weekly_Return" in y_test.columns:
    y_test = y_test["Weekly_Return"]
elif "Target" in y_test.columns:
    y_test = y_test["Target"]
elif y_test.shape[1] == 1:  # ✅ FIX: replace y_test.shape,[object Object],
    y_test = y_test.iloc[:, 0]
else:
    print(f"⚠️  Warning: Unexpected y_test format: {list(y_test.columns)}")
    y_test = y_test.iloc[:, 0]

print(f"   ✅ y_test loaded: {y_test.shape}")

# Align indices
y_test.index = X_test.index
print(f"   ✅ Number of weeks: {len(X_test)}")

# ============================================================================
# LOAD NEURAL NETWORK MODEL
# ============================================================================

print("\n[2/6] Loading Neural Network model...")

# Try different possible filenames for Neural Network
nn_candidates = [
    "neuralnetwork.pkl",
    "neural_network.pkl",
    "NeuralNetwork.pkl",
    "nn.pkl",
]

model = None
model_filename = None

for filename in nn_candidates:
    model_path = MODEL_DIR / filename
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_filename = filename
            print(f"   ✅ Neural Network loaded: {filename}")
            break
        except Exception as e:
            print(f"   ⚠️  Failed to load {filename}: {e}")
            continue

if model is None:
    print("\n❌ Error: Neural Network model not found!")
    print(f"   Searched in: {MODEL_DIR}")
    print(f"   Tried filenames: {nn_candidates}")
    print("\n💡 Available models:")
    for f in MODEL_DIR.glob("*.pkl"):
        print(f"   - {f.name}")
    sys.exit(1)

# ============================================================================
# FEATURE ALIGNMENT
# ============================================================================

print("\n[2.5/6] Checking feature alignment...")

if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
    current_features = X_test.columns.tolist()
    missing_features = set(expected_features) - set(current_features)
    extra_features = set(current_features) - set(expected_features)

    if missing_features:
        print(f"   ⚠️  Missing features: {missing_features}")

        # Try to load from full dataset
        possible_paths = [
            DATA_DIR / "features_engineered.csv",
            BASE_DIR / "data" / "features_engineered.csv",
            DATA_DIR / "full_dataset.csv",
        ]

        features_loaded = False
        for path in possible_paths:
            if path.exists():
                print(f"   → Loading from: {path}")
                df_full = pd.read_csv(path)

                if "Date" in df_full.columns:
                    df_full = df_full.set_index("Date")

                for feature in missing_features:
                    if feature in df_full.columns:
                        X_test[feature] = df_full[feature].reindex(X_test.index).fillna(0)
                        print(f"   ✓ Added '{feature}'")
                        features_loaded = True
                    else:
                        X_test[feature] = 0
                        print(f"   ⚠️  '{feature}' filled with 0")
                break

        if not features_loaded:
            print("   ⚠️  Filling missing features with 0")
            for feature in missing_features:
                X_test[feature] = 0

    if extra_features:
        print(f"   ⚠️  Extra features (will be removed): {extra_features}")

    # Reorder columns to match model
    X_test = X_test[expected_features]
    print(f"   ✅ Features aligned: {X_test.shape}")
else:
    print("   ⚠️  Model doesn't have feature_names_in_ attribute")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n[3/6] Generating Neural Network predictions...")

try:
    y_pred = model.predict(X_test)
    print(f"   ✅ Predictions generated: {len(y_pred)} weeks")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    sys.exit(1)

# Create series
y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predicted_Return")
y_actual_series = pd.Series(y_test.values, index=X_test.index, name="Actual_Return")

# Prediction statistics
print("\n   📊 Neural Network Prediction Statistics:")
print(f"      Mean prediction:  {y_pred_series.mean():.6f}")
print(f"      Std prediction:   {y_pred_series.std():.6f}")
print(f"      Min prediction:   {y_pred_series.min():.6f}")
print(f"      Max prediction:   {y_pred_series.max():.6f}")

print("\n   📊 Actual Return Statistics:")
print(f"      Mean actual:      {y_actual_series.mean():.6f}")
print(f"      Std actual:       {y_actual_series.std():.6f}")
print(f"      Min actual:       {y_actual_series.min():.6f}")
print(f"      Max actual:       {y_actual_series.max():.6f}")

# Correlation
correlation = np.corrcoef(y_pred_series, y_actual_series)[0, 1]
print(f"\n   📈 Prediction-Actual Correlation: {correlation:.4f}")

# ============================================================================
# BACKTESTING FUNCTIONS
# ============================================================================

def backtest_strategy(y_pred, y_actual, dates, transaction_cost=0.0, threshold=0.0):
    """
    Backtest Neural Network trading strategy.

    Strategy:
    - Go LONG (position=1) when predicted return > threshold
    - Stay in CASH (position=0) when predicted return <= threshold
    """
    # Generate positions (1 = long, 0 = cash)
    positions = np.where(y_pred > threshold, 1, 0)

    # Calculate strategy returns (gross)
    strategy_returns_gross = positions * y_actual

    # Calculate transaction costs
    position_changes = np.diff(positions, prepend=0)
    trades = (position_changes != 0).astype(int)
    transaction_costs = trades * transaction_cost

    # Net returns after costs
    strategy_returns_net = strategy_returns_gross - transaction_costs

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns_net)

    # Performance metrics
    final_value = INITIAL_CAPITAL * cumulative_returns[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    num_trades = int(trades.sum())
    winning_trades = int((strategy_returns_net > 0).sum())
    win_rate = winning_trades / len(strategy_returns_net) if len(strategy_returns_net) > 0 else 0

    mean_return = float(strategy_returns_net.mean())
    std_return = float(strategy_returns_net.std())
    sharpe_ratio = (mean_return / std_return) * np.sqrt(52) if std_return > 0 else 0

    # Drawdown
    cumulative_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = float(drawdowns.min())

    # Volatility
    volatility = std_return * np.sqrt(52)

    # Total costs
    total_costs = float(transaction_costs.sum())

    # Time in market
    time_in_market = float(positions.sum() / len(positions))

    return {
        "final_value": final_value,
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "num_trades": num_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "volatility": volatility,
        "volatility_pct": volatility * 100,
        "total_costs": total_costs,
        "total_costs_pct": total_costs * 100,
        "time_in_market": time_in_market,
        "time_in_market_pct": time_in_market * 100,
        "cumulative_returns": cumulative_returns,
        "strategy_returns_net": strategy_returns_net,
        "positions": positions,
        "dates": dates,
    }

def backtest_buy_hold(y_actual, dates, transaction_cost=0.005):
    """
    Backtest Buy & Hold strategy.

    Key fix: Transaction cost applied ONCE at entry, not every week.

    Args:
        y_actual: Array of actual weekly returns
        dates: Array of dates
        transaction_cost: One-time entry cost (default 0.5%)

    Returns:
        Dictionary with performance metrics
    """
    # Convert to numpy array
    returns = np.array(y_actual, dtype=float)

    # Apply transaction cost ONLY at entry (first week)
    returns_with_cost = returns.copy()
    returns_with_cost[0] -= transaction_cost  # ✅ ONE-TIME COST

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns_with_cost)

    # Final value
    final_value = INITIAL_CAPITAL * cumulative_returns[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Performance metrics
    mean_return = float(returns_with_cost.mean())
    std_return = float(returns_with_cost.std())
    sharpe_ratio = (mean_return / std_return) * np.sqrt(52) if std_return > 0 else 0

    # Maximum drawdown
    cumulative_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = float(drawdowns.min())

    # Volatility (annualized)
    volatility = std_return * np.sqrt(52)

    # Winning weeks
    winning_weeks = int((returns_with_cost > 0).sum())
    win_rate = winning_weeks / len(returns_with_cost) if len(returns_with_cost) > 0 else 0

    return {
        "final_value": final_value,
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "num_trades": 1,  # Buy once and hold
        "winning_trades": winning_weeks,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "volatility": volatility,
        "volatility_pct": volatility * 100,
        "total_costs": transaction_cost,  # ONE-TIME
        "total_costs_pct": transaction_cost * 100,
        "time_in_market": 1.0,
        "time_in_market_pct": 100.0,
        "cumulative_returns": cumulative_returns,
        "strategy_returns_net": returns_with_cost,
        "positions": np.ones(len(returns)),
        "dates": dates,
    }

# ============================================================================
# RUN BACKTESTS
# ============================================================================

print("\n[4/6] Running backtests with Neural Network...")

nn_results = {}

for tc in TRANSACTION_COSTS:
    strategy_name = f"NN_{tc*100:.1f}%"
    print(f"   → {strategy_name} (Transaction cost: {tc*100:.1f}%)")
    nn_results[strategy_name] = backtest_strategy(
        y_pred,
        y_actual_series.values,
        X_test.index,
        transaction_cost=tc,
    )

print("   → Buy & Hold (Transaction cost: 0.5%)")
buy_hold_result = backtest_buy_hold(y_actual_series.values, X_test.index)

print("   ✅ All backtests completed")

# ============================================================================
# CREATE COMPARISONS
# ============================================================================

print("\n[5/6] Creating performance comparisons...")

all_comparisons = []

for nn_name, nn_result in nn_results.items():
    comparison = {
        "Strategy": nn_name,
        "Model": "Neural Network",
        "Transaction_Cost": nn_name.split("_")[1],  # ✅ FIX: replace nn_name.split("_"),[object Object],,
        "Final_Value": nn_result["final_value"],
        "Total_Return_pct": nn_result["total_return_pct"],
        "Sharpe_Ratio": nn_result["sharpe_ratio"],
        "Max_Drawdown_pct": nn_result["max_drawdown_pct"],
        "Num_Trades": nn_result["num_trades"],
        "Win_Rate_pct": nn_result["win_rate_pct"],
        "Volatility_pct": nn_result["volatility_pct"],
        "Time_in_Market_pct": nn_result["time_in_market_pct"],
        "Total_Costs_pct": nn_result["total_costs_pct"],
    }
    all_comparisons.append(comparison)

# Add Buy & Hold
all_comparisons.append(
    {
        "Strategy": "Buy & Hold",
        "Model": "Benchmark",
        "Transaction_Cost": "0.5%",
        "Final_Value": buy_hold_result["final_value"],
        "Total_Return_pct": buy_hold_result["total_return_pct"],
        "Sharpe_Ratio": buy_hold_result["sharpe_ratio"],
        "Max_Drawdown_pct": buy_hold_result["max_drawdown_pct"],
        "Num_Trades": buy_hold_result["num_trades"],
        "Win_Rate_pct": buy_hold_result["win_rate_pct"],
        "Volatility_pct": buy_hold_result["volatility_pct"],
        "Time_in_Market_pct": buy_hold_result["time_in_market_pct"],
        "Total_Costs_pct": buy_hold_result["total_costs_pct"],
    }
)

comparison_df = pd.DataFrame(all_comparisons)
comparison_path = RESULTS_DIR / "nn_backtest_results.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"   ✅ Results saved: {comparison_path}")

# ============================================================================
# VISUALIZATIONS (placeholder as in your file)
# ============================================================================

print("\n[6/6] Creating visualizations...")
# [Continue with all the visualization code from the previous version]
# ... (same as before)

print("\n" + "=" * 80)
print("✅ NEURAL NETWORK BACKTEST COMPLETED!")
print("=" * 80)
