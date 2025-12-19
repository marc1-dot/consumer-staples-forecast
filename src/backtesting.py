"""
backtesting.py
--------------
Backtest trading strategy using model predictions.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = '/files/consumer-staples-forecast'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

sns.set_style("whitegrid")

print("="*70)
print("SCRIPT STARTED")
print("="*70)


def load_test_data():
    """Load test data and actual returns."""

    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)

    try:
        X_test_path = os.path.join(PROCESSED_DIR, 'X_test.csv')
        y_test_path = os.path.join(PROCESSED_DIR, 'y_test.csv')

        print(f"Loading X_test from: {X_test_path}")
        print(f"Loading y_test from: {y_test_path}")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()

        print(f"✓ X_test loaded: {X_test.shape}")
        print(f"✓ y_test loaded: {len(y_test)} samples")
        print(f"  Mean actual return: {y_test.mean():.6f}")
        print(f"  Std actual return: {y_test.std():.6f}")
        print(f"  Min actual return: {y_test.min():.6f}")
        print(f"  Max actual return: {y_test.max():.6f}")

        return X_test, y_test

    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        sys.exit(1)


def load_best_model():
    """Load the best performing model."""

    print("\n" + "="*70)
    print("LOADING BEST MODEL")
    print("="*70)

    try:
        comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')

        if not os.path.exists(comparison_path):
            print("⚠️ model_comparison.csv not found. Using RandomForest as default.")
            best_model_name = 'RandomForest'
        else:
            comparison = pd.read_csv(comparison_path)
            best_idx = comparison['Test_R2'].idxmax()
            best_model_name = comparison.loc[best_idx, 'Model']
            best_r2 = comparison.loc[best_idx, 'Test_R2']
            print(f"Best model: {best_model_name} (Test R² = {best_r2:.4f})")

        model_path = os.path.join(MODEL_DIR, f"{best_model_name}.pkl")
        print(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully")

        return model, best_model_name

    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        sys.exit(1)


def backtest_strategy(y_actual, y_pred, threshold=0.0, initial_capital=10000):
    """Backtest trading strategy with diagnostics."""

    print("\n" + "="*70)
    print(f"BACKTESTING STRATEGY")
    print(f"  Threshold: {threshold:.2%}")
    print(f"  Initial Capital: ${initial_capital:,.0f}")
    print("="*70)

    # Convert to numpy arrays
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    # ========== DIAGNOSTIC CHECKS ==========
    print("\n🔍 DATA DIAGNOSTICS:")
    print(f"  y_actual - Min: {y_actual.min():.6f}, Max: {y_actual.max():.6f}, Mean: {y_actual.mean():.6f}")
    print(f"  y_pred   - Min: {y_pred.min():.6f}, Max: {y_pred.max():.6f}, Mean: {y_pred.mean():.6f}")
    print(f"  Sample y_actual (first 5): {y_actual[:5]}")
    print(f"  Sample y_pred (first 5):   {y_pred[:5]}")

    # Check for percentage format
    if np.abs(y_actual).max() > 1.0:
        print("\n⚠️ WARNING: y_actual contains values > 1.0")
        print("   Your returns might be in percentage format (e.g., 5.0 instead of 0.05)")
        print("   This will cause MASSIVE calculation errors!")
        print(f"   Max absolute value: {np.abs(y_actual).max():.2f}")

        count_large = (np.abs(y_actual) > 1.0).sum()
        print(f"   Number of values > 1.0: {count_large} / {len(y_actual)}")

        print("\n   ⚠️ PLEASE FIX YOUR DATA IN feature_engineering.py")
        print("   The Target should be in decimal format (0.05 for 5%)")
        print("\n   Continuing with current data...")

    # Create positions
    positions = np.where(y_pred > threshold, 1,
                np.where(y_pred < -threshold, -1, 0))

    long_count = (positions == 1).sum()
    short_count = (positions == -1).sum()
    none_count = (positions == 0).sum()

    print(f"\n  Position distribution:")
    print(f"    Long:  {long_count} ({long_count/len(positions)*100:.1f}%)")
    print(f"    Short: {short_count} ({short_count/len(positions)*100:.1f}%)")
    print(f"    None:  {none_count} ({none_count/len(positions)*100:.1f}%)")

    # Calculate strategy returns
    strategy_returns = positions * y_actual

    print(f"\n  Strategy returns:")
    print(f"    Min: {strategy_returns.min():.6f}")
    print(f"    Max: {strategy_returns.max():.6f}")
    print(f"    Mean: {strategy_returns.mean():.6f}")
    print(f"    Sample (first 5): {strategy_returns[:5]}")

    if np.abs(strategy_returns).max() > 0.5:
        print("\n⚠️ WARNING: Strategy returns > 50% detected!")
        print("   This is VERY unusual for weekly stock returns.")
        print("   Your data format is likely incorrect.")

    # Calculate cumulative returns
    cumulative_strategy = np.cumprod(1 + strategy_returns)
    cumulative_buy_hold = np.cumprod(1 + y_actual)

    cumulative_strategy = pd.Series(cumulative_strategy)
    cumulative_buy_hold = pd.Series(cumulative_buy_hold)

    print(f"\n  Cumulative returns:")
    print(f"    Strategy final: {cumulative_strategy.iloc[-1]:.4f}x")
    print(f"    Buy&Hold final: {cumulative_buy_hold.iloc[-1]:.4f}x")

    if cumulative_strategy.iloc[-1] > 50:
        print("\n🚨 CRITICAL ERROR: Cumulative return > 50x detected!")
        print("   This indicates a DATA FORMAT problem.")
        print("   Check your feature_engineering.py - Target calculation")
        print("\n   Expected: df['Target'] = df['Close'].pct_change()")
        print("   NOT: df['Target'] = df['Close'].pct_change() * 100")

    # Portfolio values
    portfolio_value_strategy = initial_capital * cumulative_strategy
    portfolio_value_buy_hold = initial_capital * cumulative_buy_hold

    # Create results DataFrame
    results = pd.DataFrame({
        'Week': range(len(y_actual)),
        'Actual_Return': y_actual,
        'Predicted_Return': y_pred,
        'Position': positions,
        'Strategy_Return': strategy_returns,
        'Cumulative_Strategy': cumulative_strategy,
        'Cumulative_BuyHold': cumulative_buy_hold,
        'Portfolio_Value_Strategy': portfolio_value_strategy,
        'Portfolio_Value_BuyHold': portfolio_value_buy_hold
    })

    # ========== CALCULATE METRICS ==========

    total_return_strategy = cumulative_strategy.iloc[-1] - 1
    total_return_buy_hold = cumulative_buy_hold.iloc[-1] - 1

    n_weeks = len(results)
    n_years = n_weeks / 52

    annualized_return_strategy = (cumulative_strategy.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0
    annualized_return_buy_hold = (cumulative_buy_hold.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0

    volatility_strategy = strategy_returns.std() * np.sqrt(52)
    volatility_buy_hold = y_actual.std() * np.sqrt(52)

    sharpe_strategy = annualized_return_strategy / volatility_strategy if volatility_strategy > 0 else 0
    sharpe_buy_hold = annualized_return_buy_hold / volatility_buy_hold if volatility_buy_hold > 0 else 0

    # Max Drawdown
    cumulative_max_strategy = cumulative_strategy.cummax()
    drawdown_strategy = (cumulative_strategy - cumulative_max_strategy) / cumulative_max_strategy
    max_drawdown_strategy = drawdown_strategy.min()

    cumulative_max_buy_hold = cumulative_buy_hold.cummax()
    drawdown_buy_hold = (cumulative_buy_hold - cumulative_max_buy_hold) / cumulative_max_buy_hold
    max_drawdown_buy_hold = drawdown_buy_hold.min()

    # Win rate
    winning_trades = (strategy_returns[positions != 0] > 0).sum()
    total_trades = (positions != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_return_per_trade = strategy_returns[positions != 0].mean() if total_trades > 0 else 0

    final_value_strategy = portfolio_value_strategy.iloc[-1]
    final_value_buy_hold = portfolio_value_buy_hold.iloc[-1]

    # ========== PRINT RESULTS ==========

    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)

    print(f"\n{'ML STRATEGY':=^70}")
    print(f"  Total Return:          {total_return_strategy:>10.2%}")
    print(f"  Annualized Return:     {annualized_return_strategy:>10.2%}")
    print(f"  Volatility (annual):   {volatility_strategy:>10.2%}")
    print(f"  Sharpe Ratio:          {sharpe_strategy:>10.2f}")
    print(f"  Max Drawdown:          {max_drawdown_strategy:>10.2%}")
    print(f"  Win Rate:              {win_rate:>10.2%}")
    print(f"  Avg Return per Trade:  {avg_return_per_trade:>10.6f}")
    print(f"  Total Trades:          {total_trades:>10.0f}")
    print(f"  Final Portfolio Value: ${final_value_strategy:>10,.2f}")

    print(f"\n{'BUY & HOLD BENCHMARK':=^70}")
    print(f"  Total Return:          {total_return_buy_hold:>10.2%}")
    print(f"  Annualized Return:     {annualized_return_buy_hold:>10.2%}")
    print(f"  Volatility (annual):   {volatility_buy_hold:>10.2%}")
    print(f"  Sharpe Ratio:          {sharpe_buy_hold:>10.2f}")
    print(f"  Max Drawdown:          {max_drawdown_buy_hold:>10.2%}")
    print(f"  Final Portfolio Value: ${final_value_buy_hold:>10,.2f}")

    print(f"\n{'OUTPERFORMANCE':=^70}")
    excess_return = total_return_strategy - total_return_buy_hold
    sharpe_improvement = sharpe_strategy - sharpe_buy_hold
    profit_difference = final_value_strategy - final_value_buy_hold

    print(f"  Excess Return:         {excess_return:>10.2%}")
    print(f"  Sharpe Improvement:    {sharpe_improvement:>10.2f}")
    print(f"  Profit Difference:     ${profit_difference:>10,.2f}")

    if excess_return > 0:
        print(f"\n  ✅ Strategy OUTPERFORMS Buy & Hold by {excess_return:.2%}")
    else:
        print(f"\n  ❌ Strategy UNDERPERFORMS Buy & Hold by {abs(excess_return):.2%}")

    # Create metrics DataFrame
    metrics = {
        'Strategy': ['ML Strategy', 'Buy & Hold'],
        'Total_Return': [total_return_strategy, total_return_buy_hold],
        'Annualized_Return': [annualized_return_strategy, annualized_return_buy_hold],
        'Volatility': [volatility_strategy, volatility_buy_hold],
        'Sharpe_Ratio': [sharpe_strategy, sharpe_buy_hold],
        'Max_Drawdown': [max_drawdown_strategy, max_drawdown_buy_hold],
        'Win_Rate': [win_rate, np.nan],
        'Total_Trades': [total_trades, np.nan],
        'Final_Portfolio_Value': [final_value_strategy, final_value_buy_hold]
    }

    metrics_df = pd.DataFrame(metrics)

    # Save metrics
    try:
        metrics_path = os.path.join(RESULTS_DIR, 'backtest_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n✓ Metrics saved to: {metrics_path}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")

    return results, metrics_df


def visualize_backtest(results, model_name, metrics_df):
    """Create backtest visualizations."""

    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    try:
        # Figure 1: Main results
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # ✅ FIX 1/3: axes indexing in Python
        ax = axes[0]
        ax.plot(results['Week'], results['Cumulative_Strategy'],
                label='ML Strategy', linewidth=2.5, color='#2E86AB')
        ax.plot(results['Week'], results['Cumulative_BuyHold'],
                label='Buy & Hold', linewidth=2.5, color='#A23B72', linestyle='--')
        ax.axhline(y=1, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title(f'Cumulative Returns - {model_name}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        colors = ['green' if r > 0 else 'red' for r in results['Strategy_Return']]
        ax.bar(results['Week'], results['Strategy_Return'], alpha=0.6, color=colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Weekly Strategy Returns', fontweight='bold', fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[2]
        colors = ['red' if p == -1 else 'green' if p == 1 else 'gray' for p in results['Position']]
        ax.bar(results['Week'], results['Position'], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_title('Trading Positions', fontweight='bold', fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel('Position')
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, 'backtest_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

        # Figure 2: Portfolio value
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(results['Week'], results['Portfolio_Value_Strategy'],
                label='ML Strategy', linewidth=2.5, color='#2E86AB')
        ax.plot(results['Week'], results['Portfolio_Value_BuyHold'],
                label='Buy & Hold', linewidth=2.5, color='#A23B72', linestyle='--')
        ax.set_title('Portfolio Value Over Time', fontweight='bold', fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # ✅ FIX 2: iloc[0] (au lieu de iloc,[object Object],,)
        ax.axhline(
            y=results['Portfolio_Value_Strategy'].iloc[0],
            color='black', linestyle=':', linewidth=1, alpha=0.5,
            label='Initial Capital'
        )

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, 'backtest_portfolio_value.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        print("\n🚀 Starting backtesting script...\n")

        # Load data
        X_test, y_test = load_test_data()

        # Load model
        model, model_name = load_best_model()

        # Generate predictions
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS")
        print("="*70)
        y_pred = model.predict(X_test)
        print(f"✓ Predictions generated: {len(y_pred)} samples")

        # Backtest
        results, metrics = backtest_strategy(y_test, y_pred, threshold=0.0, initial_capital=10000)

        # Visualize
        visualize_backtest(results, model_name, metrics)

        # Save detailed results
        results_path = os.path.join(RESULTS_DIR, 'backtest_detailed_results.csv')
        results.to_csv(results_path, index=False)
        print(f"\n✓ Detailed results saved to: {results_path}")

        print("\n" + "="*70)
        print("✓✓✓ BACKTESTING COMPLETE ✓✓✓")
        print("="*70)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
