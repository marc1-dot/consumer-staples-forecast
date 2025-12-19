"""
Advanced Backtesting Script - INDIVIDUAL COMPARISONS
Compares each ML strategy (0.0%, 0.1%, 0.5%) separately with Buy & Hold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/files/consumer-staples-forecast"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

TRANSACTION_COSTS = [0.0, 0.001, 0.005]  # 0%, 0.1%, 0.5%
INITIAL_CAPITAL = 10000

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("ADVANCED BACKTESTING - ML STRATEGY VS BUY & HOLD")
print("=" * 70)

print("\n[1/6] Loading test data...")

X_test_path = os.path.join(PROCESSED_DIR, "X_test.csv")
y_test_path = os.path.join(PROCESSED_DIR, "y_test.csv")

if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
    print("❌ Error: Test files not found!")
    exit(1)

# Load X_test
X_test = pd.read_csv(X_test_path, index_col=0)
print(f"✓ X_test loaded: {X_test.shape}")

# Load y_test
y_test = pd.read_csv(y_test_path)

if "Weekly_Return" in y_test.columns:
    y_test = y_test["Weekly_Return"]
elif y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]
else:
    print(f"❌ Error: Unexpected y_test format: {y_test.columns}")
    exit(1)

print(f"✓ y_test loaded: {y_test.shape}")

# Align indices
y_test.index = X_test.index
print(f"✓ Number of weeks: {len(X_test)}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n[2/6] Loading Random Forest model...")
model_path = os.path.join(MODELS_DIR, "RandomForest.pkl")

if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found!")
    exit(1)

model = joblib.load(model_path)
print("✓ Model loaded")

# ============================================================================
# FIX FEATURES
# ============================================================================

print("\n[2.5/6] Checking feature alignment...")

if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
    current_features = X_test.columns.tolist()
    missing_features = set(expected_features) - set(current_features)

    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")

        possible_paths = [
            "/files/consumer-staples-forecast/data/processed/features_engineered.csv",
            "/files/consumer-staples-forecast/data/features_engineered.csv",
            "/files/consumer-staples-forecast/features_engineered.csv",
        ]

        features_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   → Loading from: {path}")
                df_full = pd.read_csv(path, index_col=0)

                for feature in missing_features:
                    if feature in df_full.columns:
                        X_test[feature] = df_full[feature].iloc[-len(X_test):].values
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

    X_test = X_test[expected_features]
    print(f"✓ Features aligned: {X_test.shape}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n[3/6] Generating predictions...")

try:
    y_pred = model.predict(X_test)
    print(f"✓ Predictions generated: {len(y_pred)} weeks")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predicted_Return")
y_actual_series = pd.Series(y_test.values, index=X_test.index, name="Actual_Return")

# ============================================================================
# BACKTESTING FUNCTIONS
# ============================================================================

def backtest_strategy(y_pred, y_actual, dates, transaction_cost=0.0, threshold=0.0):
    """Backtest ML trading strategy"""

    positions = np.where(y_pred > threshold, 1, 0)
    strategy_returns_gross = positions * y_actual
    position_changes = np.diff(positions, prepend=0)
    trades = (position_changes != 0).astype(int)
    transaction_costs = trades * transaction_cost
    strategy_returns_net = strategy_returns_gross - transaction_costs
    cumulative_returns = np.cumprod(1 + strategy_returns_net)

    final_value = INITIAL_CAPITAL * cumulative_returns[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    num_trades = trades.sum()
    winning_trades = (strategy_returns_net > 0).sum()
    win_rate = winning_trades / len(strategy_returns_net) if len(strategy_returns_net) > 0 else 0

    mean_return = strategy_returns_net.mean()
    std_return = strategy_returns_net.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(52) if std_return > 0 else 0

    cumulative_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    volatility = std_return * np.sqrt(52)
    total_costs = transaction_costs.sum()

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
        "cumulative_returns": cumulative_returns,
        "strategy_returns_net": strategy_returns_net,
        "positions": positions,
        "dates": dates,
    }


def backtest_buy_hold(y_actual, dates, transaction_cost=0.005):
    """Backtest Buy & Hold strategy"""

    positions = np.ones(len(y_actual))
    strategy_returns = y_actual.copy()
    strategy_returns[0] -= transaction_cost  # <-- fixed

    cumulative_returns = np.cumprod(1 + strategy_returns)

    final_value = INITIAL_CAPITAL * cumulative_returns[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(52) if std_return > 0 else 0

    cumulative_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    volatility = std_return * np.sqrt(52)
    winning_weeks = (strategy_returns > 0).sum()
    win_rate = winning_weeks / len(strategy_returns)

    return {
        "final_value": final_value,
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "num_trades": 1,
        "winning_trades": winning_weeks,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "volatility": volatility,
        "volatility_pct": volatility * 100,
        "total_costs": transaction_cost,
        "total_costs_pct": transaction_cost * 100,
        "cumulative_returns": cumulative_returns,
        "strategy_returns_net": strategy_returns,
        "positions": positions,
        "dates": dates,
    }

# ============================================================================
# RUN BACKTESTS
# ============================================================================

print("\n[4/6] Running backtests...")

ml_results = {}

for tc in TRANSACTION_COSTS:
    strategy_name = f"ML_{tc*100:.1f}%"
    print(f"  → {strategy_name}")
    ml_results[strategy_name] = backtest_strategy(
        y_pred,
        y_actual_series.values,
        X_test.index,
        transaction_cost=tc,
    )

print("  → Buy & Hold")
buy_hold_result = backtest_buy_hold(y_actual_series.values, X_test.index)

print("✓ All backtests completed")

# ============================================================================
# CREATE INDIVIDUAL COMPARISONS
# ============================================================================

print("\n[5/6] Creating individual comparisons...")

all_comparisons = []

for ml_name, ml_result in ml_results.items():
    comparison = {
        "ML_Strategy": ml_name,
        "ML_Final_Value": ml_result["final_value"],
        "ML_Total_Return": ml_result["total_return_pct"],
        "ML_Sharpe": ml_result["sharpe_ratio"],
        "ML_Max_Drawdown": ml_result["max_drawdown_pct"],
        "ML_Num_Trades": ml_result["num_trades"],
        "ML_Win_Rate": ml_result["win_rate_pct"],
        "BH_Final_Value": buy_hold_result["final_value"],
        "BH_Total_Return": buy_hold_result["total_return_pct"],
        "BH_Sharpe": buy_hold_result["sharpe_ratio"],
        "BH_Max_Drawdown": buy_hold_result["max_drawdown_pct"],
        "BH_Num_Trades": buy_hold_result["num_trades"],
        "BH_Win_Rate": buy_hold_result["win_rate_pct"],
        "Diff_Return": ml_result["total_return_pct"] - buy_hold_result["total_return_pct"],
        "Diff_Sharpe": ml_result["sharpe_ratio"] - buy_hold_result["sharpe_ratio"],
        "Diff_Drawdown": ml_result["max_drawdown_pct"] - buy_hold_result["max_drawdown_pct"],
    }
    all_comparisons.append(comparison)

comparison_df = pd.DataFrame(all_comparisons)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "ml_vs_buyhold_comparisons.csv"), index=False)
print("✓ Comparisons saved to: results/ml_vs_buyhold_comparisons.csv")

# ============================================================================
# VISUALIZATIONS - INDIVIDUAL COMPARISONS
# ============================================================================

print("\n[6/6] Creating visualizations...")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

for ml_name, ml_result in ml_results.items():
    print(f"  → Creating comparison for {ml_name}")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Subplot 1: Cumulative Returns
    ax1 = fig.add_subplot(gs[0, :])

    portfolio_ml = INITIAL_CAPITAL * ml_result["cumulative_returns"]
    portfolio_bh = INITIAL_CAPITAL * buy_hold_result["cumulative_returns"]

    ax1.plot(
        range(len(ml_result["dates"])),
        portfolio_ml,
        label=ml_name,
        linewidth=2.5,
        color="#2ecc71",
        alpha=0.9,
    )
    ax1.plot(
        range(len(buy_hold_result["dates"])),
        portfolio_bh,
        label="Buy & Hold",
        linewidth=2,
        color="#95a5a6",
        linestyle="--",
        alpha=0.9,
    )

    ax1.set_xlabel("Week Number", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Cumulative Portfolio Value: {ml_name} vs Buy & Hold",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=INITIAL_CAPITAL, color="black", linestyle=":", alpha=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Subplot 2: Total Return
    ax2 = fig.add_subplot(gs[1, 0])
    strategies = [ml_name.replace("_", "\n"), "Buy &\nHold"]
    returns = [ml_result["total_return_pct"], buy_hold_result["total_return_pct"]]
    colors_bar = ["#2ecc71", "#95a5a6"]

    bars = ax2.bar(strategies, returns, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Total Return (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Total Return Comparison", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Sharpe Ratio
    ax3 = fig.add_subplot(gs[1, 1])
    sharpe_values = [ml_result["sharpe_ratio"], buy_hold_result["sharpe_ratio"]]

    bars = ax3.bar(strategies, sharpe_values, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, sharpe_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax3.set_ylabel("Sharpe Ratio", fontsize=10, fontweight="bold")
    ax3.set_title("Sharpe Ratio Comparison", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # Subplot 4: Max Drawdown
    ax4 = fig.add_subplot(gs[1, 2])
    drawdown_values = [ml_result["max_drawdown_pct"], buy_hold_result["max_drawdown_pct"]]

    bars = ax4.bar(strategies, drawdown_values, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, drawdown_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2f}%", ha="center", va="bottom" if height > 0 else "top", fontsize=10, fontweight="bold")

    ax4.set_ylabel("Max Drawdown (%)", fontsize=10, fontweight="bold")
    ax4.set_title("Max Drawdown Comparison", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # Subplot 5: Number of Trades
    ax5 = fig.add_subplot(gs[2, 0])
    trades = [ml_result["num_trades"], buy_hold_result["num_trades"]]

    bars = ax5.bar(strategies, trades, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, trades):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(val)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax5.set_ylabel("Number of Trades", fontsize=10, fontweight="bold")
    ax5.set_title("Trading Activity", fontsize=11, fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)

    # Subplot 6: Win Rate
    ax6 = fig.add_subplot(gs[2, 1])
    win_rates = [ml_result["win_rate_pct"], buy_hold_result["win_rate_pct"]]

    bars = ax6.bar(strategies, win_rates, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, win_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax6.set_ylabel("Win Rate (%)", fontsize=10, fontweight="bold")
    ax6.set_title("Win Rate Comparison", fontsize=11, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)

    # Subplot 7: Drawdown Evolution
    ax7 = fig.add_subplot(gs[2, 2])

    cumulative_max_ml = np.maximum.accumulate(ml_result["cumulative_returns"])
    drawdowns_ml = (ml_result["cumulative_returns"] - cumulative_max_ml) / cumulative_max_ml

    cumulative_max_bh = np.maximum.accumulate(buy_hold_result["cumulative_returns"])
    drawdowns_bh = (buy_hold_result["cumulative_returns"] - cumulative_max_bh) / cumulative_max_bh

    ax7.plot(range(len(ml_result["dates"])), drawdowns_ml * 100, label=ml_name, linewidth=2, color="#2ecc71", alpha=0.9)
    ax7.fill_between(range(len(ml_result["dates"])), drawdowns_ml * 100, 0, alpha=0.2, color="#2ecc71")

    ax7.plot(range(len(buy_hold_result["dates"])), drawdowns_bh * 100, label="Buy & Hold", linewidth=2, color="#95a5a6", linestyle="--", alpha=0.9)
    ax7.fill_between(range(len(buy_hold_result["dates"])), drawdowns_bh * 100, 0, alpha=0.2, color="#95a5a6")

    ax7.set_xlabel("Week Number", fontsize=10, fontweight="bold")
    ax7.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax7.set_title("Drawdown Evolution", fontsize=11, fontweight="bold")
    ax7.legend(loc="lower left", fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color="black", linestyle="-", linewidth=1)

    plt.suptitle(f"Performance Comparison: {ml_name} vs Buy & Hold", fontsize=16, fontweight="bold", y=0.995)

    filename = f"comparison_{ml_name.replace('.', '_').replace('%', 'pct')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: results/{filename}")
    plt.close()

# ============================================================================
# SUMMARY VISUALIZATION: Transaction Costs Impact
# ============================================================================

print("  → Creating transaction costs impact summary")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

tc_labels = [k.split("_")[1] for k in ml_results.keys()]
ml_returns = [v["total_return_pct"] for v in ml_results.values()]
ml_sharpe = [v["sharpe_ratio"] for v in ml_results.values()]
ml_trades = [v["num_trades"] for v in ml_results.values()]
ml_drawdown = [v["max_drawdown_pct"] for v in ml_results.values()]

bh_return = buy_hold_result["total_return_pct"]
bh_sharpe = buy_hold_result["sharpe_ratio"]
bh_drawdown = buy_hold_result["max_drawdown_pct"]

# Subplot 1: Returns
ax = axes[0, 0]
x = np.arange(len(tc_labels))
width = 0.35

bars1 = ax.bar(x - width / 2, ml_returns, width, label="ML Strategy",
               color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=1.5)
bars2 = ax.bar(x + width / 2, [bh_return] * len(tc_labels), width, label="Buy & Hold",
               color="#95a5a6", alpha=0.8, edgecolor="black", linewidth=1.5)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Total Return (%)", fontsize=11, fontweight="bold")
ax.set_title("Total Return vs Transaction Costs", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tc_labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Subplot 2: Sharpe Ratio
ax = axes[0, 1]
bars1 = ax.bar(x - width / 2, ml_sharpe, width, label="ML Strategy",
               color="#3498db", alpha=0.8, edgecolor="black", linewidth=1.5)
bars2 = ax.bar(x + width / 2, [bh_sharpe] * len(tc_labels), width, label="Buy & Hold",
               color="#95a5a6", alpha=0.8, edgecolor="black", linewidth=1.5)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold")
ax.set_title("Sharpe Ratio vs Transaction Costs", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tc_labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Subplot 3: Number of Trades
ax = axes[1, 0]
bars = ax.bar(tc_labels, ml_trades, color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Number of Trades", fontsize=11, fontweight="bold")
ax.set_title("Trading Activity vs Transaction Costs", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Subplot 4: Max Drawdown
ax = axes[1, 1]
bars1 = ax.bar(x - width / 2, ml_drawdown, width, label="ML Strategy",
               color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=1.5)
bars2 = ax.bar(x + width / 2, [bh_drawdown] * len(tc_labels), width, label="Buy & Hold",
               color="#95a5a6", alpha=0.8, edgecolor="black", linewidth=1.5)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%",
            ha="center", va="bottom" if height > 0 else "top", fontsize=9, fontweight="bold")

ax.set_ylabel("Max Drawdown (%)", fontsize=11, fontweight="bold")
ax.set_title("Max Drawdown vs Transaction Costs", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tc_labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Impact of Transaction Costs on ML Strategy Performance",
             fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "transaction_costs_impact_summary.png"),
            dpi=300, bbox_inches="tight")
print("  ✓ Saved: results/transaction_costs_impact_summary.png")
plt.close()

# ============================================================================
# COMPREHENSIVE SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 100)
print("COMPREHENSIVE PERFORMANCE SUMMARY")
print("=" * 100)

summary_table = []

summary_table.append({
    "Strategy": "Buy & Hold",
    "Transaction Cost": "0.5%",
    "Final Value ($)": f"${buy_hold_result['final_value']:,.2f}",
    "Total Return (%)": f"{buy_hold_result['total_return_pct']:.2f}%",
    "Sharpe Ratio": f"{buy_hold_result['sharpe_ratio']:.3f}",
    "Max Drawdown (%)": f"{buy_hold_result['max_drawdown_pct']:.2f}%",
    "Win Rate (%)": f"{buy_hold_result['win_rate_pct']:.2f}%",
    "Num Trades": buy_hold_result["num_trades"],
    "Volatility (%)": f"{buy_hold_result['volatility_pct']:.2f}%"
})

summary_table.append({
    "Strategy": "─" * 15,
    "Transaction Cost": "─" * 8,
    "Final Value ($)": "─" * 15,
    "Total Return (%)": "─" * 12,
    "Sharpe Ratio": "─" * 10,
    "Max Drawdown (%)": "─" * 12,
    "Win Rate (%)": "─" * 10,
    "Num Trades": "─" * 8,
    "Volatility (%)": "─" * 11
})

for ml_name, ml_result in ml_results.items():
    tc_value = ml_name.split("_")[1]
    diff_return = ml_result["total_return_pct"] - buy_hold_result["total_return_pct"]

    summary_table.append({
        "Strategy": "ML Strategy",
        "Transaction Cost": tc_value,
        "Final Value ($)": f"${ml_result['final_value']:,.2f}",
        "Total Return (%)": f"{ml_result['total_return_pct']:.2f}% ({diff_return:+.2f}%)",
        "Sharpe Ratio": f"{ml_result['sharpe_ratio']:.3f}",
        "Max Drawdown (%)": f"{ml_result['max_drawdown_pct']:.2f}%",
        "Win Rate (%)": f"{ml_result['win_rate_pct']:.2f}%",
        "Num Trades": ml_result["num_trades"],
        "Volatility (%)": f"{ml_result['volatility_pct']:.2f}%"
    })

summary_df = pd.DataFrame(summary_table)
print("\n" + summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_summary.csv"), index=False)
print("\n✓ Comprehensive summary saved to: results/comprehensive_summary.csv")

# ============================================================================
# DETAILED BREAKDOWN BY TRANSACTION COST
# ============================================================================

print("\n" + "=" * 100)
print("DETAILED BREAKDOWN: ML STRATEGY VS BUY & HOLD BY TRANSACTION COST")
print("=" * 100)

for ml_name, ml_result in ml_results.items():
    tc_value = ml_name.split("_")[1]

    print(f"\n{'─' * 100}")
    print(f"SCENARIO: Transaction Cost = {tc_value}")
    print(f"{'─' * 100}")

    print(f"\n{'ML Strategy':<25} {'Buy & Hold':<25} {'Difference':<25}")
    print(f"{'-' * 75}")

    metrics = [
        ("Final Value",
         f"${ml_result['final_value']:,.2f}",
         f"${buy_hold_result['final_value']:,.2f}",
         f"${ml_result['final_value'] - buy_hold_result['final_value']:+,.2f}"),

        ("Total Return",
         f"{ml_result['total_return_pct']:.2f}%",
         f"{buy_hold_result['total_return_pct']:.2f}%",
         f"{ml_result['total_return_pct'] - buy_hold_result['total_return_pct']:+.2f}%"),

        ("Sharpe Ratio",
         f"{ml_result['sharpe_ratio']:.3f}",
         f"{buy_hold_result['sharpe_ratio']:.3f}",
         f"{ml_result['sharpe_ratio'] - buy_hold_result['sharpe_ratio']:+.3f}"),

        ("Max Drawdown",
         f"{ml_result['max_drawdown_pct']:.2f}%",
         f"{buy_hold_result['max_drawdown_pct']:.2f}%",
         f"{ml_result['max_drawdown_pct'] - buy_hold_result['max_drawdown_pct']:+.2f}%"),

        ("Win Rate",
         f"{ml_result['win_rate_pct']:.2f}%",
         f"{buy_hold_result['win_rate_pct']:.2f}%",
         f"{ml_result['win_rate_pct'] - buy_hold_result['win_rate_pct']:+.2f}%"),

        ("Number of Trades",
         f"{ml_result['num_trades']}",
         f"{buy_hold_result['num_trades']}",
         f"{ml_result['num_trades'] - buy_hold_result['num_trades']:+}"),

        ("Volatility",
         f"{ml_result['volatility_pct']:.2f}%",
         f"{buy_hold_result['volatility_pct']:.2f}%",
         f"{ml_result['volatility_pct'] - buy_hold_result['volatility_pct']:+.2f}%"),
    ]

    for metric_name, ml_val, bh_val, diff_val in metrics:
        print(f"{metric_name:<20} {ml_val:<25} {bh_val:<25} {diff_val:<25}")

    diff_return = ml_result["total_return_pct"] - buy_hold_result["total_return_pct"]
    diff_sharpe = ml_result["sharpe_ratio"] - buy_hold_result["sharpe_ratio"]

    print("\nPerformance Verdict:")
    if diff_return > 0:
        print(f"  ✅ ML Strategy OUTPERFORMS Buy & Hold by {diff_return:.2f}% in total return")
    else:
        print(f"  ⚠️  ML Strategy UNDERPERFORMS Buy & Hold by {abs(diff_return):.2f}% in total return")

    if diff_sharpe > 0:
        print(f"  ✅ ML Strategy has BETTER risk-adjusted returns (Sharpe: {diff_sharpe:+.3f})")
    else:
        print(f"  ⚠️  ML Strategy has WORSE risk-adjusted returns (Sharpe: {diff_sharpe:+.3f})")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

print("\n📊 Transaction Costs Impact:")
for ml_name, ml_result in ml_results.items():
    tc_value = ml_name.split("_")[1]
    diff_return = ml_result["total_return_pct"] - buy_hold_result["total_return_pct"]
    print(f"  • At {tc_value} transaction cost: ML Strategy return difference = {diff_return:+.2f}%")

print("\n📈 Risk-Adjusted Performance:")
for ml_name, ml_result in ml_results.items():
    tc_value = ml_name.split("_")[1]
    diff_sharpe = ml_result["sharpe_ratio"] - buy_hold_result["sharpe_ratio"]
    print(f"  • At {tc_value} transaction cost: Sharpe Ratio difference = {diff_sharpe:+.3f}")

print("\n⚠️  Important Notes:")
print("  • Transaction costs significantly impact ML strategy performance")
print("  • Lower transaction costs favor active trading strategies")
print("  • Buy & Hold has minimal transaction costs (one-time entry)")
print("  • ML Strategy requires frequent trading over the test period")
print(f"  • Test period: {len(X_test)} weeks")

print("\n" + "=" * 100)
print("✅ BACKTEST COMPLETED SUCCESSFULLY!")
print("=" * 100)

print("\n📁 Generated Files:")
print("  • ml_vs_buyhold_comparisons.csv - Individual comparisons")
print("  • comprehensive_summary.csv - Overall summary table")
print("  • comparison_ML_0_0pct.png - ML 0.0% vs Buy & Hold")
print("  • comparison_ML_0_1pct.png - ML 0.1% vs Buy & Hold")
print("  • comparison_ML_0_5pct.png - ML 0.5% vs Buy & Hold")
print("  • transaction_costs_impact_summary.png - Transaction costs impact")

print(f"\n📂 All results saved to: {RESULTS_DIR}/")
