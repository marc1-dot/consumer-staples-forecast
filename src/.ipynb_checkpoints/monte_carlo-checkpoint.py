"""
monte_carlo.py
--------------
Monte Carlo simulation for ML strategy equity curve using bootstrap of weekly returns.
Simulates 4 Models: Linear Regression, Random Forest, XGBoost, Neural Network.

- Loads X_test / y_test (test set = last 10% of data)
- Iterates through all trained models.
- Builds strategy weekly net returns: long (1) if y_pred > threshold else cash (0).
- Applies transaction costs on position changes.
- Runs Monte Carlo bootstrapping to simulate 2000 equity paths per model.
- Calculates Risk Metrics (VaR, CVaR, Prob of Loss).
- Saves plots per model + consolidated summary CSV.

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

# ----------------------------
# CONFIG
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "trained_models"
RESULTS_DIR = BASE_DIR / "results" / "monte_carlo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")

INITIAL_CAPITAL = 10_000
N_SIMULATIONS = 2000
RANDOM_SEED = 42

# Strategy rule
THRESHOLD = 0.0  # long if predicted return > threshold else cash

# Transaction costs per trade (0.1% matches your backtest scenario)
TRANSACTION_COST = 0.001

# Models to simulate
MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "Neural Network": "neuralnetwork.pkl",
}

# ----------------------------
# HELPERS
# ----------------------------

def load_xy_test(data_dir: Path):
    """Load X_test and y_test from processed directory."""
    x_path = data_dir / "X_test_clean.csv" # Using cleaned version from previous steps
    y_path = data_dir / "y_test.csv"

    if not x_path.exists():
        # Fallback to standard X_test if clean not found
        x_path = data_dir / "X_test.csv" 
    
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Missing X_test or y_test files.")

    X_test = pd.read_csv(x_path)
    y_test_df = pd.read_csv(y_path)

    # Handle Date Index if present
    if "Date" in X_test.columns:
        X_test["Date"] = pd.to_datetime(X_test["Date"], errors="coerce")
        X_test = X_test.set_index("Date")

    # Get y_test as Series
    if "Target" in y_test_df.columns:
        y_test = y_test_df["Target"]
    else:
        y_test = y_test_df.iloc[:, 0]

    # Clean Data
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = pd.to_numeric(y_test, errors="coerce").fillna(0)

    return X_test, y_test

def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """Align X columns to model expected features."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        current = list(X.columns)

        # Add missing with 0
        for c in expected:
            if c not in current:
                X[c] = 0.0
        
        # Keep only expected, in correct order
        X = X[expected]
    return X

def strategy_net_returns(y_pred: np.ndarray,
                         y_actual: np.ndarray,
                         transaction_cost: float = 0.0,
                         threshold: float = 0.0) -> np.ndarray:
    """
    Long/cash strategy:
    - position=1 if y_pred > threshold else 0
    - gross return = position * y_actual
    - cost applied when position changes
    """
    positions = (y_pred > threshold).astype(int)
    gross = positions * y_actual

    # trade occurs when position changes
    changes = np.diff(positions, prepend=0)
    trades = (changes != 0).astype(int)
    costs = trades * transaction_cost

    net = gross - costs
    return net

def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Historical VaR/CVaR of returns distribution."""
    if len(returns) == 0: return 0.0, 0.0
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return var, cvar

def simulate_paths(returns: np.ndarray, n_sims: int, initial_capital: float, seed: int = 42) -> np.ndarray:
    """Bootstrap returns and simulate equity curves."""
    rng = np.random.default_rng(seed)
    T = len(returns)

    # Sample indices (n_sims, T)
    idx = rng.integers(low=0, high=T, size=(n_sims, T))
    sampled = returns[idx]

    # Equity = Cumulative Product
    growth = np.cumprod(1.0 + sampled, axis=1)
    
    # Prepend initial capital
    equity = np.empty((n_sims, T + 1), dtype=float)
    equity[:, 0] = initial_capital
    equity[:, 1:] = initial_capital * growth
    
    return equity

def summarize_terminal_values(equity: np.ndarray, initial_cap: float) -> dict:
    """Compute summary stats on terminal equity."""
    terminal = equity[:, -1]
    stats = {
        "Mean Final Equity": float(np.mean(terminal)),
        "Median Final Equity": float(np.median(terminal)),
        "Worst 5% Case": float(np.quantile(terminal, 0.05)),
        "Best 5% Case": float(np.quantile(terminal, 0.95)),
        "Prob of Loss (%)": float(np.mean(terminal < initial_cap) * 100),
    }
    return stats

def save_plots(equity: np.ndarray, out_dir: Path, model_name: str):
    """Save 3 separate Monte Carlo plots for a specific model."""
    safe_name = model_name.replace(" ", "_")
    T = equity.shape[1] - 1
    x = np.arange(T + 1)

    # 1. Sample Paths
    plt.figure(figsize=(10, 6))
    n_show = min(50, equity.shape[0])
    for i in range(n_show):
        plt.plot(x, equity[i, :], alpha=0.3, linewidth=1)
    plt.title(f"{model_name}: Monte Carlo Equity Paths (Sample)", fontsize=14)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_name}_paths.png", dpi=200)
    plt.close()

    # 2. Percentiles
    p05 = np.quantile(equity, 0.05, axis=0)
    p50 = np.quantile(equity, 0.50, axis=0)
    p95 = np.quantile(equity, 0.95, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(x, p50, label="Median", color="blue", linewidth=2)
    plt.fill_between(x, p05, p95, color="blue", alpha=0.1, label="90% Confidence Interval")
    plt.title(f"{model_name}: Expected Equity Range (90% CI)", fontsize=14)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_name}_percentiles.png", dpi=200)
    plt.close()

    # 3. Terminal Histogram
    terminal = equity[:, -1]
    plt.figure(figsize=(10, 6))
    plt.hist(terminal, bins=50, color="teal", alpha=0.7, edgecolor="black")
    plt.axvline(INITIAL_CAPITAL, color="red", linestyle="--", label="Initial Capital")
    plt.title(f"{model_name}: Final Equity Distribution (N={N_SIMULATIONS})", fontsize=14)
    plt.xlabel("Final Equity Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_name}_histogram.png", dpi=200)
    plt.close()

# ----------------------------
# MAIN
# ----------------------------

def main():
    print("\n" + "=" * 80)
    print("🎲 MONTE CARLO SIMULATION (MULTI-MODEL)")
    print("=" * 80)
    
    # 1. Load Data
    try:
        X_test, y_test = load_xy_test(DATA_DIR)
        print(f"✅ Data Loaded: {len(X_test)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    all_results = []

    # 2. Loop through models
    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            print(f"⚠️  Skipping {model_name}: File not found ({filename})")
            continue

        print(f"\n🔹 Processing: {model_name}...")
        
        # Load Model
        model = joblib.load(model_path)
        
        # Prepare Data & Predict
        X_aligned = align_features(model, X_test.copy())
        y_pred = model.predict(X_aligned)
        
        # Calculate Strategy Returns
        net_returns = strategy_net_returns(
            y_pred=np.asarray(y_pred, dtype=float),
            y_actual=np.asarray(y_test.values, dtype=float),
            transaction_cost=TRANSACTION_COST,
            threshold=THRESHOLD
        )
        
        # Basic Stats
        mean_ret = net_returns.mean()
        var_95, cvar_95 = var_cvar(net_returns, alpha=0.05)
        
        # Run Monte Carlo
        equity = simulate_paths(
            returns=net_returns,
            n_sims=N_SIMULATIONS,
            initial_capital=INITIAL_CAPITAL,
            seed=RANDOM_SEED
        )
        
        # Terminal Stats
        term_stats = summarize_terminal_values(equity, INITIAL_CAPITAL)
        
        # Save Plots
        save_plots(equity, RESULTS_DIR, model_name)
        print(f"   📊 Plots saved.")

        # Collect Results
        result_row = {
            "Model": model_name,
            "Avg Daily Return": mean_ret,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            **term_stats
        }
        all_results.append(result_row)

    # 3. Save Summary CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        # Reorder columns for readability
        cols = ["Model", "Mean Final Equity", "Prob of Loss (%)", "VaR 95%", "CVaR 95%", "Worst 5% Case", "Best 5% Case"]
        # Ensure only existing columns are selected (in case of dict key mismatch)
        cols = [c for c in cols if c in df_results.columns]
        df_results = df_results[cols]
        
        summary_path = RESULTS_DIR / "mc_summary_all_models.csv"
        df_results.to_csv(summary_path, index=False)
        
        print("\n" + "=" * 80)
        print("📋 SUMMARY OF RESULTS")
        print("=" * 80)
        print(df_results.to_string(index=False))
        print(f"\n✅ Detailed summary saved to: {summary_path}")
    else:
        print("\n❌ No models were processed successfully.")

    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
