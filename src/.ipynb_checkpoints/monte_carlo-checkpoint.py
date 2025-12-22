"""
monte_carlo.py
--------------
Monte Carlo simulation for ML strategy equity curve using bootstrap of weekly returns.

- Loads X_test / y_test (test set = last 10% of data if you used time split)
- Loads a trained model (default: NeuralNetwork if present)
- Builds strategy weekly net returns: long (1) if y_pred > threshold else cash (0)
- Applies transaction costs on position changes
- Runs Monte Carlo bootstrapping to simulate many equity paths
- Saves plots + summary CSV

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "trained_models"
RESULTS_DIR = BASE_DIR / "results" / "monte_carlo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000
N_SIMULATIONS = 2000
RANDOM_SEED = 42

# Strategy rule
THRESHOLD = 0.0  # long if predicted return > threshold else cash

# Transaction costs per trade (e.g. 0.001 = 0.1%)
TRANSACTION_COST = 0.001

# Model filename priority
MODEL_CANDIDATES = [
    "NeuralNetwork.pkl",
    "neuralnetwork.pkl",
    "neural_network.pkl",
]

# ----------------------------
# HELPERS
# ----------------------------

def load_xy_test(data_dir: Path):
    """Load X_test and y_test from processed directory."""
    x_path = data_dir / "X_test.csv"
    y_path = data_dir / "y_test.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    X_test = pd.read_csv(x_path)
    y_test_df = pd.read_csv(y_path)

    # Set Date as index if present
    if "Date" in X_test.columns:
        X_test["Date"] = pd.to_datetime(X_test["Date"], errors="coerce")
        X_test = X_test.set_index("Date")

    # Get y_test as Series
    if "Weekly_Return" in y_test_df.columns:
        y_test = y_test_df["Weekly_Return"]
    elif "Target" in y_test_df.columns:
        y_test = y_test_df["Target"]
    elif y_test_df.shape[1] == 1:
        y_test = y_test_df.iloc[:, 0]
    else:
        y_test = y_test_df.iloc[:, 0]

    # Align index if X_test has datetime index
    if isinstance(X_test.index, pd.DatetimeIndex):
        y_test.index = X_test.index

    # Basic cleaning
    X_test = X_test.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    y_test = pd.to_numeric(y_test, errors="coerce").fillna(0)

    return X_test, y_test


def load_model(model_dir: Path):
    """Load a trained model, preferring NeuralNetwork if available."""
    for name in MODEL_CANDIDATES:
        p = model_dir / name
        if p.exists():
            return joblib.load(p), p.name

    # fallback: first pkl
    pkls = sorted(model_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl models found in {model_dir}")
    return joblib.load(pkls[0]), pkls[0].name


def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align X columns to model expected features if model exposes feature_names_in_.
    Adds missing columns filled with 0 and drops extra columns.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        current = list(X.columns)

        missing = [c for c in expected if c not in current]
        extra = [c for c in current if c not in expected]

        if missing:
            for c in missing:
                X[c] = 0.0

        if extra:
            X = X.drop(columns=extra)

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
    - cost applied when position changes (enter/exit): cost per change
    """
    positions = (y_pred > threshold).astype(int)
    gross = positions * y_actual

    # trade occurs when position changes
    changes = np.diff(positions, prepend=0)
    trades = (changes != 0).astype(int)
    costs = trades * transaction_cost

    net = gross - costs
    return net


def simulate_paths(weekly_returns: np.ndarray,
                   n_sims: int,
                   initial_capital: float,
                   seed: int = 42) -> np.ndarray:
    """
    Bootstrap weekly returns with replacement and simulate equity curves.
    Returns array shape: (n_sims, T+1) with equity starting at initial_capital.
    """
    rng = np.random.default_rng(seed)
    T = len(weekly_returns)

    # sample indices (n_sims, T)
    idx = rng.integers(low=0, high=T, size=(n_sims, T))
    sampled = weekly_returns[idx]  # (n_sims, T)

    # equity: cumulative product of (1+r)
    growth = np.cumprod(1.0 + sampled, axis=1)
    equity = np.empty((n_sims, T + 1), dtype=float)
    equity[:, 0] = initial_capital
    equity[:, 1:] = initial_capital * growth
    return equity


def summarize_terminal_values(equity: np.ndarray) -> dict:
    """Compute summary statistics on terminal equity."""
    terminal = equity[:, -1]
    stats = {
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_p05": float(np.quantile(terminal, 0.05)),
        "terminal_p25": float(np.quantile(terminal, 0.25)),
        "terminal_p75": float(np.quantile(terminal, 0.75)),
        "terminal_p95": float(np.quantile(terminal, 0.95)),
        "prob_loss": float(np.mean(terminal < equity[0, 0])),  # < initial capital
    }
    return stats


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Historical VaR/CVaR of weekly returns distribution.
    VaR is quantile at alpha; CVaR is mean of tail returns <= VaR.
    """
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    cvar = float(np.mean(tail)) if len(tail) else float("nan")
    return var, cvar


def save_plots(equity: np.ndarray, out_dir: Path, title_prefix: str = ""):
    """Save Monte Carlo plots."""
    T = equity.shape[1] - 1
    x = np.arange(T + 1)

    # 1) sample of paths
    plt.figure()
    n_show = min(50, equity.shape[0])
    for i in range(n_show):
        plt.plot(x, equity[i, :], alpha=0.2)
    plt.title(f"{title_prefix}Monte Carlo Equity Paths (sample)")
    plt.xlabel("Step (weeks)")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_paths_sample.png", dpi=200)
    plt.close()

    # 2) percentile band
    p05 = np.quantile(equity, 0.05, axis=0)
    p50 = np.quantile(equity, 0.50, axis=0)
    p95 = np.quantile(equity, 0.95, axis=0)

    plt.figure()
    plt.plot(x, p50, label="Median")
    plt.plot(x, p05, label="5th pct")
    plt.plot(x, p95, label="95th pct")
    plt.title(f"{title_prefix}Equity Percentile Bands")
    plt.xlabel("Step (weeks)")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mc_percentiles.png", dpi=200)
    plt.close()

    # 3) terminal distribution
    terminal = equity[:, -1]
    plt.figure()
    plt.hist(terminal, bins=60, alpha=0.8, edgecolor="black")
    plt.title(f"{title_prefix}Terminal Equity Distribution")
    plt.xlabel("Terminal equity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_terminal_hist.png", dpi=200)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

def main():
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION - ML STRATEGY (BOOTSTRAP)")
    print("=" * 80)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")

    # Load test data
    print("\n[1/5] Loading X_test / y_test...")
    X_test, y_test = load_xy_test(DATA_DIR)
    print(f"   ✅ X_test: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(f"   ✅ y_test: {len(y_test)} samples")

    # Load model
    print("\n[2/5] Loading model...")
    model, model_file = load_model(MODEL_DIR)
    print(f"   ✅ Loaded: {model_file}")

    # Align features
    print("\n[3/5] Aligning features + predicting...")
    X_aligned = align_features(model, X_test.copy())
    y_pred = model.predict(X_aligned)

    # Strategy returns
    net = strategy_net_returns(
        y_pred=np.asarray(y_pred, dtype=float),
        y_actual=np.asarray(y_test.values, dtype=float),
        transaction_cost=TRANSACTION_COST,
        threshold=THRESHOLD,
    )

    print("\n   📊 Strategy weekly net returns:")
    print(f"      Mean: {net.mean():.6f}")
    print(f"      Std:  {net.std():.6f}")
    v, c = var_cvar(net, alpha=0.05)
    print(f"      VaR 5% (weekly):  {v:.6f}")
    print(f"      CVaR 5% (weekly): {c:.6f}")

    # Monte Carlo
    print("\n[4/5] Running Monte Carlo...")
    equity = simulate_paths(
        weekly_returns=net,
        n_sims=N_SIMULATIONS,
        initial_capital=INITIAL_CAPITAL,
        seed=RANDOM_SEED,
    )

    stats = summarize_terminal_values(equity)
    print("\n   🎯 Terminal equity stats:")
    for k, val in stats.items():
        print(f"      {k}: {val:.4f}" if isinstance(val, float) else f"      {k}: {val}")

    # Save outputs
    print("\n[5/5] Saving outputs...")
    # CSV summary
    summary = {
        "model_file": model_file,
        "initial_capital": INITIAL_CAPITAL,
        "n_simulations": N_SIMULATIONS,
        "transaction_cost": TRANSACTION_COST,
        "threshold": THRESHOLD,
        "weekly_mean_return": float(net.mean()),
        "weekly_std_return": float(net.std()),
        "weekly_var_5": v,
        "weekly_cvar_5": c,
        **stats,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = RESULTS_DIR / "mc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ Saved: {summary_path}")

    # Also save terminal values
    terminal_path = RESULTS_DIR / "mc_terminal_values.csv"
    pd.DataFrame({"terminal_equity": equity[:, -1]}).to_csv(terminal_path, index=False)
    print(f"   ✅ Saved: {terminal_path}")

    # Plots
    save_plots(equity, RESULTS_DIR, title_prefix=f"{model_file} | ")
    print(f"   ✅ Saved plots in: {RESULTS_DIR}")

    print("\n" + "=" * 80)
    print("✅ MONTE CARLO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
