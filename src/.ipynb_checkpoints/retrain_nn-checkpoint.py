import os
import pandas as pd
import numpy as np
import joblib
from models.neural_network import train_neural_network

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "combined_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "trained", "NeuralNetwork.pkl")

print("\n🔄 RETRAINING NEURAL NETWORK WITH FEATURE SCALING\n")

# Load and clean data
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['Weekly_Return'])

exclude_cols = ['Weekly_Return', 'Close', 'Return', 'Date', 'Ticker']
X = df.drop(columns=exclude_cols, errors='ignore')
y = df['Weekly_Return']

# Clean numeric data
X = X.apply(pd.to_numeric, errors='coerce')
X = X.ffill().bfill().fillna(0)
X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# Train/test split
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features\n")

# Train model
model = train_neural_network(X_train, y_train)

# Save trained model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}\n")
