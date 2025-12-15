"""
neural_network.py
-----------------
Train an optimized feed-forward neural network (MLP) to predict weekly stock returns.
Includes normalization, early stopping, and a compact architecture for small datasets.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_neural_network(X_train, y_train):
    """
    Trains a compact and stable Multi-Layer Perceptron (feed-forward neural network).
    - Scales inputs for gradient stability
    - Uses smaller architecture to prevent overfitting
    - Implements early stopping based on validation loss
    """

    # Build the pipeline: Scaling + Neural Network
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16, 8),     # simpler architecture → better generalization
            activation="relu",
            solver="adam",
            learning_rate_init=0.0005,      # slower, more stable learning
            max_iter=300,                   # fewer iterations (enough for convergence)
            early_stopping=True,            # stops when validation score stops improving
            validation_fraction=0.1,        # 10% of training data used for validation
            alpha=0.001,                    # L2 regularization to reduce overfitting
            random_state=42,
            verbose=False
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    print("✅ Optimized Neural Network model trained successfully.")
    return model
