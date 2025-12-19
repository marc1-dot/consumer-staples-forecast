"""
neural_network.py
-----------------
Multi-layer perceptron (MLP) regressor with proper scaling and regularization.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_neural_network(X_train, y_train):
    """
    Train a neural network with:
    - StandardScaler for feature normalization (CRITICAL for NN)
    - Simplified architecture (2 hidden layers)
    - Early stopping to prevent overfitting
    - Adam optimizer with adaptive learning rate
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model (Pipeline with scaler + MLP)
    """
    # Create model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),  # Normalize features
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32),     # 2 layers: 64 → 32 neurons
            activation='relu',                # ReLU activation
            solver='adam',                    # Adam optimizer
            alpha=0.01,                       # L2 regularization (prevent overfitting)
            batch_size=32,                    # Mini-batch size
            learning_rate='adaptive',         # Adaptive learning rate
            learning_rate_init=0.001,         # Initial learning rate
            max_iter=500,                     # Max epochs
            early_stopping=True,              # Stop if validation loss doesn't improve
            validation_fraction=0.15,         # 15% for validation
            n_iter_no_change=20,              # Stop after 20 epochs without improvement
            random_state=42,
            verbose=False
        ))
    ])
    
    # Train model
    print("   Training Neural Network with {} samples...".format(len(X_train)))
    model.fit(X_train, y_train)
    print("   Training complete!")
    
    return model