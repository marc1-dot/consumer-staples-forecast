"""
Neural Network Model
--------------------
Simple MLP for financial time series with limited data.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_neural_network(X_train, y_train):
    """
    Train a simple neural network optimized for small datasets.
    
    Architecture:
    - Input layer → 32 neurons (ReLU) → 16 neurons (ReLU) → Output
    - L2 regularization to prevent overfitting
    - Early stopping on validation set
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained Pipeline (StandardScaler + MLPRegressor)
    """
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            tol=1e-4,
            verbose=False
        ))
    ])
    
    model.fit(X_train, y_train)
    
    return model