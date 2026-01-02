"""
neural_network.py
-----------------
Multi-layer perceptron (MLP) regressor with proper scaling and regularization.

Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""

import os


os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import random
import numpy as np


random.seed(42)
np.random.seed(42)

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_neural_network(X_train, y_train):
    """
    Train a neural network with:
    - StandardScaler for feature normalization (CRITICAL for NN)
    - Simplified architecture (2 hidden layers)
    - NO early stopping (for reproducibility)
    - Adam optimizer with adaptive learning rate
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model (Pipeline with scaler + MLP)
    """
    
    # Create model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),  # Normalize features (Mean=0, Std=1)
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32),     # 2 layers: 64 → 32 neurons
            activation='relu',                # ReLU activation
            solver='adam',                    # Adam optimizer
            alpha=0.02,                       # L2 regularization (increased to prevent overfitting)
            batch_size=32,                    # Mini-batch size
            learning_rate='adaptive',         # Adaptive learning rate
            learning_rate_init=0.001,         # Initial learning rate
            max_iter=800,                     # Fixed number of epochs (no early stop)
            shuffle=False,                    # DO NOT shuffle batches (for reproducibility)
            early_stopping=False,             # DISABLED for full reproducibility
            random_state=42,                  # Fix weight initialization
            verbose=False
        ))
    ])
    
    # Train model
    print("   Training Neural Network with {} samples...".format(len(X_train)))
    model.fit(X_train, y_train)
    print("   Training complete!")
    
    return model
