"""
neural_network.py
-----------------
Neural Network model with proper feature scaling.
"""

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_neural_network(X_train, y_train):
    """
    Train a Neural Network with standardized features.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained pipeline (scaler + model)
    """
    
    # Create pipeline with scaling
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42,
            verbose=False
        ))
    ])
    
    model.fit(X_train, y_train)
    
    print("✅ Neural Network trained with feature scaling.")
    return model