from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    """Trains a baseline Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Linear Regression model trained successfully.")
    return model
