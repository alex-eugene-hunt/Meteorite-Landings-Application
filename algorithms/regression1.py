# Team 1: Abdul Aziz, Alex Hunt, Barjinder Singh
# CS-5593-995: Professor Gruenwald, FA24
# Data Mining Project
# This python code implements the regression of our project

import pandas as pd
import numpy as np
import dill as pickle  # To save and load the model

# Linear Regression Model from Scratch
class LinearRegressionScratch:
    def __init__(self):
        self.weights = None  # Model weights

    def fit(self, X, y):
        """
        Train the model using the Normal Equation.
        X: Feature matrix (m x n)
        y: Target vector (m x 1)
        """
        # Add a bias term (column of ones) to the feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # Normal Equation: (X.T * X)^(-1) * X.T * y
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predict using the trained model.
        X: Feature matrix (m x n)
        Returns: Predictions (m x 1)
        """
        # Add a bias term (column of ones) to the feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights

# Train and save the model
def train_and_save_model():
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["mass (g)", "year", "reclong", "reclat"])
    X = df[["mass (g)", "year"]].values
    y_lat = df["reclat"].values
    y_long = df["reclong"].values

    # Train models for latitude and longitude
    lat_model = LinearRegressionScratch()
    long_model = LinearRegressionScratch()
    lat_model.fit(X, y_lat)
    long_model.fit(X, y_long)

    # Save models
    with open("algorithms/regression1_lat_model.pkl", "wb") as f:
        pickle.dump(lat_model, f)
    with open("algorithms/regression1_long_model.pkl", "wb") as f:
        pickle.dump(long_model, f)

# Predict using the model
def predict(input_data):
    # Load the saved models
    with open("algorithms/regression1_lat_model.pkl", "rb") as f:
        lat_model = pickle.load(f)
    with open("algorithms/regression1_long_model.pkl", "rb") as f:
        long_model = pickle.load(f)

    # Convert input_data to a NumPy array
    input_array = np.array(input_data).reshape(1, -1)

    # Predict latitude and longitude
    predicted_lat = lat_model.predict(input_array)[0]
    predicted_long = long_model.predict(input_array)[0]

    # Clamp predictions to valid geographical bounds
    predicted_lat = max(min(predicted_lat, 90), -90)
    predicted_long = max(min(predicted_long, 180), -180)

    return predicted_lat, predicted_long

# Train and save the model when the script is run
if __name__ == "__main__":
    train_and_save_model()
