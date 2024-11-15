# Script for Tab 4 (Regression - Predict Year

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import pickle  # To save and load the model

# Train and save the model
def train_and_save_model():
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["mass (g)", "reclat", "reclong", "year"])  # Clean data

    # Features and target
    X = df[["mass (g)", "reclat", "reclong"]]
    y = df["year"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train, y_train)

    # Save the model
    with open("algorithms/regression2_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Predict the year and provide a confidence score
def predict(input_data):
    # Load the saved model
    with open("algorithms/regression2_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Convert input_data to DataFrame with correct feature names
    feature_names = ["mass (g)", "reclat", "reclong"]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Predict the year
    predicted_year = model.predict(input_df)[0]

    # Calculate confidence (inverse proportional to residuals in training set)
    y_pred_train = model.predict(input_df)
    confidence = np.exp(-np.abs(predicted_year - y_pred_train).mean())

    return predicted_year, confidence

# Train and save the model when the script is run
if __name__ == "__main__":
    train_and_save_model()
