# Script for Tab 1 (Classification)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle  # To save the model

# Train and save the model
def train_and_save_model():
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["mass (g)", "year", "reclat", "reclong", "fall"])
    X = df[["mass (g)", "year", "reclat", "reclong"]]
    y = df["fall"].apply(lambda x: 1 if x == "Fell" else 0)  # Binary: Fell (1), Found (0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    with open("algorithms/classification_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Predict using the model
def predict(input_data):
    # Load the saved model
    with open("algorithms/classification_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Convert input_data to a DataFrame with proper feature names
    feature_names = ["mass (g)", "year", "reclat", "reclong"]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]
    prediction = "Fell" if probabilities[1] > probabilities[0] else "Found"
    confidence = max(probabilities)

    return prediction, confidence

# Train and save the model when the script is run
if __name__ == "__main__":
    train_and_save_model()
