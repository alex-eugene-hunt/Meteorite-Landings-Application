import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle  # To save and load the model

# Train and save the model
def train_and_save_model():
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["mass (g)", "year", "reclong", "reclat"])
    X = df[["mass (g)", "year"]]
    y_lat = df["reclat"]
    y_long = df["reclong"]

    # Train models
    lat_model = LinearRegression()
    long_model = LinearRegression()
    lat_model.fit(X, y_lat)
    long_model.fit(X, y_long)

    # Save models
    with open("algorithms/regression1_lat_model.pkl", "wb") as f:
        pickle.dump(lat_model, f)
    with open("algorithms/regression1_long_model.pkl", "wb") as f:
        pickle.dump(long_model, f)

def predict(input_data):
    # Load models
    with open("algorithms/regression1_lat_model.pkl", "rb") as f:
        lat_model = pickle.load(f)
    with open("algorithms/regression1_long_model.pkl", "rb") as f:
        long_model = pickle.load(f)

    # Convert input_data to a DataFrame with proper feature names
    feature_names = ["mass (g)", "year"]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Predict latitude and longitude
    predicted_lat = lat_model.predict(input_df)[0]
    predicted_long = long_model.predict(input_df)[0]

    return predicted_lat, predicted_long

# Train and save the model when the script is run
if __name__ == "__main__":
    train_and_save_model()

