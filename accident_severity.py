"""
Accident Severity Prediction
============================

A machine learning project that predicts road traffic accident severity
using selected driver, road, weather, vehicle, and accident-cause features.

Author: Halcion Muthuri
Institution: Zetech University, Kenya
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "RTA Dataset.csv.zip"
MODEL_FILE = "accident_severity_model.pkl"

RANDOM_STATE = 50
TEST_SIZE = 0.20


# ============================================================
# 1. Load Dataset
# ============================================================

def load_dataset():
    """Load the road traffic accident dataset."""

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Make sure the dataset is in the same folder as this Python file."
        )

    data = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print(f"Number of rows: {data.shape[0]:,}")
    print(f"Number of columns: {data.shape[1]}")

    return data


# ============================================================
# 2. Convert Time to Seconds
# ============================================================

def time_to_seconds(time_value):
    """
    Convert a time value in HH:MM:SS format into seconds since midnight.
    """

    try:
        hours, minutes, seconds = map(int, str(time_value).split(":"))
        return hours * 3600 + minutes * 60 + seconds

    except (ValueError, TypeError):
        return None


# ============================================================
# 3. Prepare Data
# ============================================================

def prepare_data(data):
    """Select relevant variables and prepare the dataset."""

    features = [
        "Time",
        "Age_band_of_driver",
        "Weather_conditions",
        "Cause_of_accident",
        "Road_surface_conditions",
        "Vehicle_movement"
    ]

    target = "Accident_severity"

    # Check that required columns exist
    required_columns = features + [target]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"The following required columns are missing: {missing_columns}"
        )

    # Keep only relevant columns
    data = data[required_columns].copy()

    # Remove missing values
    data = data.dropna()

    print(f"\nRows after removing missing values: {len(data):,}")

    # Convert Time into seconds
    data["Time_seconds"] = data["Time"].apply(time_to_seconds)

    # Remove rows where time conversion failed
    data = data.dropna(subset=["Time_seconds"])

    data["Time_seconds"] = data["Time_seconds"].astype(int)

    # Define input features
    X = data[
        [
            "Time_seconds",
            "Age_band_of_driver",
            "Weather_conditions",
            "Cause_of_accident",
            "Road_surface_conditions",
            "Vehicle_movement"
        ]
    ]

    # Define target
    y = data[target]

    return X, y


# ============================================================
# 4. Build Machine Learning Pipeline
# ============================================================

def build_model():
    """Create preprocessing and Random Forest classification pipeline."""

    categorical_features = [
        "Age_band_of_driver",
        "Weather_conditions",
        "Cause_of_accident",
        "Road_surface_conditions",
        "Vehicle_movement"
    ]

    numerical_features = [
        "Time_seconds"
    ]

    # One-hot encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    # Random Forest classifier
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    # Combine preprocessing and model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier)
        ]
    )

    return pipeline


# ============================================================
# 5. Train Model
# ============================================================

def train_model(model, X_train, y_train):
    """Train the machine learning model."""

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    return model


# ============================================================
# 6. Evaluate Model
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 30)
    print("MODEL EVALUATION")
    print("=" * 30)

    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return predictions, accuracy


# ============================================================
# 7. Make Example Prediction
# ============================================================

def make_prediction(model):
    """Predict accident severity for a hypothetical accident."""

    hypothetical_accident = pd.DataFrame({
        "Time_seconds": [10 * 3600],  # 10:00 AM
        "Age_band_of_driver": ["18-30"],
        "Weather_conditions": ["Normal"],
        "Cause_of_accident": ["No distancing"],
        "Road_surface_conditions": ["Dry"],
        "Vehicle_movement": ["Going straight"]
    })

    prediction = model.predict(hypothetical_accident)

    print("=" * 30)
    print("EXAMPLE PREDICTION")
    print("=" * 30)

    print(
        f"Predicted Accident Severity: {prediction[0]}"
    )

    return prediction[0]


# ============================================================
# 8. Save Model
# ============================================================

def save_model(model):
    """Save the trained machine learning model."""

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved as: {MODEL_FILE}")


# ============================================================
# 9. Main Program
# ============================================================

def main():

    print("=" * 50)
    print("ACCIDENT SEVERITY PREDICTION")
    print("=" * 50)

    # Load dataset
    data = load_dataset()

    # Prepare data
    X, y = prepare_data(data)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDataset split successfully.")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    # Build model
    model = build_model()

    # Train model
    model = train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate model
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Example prediction
    make_prediction(model)

    # Save trained model
    save_model(model)

    print("\n" + "=" * 50)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 50)


# ============================================================
# Run Program
# ============================================================

if __name__ == "__main__":
    main()
