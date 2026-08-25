"""
Accident Severity Prediction
============================

A machine learning project that predicts road traffic accident severity
using selected driver, road, weather, vehicle, and accident-cause features.

The program also generates visualizations to help understand the dataset.

Author: Halcion Muthuri
Institution: Zetech University, Kenya
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    """Convert HH:MM:SS into seconds since midnight."""

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

    required_columns = features + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"The following required columns are missing: {missing_columns}"
        )

    data = data[required_columns].copy()

    data = data.dropna()

    print(f"\nRows after removing missing values: {len(data):,}")

    data["Time_seconds"] = data["Time"].apply(time_to_seconds)

    data = data.dropna(subset=["Time_seconds"])

    data["Time_seconds"] = data["Time_seconds"].astype(int)

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

    y = data[target]

    return data, X, y


# ============================================================
# 4. Data Visualizations
# ============================================================

def create_visualizations(data):
    """Create charts for basic exploratory data analysis."""

    print("\nCreating visualizations...")

    # --------------------------------------------------------
    # Chart 1: Accident Severity Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    data["Accident_severity"].value_counts().plot(
        kind="bar"
    )

    plt.title("Accident Severity Distribution")
    plt.xlabel("Accident Severity")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Chart 2: Accidents by Weather Condition
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    data["Weather_conditions"].value_counts().plot(
        kind="bar"
    )

    plt.title("Accidents by Weather Condition")
    plt.xlabel("Weather Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Chart 3: Accidents by Driver Age Group
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    data["Age_band_of_driver"].value_counts().plot(
        kind="bar"
    )

    plt.title("Accidents by Driver Age Group")
    plt.xlabel("Driver Age Group")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Chart 4: Accident Severity by Weather
    # --------------------------------------------------------

    severity_weather = pd.crosstab(
        data["Weather_conditions"],
        data["Accident_severity"]
    )

    severity_weather.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Accident Severity by Weather Condition")
    plt.xlabel("Weather Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Accident Severity")
    plt.tight_layout()
    plt.show()

    print("Visualizations completed.")


# ============================================================
# 5. Build Machine Learning Pipeline
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

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier)
        ]
    )

    return pipeline


# ============================================================
# 6. Train Model
# ============================================================

def train_model(model, X_train, y_train):
    """Train the machine learning model."""

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    return model


# ============================================================
# 7. Evaluate Model
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
# 8. Make Example Prediction
# ============================================================

def make_prediction(model):
    """Predict accident severity for a hypothetical accident."""

    hypothetical_accident = pd.DataFrame({
        "Time_seconds": [10 * 3600],
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
# 9. Save Model
# ============================================================

def save_model(model):
    """Save the trained machine learning model."""

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved as: {MODEL_FILE}")


# ============================================================
# 10. Main Program
# ============================================================

def main():

    print("=" * 50)
    print("ACCIDENT SEVERITY PREDICTION")
    print("=" * 50)

    # Load dataset
    data = load_dataset()

    # Prepare data
    data, X, y = prepare_data(data)

    # Create visualizations
    create_visualizations(data)

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

    # Save model
    save_model(model)

    print("\n" + "=" * 50)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 50)


# ============================================================
# Run Program
# ============================================================

if __name__ == "__main__":
    main()
