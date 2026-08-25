# Accident Severity Prediction
# Author: Halcion Muthuri
# Description:
# A machine learning project that predicts road traffic accident
# severity using selected driver, road, weather, and accident factors.

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Load Dataset
# ============================================================

DATA_PATH = "RTA Dataset.csv.zip"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Make sure 'RTA Dataset.csv.zip' "
        "is in the same folder as this Python file."
    )

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")


# ============================================================
# 2. Select Relevant Features
# ============================================================

features = [
    "Time",
    "Age_band_of_driver",
    "Weather_conditions",
    "Cause_of_accident",
    "Road_surface_conditions",
    "Vehicle_movement"
]

target = "Accident_severity"

data = data[features + [target]].copy()


# ============================================================
# 3. Remove Missing Values
# ============================================================

data = data.dropna()

print(f"\nRows after removing missing values: {len(data)}")


# ============================================================
# 4. Convert Time to Seconds
# ============================================================

def time_to_seconds(time_value):
    """Convert HH:MM:SS into seconds since midnight."""
    try:
        hours, minutes, seconds = map(int, str(time_value).split(":"))
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError):
        return None


data["Time_seconds"] = data["Time"].apply(time_to_seconds)

data = data.dropna(subset=["Time_seconds"])

data["Time_seconds"] = data["Time_seconds"].astype(int)


# ============================================================
# 5. Prepare Features and Target
# ============================================================

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


# ============================================================
# 6. Define Categorical and Numerical Features
# ============================================================

categorical_features = [
    "Age_band_of_driver",
    "Weather_conditions",
    "Cause_of_accident",
    "Road_surface_conditions",
    "Vehicle_movement"
]

numerical_features = ["Time_seconds"]


# ============================================================
# 7. Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=50,
    stratify=y
)

print("\nDataset split successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 8. Preprocessing
# ============================================================

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


# ============================================================
# 9. Create Machine Learning Model
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=50,
    class_weight="balanced_subsample",
    n_jobs=-1
)


# ============================================================
# 10. Create Pipeline
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 11. Train Model
# ============================================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 12. Evaluate Model
# ============================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# 13. Example Prediction
# ============================================================

hypothetical_accident = pd.DataFrame({
    "Time_seconds": [36000],  # 10:00 AM
    "Age_band_of_driver": ["18-30"],
    "Weather_conditions": ["Normal"],
    "Cause_of_accident": ["No distancing"],
    "Road_surface_conditions": ["Dry"],
    "Vehicle_movement": ["Going straight"]
})

prediction = pipeline.predict(hypothetical_accident)

print("\n==============================")
print("EXAMPLE PREDICTION")
print("==============================")

print(f"Predicted Accident Severity: {prediction[0]}")


# ============================================================
# 14. Save Trained Model
# ============================================================

MODEL_NAME = "accident_severity_model.pkl"

joblib.dump(pipeline, MODEL_NAME)

print(f"\nModel saved as: {MODEL_NAME}")


# ============================================================
# End of Project
# ============================================================

print("\nAccident Severity Prediction completed successfully.")
