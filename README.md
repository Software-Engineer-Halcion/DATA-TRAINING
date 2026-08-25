# DATA-TRAINING — Accident Severity Prediction

## 📊 Project Overview

This project is a Python-based machine learning project developed as part of my **Data Science Programming** studies.

The project uses a **Road Traffic Accident (RTA) dataset** to analyze accident-related information and predict **accident severity** based on selected factors such as driver age group, weather conditions, road surface conditions, vehicle movement, time, and the reported cause of the accident.

The project demonstrates the process of preparing real-world data, performing exploratory data analysis, training machine learning models, evaluating their performance, and comparing different approaches to handling class imbalance.

---

## 🎯 Objectives

The main objectives of this project are to:

* Explore and understand a real-world road traffic accident dataset.
* Clean and prepare data for machine learning.
* Convert time information into a numerical format.
* Process categorical variables using encoding techniques.
* Perform exploratory data analysis using visualizations.
* Split the dataset into training and testing sets.
* Train a machine learning model to predict accident severity.
* Evaluate model performance using accuracy, precision, recall, and F1-score.
* Compare standard and class-balanced machine learning approaches.
* Make predictions using hypothetical accident information.
* Save the trained model for future use.

---

## 🧠 Machine Learning Models

The project uses **Random Forest Classifiers** to predict accident severity.

Two approaches were evaluated:

1. **Standard Random Forest**
2. **Balanced Random Forest**

The Standard Random Forest was selected as the primary model because it achieved the highest overall accuracy.

The Balanced Random Forest was tested to investigate whether class weighting could improve the detection of minority classes such as Fatal Injury and Serious Injury.

---

## 🔍 Selected Features

The models use the following features:

* Time
* Age band of driver
* Weather conditions
* Cause of accident
* Road surface conditions
* Vehicle movement

### Target Variable

**Accident severity**

The target variable contains three categories:

* Fatal Injury
* Serious Injury
* Slight Injury

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation and analysis
* **Scikit-learn** — machine learning, preprocessing, and model evaluation
* **Matplotlib** — data visualization
* **Joblib** — model persistence
* **CSV** — dataset format

---

## 🔄 Project Workflow

```text
Data Collection
      ↓
Data Loading
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Time Conversion
      ↓
Categorical Data Encoding
      ↓
Exploratory Data Analysis
      ↓
Train/Test Split
      ↓
Random Forest Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Accident Severity Prediction
      ↓
Model Saving
