# DATA-TRAINING — Accident Severity Prediction

## 📊 Project Overview

This project is a Python-based machine learning project developed as part of my **Data Science Programming** studies.

The project uses a **Road Traffic Accident (RTA) dataset** to analyze accident-related information and predict **accident severity** based on selected factors such as driver age group, weather conditions, road surface conditions, vehicle movement, time, and the reported cause of the accident.

The project demonstrates the process of preparing real-world data and applying machine learning techniques to a classification problem.

---

## 🎯 Objectives

The main objectives of this project are to:

* Explore and understand a real-world road traffic accident dataset.
* Clean and prepare data for machine learning.
* Convert time information into a numerical format.
* Process categorical variables using encoding techniques.
* Split the dataset into training and testing sets.
* Train a machine learning model to predict accident severity.
* Evaluate the performance of the trained model.
* Make predictions using hypothetical accident information.
* Save the trained model for future use.

---

## 🧠 Machine Learning Approach

The project uses a **Random Forest Classifier** to predict accident severity.

Random Forest was selected because it is well suited to classification problems and can work effectively with a combination of numerical and categorical features.

### Selected Features

The model uses the following variables:

* Time
* Age band of driver
* Weather conditions
* Cause of accident
* Road surface conditions
* Vehicle movement

### Target Variable

The target variable is:

**Accident severity**

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation and analysis
* **Scikit-learn** — machine learning and model evaluation
* **Joblib** — saving the trained machine learning model
* **CSV** — dataset format

---

## 📁 Repository Contents

| File                     | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `accident-severity.py`   | Python source code for data preparation, model training, evaluation, and prediction |
| `RTA Dataset.csv.zip`    | Road Traffic Accident dataset used for the project                                  |
| `Accident Severity.docx` | Project documentation and report                                                    |
| `README.md`              | Project overview and documentation                                                  |

---

## 🔄 Project Workflow

The project follows these main steps:

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
Train/Test Split
      ↓
Random Forest Model Training
      ↓
Model Evaluation
      ↓
Accident Severity Prediction
      ↓
Model Saving
```

---

## 📈 Model Evaluation

The dataset is divided into:

* **80% training data**
* **20% testing data**

The model is evaluated using:

* Accuracy
* Classification Report
* Precision
* Recall
* F1-score

The exact performance results are generated when the Python program is executed.

---

## 🔮 Example Prediction

The program includes a hypothetical accident scenario using information such as:

* Time: 10:00 AM
* Driver age band: 18–30
* Weather: Normal
* Road surface: Dry
* Cause: No distancing
* Vehicle movement: Going straight

The trained model uses these characteristics to predict the expected accident severity.

---

## 💡 Learning Outcomes

Through this project, I gained practical experience in:

* Python programming
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Categorical data encoding
* Machine learning classification
* Training and testing machine learning models
* Model evaluation
* Saving trained machine learning models
* Working with real-world datasets

---

## 🚀 Future Improvements

Future versions of this project could include:

* Comparing multiple machine learning algorithms.
* Performing hyperparameter tuning.
* Improving feature engineering.
* Adding more data visualizations.
* Handling class imbalance using additional techniques.
* Improving model performance through cross-validation.
* Developing a web interface where users can enter accident information and receive a prediction.
* Deploying the model as an API.

---

## 📚 Data Source

The dataset used in this project was obtained from **Kaggle** and contains road traffic accident information used for educational machine learning purposes.

---

## 👨‍💻 Author

**Halcion Muthuri**

Software Engineering Student
**Zetech University, Kenya**

---

## ⚠️ Disclaimer

This project was developed for **academic and learning purposes**. The predictions produced by the model should not be considered a substitute for professional road-safety analysis or official accident investigation.

---

⭐ **Thank you for visiting this project.**
