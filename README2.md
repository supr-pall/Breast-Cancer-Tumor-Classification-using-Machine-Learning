# Breast Cancer Tumor Classification (ML Project)

This project is a **machine learning classifier** that predicts whether a breast tumor is **benign** or **malignant** using the built-in **Breast Cancer Wisconsin dataset** from `scikit-learn`.

## ✨ Features

- Loads standard breast cancer dataset from `sklearn.datasets`
- Exploratory data analysis (basic statistics & feature importance)
- Train–test split
- Feature scaling
- Trains two models:
  - Logistic Regression
  - Random Forest Classifier
- Evaluates models using:
  - Accuracy
  - Classification report
  - Confusion matrix
  - ROC-AUC score
- Saves the best model using `joblib`
- Simple prediction function to test with sample input

## 🧠 Tech Stack

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- joblib

## 📂 Project Structure

```text
.
├── main.py           # Main script: training, evaluation, saving model
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
