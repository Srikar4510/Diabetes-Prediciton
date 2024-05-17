# PIMA Diabetes Prediction

This project aims to classify individuals as either diabetic or non-diabetic using the PIMA Diabetes dataset and a Support Vector Machine (SVM) model. The dataset includes various health measurements, and the goal is to predict the diabetes status of individuals.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Making Predictions](#making-predictions)
- [Results](#results)


## Introduction

In this project, we use a Support Vector Machine (SVM) to classify individuals as diabetic or non-diabetic based on various health metrics provided in the PIMA Diabetes dataset. 

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/PIMA-Diabetes-Prediction.git
    cd PIMA-Diabetes-Prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used in this project is the PIMA Diabetes dataset, which contains 768 instances with 8 attributes each. The attributes are numerical values representing various health metrics.

- **File**: `diabetes.csv`
- **Attributes**: 
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0: Non-Diabetic, 1: Diabetic)

## Data Processing

1. Load the dataset into a pandas DataFrame.
2. Explore the dataset: view the first few rows, check the shape, generate descriptive statistics, and check the distribution of the outcome variable.
3. Separate the features (X) and the labels (Y).
4. Standardize the data using `StandardScaler`.

## Model Training

1. Split the dataset into training and testing sets using `train_test_split` from scikit-learn.
2. Train a Support Vector Machine (SVM) model with a linear kernel on the training data.

## Model Evaluation

1. Predict the labels for the training data and calculate the accuracy.
2. Predict the labels for the test data and calculate the accuracy.

## Making Predictions

1. Define the input data for prediction.
2. Convert the input data to a numpy array and reshape it for prediction.
3. Standardize the input data.
4. Use the trained model to predict the label.
5. Output whether the person is diabetic or not.

## Results

- **Accuracy score on the training data**: 78.66%
- **Accuracy score on the test data**: 77.27%

