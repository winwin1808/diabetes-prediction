# Diabetes Prediction Using LSTM

## Overview
This project is a machine-learning model for predicting diabetes using patient data. The model is built using a traditional ML model,  a Long Short-Term Memory (LSTM) network and implemented in PyTorch. It is designed to work with sequential or time-series patient data.
![image](https://github.com/winwin1808/diabetes-prediction/assets/78141233/65560e98-0e2d-427a-9186-90a0af34fcf2)
![image](https://github.com/winwin1808/diabetes-prediction/assets/78141233/afc85333-0be9-43c5-9fed-8a7c930e9f25)


## Features
- Predicts the likelihood of diabetes based on health indicators.
- Uses LSTM to capture temporal dependencies in patient data.
- Implemented in Python using PyTorch and Flask.

## Installation

Before you begin, ensure you have Python installed on your system.

1. **Clone the Repository**
git@github.com:winwin1808/diabetes-prediction.git


## Usage

1. **Start the Flask Server**
This will start the web application on your local server.

2. **Access the Web Application**
Navigate to `http://127.0.0.1:5000/` in your web browser.

3. **Using the Application**
- Input the patient's health data or upload a CSV file for prediction.
- Submit the data to receive the model's prediction.

## Model Training and Evaluation

- The LSTM model is trained on a dataset of patient records.
- Evaluate the model using standard metrics like accuracy, ROC, and precision-recall curves.

