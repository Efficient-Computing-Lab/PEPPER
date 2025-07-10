# -*- coding: utf-8 -*-
"""
This script demonstrates an XGBoost regression example for predicting
the actual inference times based on system and model characteristics.

It includes:
1. Data loading from multiple CSV files in a specified directory.
2. Data preprocessing, including converting execution time strings to numerical
   seconds. The 'execution_time' column will now be the direct target variable.
3. **Ten-fold cross-validation** for robust model evaluation.
4. Initializing and training an XGBoost Regressor within a scikit-learn Pipeline,
   which includes preprocessing steps (scaling and one-hot encoding).
5. Saving the trained model to a file (from a single train/test split for later specific prediction).
6. Loading the trained model in the evaluation function.
7. Making predictions and evaluating the model's performance with regression metrics
   (MAE, MSE, RMSE, R2-score) and visualization of predicted vs. actual values,
   and feature importance.
   All generated figures will now be saved to files instead of being displayed.
8. Added a section to demonstrate how to predict inference time for a specific,
   user-defined set of model features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib # Import joblib for saving/loading models
from xgboost import XGBRegressor # Import XGBoost Regressor

# --- Configuration Constants ---
# Ensure this is correct for your environment
# NOTE: Replace with a valid path on your system where CSVs are located.
# For example, CSV_DIRECTORY = 'data/' if your CSVs are in a 'data' folder
# relative to your script.
CSV_DIRECTORY = '/home/gkorod/Downloads/mydataset/'
TARGET_COLUMN_NAME = 'execution_time' # Now a continuous target

ALL_CSV_COLUMNS = [
    'conv_layers', 'cpu_usage_percent', 'device', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'disk_usage_percent', 'end_timestamp', 'execution_number',
    'execution_time', 'filter_details', 'fully_conn_layers', 'memory_usage_percent',
    #'network_type'
    'pool_layers', 'start_timestamp', 'total_parameters'
]
FEATURE_COLUMNS = [
    'conv_layers', 'cpu_usage_percent', 'device', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'disk_usage_percent', 'fully_conn_layers',
    'memory_usage_percent',
    #'network_type', # Include network_type for demonstration if it's in your data
    'pool_layers',
    'total_parameters'
]
NUMERICAL_FEATURES = [
    'conv_layers', 'cpu_usage_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'disk_usage_percent', 'fully_conn_layers',
    'memory_usage_percent', 'pool_layers', 'total_parameters'
]
CATEGORICAL_FEATURES = [
    'device',
    #'network_type' # Include network_type for demonstration if it's in your data
]


# --- Main Execution Flow ---
def main():
    model_path = 'best_trained_xgboost_model.joblib'  # Path to save/load the trained model
    given_model_name = "ONNX-model"
    specific_model_features = pd.DataFrame([{
        'conv_layers': 7,
        'cpu_usage_percent': 45.2,
        'device': 0, # 0 means that the device is a RPi
        'disk_io_read_bytes': 15000,
        'disk_io_write_bytes': 8000,
        'disk_usage_percent': 55.0,
        'fully_conn_layers': 3,
        'memory_usage_percent': 70.0,
        'pool_layers': 4,
        'total_parameters': 25000000
    }])

    specific_model_features = specific_model_features[FEATURE_COLUMNS]

    try:
        loaded_for_prediction_pipeline = joblib.load(model_path)
        predicted_inference_time = loaded_for_prediction_pipeline.predict(specific_model_features)

        print(f"\nFeatures for specific model:\n{specific_model_features}")
        output = {"model": given_model_name, "predicted_inference_time_seconds": float(predicted_inference_time[0])}
        print(f"Predicted inference time: {output}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Cannot make specific prediction.")
    except Exception as e:
        print(f"Error making specific prediction: {e}")

    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()