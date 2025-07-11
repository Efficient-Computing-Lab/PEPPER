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
import sys
import joblib # Import joblib for saving/loading models
from xgboost import XGBRegressor # Import XGBoost Regressor
import random
import argparse
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

def model_characteristics(model_name,device_type):
    if model_name == "alexnet.onnx":
        characteristics = {
            'conv_layers': 5,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 3,
            'memory_usage_percent': 70.0,
            'pool_layers': 3,
            'total_parameters': 60965228
        }
    elif model_name == "densenet.onnx":
        characteristics = {
            'conv_layers': 121,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 1,
            'memory_usage_percent': 70.0,
            'pool_layers': 5,
            'total_parameters': 8146152
        }
    elif model_name == "efficientnet.onnx":
        characteristics = {
            'conv_layers': 55,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 1,
            'memory_usage_percent': 70.0,
            'pool_layers': 1,
            'total_parameters': 12966034
        }
    elif model_name == "epos.onnx":
        characteristics = {
            'conv_layers': 149,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 1,
            'memory_usage_percent': 70.0,
            'pool_layers': 1,
            'total_parameters': 41402464
        }
    elif model_name == "googlenet.onnx":
        characteristics = {
            'conv_layers': 57,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 1,
            'memory_usage_percent': 70.0,
            'pool_layers': 14,
            'total_parameters': 6998555
        }
    elif model_name == "mobilenet.onnx":
        characteristics = {
            'conv_layers': 54,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 1,
            'memory_usage_percent': 70.0,
            'pool_layers': 1,
            'total_parameters': 3539138
        }
    elif model_name == "resnet.onnx":
        characteristics = {
            'conv_layers': 151,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 1,
            'memory_usage_percent': 70.0,
            'pool_layers': 2,
            'total_parameters': 60404072
        }
    elif model_name == "vgg.onnx":
        characteristics = {
            'conv_layers': 13,
            'cpu_usage_percent': 45.2,
            'device': device_type,
            'disk_io_read_bytes': 15000,
            'disk_io_write_bytes': 8000,
            'disk_usage_percent': 55.0,
            'fully_conn_layers': 3,
            'memory_usage_percent': 70.0,
            'pool_layers': 5,
            'total_parameters': 138357544
        }
    else:
        print("You provided a wrong model")
        sys.exit(0)
    return characteristics
# --- Main Execution Flow ---
def main():

    model_path = 'best_trained_xgboost_model.joblib'  # Path to save/load the trained model
    given_model_names = ["alexnet.onnx","densenet.onnx"]
    characteristics_list = []
    # device_type 0 = RaspberryPi 4B
    # device_type 1 = Jetson Nano
    for given_model_name in given_model_names:
        rpi_characteristics = model_characteristics(given_model_name,0)
        jetson_characteristics = model_characteristics(given_model_name, 1)
        characteristics_list.append(rpi_characteristics)
        characteristics_list.append(jetson_characteristics)

        for characteristics_entry in characteristics_list:
            specific_model_features = pd.DataFrame([characteristics_entry])

            specific_model_features = specific_model_features[FEATURE_COLUMNS]

            try:
                loaded_for_prediction_pipeline = joblib.load(model_path)
                predicted_inference_time = loaded_for_prediction_pipeline.predict(specific_model_features)

                if characteristics_entry.get("device") == 0:
                    given_device = "RaspberryPi 4B"
                if characteristics_entry.get("device") == 1:
                    given_device = "Jetson Nano"
                output = {"device": given_device, "model": given_model_name, "predicted_inference_time_seconds": float(predicted_inference_time[0])}
                print(f"Predicted inference time: {output}")

                select_random_device = random.choice(["Raspberrypi 4B", "Jetson Nano"])
                print(f"Random Selection: The {given_model_name} should be executed on {select_random_device}")
            except FileNotFoundError:
                print(f"Error: Model file '{model_path}' not found. Cannot make specific prediction.")
            except Exception as e:
                print(f"Error making specific prediction: {e}")

        print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()