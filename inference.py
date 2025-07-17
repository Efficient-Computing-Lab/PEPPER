import pandas as pd
import numpy as np
from sklearn.externals.array_api_compat import device
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
    'conv_layers', 'total_device_load_percent','device_cpu_cores', 'device_load_percent', 'device', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent', 'end_timestamp', 'execution_number',
    'execution_time', 'filter_details', 'fully_conn_layers', 'total_memory_usage_percent',

    'pool_layers', 'start_timestamp', 'total_parameters'
]
FEATURE_COLUMNS = [
    'conv_layers', 'device_load_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent',
    'filter_details', 'fully_conn_layers',
    'device',
    'pool_layers', 'total_parameters'
]
NUMERICAL_FEATURES = [
    'conv_layers', 'disk_io_read_bytes', 'device_load_percent', 'device_cpu_cores',
    'disk_io_write_bytes', 'device_disk_usage_percent', 'fully_conn_layers',
    'pool_layers', 'total_parameters'
]
CATEGORICAL_FEATURES = [
    'device',
]


def model_characteristics(model_name,device_type,device_cpu_usage,disk_usage):
    if model_name == "alexnet.onnx":
        characteristics = {
            'conv_layers': 5,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 1376,
            'fully_conn_layers': 3,
            'pool_layers': 3,
            'total_parameters': 60965228
        }
    elif model_name == "densenet.onnx":
        characteristics = {
            'conv_layers': 120,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 11240,
            'fully_conn_layers': 1,
            'pool_layers': 4,
            'total_parameters': 8146152
        }
    elif model_name == "efficientnet.onnx":
        characteristics = {
            'conv_layers': 56,
            'device_load_percent':  device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 56184,
            'fully_conn_layers': 1,
            'pool_layers': 2,
            'total_parameters': 12966034
        }
    elif model_name == "deeplab.onnx":
        characteristics = {
            'conv_layers': 105,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 456500,
            'disk_io_write_bytes': 98918400,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 102686,
            'fully_conn_layers': 0,
            'pool_layers': 2,
            'total_parameters': 41402464
        }
    elif model_name == "googlenet.onnx":
        characteristics = {
            'conv_layers': 59,
            'device_load_percent':  device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 7280,
            'fully_conn_layers': 1,
            'pool_layers': 15,
            'total_parameters': 6998555
        }
    elif model_name == "mobilenet.onnx":
        characteristics = {
            'conv_layers': 53,
            'device_load_percent':  device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 18088,
            'fully_conn_layers': 1,
            'pool_layers': 2,
            'total_parameters': 3539138
        }
    elif model_name == "resnet.onnx":
        characteristics = {
            'conv_layers': 152,
            'device_load_percent':  device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 75712,
            'fully_conn_layers': 1,
            'pool_layers': 1,
            'total_parameters': 60404072
        }
    elif model_name == "vgg.onnx":
        characteristics = {
            'conv_layers': 13,
            'device_load_percent':  device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 4224,
            'fully_conn_layers': 3,
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
    given_model_names = ["vgg.onnx"]
    characteristics_list = []
    # device_type 0 = RaspberryPi 4B
    # device_type 1 = Jetson Nano
    device_types =[0,1]
    for given_model_name in given_model_names:
        for device_type in device_types:
            if device_type == 0:
                print("RaspberryPi 4B")
            if device_type == 1:
                print("Jetson Nano")
            cpu_usage_input = input("Enter CPU usage percentage (e.g., 55.0): ").strip()
            device_cpu_usage = float(cpu_usage_input)
            disk_usage_input = input("Enter Disk usage percentage (e.g., 70.0): ").strip()
            disk_usage = float(disk_usage_input)
            input_characteristics={}
            if device_type == 0:
                input_characteristics = model_characteristics(given_model_name,0,device_cpu_usage,disk_usage)
            if device_type == 1:
                input_characteristics = model_characteristics(given_model_name, 1,device_cpu_usage,disk_usage)
            characteristics_list.append(input_characteristics)

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
                print(output)


            except FileNotFoundError:
                print(f"Error: Model file '{model_path}' not found. Cannot make specific prediction.")
            except Exception as e:
                print(f"Error making specific prediction: {e}")

        select_random_device = random.choice(["Raspberrypi 4B", "Jetson Nano"])
        print(f"Random Selection: The {given_model_name} should be executed on {select_random_device}")
        print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()