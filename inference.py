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
from sklearn.linear_model import LinearRegression
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

def load_device_data(csv_path, device_type, device_label):
    try:
        df = pd.read_csv(csv_path)
        df["device_type"] = device_type
        df["device_label"] = device_label
        return df
    except FileNotFoundError:
        print(f"CSV file '{csv_path}' not found.")
        return pd.DataFrame()  # Empty dataframe

def main():
    model_path = 'best_trained_xgboost_model.joblib'

    model_names = [
        "deeplab.onnx"]

    device_csvs = [
        ("raspberrypi_master.csv", 0, "RaspberryPi 4B master"),
        ("raspberrypi_worker.csv", 0, "RaspberryPi 4B worker"),
        ("jetson_nano.csv", 1, "Jetson Nano"),
    ]

    try:
        loaded_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    output_results = []

    for model_name in model_names:
        print(f"\n--- Processing model: {model_name} ---")
        characteristics_list = []

        # Collect data for this model from all devices
        for csv_file, device_type, device_label in device_csvs:
            try:
                df = pd.read_csv(csv_file)
            except FileNotFoundError:
                print(f"Error: CSV file '{csv_file}' not found.")
                continue

            for _, row in df.iterrows():
                cpu_usage = float(row["cpu_usage"])
                disk_usage = float(row["disk_usage"])
                exec_time = float(row["execution_time"])

                characteristics = model_characteristics(
                    model_name, device_type, cpu_usage, disk_usage
                )

                characteristics_list.append((characteristics, device_label, exec_time))

        # Random device selection
        select_random_device = random.choice([label for _, _, label in device_csvs])
        print(f"🎲 Random suggestion for {model_name}: {select_random_device}")

        model_predictions = []

        for characteristics, label, exec_time in characteristics_list:
            try:
                features_df = pd.DataFrame([characteristics])[FEATURE_COLUMNS]
                prediction = loaded_model.predict(features_df)

                result = {
                    "model": model_name,
                    "device": label,
                    "cpu_usage": characteristics["device_load_percent"],
                    "disk_usage": characteristics["device_disk_usage_percent"],
                    "actual_execution_time_seconds": exec_time,
                    "predicted_inference_time_seconds": float(prediction[0]),
                    "random_selection_device": select_random_device
                }

                model_predictions.append(result)

            except Exception as e:
                print(f"Error during prediction for {label}: {e}")

        # Determine fastest actual device
        if model_predictions:
            fastest_actual_entry = min(model_predictions, key=lambda x: x["actual_execution_time_seconds"])
            fastest_predicted_entry = min(model_predictions, key=lambda x: x["predicted_inference_time_seconds"])

            fastest_actual_device = fastest_actual_entry["device"]
            fastest_predicted_device = fastest_predicted_entry["device"]

            for entry in model_predictions:
                entry["fastest_device_by_execution_time"] = fastest_actual_device
                entry["fastest_device_by_prediction"] = fastest_predicted_device
                entry["random_was_fastest"] = (entry["random_selection_device"] == fastest_actual_device)
                entry["prediction_was_correct"] = (fastest_predicted_device == fastest_actual_device)
                output_results.append(entry)

            print(f"⚡ Fastest actual device for {model_name}: {fastest_actual_device} ({fastest_actual_entry['actual_execution_time_seconds']:.4f}s)")
            print(f"🧠 Fastest predicted device: {fastest_predicted_device} ({fastest_predicted_entry['predicted_inference_time_seconds']:.4f}s)")
            print(f"✅ Random was fastest: {select_random_device == fastest_actual_device}")
            print(f"✅ Prediction was correct: {fastest_predicted_device == fastest_actual_device}")

        print("--- Done ---")

    # Save output to CSV
    output_df = pd.DataFrame(output_results)
    output_df.to_csv("model_inference_predictions.csv", index=False)
    print("\n✅ All results saved to 'model_inference_predictions.csv'")


if __name__ == "__main__":
    main()