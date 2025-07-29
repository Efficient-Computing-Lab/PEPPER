import pandas as pd
import numpy as np
import os
import joblib
import random

# --- Force TensorFlow to use CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ----------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration Constants for LSTM Model Assets ---
MODEL_PATH = 'best_trained_lstm_model_cpu.h5'
PREPROCESSOR_PATH = 'preprocessor_lstm_cpu.joblib'
TARGET_SCALER_PATH = 'target_scaler_lstm_cpu.joblib'

# General Data Configuration (from your original script)
CSV_DIRECTORY = '/home/gkorod/Downloads/mydataset/'
TARGET_COLUMN_NAME = 'execution_time'

FEATURE_COLUMNS = [
    'conv_layers', 'device_load_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent',
    'filter_details',
    'fully_conn_layers',
    'device',
    'pool_layers', 'total_parameters'
]
NUMERICAL_FEATURES = [
    'conv_layers', 'disk_io_read_bytes', 'device_load_percent',
    'disk_io_write_bytes', 'device_disk_usage_percent', 'fully_conn_layers',
    'pool_layers', 'total_parameters'
]
CATEGORICAL_FEATURES = [
    'device',
    'filter_details'
]

def model_characteristics(model_name, device_type, device_cpu_usage, disk_usage):
    characteristics = {}
    if model_name == "alexnet.onnx":
        characteristics = {
            'conv_layers': 5, 'device_load_percent': device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '1376',
            'fully_conn_layers': 3, 'pool_layers': 3, 'total_parameters': 60965228
        }
    elif model_name == "densenet.onnx":
        characteristics = {
            'conv_layers': 120, 'device_load_percent': device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '11240',
            'fully_conn_layers': 1, 'pool_layers': 4, 'total_parameters': 8146152
        }
    elif model_name == "efficientnet.onnx":
        characteristics = {
            'conv_layers': 56, 'device_load_percent':  device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '56184',
            'fully_conn_layers': 1, 'pool_layers': 2, 'total_parameters': 12966034
        }
    elif model_name == "deeplab.onnx":
        characteristics = {
            'conv_layers': 105, 'device_load_percent': device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 456500, 'disk_io_write_bytes': 98918400, 'device_disk_usage_percent': disk_usage,
            'filter_details': '102686',
            'fully_conn_layers': 0, 'pool_layers': 2, 'total_parameters': 41402464
        }
    elif model_name == "googlenet.onnx":
        characteristics = {
            'conv_layers': 59, 'device_load_percent':  device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '7280',
            'fully_conn_layers': 1, 'pool_layers': 15, 'total_parameters': 6998555
        }
    elif model_name == "mobilenet.onnx":
        characteristics = {
            'conv_layers': 53, 'device_load_percent':  device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '18088',
            'fully_conn_layers': 1, 'pool_layers': 2, 'total_parameters': 3539138
        }
    elif model_name == "resnet.onnx":
        characteristics = {
            'conv_layers': 152, 'device_load_percent':  device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '75712',
            'fully_conn_layers': 1, 'pool_layers': 1, 'total_parameters': 60404072
        }
    elif model_name == "vgg.onnx":
        characteristics = {
            'conv_layers': 13, 'device_load_percent':  device_cpu_usage, 'device': device_type,
            'disk_io_read_bytes': 74018, 'disk_io_write_bytes': 4128, 'device_disk_usage_percent': disk_usage,
            'filter_details': '4224',
            'fully_conn_layers': 3, 'pool_layers': 5, 'total_parameters': 138357544
        }
    else:
        print("You provided a wrong model")
        exit(0)
    return characteristics

# --- Main Execution Flow ---
def main():
    print("Loading LSTM model and preprocessors for prediction...")
    try:
        # Define custom objects for loading the Keras model.
        # Using tf.keras.losses.get and tf.keras.metrics.get for robustness.
        custom_objects = {
            'mse': tf.keras.losses.get('mse'),
            'mae': tf.keras.metrics.get('mae')
        }
        loaded_lstm_model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        loaded_target_scaler = joblib.load(TARGET_SCALER_PATH)
        print("LSTM model and preprocessors loaded successfully.")
    except FileNotFoundError:
        print(f"Error: One or more model files not found. Please ensure {MODEL_PATH}, {PREPROCESSOR_PATH}, and {TARGET_SCALER_PATH} exist.")
        return
    except Exception as e:
        print(f"An error occurred while loading model components: {e}")
        print("\n**Important Note:** This error often occurs due to a mismatch in TensorFlow/Keras versions between when the model was saved and when it's being loaded.")
        print("The most robust solution is to:")
        print("1. Ensure your TensorFlow/Keras version is the same for both training and prediction.")
        print("2. Re-train your model (using `lstm_training.py`) with explicit loss and metric objects:")
        print("   Instead of `model.compile(..., loss='mse', metrics=['mae'])`, use:")
        print("   `model.compile(..., loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])`")
        print("   Then, save the newly trained model.")
        return

    given_model_names = ["efficientnet.onnx"]
    characteristics_list = []
    device_types =[0, 0,1]
    count_device = 0

    for given_model_name in given_model_names:
        for device_type in device_types:
            if device_type == 0:
                if count_device == 0:
                    string_ending = "master"
                if count_device == 1:
                    string_ending = "worker"
                print("RaspberryPi 4B " + string_ending)
            if device_type == 1:
                print("Jetson Nano")
            cpu_usage_input = input("Enter CPU usage percentage (e.g., 55.0): ").strip()
            device_cpu_usage = float(cpu_usage_input)
            disk_usage_input = input("Enter Disk usage percentage (e.g., 70.0): ").strip()
            disk_usage = float(disk_usage_input)
            input_characteristics = {}
            if device_type == 0:
                input_characteristics = model_characteristics(given_model_name, 0, device_cpu_usage, disk_usage)
            if device_type == 1:
                input_characteristics = model_characteristics(given_model_name, 1, device_cpu_usage, disk_usage)
            characteristics_list.append(input_characteristics)
            count_device = count_device +1
        for characteristics_entry in characteristics_list:
            specific_model_features = pd.DataFrame([characteristics_entry], columns=FEATURE_COLUMNS)

            if 'device' in specific_model_features.columns:
                specific_model_features['device'] = specific_model_features['device'].map({'raspberrypi': 0, 'jetson': 1, 0:0, 1:1}).fillna(-1).astype(int)
            if 'filter_details' in specific_model_features.columns:
                specific_model_features['filter_details'] = specific_model_features['filter_details'].astype(str)

            processed_features = loaded_preprocessor.transform(specific_model_features)
            processed_features_lstm = processed_features.reshape(1, 1, processed_features.shape[1])

            try:
                predicted_scaled = loaded_lstm_model.predict(processed_features_lstm, verbose=0)
                predicted_inference_time = loaded_target_scaler.inverse_transform(predicted_scaled)

                if characteristics_entry.get("device") == 0:
                    given_device = "RaspberryPi 4B"
                elif characteristics_entry.get("device") == 1:
                    given_device = "Jetson Nano"
                else:
                    given_device = "Unknown Device"

                output = {
                    "device": given_device,
                    "model": given_model_name,
                    "predicted_inference_time_seconds": float(predicted_inference_time[0][0])
                }
                print(output)

            except Exception as e:
                print(f"Error making specific prediction with LSTM: {e}")

        select_random_device = random.choice(["Raspberrypi 4B master", "Raspberrypi 4B worker", "Jetson nano"])
        print(f"\nRandom Selection: The {given_model_name} should be executed on {select_random_device}")
        print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()