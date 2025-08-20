import pandas as pd
import numpy as np
import joblib
import random
import sys
from datetime import datetime
from collections import defaultdict

CSV_DIRECTORY = '/home/gkorod/Downloads/mydataset/'
TARGET_COLUMN_NAME = 'execution_time'  # continuous target

FEATURE_COLUMNS = [
    'conv_layers', 'device_load_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent',
    'filter_details', 'fully_conn_layers',
    'device',
    'pool_layers', 'total_parameters'
]


def model_characteristics(model_name, device_type, device_cpu_usage, disk_usage):
    # This function is already correct and doesn't need changes.
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
            'device_load_percent': device_cpu_usage,
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
            'conv_layers': 149,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 456500,
            'disk_io_write_bytes': 98918400,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 102686,
            'fully_conn_layers': 1,
            'pool_layers': 1,
            'total_parameters': 41402464
        }
    elif model_name == "deeplab_part1.onnx":
        characteristics = {
            'conv_layers': 143,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 456500,
            'disk_io_write_bytes': 23347328,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 100632,
            'fully_conn_layers': 1,
            'pool_layers': 1,
            'total_parameters': 40925519
        }
    elif model_name == "deeplab_part2.onnx":
        characteristics = {
            'conv_layers': 6,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 23347328,
            'disk_io_write_bytes': 98918400,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 2054,
            'fully_conn_layers': 0,
            'pool_layers': 0,
            'total_parameters': 476949
        }
    elif model_name == "googlenet.onnx":
        characteristics = {
            'conv_layers': 59,
            'device_load_percent': device_cpu_usage,
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
            'device_load_percent': device_cpu_usage,
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
            'device_load_percent': device_cpu_usage,
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
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 4224,
            'fully_conn_layers': 3,
            'pool_layers': 5,
            'total_parameters': 138357544
        }
    elif model_name == "regnet.onnx":
        characteristics = {
            'conv_layers': 74,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 81008,
            'fully_conn_layers': 1,
            'pool_layers': 1,
            'total_parameters': 107730552
        }
    elif model_name == "nasnet.onnx":
        characteristics = {
            'conv_layers': 488,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 456500,
            'disk_io_write_bytes': 4000,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 180396,
            'fully_conn_layers': 1,
            'pool_layers': 57,
            'total_parameters': 88655162
        }
    elif model_name == "convnext.onnx":
        characteristics = {
            'conv_layers': 40,
            'device_load_percent': device_cpu_usage,
            'device': device_type,
            'disk_io_read_bytes': 74018,
            'disk_io_write_bytes': 4128,
            'device_disk_usage_percent': disk_usage,
            'filter_details': 19968,
            'fully_conn_layers': 73,
            'pool_layers': 1,
            'total_parameters': 88591464
        }
    else:
        print("You provided a wrong model")
        sys.exit(0)
    return characteristics


def main():
    model_path = 'best_trained_xgboost_model.joblib'

    model_names = ["deeplab_part2.onnx"]

    device_csvs = [
        ("/home/gkorod/evaluation/inference-deeplab_part2.onnx-raspberrypi.csv", 0, "RaspberryPi 4B master"),
        ("/home/gkorod/evaluation/inference-deeplab_part2.onnx-raspberrypi-worker.csv", 0, "RaspberryPi 4B worker"),
        ("/home/gkorod/evaluation/inference-deeplab_part2.onnx-jetson.csv", 1, "Jetson Nano"),
    ]

    try:
        loaded_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    all_runs_results = []

    for model_name in model_names:
        print(f"\n--- Processing model: {model_name} ---")
        characteristics_list = []

        # Read all CSVs and combine rows with characteristics
        for csv_file, device_type, device_label in device_csvs:
            try:
                df = pd.read_csv(csv_file)
            except FileNotFoundError:
                print(f"Error: CSV file '{csv_file}' not found.")
                continue

            for _, row in df.iterrows():
                try:
                    cpu_usage = float(row["cpu_usage"])
                    disk_usage = float(row["disk_usage"])
                except Exception as e:
                    print(f"Skipping row due to error parsing 'cpu_usage' or 'disk_usage': {e}")
                    continue

                # Convert execution_time string to seconds float
                try:
                    time_str = str(row["execution_time"])
                    if '.' in time_str:
                        h, m, s_micro = time_str.split(':')
                        s, micro = s_micro.split('.')
                        time_obj = datetime(1, 1, 1, int(h), int(m), int(s), int(micro.ljust(6, '0')))
                    else:
                        h, m, s = time_str.split(':')
                        time_obj = datetime(1, 1, 1, int(h), int(m), int(s))

                    exec_time = (
                            time_obj.hour * 3600
                            + time_obj.minute * 60
                            + time_obj.second
                            + time_obj.microsecond / 1e6
                    )
                except Exception as e:
                    print(f"Skipping row due to time parse error: {e}. Time string: '{row['execution_time']}'")
                    continue

                characteristics = model_characteristics(
                    model_name, device_type, cpu_usage, disk_usage
                )
                characteristics_list.append((characteristics, device_label, exec_time))

        # Group entries per run (each run has N devices)
        num_devices = len(device_csvs)
        runs = [
            characteristics_list[i: i + num_devices]
            for i in range(0, len(characteristics_list), num_devices)
        ]

        for run in runs:
            if len(run) != num_devices:
                print(f"Warning: Skipping an incomplete run with {len(run)} devices.")
                continue

            run_results = []

            for characteristics, label, exec_time in run:
                features_df = pd.DataFrame([characteristics])[FEATURE_COLUMNS]
                prediction = loaded_model.predict(features_df)

                run_results.append(
                    {
                        "device": label,
                        "actual_execution_time": exec_time,
                        "predicted_execution_time": float(prediction[0]),
                    }
                )

            # Find fastest devices for this specific run
            actual_fastest = min(run_results, key=lambda x: x["actual_execution_time"])["device"]
            predicted_fastest = min(run_results, key=lambda x: x["predicted_execution_time"])["device"]

            # This is the correct random choice logic, performed once per run
            random_choice = random.choice(["RaspberryPi 4B master", "RaspberryPi 4B worker", "Jetson Nano"])

            all_runs_results.append(
                {
                    "model": model_name,
                    "run_number": len(all_runs_results) + 1,
                    "actual_fastest_device": actual_fastest,
                    "predicted_fastest_device": predicted_fastest,
                    "random_selection_device": random_choice,
                    "prediction_was_correct": (predicted_fastest == actual_fastest),
                    "random_was_fastest": (random_choice == actual_fastest),
                }
            )

        print("--- Done ---")

    output_df = pd.DataFrame(all_runs_results)
    output_df.to_csv("model_inference_predictions.csv", index=False)
    print("\n✅ All results saved to 'model_inference_predictions.csv'")

    # Statistics
    total_runs = len(output_df)
    if total_runs > 0:
        prediction_correct_count = output_df["prediction_was_correct"].sum()
        random_correct_count = output_df["random_was_fastest"].sum()

        prediction_accuracy = 100.0 * prediction_correct_count / total_runs
        random_accuracy = 100.0 * random_correct_count / total_runs

        print("\n📊 Overall Statistics:")
        print(f"Total number of runs: {total_runs}")
        print(
            f"✅ Prediction correctly identified the fastest device: {prediction_accuracy:.2f}% ({prediction_correct_count}/{total_runs})"
        )
        print(
            f"🎲 Random selection identified the fastest device: {random_accuracy:.2f}% ({random_correct_count}/{total_runs})"
        )

        # Add a note about the expected random accuracy to clarify
        num_devices = len(device_csvs)
        if num_devices > 0:
            expected_random_accuracy = 100.0 / num_devices
            print(f"\n💡 Note: Expected random accuracy for {num_devices} devices is {expected_random_accuracy:.2f}%."
                  f" The observed random accuracy should be close to this value over many runs, "
                  f"and a result of 90% is highly unlikely without an error in data or logic.")
    else:
        print("⚠️ No prediction results available to compute statistics.")


if __name__ == "__main__":
    main()