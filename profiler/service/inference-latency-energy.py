# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import numpy as np
from model_class import MLPRegressor
# --- Load trained model components ---
# This path should point to the file created by the skorch-only training script
MODEL_PATH = "../training/best_nn_model_skorch_only.pkl"
try:
    model_data = joblib.load(MODEL_PATH)
    # The key is now 'model', not 'pipeline'
    model = model_data['model']
    y_scaler = model_data['y_scaler']
    numerical_scaler = model_data['numerical_scaler']
    categorical_encoder = model_data['categorical_encoder']
    NUMERICAL_FEATURES = model_data['numerical_features']
    CATEGORICAL_FEATURES = model_data['categorical_features']
    print("✅ Model and scalers loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}.")
    exit()
except KeyError:
    # This error occurs if the saved dictionary keys are different
    print("Error: The loaded model file does not contain the expected keys.")
    exit()

# --- Example input ---
input_dict = {
    'conv_layers': 5.0,
    'device_load_percent': 46.0,
    'disk_io_read_bytes': 137749.0,
    'disk_io_write_bytes': 4128.0,
    'device_disk_usage_percent': 58.4,
    'filter_details': 1376.0,
    'fully_conn_layers': 3.0,
    'device': 'raspberrypi',
    'pool_layers': 3.0,
    'total_parameters': 60965228.0
}
df_input = pd.DataFrame([input_dict])

# --- Preprocess input manually, as the pipeline no longer exists ---
# The same transformations from the training script must be applied here.
X_num = numerical_scaler.transform(df_input[NUMERICAL_FEATURES].values)
X_cat = categorical_encoder.transform(df_input[CATEGORICAL_FEATURES].values).toarray()
X_processed = np.concatenate((X_num, X_cat), axis=1).astype(np.float32)

# --- Predict ---
# The model object is a NeuralNetRegressor
y_pred_scaled = model.predict(X_processed)

# --- Inverse-transform targets ---
y_pred = y_scaler.inverse_transform(y_pred_scaled)

y_pred_df = pd.DataFrame(y_pred, columns=['execution_time_pred', 'cpu_energy_pred'])
print("Device: "+ input_dict.get("device"))
print("\nPredictions:")
print(y_pred_df)