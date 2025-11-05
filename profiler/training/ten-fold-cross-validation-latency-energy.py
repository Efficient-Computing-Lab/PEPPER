# -*- coding: utf-8 -*-
"""
Optimized Neural Network Multi-Output Regression for predicting inference latency and CPU energy.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

# --- Configuration ---
CSV_DIRECTORY = "/home/gkorod/energy_dataset"
TARGET_COLUMNS = ["cpu_energy_consumption"]
FEATURE_COLUMNS = [
    'conv_layers', 'device_load_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent',
    'filter_details', 'fully_conn_layers',
    'device',
    'pool_layers', 'total_parameters'
]
NUMERICAL_FEATURES = [
    'conv_layers', 'disk_io_read_bytes', 'device_load_percent',
    'disk_io_write_bytes', 'device_disk_usage_percent', 'fully_conn_layers',
    'pool_layers', 'total_parameters', 'filter_details'
]
CATEGORICAL_FEATURES = ['device']


# --- Normalize cpu_energy_consumption ---
def normalize_energy(val):
    val = round(float(val), 3)
    int_part, dec_part = str(val).split(".")
    if len(int_part) == 2:
        return float(f"3.{dec_part}")
    return val


# --- Load CSVs ---
def load_data_from_csvs(directory: str):
    all_dataframes = []
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' not found.")
        return None
    found_csv = False
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.csv'):
                filepath = os.path.join(root, filename)
                df_temp = pd.read_csv(filepath, low_memory=False)
                all_dataframes.append(df_temp)
                found_csv = True
    if not found_csv:
        print(f"No CSV files found in '{directory}'.")
        return None
    return pd.concat(all_dataframes, ignore_index=True)


# --- Preprocessing ---
def preprocess_data(df: pd.DataFrame):
    df.dropna(subset=TARGET_COLUMNS, inplace=True)
    if df.empty:
        return None, None, None, None, None
    df['device'] = df['device'].astype(str).str.lower()
    if 'device_load_percent' in df.columns and 'device_cpu_cores' in df.columns:
        df['device_cpu_cores'] = df['device_cpu_cores'].replace(0, 4)
        df['device_load_percent'] = (df['device_cpu_cores'] * df['device_load_percent']) / 4
    if 'execution_time' in df.columns:
        df['execution_time'] = pd.to_timedelta(df['execution_time'], errors='coerce').dt.total_seconds()
    if 'cpu_energy_consumption' in df.columns:
        df['cpu_energy_consumption'] = df['cpu_energy_consumption'].apply(normalize_energy)
    for col in ['total_memory_usage_percent', 'total_cpu_usage_percent', 'device_cpu_cores']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()
    for col in NUMERICAL_FEATURES:
        if col in X.columns and X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)
    for col in CATEGORICAL_FEATURES:
        if col in X.columns and X[col].isnull().any():
            X[col].fillna(X[col].mode()[0], inplace=True)
    y_scaler = RobustScaler()
    return X, y, y_scaler


# --- Neural Network ---
class MLPRegressor(nn.Module):
    def __init__(self, num_features, hidden_size=128, num_outputs=2):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Cross-validation with skorch only ---
def perform_cross_validation_nn_skorch_only(X, y, y_scaler, save_path="best_nn_model_skorch_only.pkl"):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_mae, best_model, best_y_scaler = float('inf'), None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize scalers and encoders once
    numerical_scaler = RobustScaler()
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit scalers and encoder on the entire dataset to ensure consistency across folds
    X_num = X[NUMERICAL_FEATURES].values
    X_cat = X[CATEGORICAL_FEATURES].values
    numerical_scaler.fit(X_num)
    categorical_encoder.fit(X_cat)
    y_scaler.fit(y.values)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply transformations manually for this fold
        X_train_num_scaled = numerical_scaler.transform(X_train[NUMERICAL_FEATURES].values)
        X_train_cat_encoded = categorical_encoder.transform(X_train[CATEGORICAL_FEATURES].values).toarray()
        X_train_processed = np.concatenate((X_train_num_scaled, X_train_cat_encoded), axis=1).astype(np.float32)

        X_val_num_scaled = numerical_scaler.transform(X_val[NUMERICAL_FEATURES].values)
        X_val_cat_encoded = categorical_encoder.transform(X_val[CATEGORICAL_FEATURES].values).toarray()
        X_val_processed = np.concatenate((X_val_num_scaled, X_val_cat_encoded), axis=1).astype(np.float32)

        y_train_scaled = y_scaler.transform(y_train.values).astype(np.float32)
        y_val_scaled = y_scaler.transform(y_val.values).astype(np.float32)

        num_features = X_train_processed.shape[1]

        # Initialize and fit the skorch model
        net = NeuralNetRegressor(
            MLPRegressor,
            module__num_features=num_features,
            module__hidden_size=128,
            module__num_outputs=y.shape[1],
            max_epochs=100,
            lr=0.001,
            optimizer=torch.optim.Adam,
            criterion=nn.MSELoss,
            batch_size=32,
            iterator_train__shuffle=True,
            verbose=0,
            device=device
        )
        net.fit(X_train_processed, y_train_scaled)

        # Predict and inverse transform
        y_pred_scaled = net.predict(X_val_processed)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        mae = mean_absolute_error(y_val, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_val, y_pred, multioutput='raw_values'))
        r2 = r2_score(y_val, y_pred, multioutput='raw_values')
        print(f"Fold {fold} → MAE={mae}, RMSE={rmse}, R2={r2}")

        if np.mean(mae) < best_mae:
            best_mae = np.mean(mae)
            best_model = net
            best_y_scaler = y_scaler

    if best_model:
        joblib.dump({
            'model': best_model,
            'y_scaler': best_y_scaler,
            'numerical_scaler': numerical_scaler,
            'categorical_encoder': categorical_encoder,
            'numerical_features': NUMERICAL_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
        }, save_path)
        print(f"✅ Best NN model saved to {save_path}")


# --- Main ---
def main():
    df = load_data_from_csvs(CSV_DIRECTORY)
    if df is None:
        print("Data loading failed.")
        return
    X, y, y_scaler = preprocess_data(df)
    if X is None or y is None:
        print("Preprocessing failed.")
        return
    perform_cross_validation_nn_skorch_only(X, y, y_scaler)


if __name__ == "__main__":
    main()