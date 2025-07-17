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
MODEL_PATH = 'best_trained_xgboost_model.joblib' # Path to save/load the trained model
ALL_CSV_COLUMNS = [
    'conv_layers', 'total_cpu_usage_percent','device_cpu_cores', 'device_load_percent', 'device', 'disk_io_read_bytes',
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
    'conv_layers', 'disk_io_read_bytes', 'device_load_percent',
    'disk_io_write_bytes', 'device_disk_usage_percent', 'fully_conn_layers',
    'pool_layers', 'total_parameters'
]
CATEGORICAL_FEATURES = [
    'device',
]

# --- Functions ---

def load_data_from_csvs(directory: str) -> pd.DataFrame | None:
    """
    Loads and concatenates all CSV files from a specified directory into a single DataFrame.
    Returns None if no files are found or directory error.
    """
    print(f"\n--- 1. Loading Data from CSVs in '{directory}' ---")
    all_dataframes = []
    try:
        # Check if the directory exists
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' not found. Please ensure the path is correct.")
            return None

        # Iterate through files in the directory
        found_csv = False
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                print(f"Loading {filepath}...")
                df_temp = pd.read_csv(filepath, low_memory=False)
                all_dataframes.append(df_temp)
                found_csv = True

        if not found_csv:
            print(f"No CSV files found in '{directory}'. Please ensure your CSVs are there.")
            return None

    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None

    if not all_dataframes:
        print("No CSV files were successfully loaded.")
        return None

    df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Successfully loaded {len(all_dataframes)} CSV files.")
    print(f"Total rows in combined dataset: {df.shape[0]}")
    print(f"Total columns in combined dataset: {df.shape[1]}")
    print("\nFirst 5 rows of the combined dataset:")
    print(df.head())
    return df

def preprocess_data(df: pd.DataFrame, target_col: str,
                    feature_cols: list, numerical_cols: list, categorical_cols: list) -> tuple:
    """
    Performs data cleaning, converts target_col (time string) to float (total seconds),
    and prepares features for regression.
    Returns (X, y, preprocessor) or (None, None, None) on error.
    """
    print("\n--- 2. Data Cleaning and Preprocessing ---")
    # Map device column manually
    if 'device' in df.columns:
        df['device'] = df['device'].astype(str).str.lower()  # make lowercase and ensure string type
        df['device'] = df['device'].map({'raspberrypi': 0, 'jetson': 1}).fillna(-1)
    if 'device_cpu_cores' in df.columns:
        df['device_cpu_cores'] = df['device_cpu_cores'].replace(0, 4) # rpi 4B and jetson nano have 4 cores
        df['device_load_percent'] = (df['device_cpu_cores'] * df['device_load_percent']) / 4
        df = df.drop('device_cpu_cores', axis=1)

    if 'total_memory_usage_percent' in df.columns:
        df = df.drop('total_memory_usage_percent', axis=1)
    if 'total_cpu_usage_percent' in df.columns:
        df = df.drop('total_cpu_usage_percent', axis=1)



    # --- Debugging: Inspect the target column BEFORE conversion (now with timedelta context) ---
    print(f"\n--- Debugging '{target_col}' column (BEFORE time parsing) ---")
    print(f"Dtype of '{target_col}' before parsing: {df[target_col].dtype}")
    print(f"First 10 values of '{target_col}':")
    print(df[target_col].head(10))
    print(f"Number of unique values in '{target_col}': {df[target_col].nunique()}")
    print("-------------------------------------------\n")
    # --- END Debugging ---

    # --- Convert target_col from 'HH:MM:SS.microseconds' string to total seconds (float) ---
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame for conversion.")
        return None, None, None

    print(f"Attempting to convert '{target_col}' from time string to total seconds...")
    try:
        df[target_col] = pd.to_timedelta(df[target_col], errors='coerce')
        df[target_col] = df[target_col].dt.total_seconds()
        print(f"Successfully converted '{target_col}' to total seconds (float).")

    except Exception as e:
        print(f"Error parsing or converting '{target_col}' to total seconds: {e}")
        print("Please ensure 'execution_time' is in 'HH:MM:SS.microseconds' format or compatible.")
        return None, None, None
    # --- End time conversion ---

    # --- NEW DEBUGGING STEP: Inspect the target column AFTER conversion to seconds ---
    print(f"\n--- Debugging '{target_col}' column (AFTER conversion to seconds) ---")
    print(f"Dtype of '{target_col}' AFTER conversion: {df[target_col].dtype}")
    print(f"First 10 values of '{target_col}' AFTER conversion:")
    print(df[target_col].head(10))
    print(f"Number of NaNs in '{target_col}' AFTER conversion: {df[target_col].isnull().sum()}")
    print("-------------------------------------------\n")
    # --- END NEW DEBUGGING STEP ---


    # --- Handle missing values in the target column AFTER conversion ---
    if df[target_col].isnull().any():
        initial_rows = df.shape[0]
        df.dropna(subset=[target_col], inplace=True)
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to missing values (e.g., unparseable time strings) in target column '{target_col}'.")
        else:
            print(f"No missing values in target column '{target_col}' after time conversion.")
    else:
        print(f"No missing values in target column '{target_col}' after time conversion.")


    # --- IMPORTANT: Check if DataFrame is empty AFTER dropping rows ---
    if df.empty:
        print("Error: DataFrame became empty after handling missing values in the target column. Cannot proceed.")
        return None, None, None

    # --- Handle missing numerical feature values ---
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            print(f"Filled missing numerical values in '{col}' with its mean.")

    # --- Handle missing categorical feature values ---
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                print(f"Filled missing categorical values in '{col}' with its mode.")
            else:
                print(f"Warning: Could not determine mode for '{col}' (possibly all NaNs or empty). Not filling.")


    X = df[feature_cols].copy()
    y = df[target_col]

    print("Features (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)
    print(f"Target variable '{target_col}' summary statistics:\n{y.describe()}")


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    print("\nFeature preprocessing setup complete (MinMaxScaler for numerical, OneHotEncoder for categorical).")
    return X, y, preprocessor



def perform_cross_validation(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, save_path=MODEL_PATH):
    print("\n--- Performing 10-Fold Cross-Validation ---")

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100,
                                   learning_rate=0.1, max_depth=3, random_state=42))
    ])

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    best_mae = float('inf')
    best_model = None
    best_y_val = None
    best_y_pred = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}")
        print(f"Training indices shape: {train_idx.shape}")
        print(f"Validation indices shape: {val_idx.shape}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)

        print(f"Fold {fold} metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model_pipeline
            best_y_val = y_val
            best_y_pred = y_pred
            best_fold = fold

    # Final metrics of best model
    final_mae = mean_absolute_error(best_y_val, best_y_pred)
    final_mse = mean_squared_error(best_y_val, best_y_pred)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(best_y_val, best_y_pred)

    print(f"\n✅ Best model from Fold {best_fold} metrics:")
    print(f"MAE:  {final_mae:.4f}")
    print(f"MSE:  {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"R2:   {final_r2:.4f}")

    # Save residuals
    errors = best_y_val - best_y_pred
    plt.figure(figsize=(30, 20))
    sns.histplot(errors, kde=True, bins=30)
    plt.title(f'Residuals of Best Model (Fold {best_fold})',fontsize=40)
    plt.xlabel('Prediction Error', fontsize=40,fontweight="bold")
    plt.ylabel('Frequency', fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_model_residuals.pdf')
    print("Saved: best_model_residuals.pdf")
    plt.close()

    # Save predicted vs actual
    plt.figure(figsize=(30, 20))

    # Blue, larger bubbles with edge color for visibility
    sns.scatterplot(
        x=best_y_val,
        y=best_y_pred,
        alpha=0.6,
        color='blue',
        s=200,  # bubble size
        edgecolor='black'  # optional: to improve visibility
    )

    # Heavier red dashed diagonal line
    plt.plot(
        [best_y_val.min(), best_y_val.max()],
        [best_y_val.min(), best_y_val.max()],
        color='red',
        linestyle='--',
        linewidth=4
    )

    plt.xlabel("Actual", fontsize=40, fontweight="bold")
    plt.ylabel("Predicted", fontsize=40, fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.title("Predicted vs Actual (Best Model)", fontsize=40)
    plt.tight_layout()
    plt.savefig('best_model_pred_vs_actual.pdf')
    print("Saved: best_model_pred_vs_actual.pdf")
    plt.close()

    # Save MAE error distribution (absolute errors)
    abs_errors = np.abs(best_y_val - best_y_pred)
    plt.figure(figsize=(30, 20))
    sns.histplot(abs_errors, kde=True, bins=30)
    plt.title("Mean Absolute Errors (MAE) - Best Model",fontsize=40)
    plt.xlabel("Mean Absolute Error", fontsize=40,fontweight="bold")
    plt.ylabel("Frequency", fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_model_mae_distribution.pdf')
    print("Saved: best_model_mae_distribution.pdf")
    plt.close()

    # Save squared error distribution (MSE)
    squared_errors = (best_y_val - best_y_pred) ** 2
    plt.figure(figsize=(30, 20))
    sns.histplot(squared_errors, kde=True, bins=30)
    plt.title("Mean Squared Errors (MSE) - Best Model", fontsize=40)
    plt.xlabel("Mean Squared Error", fontsize=40,fontweight="bold")
    plt.ylabel("Frequency", fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_model_mse_distribution.pdf')
    print("Saved: best_model_mse_distribution.pdf")
    plt.close()

    # Save RMSE distribution (root of squared errors)
    rmse_errors = np.sqrt(squared_errors)
    plt.figure(figsize=(30, 20))
    sns.histplot(rmse_errors, kde=True, bins=30)
    plt.title("Root Mean Squared Errors (RMSE) - Best Model", fontsize=40)
    plt.xlabel("Root Mean Squared Error", fontsize=40,fontweight="bold")
    plt.ylabel("Frequency", fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_model_rmse_distribution.pdf')
    print("Saved: best_model_rmse_distribution.pdf")
    plt.close()

    # Save the model
    if best_model:
        joblib.dump(best_model, save_path)
        print(f"✅ Best model saved to {save_path}")

    return best_model

# --- Main Execution Flow ---
def main():
    """
    Main function to orchestrate the entire model training and evaluation process.
    """
    # Load Data
    df = load_data_from_csvs(CSV_DIRECTORY)
    if df is None:
        print("Data loading failed. Exiting.")
        return

    # Preprocess Data
    X, y, preprocessor = preprocess_data(df, TARGET_COLUMN_NAME,
                                         FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    if X is None or y is None or preprocessor is None:
        print("Data preprocessing failed. Exiting.")
        return

    # Ensure consistent column order for preprocessing
    X = X[FEATURE_COLUMNS]

    # --- Perform Ten-Fold Cross-Validation ---
    # This provides a more robust estimate of model performance
    # This method will train and run model 10 times and find the one with the best performance
    perform_cross_validation(X, y, preprocessor)
    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()