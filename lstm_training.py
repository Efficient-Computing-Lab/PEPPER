import os

# --- Force TensorFlow to use CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Alternatively, you can use: os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import TensorFlow and Keras AFTER setting the environment variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- Configuration Constants ---
CSV_DIRECTORY = '/home/gkorod/Downloads/mydataset/'
TARGET_COLUMN_NAME = 'execution_time'
MODEL_PATH = 'best_trained_lstm_model_cpu.h5' # Changed path to indicate CPU model
PREPROCESSOR_PATH = 'preprocessor_lstm_cpu.joblib'
TARGET_SCALER_PATH = 'target_scaler_lstm_cpu.joblib'

ALL_CSV_COLUMNS = [
    'conv_layers', 'total_cpu_usage_percent','device_cpu_cores', 'device_load_percent', 'device', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent', 'end_timestamp', 'execution_number',
    'execution_time', 'filter_details', 'fully_conn_layers', 'total_memory_usage_percent',
    'pool_layers', 'start_timestamp', 'total_parameters'
]
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

# --- Functions ---

def load_data_from_csvs(directory: str) -> pd.DataFrame | None:
    """
    Loads and concatenates all CSV files from a specified directory into a single DataFrame.
    Returns None if no files are found or directory error.
    """
    print(f"\n--- 1. Loading Data from CSVs in '{directory}' ---")
    all_dataframes = []
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' not found. Please ensure the path is correct.")
            return None

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

    # Map device column manually and filter out unmapped values
    if 'device' in df.columns:
        df['device'] = df['device'].astype(str).str.lower()
        df['device'] = df['device'].map({'raspberrypi': 0, 'jetson': 1}).fillna(-1)
        # Filter out rows where device was not 'raspberrypi' or 'jetson' (i.e., -1)
        df = df[df['device'] != -1]
        if not df['device'].empty:
            df['device'] = df['device'].astype(int)

    # Handle 'filter_details' if it exists and ensure it's string type for OneHotEncoder
    if 'filter_details' in df.columns:
        df['filter_details'] = df['filter_details'].astype(str)
        df['filter_details'].fillna('UNKNOWN_FILTER', inplace=True) # Fill NaNs for categorical

    if 'device_cpu_cores' in df.columns:
        df['device_cpu_cores'] = df['device_cpu_cores'].replace(0, 4) # Assume 4 cores for these devices
        df['device_load_percent'] = (df['device_cpu_cores'] * df['device_load_percent']) / 4
        # Apply filter *after* calculation as per original logic
        df = df[(df['device_load_percent'] < 35) | (df['device_load_percent'] > 65)]
        df = df.drop('device_cpu_cores', axis=1)

    # Drop columns not used as features but present in ALL_CSV_COLUMNS and not needed.
    # Be careful not to drop TARGET_COLUMN_NAME.
    columns_to_drop_from_df = ['total_memory_usage_percent', 'total_cpu_usage_percent',
                               'end_timestamp', 'execution_number', 'start_timestamp']
    for col in columns_to_drop_from_df:
        if col in df.columns:
            df = df.drop(col, axis=1)
            print(f"Dropped column: {col}")


    print(f"\n--- Debugging '{target_col}' column (BEFORE time parsing) ---")
    print(f"Dtype of '{target_col}' before parsing: {df[target_col].dtype}")
    print(f"First 10 values of '{target_col}':")
    print(df[target_col].head(10))
    print(f"Number of unique values in '{target_col}': {df[target_col].nunique()}")
    print("-------------------------------------------\n")

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

    print(f"\n--- Debugging '{target_col}' column (AFTER conversion to seconds) ---")
    print(f"Dtype of '{target_col}' AFTER conversion: {df[target_col].dtype}")
    print(f"First 10 values of '{target_col}' AFTER conversion:")
    print(df[target_col].head(10))
    print(f"Number of NaNs in '{target_col}' AFTER conversion: {df[target_col].isnull().sum()}")
    print("-------------------------------------------\n")

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

    if df.empty:
        print("Error: DataFrame became empty after handling missing values in the target column. Cannot proceed.")
        return None, None, None

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            print(f"Filled missing numerical values in '{col}' with its mean.")

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

    # OneHotEncoder `sparse_output=False` is important for direct use with Keras
    # The preprocessor is only defined here; it will be fitted within the CV loop.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )

    print("\nFeature preprocessing setup complete (RobustScaler for numerical, OneHotEncoder for categorical).")
    return X, y, preprocessor

# Function to create the LSTM model
def create_lstm_model(input_shape):
    """
    Creates a Keras Sequential model with LSTM layers for regression.
    """
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def perform_cross_validation(X: pd.DataFrame, y: pd.Series, preprocessor_definition: ColumnTransformer,
                             model_save_path=MODEL_PATH, preprocessor_save_path=PREPROCESSOR_PATH,
                             target_scaler_save_path=TARGET_SCALER_PATH):
    """
    Performs 10-fold cross-validation with an LSTM model, saving the best model and scalers.
    """
    print("\n--- Performing 10-Fold Cross-Validation with LSTM ---")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    best_mae = float('inf')
    best_model_keras = None
    best_preprocessor_fitted = None
    best_target_scaler_fitted = None
    best_y_val_unscaled = None
    best_y_pred_unscaled = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}")
        print(f"Training indices shape: {train_idx.shape}")
        print(f"Validation indices shape: {val_idx.shape}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        current_preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
            ],
            remainder='passthrough'
        )
        X_train_processed = current_preprocessor.fit_transform(X_train)
        X_val_processed = current_preprocessor.transform(X_val)

        n_features = X_train_processed.shape[1]

        current_target_scaler = MinMaxScaler()
        y_train_scaled = current_target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val_scaled = current_target_scaler.transform(y_val.values.reshape(-1, 1))

        X_train_lstm = X_train_processed.reshape(X_train_processed.shape[0], 1, n_features)
        X_val_lstm = X_val_processed.reshape(X_val_processed.shape[0], 1, n_features)

        tf.keras.backend.clear_session()
        model = create_lstm_model(input_shape=(1, n_features))

        if fold == 1:
            with open('keras_model_summary_cpu.txt', 'w') as f: # Changed summary file name
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            print("Keras model summary saved to keras_model_summary_cpu.txt")

        history = model.fit(
            X_train_lstm, y_train_scaled,
            epochs=50,
            batch_size=32,
            validation_data=(X_val_lstm, y_val_scaled),
            verbose=0
        )

        y_pred_scaled = model.predict(X_val_lstm, verbose=0)
        y_pred = current_target_scaler.inverse_transform(y_pred_scaled)
        y_val_unscaled = y_val.values

        mae = mean_absolute_error(y_val_unscaled, y_pred)
        mse = mean_squared_error(y_val_unscaled, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_unscaled, y_pred)

        print(f"Fold {fold} metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model_keras = model
            best_preprocessor_fitted = current_preprocessor
            best_target_scaler_fitted = current_target_scaler
            best_y_val_unscaled = y_val_unscaled
            best_y_pred_unscaled = y_pred
            best_fold = fold

    final_mae = mean_absolute_error(best_y_val_unscaled, best_y_pred_unscaled)
    final_mse = mean_squared_error(best_y_val_unscaled, best_y_pred_unscaled)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(best_y_val_unscaled, best_y_pred_unscaled)

    print(f"\n✅ Best model from Fold {best_fold} metrics (evaluated on its validation set):")
    print(f"MAE:  {final_mae:.4f}")
    print(f"MSE:  {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"R2:   {final_r2:.4f}")

    errors = best_y_val_unscaled - best_y_pred_unscaled.flatten()
    plt.figure(figsize=(30, 20))
    sns.histplot(errors, kde=True, bins=30)
    plt.title(f'Residuals of Best LSTM Model (CPU) (Fold {best_fold})',fontsize=40)
    plt.xlabel('Prediction Error', fontsize=40,fontweight="bold")
    plt.ylabel('Frequency', fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_lstm_model_residuals_cpu.pdf') # Changed figure name
    print("Saved: best_lstm_model_residuals_cpu.pdf")
    plt.close()

    plt.figure(figsize=(30, 20))
    sns.scatterplot(
        x=best_y_val_unscaled,
        y=best_y_pred_unscaled.flatten(),
        alpha=0.6,
        color='blue',
        s=200,
        edgecolor='black'
    )
    plt.plot(
        [best_y_val_unscaled.min(), best_y_val_unscaled.max()],
        [best_y_val_unscaled.min(), best_y_val_unscaled.max()],
        color='red',
        linestyle='--',
        linewidth=4
    )
    plt.xlabel("Actual", fontsize=40, fontweight="bold")
    plt.ylabel("Predicted", fontsize=40, fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.title("Predicted vs Actual (Best LSTM Model - CPU)", fontsize=40)
    plt.tight_layout()
    plt.savefig('best_lstm_model_pred_vs_actual_cpu.pdf') # Changed figure name
    print("Saved: best_lstm_model_pred_vs_actual_cpu.pdf")
    plt.close()

    abs_errors = np.abs(best_y_val_unscaled - best_y_pred_unscaled.flatten())
    plt.figure(figsize=(30, 20))
    sns.histplot(abs_errors, kde=True, bins=30)
    plt.title("Mean Absolute Errors (MAE) - Best LSTM Model (CPU)",fontsize=40)
    plt.xlabel("Mean Absolute Error", fontsize=40,fontweight="bold")
    plt.ylabel("Frequency", fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_lstm_model_mae_distribution_cpu.pdf') # Changed figure name
    print("Saved: best_lstm_model_mae_distribution_cpu.pdf")
    plt.close()

    squared_errors = (best_y_val_unscaled - best_y_pred_unscaled.flatten()) ** 2
    plt.figure(figsize=(30, 20))
    sns.histplot(squared_errors, kde=True, bins=30)
    plt.title("Mean Squared Errors (MSE) - Best LSTM Model (CPU)", fontsize=40)
    plt.xlabel("Mean Squared Error", fontsize=40,fontweight="bold")
    plt.ylabel("Frequency", fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_lstm_model_mse_distribution_cpu.pdf') # Changed figure name
    print("Saved: best_lstm_model_mse_distribution_cpu.pdf")
    plt.close()

    rmse_errors = np.sqrt(squared_errors)
    plt.figure(figsize=(30, 20))
    sns.histplot(rmse_errors, kde=True, bins=30)
    plt.title("Root Mean Squared Errors (RMSE) - Best LSTM Model (CPU)", fontsize=40)
    plt.xlabel("Root Mean Squared Error", fontsize=40,fontweight="bold")
    plt.ylabel("Frequency", fontsize=40,fontweight="bold")
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig('best_lstm_model_rmse_distribution_cpu.pdf') # Changed figure name
    print("Saved: best_lstm_model_rmse_distribution_cpu.pdf")
    plt.close()

    if best_model_keras:
        best_model_keras.save(model_save_path)
        print(f"✅ Best LSTM model saved to {model_save_path}")
        joblib.dump(best_preprocessor_fitted, preprocessor_save_path)
        print(f"✅ Fitted preprocessor saved to {preprocessor_save_path}")
        joblib.dump(best_target_scaler_fitted, target_scaler_save_path)
        print(f"✅ Fitted target scaler saved to {target_scaler_save_path}")

    return best_model_keras

def predict_single_inference_time(model_path: str, preprocessor_path: str, target_scaler_path: str, input_data: dict, feature_columns: list):
    """
    Loads the trained LSTM model, preprocessor, and target scaler to predict
    inference time for a single set of input features.
    """
    print("\n--- Predicting Single Inference Time with Trained LSTM Model ---")
    try:
        loaded_model = keras.models.load_model(model_path)
        print(f"Loaded LSTM model from {model_path}")

        loaded_preprocessor = joblib.load(preprocessor_path)
        print(f"Loaded preprocessor from {preprocessor_path}")

        loaded_target_scaler = joblib.load(target_scaler_path)
        print(f"Loaded target scaler from {target_scaler_path}")

        input_df = pd.DataFrame([input_data], columns=feature_columns)
        print("\nInput Data for Prediction:")
        print(input_df)

        if 'device' in input_df.columns:
            input_df['device'] = input_df['device'].astype(str).str.lower().map({'raspberrypi': 0, 'jetson': 1}).fillna(-1).astype(int)
        if 'filter_details' in input_df.columns:
            input_df['filter_details'] = input_df['filter_details'].astype(str).fillna('UNKNOWN_FILTER')

        processed_input = loaded_preprocessor.transform(input_df)

        processed_input_lstm = processed_input.reshape(1, 1, processed_input.shape[1])

        scaled_prediction = loaded_model.predict(processed_input_lstm, verbose=0)

        actual_prediction = loaded_target_scaler.inverse_transform(scaled_prediction)

        print(f"\nPredicted Inference Time: {actual_prediction[0][0]:.4f} seconds")
        return actual_prediction[0][0]

    except FileNotFoundError:
        print(f"Error: Model or preprocessor/scaler files not found. Please ensure {model_path}, {preprocessor_path}, and {target_scaler_path} exist.")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# --- Main Execution Flow ---
def main():
    """
    Main function to orchestrate the entire model training and evaluation process.
    """
    df = load_data_from_csvs(CSV_DIRECTORY)
    if df is None:
        print("Data loading failed. Exiting.")
        return

    X, y, preprocessor_definition = preprocess_data(df, TARGET_COLUMN_NAME,
                                                    FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    if X is None or y is None or preprocessor_definition is None:
        print("Data preprocessing failed. Exiting.")
        return

    X = X[FEATURE_COLUMNS]

    best_lstm_model = perform_cross_validation(X, y, preprocessor_definition,
                                               model_save_path=MODEL_PATH,
                                               preprocessor_save_path=PREPROCESSOR_PATH,
                                               target_scaler_save_path=TARGET_SCALER_PATH)

    if best_lstm_model:
        print("\n" + "="*50)
        print(" DEMONSTRATING SINGLE PREDICTION ")
        print("="*50)

        sample_input = {
            'conv_layers': 10,
            'device_load_percent': 70.0,
            'disk_io_read_bytes': 1024000,
            'disk_io_write_bytes': 512000,
            'device_disk_usage_percent': 30.5,
            'filter_details': 'filter_1',
            'fully_conn_layers': 2,
            'device': 'jetson',
            'pool_layers': 3,
            'total_parameters': 5000000
        }

        predict_single_inference_time(
            MODEL_PATH,
            PREPROCESSOR_PATH,
            TARGET_SCALER_PATH,
            sample_input,
            FEATURE_COLUMNS
        )
    else:
        print("\nNo best LSTM model was trained. Cannot perform single prediction.")

    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()