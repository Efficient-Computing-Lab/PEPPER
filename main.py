# -*- coding: utf-8 -*-
"""
This script demonstrates an XGBoost regression example for predicting
the actual inference times based on system and model characteristics.

It includes:
1. Data loading from multiple CSV files in a specified directory.
2. Data preprocessing, including converting execution time strings to numerical
   seconds. The 'execution_time' column will now be the direct target variable.
3. Splitting data into training and testing sets.
4. Initializing and training an XGBoost Regressor within a scikit-learn Pipeline,
   which includes preprocessing steps (scaling and one-hot encoding).
5. Saving the trained model to a file.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
CSV_DIRECTORY = '/home/gkorod/Downloads/Raspberrypi 4B/'
TARGET_COLUMN_NAME = 'execution_time' # Now a continuous target
MODEL_PATH = 'trained_xgboost_model_pipeline.joblib' # Path to save/load the trained model
# INFERENCE_TIME_THRESHOLD_SECONDS is no longer needed for regression
ALL_CSV_COLUMNS = [
    'conv_layers', 'cpu_usage_percent', 'device', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'disk_usage_percent', 'end_timestamp', 'execution_number',
    'execution_time', 'filter_details', 'fully_conn_layers', 'memory_usage_percent',
    # 'model_name', # Removed from ALL_CSV_COLUMNS if you truly don't need it loaded
    'network_type', 'pool_layers', 'start_timestamp', 'total_parameters'
]
FEATURE_COLUMNS = [
    'conv_layers', 'cpu_usage_percent', 'device', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'disk_usage_percent', 'fully_conn_layers',
    'memory_usage_percent',
    'network_type', 'pool_layers',
    'total_parameters'
]
NUMERICAL_FEATURES = [
    'conv_layers', 'cpu_usage_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'disk_usage_percent', 'fully_conn_layers',
    'memory_usage_percent', 'pool_layers', 'total_parameters'
]
CATEGORICAL_FEATURES = [
    'device',
    'network_type'
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

    # Check for required columns
    missing_essential_cols = [col for col in [target_col] + feature_cols if col not in df.columns]
    if missing_essential_cols:
        print(f"Error: The following essential columns are missing from your data: {missing_essential_cols}")
        print("Available columns:", df.columns.tolist())
        return None, None, None

    # --- Debugging: Inspect the target column BEFORE conversion (now with timedelta context) ---
    print(f"\n--- Debugging '{target_col}' column (BEFORE time parsing) ---")
    print(f"Dtype of '{target_col}' before parsing: {df[target_col].dtype}")
    print(f"First 10 values of '{target_col}':")
    print(df[target_col].head(10))
    print(f"Number of unique values in '{target_col}': {df[target_col].nunique()}")
    print("-------------------------------------------\n")
    # --- END Debugging ---

    # --- Convert target_col from 'HH:MM:SS.microseconds' string to total seconds (float) ---
    # This step transforms the 'execution_time' column (e.g., '00:00:01.234567')
    # from a string representation of time duration into a numerical value representing
    # the total seconds. This is crucial because machine learning models require
    # numerical inputs.
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame for conversion.")
        return None, None, None

    print(f"Attempting to convert '{target_col}' from time string to total seconds...")
    try:
        # Step 1: Convert the string 'execution_time' to timedelta objects
        # `pd.to_timedelta` converts strings like 'HH:MM:SS.microseconds' into
        # pandas Timedelta objects, which represent a duration.
        # `errors='coerce'` ensures that any unparseable strings are converted to NaT (Not a Time),
        # preventing errors and allowing us to handle them later.
        df[target_col] = pd.to_timedelta(df[target_col], errors='coerce')

        # Step 2: Convert timedelta objects to total seconds (float)
        # The `.dt.total_seconds()` accessor on Timedelta objects extracts the total duration
        # in seconds as a float. NaT values (from failed timedelta conversion) will become NaN (Not a Number).
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
    # Rows where 'execution_time' could not be parsed (resulting in NaN after conversion)
    # are dropped to ensure a clean target variable for model training.
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
    # For numerical features, missing values (NaN) are filled with the mean of their
    # respective columns. This is a common imputation strategy to prevent errors
    # during model training and ensure all numerical data is complete.
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            print(f"Filled missing numerical values in '{col}' with its mean.")

    # --- Handle missing categorical feature values ---
    # For categorical features, missing values (NaN) are filled with the mode (most frequent value)
    # of their respective columns. This is a common imputation strategy for categorical data.
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            # Check if mode() returns multiple values (in case of ties) and pick the first one
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                print(f"Filled missing categorical values in '{col}' with its mode.")
            else:
                print(f"Warning: Could not determine mode for '{col}' (possibly all NaNs or empty). Not filling.")


    # For regression, the target 'y' is simply the 'execution_time' column (now numerical)
    # X contains the features used to make predictions.
    X = df[feature_cols].copy()
    y = df[target_col] # Direct use of numerical execution time as target

    print("Features (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)
    print(f"Target variable '{target_col}' summary statistics:\n{y.describe()}")


    # --- Define the preprocessing pipeline for features (ColumnTransformer) ---
    # This `ColumnTransformer` is a powerful tool from scikit-learn that allows
    # different preprocessing steps to be applied to different columns of your data.
    # It ensures that numerical features are scaled and categorical features are
    # converted into a numerical format suitable for machine learning algorithms.
    preprocessor = ColumnTransformer(
        transformers=[
            # 'num': Applies StandardScaler to all columns specified in numerical_cols.
            # StandardScaler transforms numerical features to have a mean of 0 and a
            # standard deviation of 1. This helps algorithms that are sensitive to
            # the scale of features (like gradient boosting methods or neural networks)
            # to converge faster and perform better.
            ('num', StandardScaler(), numerical_cols),
            # 'cat': Applies OneHotEncoder to all columns specified in categorical_cols.
            # OneHotEncoder converts categorical (non-numerical) features into a
            # numerical format where each category becomes a new binary (0 or 1) column.
            # `handle_unknown='ignore'` prevents errors if a category unseen during training
            # appears in the test set; it will encode it as all zeros.
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        # `remainder='passthrough'` ensures that any columns in X that are NOT
        # explicitly listed in numerical_cols or categorical_cols are kept as is
        # and passed through the transformer. (In this specific setup, all FEATURE_COLUMNS
        # are handled, so 'passthrough' might not apply to many columns but is good practice).
        remainder='passthrough'
    )

    print("\nFeature preprocessing setup complete (StandardScaler for numerical, OneHotEncoder for categorical).")
    return X, y, preprocessor

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer, model_path: str) -> Pipeline:
    """
    Trains an XGBoost Regressor within a scikit-learn Pipeline and saves the trained model.
    """
    print("\n--- 3. Training the XGBoost Model ---")

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100,
                                   learning_rate=0.1, max_depth=3, random_state=42))
    ])

    # Fit the entire pipeline (preprocessor + regressor) on the training data
    model_pipeline.fit(X_train, y_train)

    print("XGBoost Regressor model (within pipeline) trained successfully.")

    # Save the trained model pipeline
    try:
        joblib.dump(model_pipeline, model_path)
        print(f"Trained model saved to '{model_path}'.")
    except Exception as e:
        print(f"Error saving the model to '{model_path}': {e}")

    return model_pipeline

def evaluate_model(model_path: str, X_test: pd.DataFrame, y_test: pd.Series,
                   numerical_features: list, categorical_features: list, original_df: pd.DataFrame) -> None:
    """
    Loads a trained model and evaluates it using various regression metrics and plots.
    All generated plots are saved as PNG files.
    """
    print("\n--- 4. Evaluating the Model ---")

    # Load the trained model pipeline
    try:
        model_pipeline = joblib.load(model_path)
        print(f"Trained model loaded from '{model_path}'.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please ensure the model was trained and saved.")
        return
    except Exception as e:
        print(f"Error loading the model from '{model_path}': {e}")
        return


    if X_test.empty or y_test.empty:
        print("Error: Test set is empty. Cannot make predictions or evaluate.")
        return

    # Make predictions using the *loaded and fitted* pipeline
    y_pred = model_pipeline.predict(X_test)

    print("First 10 predicted inference times (seconds):")
    print(np.round(y_pred[:10], 4))
    print("First 10 actual inference times (seconds, for comparison):")
    print(y_test.values[:10])

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

    # --- Plot: Predicted vs. Actual Values ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
    plt.xlabel('Actual Inference Time (seconds)')
    plt.ylabel('Predicted Inference Time (seconds)')
    plt.title('Actual vs. Predicted Inference Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('actual_vs_predicted_inference_time.png') # Save figure
    print("Saved 'actual_vs_predicted_inference_time.png'")
    plt.close() # Close the figure to free memory

    # --- Plot: Residuals Histogram ---
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    plt.savefig('residuals_histogram.png') # Save figure
    print("Saved 'residuals_histogram.png'")
    plt.close() # Close the figure to free memory

    # --- Plot: Residuals vs. Predicted Values ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Inference Time')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals vs. Predicted Values')
    plt.grid(True)
    plt.savefig('residuals_vs_predicted.png') # Save figure
    print("Saved 'residuals_vs_predicted.png'")
    plt.close() # Close the figure to free memory


    # --- Feature Importance Visualization for XGBoost ---
    print("\n--- Optional: Feature Importance Visualization (XGBoost) ---")
    try:
        # Get feature names after preprocessing
        # Access the preprocessor from the loaded model_pipeline
        cat_transformer = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            ohe_feature_names = list(cat_transformer.get_feature_names_out(categorical_features))
        else:
            # Fallback for older scikit-learn versions or if not OneHotEncoder
            ohe_feature_names = []
            for col in categorical_features:
                if col in original_df.columns:
                    unique_vals = original_df[col].dropna().astype(str).unique()
                    ohe_feature_names.extend([f"{col}_{val}" for val in unique_vals])
            print("Warning: get_feature_names_out not found on categorical transformer. Using fallback feature names.")


        # Combine numerical feature names with one-hot encoded feature names
        all_processed_features = numerical_features + ohe_feature_names

        # Get feature importances from the trained XGBoost regressor
        xgb_importances = model_pipeline.named_steps['regressor'].feature_importances_

        # Ensure the number of feature importances matches the number of processed features
        if len(all_processed_features) != len(xgb_importances):
            print(f"Warning: Mismatch between number of processed features ({len(all_processed_features)}) "
                  f"and feature importances ({len(xgb_importances)}). "
                  f"Cannot accurately plot feature importance.")
            print(f"Processed Features: {all_processed_features}")
            print(f"Importances: {xgb_importances}") # Print for debugging
        else:
            # Create a DataFrame for visualization
            feature_importance_df = pd.DataFrame({
                'Feature': all_processed_features,
                'Importance': xgb_importances
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
            plt.title('Top 20 Feature Importances in XGBoost Regressor (After Preprocessing)')
            plt.xlabel('Feature Importance (Gain)')
            plt.ylabel('Feature')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.savefig('feature_importances.png') # Save figure
            print("Saved 'feature_importances.png'")
            plt.close() # Close the figure to free memory
    except Exception as e:
        print(f"Could not visualize feature importances due to: {e}")
        print("This might happen if the preprocessor or regressor structure is unexpected, or data is insufficient.")


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

    # Split Data
    print("\n--- Splitting Data into Training and Testing Sets ---")
    if X.empty or y.empty: # This check is crucial
        print("Error: Dataset is empty after preprocessing. Cannot split data.")
        return

    # Ensure consistent column order for preprocessing
    X = X[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Train Model (and save it)
    if X_train.empty or y_train.empty: # Check for empty training set
        print("Error: Training set is empty. Cannot train model.")
        return
    trained_pipeline = train_model(X_train, y_train, preprocessor, MODEL_PATH)

    # Evaluate Model (by loading the saved model)
    evaluate_model(MODEL_PATH, X_test, y_test, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, df)

    # --- Prediction for a Specific Model ---
    print("\n--- 5. Prediction for a Specific User-Defined Model ---")
    print("Please manually edit the 'specific_model_features' DataFrame below within the code")
    print("to define the characteristics of the model you want to predict for.")
    model_name = "some_super_model"
    # Define the characteristics for the *specific model* you want to predict
    specific_model_features = pd.DataFrame([{
        'conv_layers': 7,
        'cpu_usage_percent': 45.2,
        'device': 'raspberrypi',
        'disk_io_read_bytes': 15000,
        'disk_io_write_bytes': 8000,
        'disk_usage_percent': 55.0,
        'fully_conn_layers': 3,
        'memory_usage_percent': 70.0,
        'network_type': 'RNN',
        'pool_layers': 4,
        'total_parameters': 25000000
    }])

    # Ensure the columns are in the same order as your FEATURE_COLUMNS
    specific_model_features = specific_model_features[FEATURE_COLUMNS]

    # Load the model here for the specific prediction as well,
    # or you could pass `trained_pipeline` directly if you're sure it's available.
    # Loading it again here for demonstration of standalone prediction with a loaded model.
    try:
        loaded_for_prediction_pipeline = joblib.load(MODEL_PATH)
        predicted_inference_time = loaded_for_prediction_pipeline.predict(specific_model_features)

        print(f"\nFeatures for specific model:\n{specific_model_features}")
        print(f"Predicted inference time for {model_name}: {predicted_inference_time[0]:.4f} seconds")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Cannot make specific prediction.")
    except Exception as e:
        print(f"Error making specific prediction: {e}")

    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()
