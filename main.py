import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier # Import XGBoost classifier

# --- Configuration Constants ---
# Ensure this is correct for your environment
CSV_DIRECTORY = '/home/gkorod/Downloads/Raspberrypi 4B/'
TARGET_COLUMN_NAME = 'execution_time'
INFERENCE_TIME_THRESHOLD_SECONDS = 0.5 # Adjust as needed
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
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                print(f"Loading {filepath}...")
                df_temp = pd.read_csv(filepath, low_memory=False)
                all_dataframes.append(df_temp)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found. Please create it and place your CSVs, or correct the path.")
        return None

    if not all_dataframes:
        print("No CSV files found in the specified directory. Please ensure your CSVs are there.")
        return None

    df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Successfully loaded {len(all_dataframes)} CSV files.")
    print(f"Total rows in combined dataset: {df.shape[0]}")
    print(f"Total columns in combined dataset: {df.shape[1]}")
    print("\nFirst 5 rows of the combined dataset:")
    print(df.head())
    return df

def preprocess_data(df: pd.DataFrame, target_col: str, threshold: float,
                    feature_cols: list, numerical_cols: list, categorical_cols: list) -> tuple | tuple[None, None, None]:
    """
    Performs data cleaning, converts target_col (time string) to float (total seconds),
    creates the target variable, and prepares features.
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
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame for conversion.")
        return None, None, None

    print(f"Attempting to convert '{target_col}' from time string to total seconds...")
    try:
        # Step 1: Convert the string 'execution_time' to timedelta objects
        # errors='coerce' will turn any invalid time strings into NaT (Not a Time)
        df[target_col] = pd.to_timedelta(df[target_col], errors='coerce')

        # Step 2: Convert timedelta objects to total seconds (float)
        # This will turn NaT values (from failed timedelta conversion) into NaN
        df[target_col] = df[target_col].dt.total_seconds()
        print(f"Successfully converted '{target_col}' to total seconds (float).")

    except Exception as e:
        print(f"Error parsing or converting '{target_col}' to total seconds: {e}")
        print("Please ensure 'execution_time' is in 'HH:MM:SS.microseconds' format.")
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
    # This will now drop rows where time parsing failed and resulted in NaN
    if df[target_col].isnull().any():
        initial_rows = df.shape[0]
        df.dropna(subset=[target_col], inplace=True)
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to missing values (unparseable time strings) in target column '{target_col}'.")
        else:
            print(f"No missing values in target column '{target_col}' after time conversion.")
    else:
        print(f"No missing values in target column '{target_col}' after time conversion.")


    # --- IMPORTANT: Check if DataFrame is empty AFTER dropping rows ---
    if df.empty:
        print("Error: DataFrame became empty after handling missing values in the target column. Cannot proceed.")
        return None, None, None

    # Fill missing numerical features with the mean (for feature columns)
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            print(f"Filled missing numerical values in '{col}' with its mean.")

    # Fill missing categorical features with the mode (for feature columns)
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"Filled missing categorical values in '{col}' with its mode.")

    # Create the binary target variable 'is_slow'
    # This line should now work correctly as df[target_col] will be float
    df['is_slow'] = (df[target_col] > threshold).astype(int)

    print(f"\nDistribution of 'is_slow' target variable (threshold: {threshold} seconds):")
    print(df['is_slow'].value_counts())

    X = df[feature_cols].copy()
    y = df['is_slow']

    print("Features (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)

    # Define the preprocessing pipeline for features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    print("\nFeature preprocessing setup complete (StandardScaler for numerical, OneHotEncoder for categorical).")
    return X, y, preprocessor

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Trains an XGBoost model within a scikit-learn Pipeline.
    """
    print("\n--- 3. Training the XGBoost Model ---")

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    print("XGBoost model (within pipeline) trained successfully.")
    return model_pipeline

def evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                   numerical_features: list, categorical_features: list) -> None:
    """
    Evaluates the trained model using various classification metrics and plots.
    """
    print("\n--- 4. Evaluating the Model ---")

    if X_test.empty or y_test.empty:
        print("Error: Test set is empty. Cannot make predictions or evaluate.")
        return

    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = model_pipeline.predict(X_test)

    print("First 10 predicted probabilities (of having 'slow' inference time):")
    print(np.round(y_pred_proba[:10], 4))
    print("First 10 predicted classes:")
    print(y_pred[:10])
    print("First 10 actual classes (for comparison):")
    print(y_test.values[:10])

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fast (0)', 'Slow (1)']))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # --- Feature Importance Visualization for XGBoost ---
    print("\n--- Optional: Feature Importance Visualization (XGBoost) ---")
    try:
        # Get feature names after preprocessing
        # This gets the names from the OneHotEncoder (for categorical features)
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        # Combine numerical feature names with one-hot encoded feature names
        all_processed_features = numerical_features + list(ohe_feature_names)

        # Get feature importances from the trained XGBoost classifier
        xgb_importances = model_pipeline.named_steps['classifier'].feature_importances_

        # Create a DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'Feature': all_processed_features,
            'Importance': xgb_importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title('Top 20 Feature Importances in XGBoost (After Preprocessing)')
        plt.xlabel('Feature Importance (Gain)')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()
    except Exception as e:
        print(f"Could not visualize feature importances due to: {e}")
        print("This might happen if the preprocessor or classifier structure is unexpected.")


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
    X, y, preprocessor = preprocess_data(df, TARGET_COLUMN_NAME, INFERENCE_TIME_THRESHOLD_SECONDS,
                                         FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    if X is None or y is None or preprocessor is None:
        print("Data preprocessing failed. Exiting.")
        return

    # Split Data
    print("\n--- Splitting Data into Training and Testing Sets ---")
    if X.empty or y.empty: # This check is crucial
        print("Error: Dataset is empty after preprocessing. Cannot split data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Proportion of 'slow' in y_train: {y_train.mean():.2f}")
    print(f"Proportion of 'slow' in y_test: {y_test.mean():.2f}")

    # Train Model
    if X_train.empty or y_train.empty: # Check for empty training set
        print("Error: Training set is empty. Cannot train model.")
        return
    model_pipeline = train_model(X_train, y_train, preprocessor)

    # Evaluate Model
    evaluate_model(model_pipeline, X_test, y_test, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()
