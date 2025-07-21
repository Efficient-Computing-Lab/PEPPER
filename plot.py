import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration Constants ---
# !!! IMPORTANT: YOU MUST CHANGE THIS TO THE ACTUAL PATH OF YOUR CSV FILES !!!
CSV_DIRECTORY = '/home/gkorod/Downloads/mydataset/' # <--- CORRECT THIS PATH IF IT'S STILL WRONG
TARGET_COLUMN_NAME = 'execution_time'

def load_data_from_csvs(directory: str) -> pd.DataFrame | None:
    """
    Loads and concatenates all CSV files from a specified directory into a single DataFrame.
    Returns None if no files are found or directory error.
    """
    print(f"\n--- Loading Data from CSVs in '{directory}' ---")
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

def prepare_data_for_plot_per_model(df: pd.DataFrame, target_col: str) -> pd.DataFrame | None:
    """
    Prepares the DataFrame by calculating device_load_percent, converting
    execution_time, and ensuring 'model_name' and 'device' are ready.
    """
    if df is None:
        return None

    df_plot = df.copy()

    print("\n--- Preparing data for plotting device_load_percent vs. inference time per model ---")

    # Check for 'model_name' column directly
    if 'model_name' not in df_plot.columns:
        print("Error: 'model_name' column not found in the dataset. Cannot plot per model.")
        return None
    # Ensure model_name is string type for plotting
    df_plot['model_name'] = df_plot['model_name'].astype(str)
    # Convert '.onnx' suffix to a more readable format for titles
    df_plot['model_name'] = df_plot['model_name'].str.replace('.onnx', '', regex=False).str.title()


    # 1. Handle 'device' column mapping for better plot labels
    if 'device' in df_plot.columns:
        df_plot['device'] = df_plot['device'].astype(str).str.lower()
        df_plot['device'] = df_plot['device'].map({'raspberrypi': 'RaspberryPi 4B', 'jetson': 'Jetson Nano'}).fillna('Other')
        df_plot = df_plot[df_plot['device'] != 'Other'] # Filter out unmapped devices
    else:
        print("Warning: 'device' column not found. Cannot categorize by device.")
        df_plot['device'] = 'Unknown'

    # 2. Calculate 'device_load_percent'
    if 'device_cpu_cores' in df_plot.columns and 'device_load_percent' in df_plot.columns:
        df_plot['device_cpu_cores'] = df_plot['device_cpu_cores'].replace(0, 4) # Assume 4 cores for these devices
        df_plot['device_load_percent'] = (df_plot['device_cpu_cores'] * df_plot['device_load_percent']) / 4
        # Apply the filter as per original logic (from the training script)
        df_plot = df_plot[(df_plot['device_load_percent'] < 35) | (df_plot['device_load_percent'] > 65)]
        df_plot = df_plot.drop('device_cpu_cores', axis=1, errors='ignore')
    else:
        print("Error: 'device_cpu_cores' or 'device_load_percent' not found. Cannot calculate effective device load.")
        return None

    # 3. Convert 'execution_time' to total seconds
    if target_col not in df_plot.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return None
    try:
        df_plot[target_col] = pd.to_timedelta(df_plot[target_col], errors='coerce')
        df_plot[target_col] = df_plot[target_col].dt.total_seconds()
    except Exception as e:
        print(f"Error converting '{target_col}' to total seconds: {e}")
        return None

    # Drop rows with NaN in the core columns for plotting
    df_plot.dropna(subset=['device_load_percent', target_col, 'device', 'model_name'], inplace=True)

    if df_plot.empty:
        print("DataFrame is empty after preprocessing and filtering for plotting.")
        return None

    print(f"Data prepared. Final rows for plotting: {df_plot.shape[0]}")
    return df_plot

def plot_device_load_vs_inference_time_per_model_simplified(df: pd.DataFrame):
    """
    Generates and saves a line plot of average device_load_percent vs. average inference time
    for each unique model, using Seaborn's lineplot for clearer trends.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    unique_models = df['model_name'].unique()
    if len(unique_models) == 0:
        print("No identifiable models in the dataset to plot.")
        return

    print(f"\n--- Generating simplified plots for {len(unique_models)} models ---")

    for model_name in unique_models:
        model_df = df[df['model_name'] == model_name].copy()

        if model_df.empty:
            print(f"No data for model: {model_name}. Skipping plot.")
            continue

        plt.figure(figsize=(15, 10))
        # Use lineplot to show the mean and confidence interval, which is easier to read
        sns.lineplot(
            data=model_df,
            x='device_load_percent',
            y=TARGET_COLUMN_NAME,
            hue='device',
            marker='o', # Add markers for data points
            errorbar='sd', # Show standard deviation as error band
            palette='viridis',
            linewidth=2
        )

        plt.title(f'Average Device Load Percentage vs. Inference Time ({model_name})', fontsize=20)
        plt.xlabel('Device Load Percentage', fontsize=15)
        plt.ylabel('Average Inference Time (seconds)', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Device Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Sanitize model name for filename
        filename_model = model_name.lower().replace(' ', '_')
        filename = f"avg_device_load_vs_inference_time_{filename_model}.pdf"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close()

# --- Main Execution Flow for Plotting ---
print("Attempting to generate simplified plots for Device Load vs. Inference Time per model...")
df_raw = load_data_from_csvs(CSV_DIRECTORY)
if df_raw is not None:
    df_plot_ready = prepare_data_for_plot_per_model(df_raw, TARGET_COLUMN_NAME)
    if df_plot_ready is not None:
        plot_device_load_vs_inference_time_per_model_simplified(df_plot_ready)
    else:
        print("Failed to prepare data for plotting per model.")
else:
    print("Failed to load raw data from CSVs. Please check CSV_DIRECTORY path.")