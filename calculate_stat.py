import pandas as pd

# Load the output CSV
df = pd.read_csv("profiler/training/model_inference_predictions.csv")

# Convert to boolean comparison columns if not already present
df["prediction_correct"] = df["fastest_device_by_prediction"] == df["fastest_device_by_execution_time"]
df["random_correct"] = df["random_selection_device"] == df["fastest_device_by_execution_time"]

# Count of total predictions
total_predictions = len(df)

# Count where prediction was correct
prediction_correct_count = df["prediction_correct"].sum()
random_correct_count = df["random_correct"].sum()

# Compute percentages
prediction_accuracy = (prediction_correct_count / total_predictions) * 100
random_accuracy = (random_correct_count / total_predictions) * 100

# Print results
print(f"✅ Prediction matched actual fastest device in {prediction_correct_count}/{total_predictions} cases ({prediction_accuracy:.2f}%)")
print(f"🎲 Random choice matched actual fastest device in {random_correct_count}/{total_predictions} cases ({random_accuracy:.2f}%)")
