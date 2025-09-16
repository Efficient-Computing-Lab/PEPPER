import subprocess
import pandas as pd
import os
import time
import numpy as np

def measure_method():
    """
    Run PowerJoular for a given duration (seconds) and return CPU power
    rounded to three decimal places.
    """
    # Run PowerJoular and output CSV (without & for proper blocking)
    output_file = os.path.join(os.getcwd(), "output.csv")
    process = subprocess.Popen(
        ["sudo", "powerjoular", "-f", output_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print(f"PowerJoular started with PID {process.pid}")

    # Wait until CSV exists (optional, usually immediate after run)
    while not os.path.exists(output_file):
        time.sleep(1)

    # Stop PowerJoular
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()

    # Read CSV into pandas
    columns = ["Date", "CPU Utilization", "Total Power", "CPU Power", "GPU Power"]
    df = pd.read_csv(output_file, header=None, names=columns)

    # Get the first CPU Power value and round to 3 decimals
    energy_value = np.float64(df['CPU Power'].iloc[-1])
    energy_value = np.round(energy_value, 3)  # keeps 3 digits after decimal
    # Delete the CSV file
    os.remove(output_file)
    print(energy_value)
    return float(energy_value)