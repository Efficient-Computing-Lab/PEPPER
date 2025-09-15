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
    subprocess.run(["powerjoular -o output.csv &"],shell=True)

    # Wait until CSV exists (optional, usually immediate after run)
    while not os.path.exists(os.getcwd() + '/output.csv'):
        time.sleep(1)

    # Read CSV into pandas
    columns = ["Date", "CPU Utilization", "Total Power", "CPU Power", "GPU Power"]
    df = pd.read_csv(os.getcwd() + '/output.csv', header=None, names=columns)

    # Get the first CPU Power value and round to 3 decimals
    energy_value = np.float64(df['CPU Power'].iloc[0])
    energy_value = np.round(energy_value, 3)  # keeps 3 digits after decimal

    return energy_value
