import subprocess
import pandas as pd
import os
import time
import numpy as np

def measure_method():
    """
    Run PowerJoular for a given duration (seconds) and return results as a pandas DataFrame.
    """
    # Run PowerJoular and stream CSV output to stdout
    process = subprocess.run(["powerjoular -o output.csv &"],shell=True)
    while not os.path.exists(os.getcwd()+'/output.csv'):
        # Wait for a short time before checking again
        time.sleep(1)
    # Read CSV output directly into pandas
    columns = ["Date", "CPU Utilization", "Total Power", "CPU Power", "GPU Power"]
    # Read the CSV file with specified column names
    df = pd.read_csv(os.getcwd() + '/output.csv', header=None, names=columns)
    energy_value = df['CPU Power'].iloc[0]
    energy_value = np.float64(energy_value)
    return energy_value