import os
import signal
import subprocess
import random
import time

import psutil

PID_FILE = "stresser_pid.txt"

def start_stresser(random_cores, random_load):
    command = f"stress-ng --cpu {random_cores} --cpu-load {random_load}"
    process = subprocess.Popen(command, shell=True, start_new_session=True)

    # Save PID to file
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))

    return process

def end_stresser():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            pid = int(f.read())
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            print(f"Terminated stress-ng process with PGID: {pgid}")
        except ProcessLookupError:
            print("Process already terminated or not found.")
        os.remove(PID_FILE)
    else:
        print("PID file not found. Is stress-ng running?")

def get_metrics(random_cpu_cores,random_cpu_load):
    print("Calculating metrics...")

    cpu_usage = (random_cpu_cores * random_cpu_load) / 4
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent
    # Create a JSON with the metrics
    metrics = {

                "total_memory_usage_percent": memory_usage,
                "total_cpu_usage_percent": cpu_usage,
                "device_disk_usage_percent": disk_usage,
                "device_cpu_cores": random_cpu_cores,
                "device_load_percent": random_cpu_load,
            }
    print(metrics)
if __name__ == '__main__':
    random_cpu_load = random.randint(20, 80)
    random_cpu_cores = random.randint(1, 4)# Use 1–4 to avoid cpu=0
    process = start_stresser(random_cpu_cores, random_cpu_load)
    time.sleep(15)
    get_metrics(random_cpu_cores,random_cpu_load)
