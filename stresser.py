import os
import signal
import subprocess
import random

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

if __name__ == '__main__':
    random_cpu_load = random.randint(20, 80)
    random_cpu_cores = random.randint(1, 4)  # Use 1–4 to avoid cpu=0
    process = start_stresser(random_cpu_cores, random_cpu_load)
