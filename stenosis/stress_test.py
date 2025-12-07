import csv
import datetime
import multiprocessing
import os
import time

import GPUtil
import psutil

# Configuration
DURATION_SECONDS = 60       # Total duration of the test
POLL_INTERVAL = 2           # Time between measurements
CSV_FILENAME = "_stress_test_log.csv"

def cpu_stress_task():
    """Performs intense calculation to stress a CPU core."""
    while True:
        _ = [x**2 for x in range(10000)]

def gpu_stress_task():
    """Attempts to stress GPU using CuPy if available."""
    try:
        import cupy as cp
        # print("GPU Stress: CuPy detected. Running matrix multiplications...")
        while True:
            a = cp.random.rand(5000, 5000)
            b = cp.random.rand(5000, 5000)
            cp.dot(a, b)
    except ImportError:
        # If CuPy is not found, we just keep the process alive but idle.
        while True:
            time.sleep(1)

def get_cpu_temp():
    """Fetches CPU temperature. Platform dependent."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        
        # specific keys often used by OS
        for name in ['coretemp', 'cpu_thermal', 'k10temp', 'znver1']:
            if name in temps:
                return temps[name][0].current
        
        # Fallback: return the first available temp sensor
        return list(temps.values())[0][0].current
    except Exception:
        return None

def get_gpu_info():
    """Fetches GPU temperature and load."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return []
        return [(gpu.name, gpu.temperature, gpu.load * 100) for gpu in gpus]
    except Exception:
        return []

def monitor_system(stop_event, csv_filename):
    """Monitors system and logs to CSV."""
    
    # Initialize CSV file with headers
    headers = ['Timestamp', 'Elapsed_Seconds', 'CPU_Temp_C', 'CPU_Usage_Pct', 'GPU_Name', 'GPU_Temp_C', 'GPU_Load_Pct']
    
    try:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            
            print(f"{'Time':<10} | {'CPU Temp':<10} | {'CPU Usage':<10} | {'GPU Info'}")
            print("-" * 70)
            
            start_time = time.time()
            
            while not stop_event.is_set():
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                elapsed = int(time.time() - start_time)
                
                # Gather Data
                cpu_temp = get_cpu_temp()
                cpu_usage = psutil.cpu_percent(interval=None)
                gpu_stats = get_gpu_info()
                
                # Handle Data for Display and Logging
                display_cpu_temp = f"{cpu_temp}°C" if cpu_temp is not None else "N/A"
                log_cpu_temp = cpu_temp if cpu_temp is not None else ""
                
                gpu_display_str = ""
                
                # If multiple GPUs, we log them as separate rows or just the first one. 
                # For simplicity in CSV, we will log the PRIMARY (first) GPU in the main columns.
                if gpu_stats:
                    first_gpu = gpu_stats[0]
                    g_name, g_temp, g_load = first_gpu
                    gpu_display_str = f"[{g_name}: {g_temp}°C, {g_load:.1f}%]"
                    
                    # Write to CSV
                    writer.writerow([current_time, elapsed, log_cpu_temp, cpu_usage, g_name, g_temp, g_load])
                else:
                    gpu_display_str = "No GPU"
                    writer.writerow([current_time, elapsed, log_cpu_temp, cpu_usage, "N/A", "", ""])
                
                print(f"{elapsed:<10} | {display_cpu_temp:<10} | {cpu_usage:<10}% | {gpu_display_str}")
                
                # Flush to ensure data is saved even if script crashes
                file.flush()
                
                time.sleep(POLL_INTERVAL)
                
    except PermissionError:
        print(f"Error: Permission denied when writing to {csv_filename}. Is the file open?")

if __name__ == "__main__":
    print(f"Starting Stress Test. Logging to {CSV_FILENAME}...")
    print(f"Duration: {DURATION_SECONDS} seconds. Press Ctrl+C to stop early.\n")

    # 1. Start CPU Stress
    cpu_count = multiprocessing.cpu_count()
    processes = []
    for _ in range(cpu_count):
        p = multiprocessing.Process(target=cpu_stress_task)
        p.daemon = True
        p.start()
        processes.append(p)

    # 2. Start GPU Stress
    p_gpu = multiprocessing.Process(target=gpu_stress_task)
    p_gpu.daemon = True
    p_gpu.start()
    processes.append(p_gpu)

    # 3. Start Monitoring
    stop_event = multiprocessing.Event()
    try:
        monitor_system(stop_event, CSV_FILENAME)
        
        # Wait loop
        start = time.time()
        while time.time() - start < DURATION_SECONDS:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stop_event.set()
        print("\nStopping processes...")
        for p in processes:
            p.terminate()
        print(f"Test Finished. Data saved to {CSV_FILENAME}")