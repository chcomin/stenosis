""" Monitor many hardware metrics during a training run."""

import contextlib
import os
import re
import subprocess
import time
from datetime import datetime

import pandas as pd
import psutil
import pynvml

# Guide to analyse the results:
# https://docs.google.com/spreadsheets/d/1q1rN3rwRRzo4uk3Rx2Ky-GUzW__-OGsRxWl0tANRjFs/edit?usp=sharing

# --- CONFIGURATION ---
DURATION_SEC = 10  # Total monitoring time
OUTPUT_FILE = "_hardware_diagnostics_full.csv"

def setup_nvml():
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as e:
        print(f"Warning: Could not initialize NVML: {e}")
        return False

def get_cpu_temp():
    """Attempts to get the highest CPU temperature."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps: 
            return 0
        max_temp = 0
        for _, entries in temps.items():
            for entry in entries:
                if entry.current and entry.current > max_temp:
                    max_temp = entry.current
        return max_temp
    except Exception: 
        return 0

def get_main_python_process_info():
    """
    Returns (pid, num_threads, process_object) of the heaviest Python process.
    Used to monitor threads and Page Faults specific to the training loop.
    """
    max_cpu = 0
    target_proc = None
    target_threads = 0
    my_pid = os.getpid()
    
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "num_threads"]):
        try:
            # Filter only python, ignore the diagnostic script itself
            if "python" in proc.info["name"].lower() and proc.pid != my_pid:
                cpu = proc.info["cpu_percent"] or 0
                if cpu > max_cpu:
                    max_cpu = cpu
                    target_proc = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    if target_proc:
        try:
            # Sum threads of parent + children (e.g., dataloaders)
            total_threads = target_proc.info["num_threads"]
            children = target_proc.children(recursive=True)
            for child in children:
                with contextlib.suppress(Exception):
                    total_threads += child.num_threads()
            return target_proc.pid, total_threads, target_proc
        except Exception:
            return None, 0, None
            
    return None, 0, None

def get_ram_faults_and_swap(process, last_swap, interval):
    """Calculates RAM 'suffering' metrics: Swap and Page Faults."""
    
    # 1. System Swap (MB/s)
    curr_swap = psutil.swap_memory()
    # Avoid division by zero
    if interval < 0.1: 
        interval = 1.0
        
    swap_in = (curr_swap.sin - last_swap.sin) / interval / 1024**2
    swap_out = (curr_swap.sout - last_swap.sout) / interval / 1024**2
    
    # 2. Process Page Faults (Major = Disk, Minor = Allocation)
    maj_flt_rate = 0
    min_flt_rate = 0
    
    if process:
        try:
            curr_mem = process.memory_info()
            
            # Initialize history if it doesn't exist
            if not hasattr(process, "_last_majflt"):
                process._last_majflt = curr_mem.majflt
                process._last_minflt = getattr(curr_mem, "minflt", 0)
            
            maj_flt_rate = (curr_mem.majflt - process._last_majflt) / interval
            
            curr_min = getattr(curr_mem, "minflt", 0)
            min_flt_rate = (curr_min - process._last_minflt) / interval
            
            # Update history
            process._last_majflt = curr_mem.majflt
            process._last_minflt = curr_min
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass # Process might have died

    return curr_swap, swap_in, swap_out, maj_flt_rate, min_flt_rate

def get_dmon_sample():
    """Gets SM% and MemCtrl% via nvidia-smi dmon (blocks for ~1s)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "dmon", "-s", "u", "-c", "1"], 
            capture_output=True, text=True, check=False
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2: 
            return 0, 0
        data_line = lines[-1].strip()
        parts = re.split(r"\s+", data_line)
        return int(parts[1]), int(parts[2]) # SM, Mem
    except Exception:
        return 0, 0

def get_nvml_metrics(handle):
    """Gets detailed GPU metrics via library."""
    m = {}
    try:
        # Temporal Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        m["gpu_util_time_pct"] = util.gpu
        
        # Memory Capacity
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        m["gpu_mem_used_mb"] = mem.used / 1024**2
        
        # Temperature and Throttling
        m["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        
        # Throttling Bitmasks
        is_therm = reasons & (pynvml.nvmlClocksThrottleReasonGpuTemp | 0x0000000000000020)
        is_pwr = reasons & pynvml.nvmlClocksThrottleReasonPower
        m["gpu_throttle_therm"] = 1 if is_therm else 0
        m["gpu_throttle_pwr"] = 1 if is_pwr else 0
        
        # PCIe
        tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        m["pcie_tx_mb_s"] = tx / 1024
        m["pcie_rx_mb_s"] = rx / 1024
        
    except pynvml.NVMLError:
        pass
    return m

def get_disk_io_rate(last_io, interval):
    curr_io = psutil.disk_io_counters()
    if interval < 0.1: 
        interval = 1.0
    r = (curr_io.read_bytes - last_io.read_bytes) / interval / 1024**2
    w = (curr_io.write_bytes - last_io.write_bytes) / interval / 1024**2
    return curr_io, r, w

def main():
    print("="*60)
    print("FULL BOTTLENECK MONITORING (GPU/CPU/RAM/DISK/PCIe)")
    print(f"Duration: {DURATION_SEC} seconds. Start your training now!")
    print("="*60)
    
    use_nvml = setup_nvml()
    gpu_handle = None
    if use_nvml:
        try:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print(f"GPU Detected: {pynvml.nvmlDeviceGetName(gpu_handle).decode('utf-8')}")
        except Exception:
            use_nvml = False

    data = []
    last_disk_io = psutil.disk_io_counters()
    last_swap = psutil.swap_memory()
    
    # Attempt to locate the training process initially
    pid, threads, proc_obj = get_main_python_process_info()
    if pid: 
        print(f"Initial Target Process: PID {pid} ({threads} threads)")
    
    start_global = datetime.now()
    
    try:
        while True:
            loop_start = time.time()
            
            # --- 1. Slow Collection (dmon ~1s) ---
            # This dictates the loop rhythm
            sm_util, mem_ctrl_util = get_dmon_sample()
            
            # --- 2. Fast Collection (NVML) ---
            nvml_stats = {}
            if use_nvml and gpu_handle:
                nvml_stats = get_nvml_metrics(gpu_handle)
            
            # --- 3. Update Process and Threads ---
            # Check again every loop as threads change dynamically
            pid, threads, proc_obj = get_main_python_process_info()
            
            # --- 4. CPU and Temp ---
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            cpu_temp = get_cpu_temp()
            io_wait = getattr(psutil.cpu_times_percent(), "iowait", 0)
            
            # --- 5. Disk and RAM (Delta Time) ---
            now = time.time()
            dt = now - loop_start
            last_disk_io, d_read, d_write = get_disk_io_rate(last_disk_io, dt)
            last_swap, sw_in, sw_out, maj_flt, min_flt = get_ram_faults_and_swap(proc_obj,
                                                                                  last_swap, dt)

            # --- CONSOLIDATION ---
            row = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                
                # GPU Processing
                "gpu_sm_util_pct": sm_util,          # Spatial (Real Load)
                "gpu_util_time_pct": nvml_stats.get("gpu_util_time_pct", 0), # Temporal
                
                # GPU Memory & Throttling
                "gpu_mem_ctrl_pct": mem_ctrl_util,
                "gpu_mem_used_mb": nvml_stats.get("gpu_mem_used_mb", 0),
                "gpu_temp_c": nvml_stats.get("gpu_temp_c", 0),
                "gpu_throttle_therm": nvml_stats.get("gpu_throttle_therm", 0),
                "gpu_throttle_pwr": nvml_stats.get("gpu_throttle_pwr", 0),
                
                # Communication
                "pcie_rx_mb_s": nvml_stats.get("pcie_rx_mb_s", 0), # CPU->GPU
                "pcie_tx_mb_s": nvml_stats.get("pcie_tx_mb_s", 0),
                
                # CPU
                "cpu_max_core_pct": max(cpu_per_core) if cpu_per_core else 0,
                "cpu_total_pct": sum(cpu_per_core)/len(cpu_per_core) if cpu_per_core else 0,
                "cpu_temp_c": cpu_temp,
                "cpu_iowait_pct": io_wait,
                "train_threads": threads,
                
                # RAM & Disk
                "ram_used_pct": psutil.virtual_memory().percent,
                "swap_activity_mb_s": sw_in + sw_out,
                "ram_major_faults": maj_flt,
                "ram_minor_faults": min_flt,
                "disk_read_mb_s": d_read,
                "disk_write_mb_s": d_write
            }
            data.append(row)
            
            # --- COMPACT VISUAL FEEDBACK ---
            # Prioritize showing what is "on fire"
            status_gpu = f"SM:{row['gpu_sm_util_pct']}%"
            if row["gpu_throttle_therm"]:
                status_gpu += " [THROTTLE!]"
            
            status_ram = "OK"
            if row["swap_activity_mb_s"] > 0: 
                status_ram = "SWAP!"
            elif row["ram_major_faults"] > 50: 
                status_ram = "FLT!"
            
            print(f"\rGPU:{status_gpu} ({int(row['gpu_temp_c'])}C) | "
                  f"CPU Max:{row['cpu_max_core_pct']}% (Th:{threads}) | "
                  f"PCIe:{int(row['pcie_rx_mb_s'])} | "
                  f"RAM:{status_ram}", end="")

            if (datetime.now() - start_global).total_seconds() > DURATION_SEC:
                break
                
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")
    finally:
        if use_nvml: 
            pynvml.nvmlShutdown()

    # --- FINAL REPORT ---
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n\n{'='*60}")
        print(f"DIAGNOSTICS SAVED: {OUTPUT_FILE}")
        print(f"{'='*60}")
        
        # Automatic Quick Summary
        print("BOTTLENECK SUMMARY:")
        
        # 1. GPU
        avg_sm = df["gpu_sm_util_pct"].mean()
        avg_time = df["gpu_util_time_pct"].mean()
        throttled = df["gpu_throttle_therm"].sum()
        print(f"* GPU Load (SM): {avg_sm:.1f}% | Occupancy (Time): {avg_time:.1f}%")
        if throttled > 0: 
            print(f"  [!] WARNING: Thermal Throttling detected in {throttled} samples!")
        
        # 2. CPU
        avg_cpu_max = df["cpu_max_core_pct"].mean()
        avg_iowait = df["cpu_iowait_pct"].mean()
        print(f"* CPU Max Core: {avg_cpu_max:.1f}% | IO Wait: {avg_iowait:.1f}%")
        
        # 3. RAM
        tot_swap = df["swap_activity_mb_s"].sum()
        avg_faults = df["ram_major_faults"].mean()
        print(f"* RAM Swap Total: {tot_swap:.1f} MB | Avg Major Faults: {avg_faults:.1f}/s")
        
        # 4. PCIe
        peak_pcie = df["pcie_rx_mb_s"].max()
        print(f"* PCIe Peak RX: {peak_pcie:.0f} MB/s")

if __name__ == "__main__":
    main()