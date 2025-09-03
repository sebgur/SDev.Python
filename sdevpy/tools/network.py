import subprocess
import time
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

TARGET = "8.8.8.8"          # You can change this to any reliable IP
INTERVAL = 5                # Seconds between pings
LOG_FILE = "wifi_1h.csv"   # Output file

RUN_LOG = False

def ping(host):
    try:
        output = subprocess.check_output(
            ["ping", "-n", "1", host],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        if "Reply from" in output:
            latency_line = [line for line in output.splitlines() if "time=" in line]
            if latency_line:
                time_ms = latency_line[0].split("time=")[-1].split("ms")[0].strip()
                return float(time_ms)
        return None  # Timeout or unreachable
    except subprocess.CalledProcessError:
        return None

def ping_log():
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Latency (ms)", "Status"])
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            latency = ping(TARGET)
            if latency is not None:
                writer.writerow([timestamp, latency, "OK"])
                print(f"{timestamp} | {latency} ms")
            else:
                writer.writerow([timestamp, "", "Timeout"])
                print(f"{timestamp} | Timeout")
            time.sleep(INTERVAL)

def ping_view():
    # Load the CSV file
    df = pd.read_csv(LOG_FILE)

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Clean latency column
    df["Latency (ms)"] = pd.to_numeric(df["Latency (ms)"], errors="coerce")

    # Plot 1: Latency over time
    plt.figure(figsize=(12, 5))
    plt.plot(df["Timestamp"], df["Latency (ms)"], label="Latency (ms)", color="blue")
    plt.title("Latency Over Time")
    plt.xlabel("Time")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Timeouts per hour
    df["Hour"] = df["Timestamp"].dt.floor("H")
    timeouts = df[df["Status"] == "Timeout"].groupby("Hour").size()

    plt.figure(figsize=(10, 4))
    timeouts.plot(kind="bar", color="red")
    plt.title("Timeouts Per Hour")
    plt.xlabel("Hour")
    plt.ylabel("Number of Timeouts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 3: Latency distribution
    plt.figure(figsize=(8, 4))
    df["Latency (ms)"].dropna().plot(kind="hist", bins=30, color="green", edgecolor="black")
    plt.title("Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # print(f"Starting ping monitor to {TARGET} every {INTERVAL} seconds...")
    if RUN_LOG:
        ping_log()
    else:
        ping_view()