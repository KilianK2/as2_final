import socket
import csv
import time
from threading import Thread, Event
import queue
import os

# IMU device settings
IMU_HOST = '192.168.203.20'  # IMU IP address
IMU_PORT = 80                # IMU port

# Create directory for data if it doesn't exist
os.makedirs("data", exist_ok=True)

# CSV settings
CSV_COLUMN_NAMES = ['sync_flag', 'IMU_stopwatch', 'GX', 'GY', 'GZ', 'AX', 'AY', 'AZ',
                    'receive_timestamp', 'buffer_save_timestamp']

print("IMU Data Collection Script")

# Collect user ID
user_id = input("Enter a user ID: ")

# Create csv writer
file_path = f"./data/{user_id}_imu_data.csv"
file = open(file_path, "w", newline='')
writer = csv.writer(file)

# Write column names to CSV file
writer.writerow(CSV_COLUMN_NAMES)

print(f"Will save data to {file_path}")
print(f"Connecting to IMU at {IMU_HOST}:{IMU_PORT}...")

# Connect to the IMU device
imu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    imu_socket.connect((IMU_HOST, IMU_PORT))
    print(f"Connected to IMU device at {IMU_HOST}:{IMU_PORT}")
except Exception as e:
    print(f"Failed to connect to IMU: {e}")
    print("Make sure your computer and the IMU are on the same network.")
    print("Check if any firewall is blocking the connection.")
    exit(1)

# Global queue for data
data_queue = queue.Queue()

# Event to signal threads to stop
stop_event = Event()

def handle_imu_data():
    """Receive data from IMU and put it in the queue"""
    buffer = ""
    while not stop_event.is_set():
        try:
            data = imu_socket.recv(1024).decode()
            if not data:
                print("No data received from IMU")
                time.sleep(0.1)
                continue
                
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                receive_timestamp = str(int(time.time() * 1000))
                data_queue.put(f"{line.strip()},{receive_timestamp}")
                
        except ConnectionResetError:
            print("Connection to IMU was reset. Stopping data collection.")
            stop_event.set()
            break
        except Exception as e:
            print(f"An error occurred while receiving data: {e}")
            time.sleep(0.1)

def write_queue_to_csv():
    """Write data from queue to CSV file"""
    while not stop_event.is_set() or not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            buffer_save_timestamp = str(int(time.time() * 1000))
            try:
                row = data.split(',') + [buffer_save_timestamp]
                writer.writerow(row)
            except Exception as e:
                print(f"Error writing data to CSV: {e}")
        except queue.Empty:
            continue

# Ask user to start data collection
input("Press ENTER to start data collection...")

# Start threads for data handling and writing
receive_thread = Thread(target=handle_imu_data)
writer_thread = Thread(target=write_queue_to_csv)

receive_thread.start()
writer_thread.start()

print("Data collection started. Press ENTER to stop.")

# Wait for user to stop
try:
    input()
except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Stopping data collection...")

# Signal threads to stop
print("Stopping data collection...")
stop_event.set()

# Send stop command to IMU if needed
try:
    imu_socket.sendall(b"STOP\n")
except:
    print("Could not send stop command to IMU")

# Wait for threads to finish (with timeout)
receive_thread.join(timeout=5)
writer_thread.join(timeout=5)

# Close resources
imu_socket.close()
file.close()

print(f"Data collection complete. Data saved to {file_path}")
print("You can now analyze the data using the visualization tools.") 