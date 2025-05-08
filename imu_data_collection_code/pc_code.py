import socket
import csv
import time
from threading import Thread, Lock, Event
import queue
import os

# Server settings
HOST = '192.168.203.11'  # Updated to your computer's current IP address
PORT = 5005         # server port

# Create directory for data if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create a socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(2)  # listen for up to 2 connections

CSV_COLUMN_NAMES = ['sync_flag', 'IMU_stopwatch', 'GX', 'GY', 'GZ', 'AX', 'AY', 'AZ',
                    'receive_timestamp', 'buffer_save_timestamp']

print("Server started, waiting for connections...")

# Collect user ID
user_id = input("Enter a user ID: ")
print("Waiting for watches to connect...")

# Initialize connections
conn1, addr1 = server.accept()
conn2, addr2 = server.accept()

# Receive hand identifiers from watches
hand1 = conn1.recv(1024).decode().strip()
hand2 = conn2.recv(1024).decode().strip()
print(hand1, hand2)

# Create csv writers for both watches
file1 = open(f"./data/{user_id}_{hand1}.csv", "w", newline='')
file2 = open(f"./data/{user_id}_{hand2}.csv", "w", newline='')

writer1 = csv.writer(file1)
writer2 = csv.writer(file2)

# Write column names to CSV files
writer1.writerow(CSV_COLUMN_NAMES)
writer2.writerow(CSV_COLUMN_NAMES)

print(f"Connected to watch1 ({hand1}) at {addr1} and watch2 ({hand2}) at {addr2}")

# Await user command to start
input("Press ENTER to start the test")

start_command = "S".encode()
conn1.sendall(start_command)
conn2.sendall(start_command)

# Global queues for data
data_queue_watch1 = queue.Queue()
data_queue_watch2 = queue.Queue()

# Event to signal threads to stop
stop_event = Event()

def handle_watch(conn, data_queue):
    buffer = ""
    while not stop_event.is_set():
        try:
            data = conn.recv(1024).decode()
            if not data:
                print("No data received!!!")
                continue
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip() == "End":
                    print("End signal received, closing connection.")
                    return
                receive_timestamp = str(int(time.time() * 1000))
                data_queue.put(f"{line.strip()},{receive_timestamp}")
        except ConnectionResetError:
            print("Connection was reset. Ending data collection for this watch.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

def write_queue_to_csv(data_queue, writer):
    while not stop_event.is_set() or not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            buffer_save_timestamp = str(int(time.time() * 1000))
            row = data.split(',') + [buffer_save_timestamp]
            writer.writerow(row)
        except queue.Empty:
            continue

# Start threads to handle both watches
thread1 = Thread(target=handle_watch, args=(conn1, data_queue_watch1))
thread2 = Thread(target=handle_watch, args=(conn2, data_queue_watch2))

# Start writer threads
writer_thread1 = Thread(target=write_queue_to_csv, args=(data_queue_watch1, writer1))
writer_thread2 = Thread(target=write_queue_to_csv, args=(data_queue_watch2, writer2))

thread1.start()
thread2.start()
writer_thread1.start()
writer_thread2.start()

# Wait for user input to stop
try:
    input("Press ENTER to end the test")
except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Stopping the test...")

# Set the stop event to signal threads to stop
stop_event.set()

# Send 'End' command to both watches
try:
    conn1.sendall(b"End\n")
    conn2.sendall(b"End\n")
except:
    print("Error sending End command to watches")

# Wait for threads to finish (with a timeout)
thread1.join(timeout=5)
thread2.join(timeout=5)
writer_thread1.join(timeout=5)
writer_thread2.join(timeout=5)

# Close csv files
file1.close()
file2.close()

# Close connections
conn1.close()
conn2.close()

print("Test ended and data saved")