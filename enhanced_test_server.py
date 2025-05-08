import socket
import sys
import time
import os
from datetime import datetime
import subprocess
import signal

# Try to get a list of occupied ports
def check_ports_in_use():
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if '0.0.0.0:5005' in line or '[::]:5005' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"Port 5005 is in use by process ID: {pid}")
                        return pid
        return None
    except Exception as e:
        print(f"Error checking ports: {e}")
        return None

# Try to kill the process using the port
def kill_process(pid):
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, text=True)
            print(result.stdout)
            return True
    except Exception as e:
        print(f"Error killing process: {e}")
    return False

# Log function
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force output to be displayed immediately

# Main server function
def run_server():
    # Check if port is in use
    pid = check_ports_in_use()
    if pid:
        response = input(f"Port 5005 is in use. Do you want to kill process {pid}? (y/n): ")
        if response.lower() == 'y':
            if kill_process(pid):
                log("Process killed. Waiting 2 seconds before starting server...")
                time.sleep(2)
            else:
                log("Failed to kill process. Please stop it manually and try again.")
                return
        else:
            log("Exiting. Please free up port 5005 and try again.")
            return

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Get IP addresses for user information
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
        log(f"Your computer hostname: {hostname}")
        log(f"Your IP address: {local_ip}")
        log("Make sure the M5StickC Plus is configured with this IP address")
    except:
        log("Could not determine local IP address")

    # Bind the socket to the port
    server_address = ('0.0.0.0', 5005)  # Listen on all available interfaces
    log(f'Starting server on {server_address}')
    try:
        server_socket.bind(server_address)
    except OSError as e:
        log(f"Error binding to port: {e}")
        log("The port might still be in use. Try restarting your computer if the problem persists.")
        return

    # Listen for incoming connections
    server_socket.listen(1)
    server_socket.settimeout(300)  # 5 minute timeout for accepting connections

    try:
        log('Waiting for M5StickC Plus to connect...')
        log('Make sure M5StickC Plus is:')
        log('1. Powered on')
        log('2. Connected to the same WiFi network (S24)')
        log('3. Configured with the correct server IP and port')
        log('4. Running the fixed watch_code_fixed.ino')
        
        # Accept connection with timeout
        try:
            connection, client_address = server_socket.accept()
            log(f'Connection from {client_address}')
            
            # Receive hand identifier with timeout
            connection.settimeout(10)
            try:
                data = connection.recv(16).decode('utf-8').strip()
                log(f'Received hand identifier: {data}')
                
                # Send start command
                time.sleep(1)  # Short delay to ensure client is ready
                log('Sending start command "S"...')
                connection.sendall(b'S\n')
                
                # Receive IMU data
                log('Waiting for IMU data...')
                count = 0
                connection.settimeout(30)  # Longer timeout for data collection
                
                # Create output directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                
                # Open a file to save the data
                filename = f"data/test_imu_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(filename, 'w') as file:
                    file.write("sync_flag,IMU_stopwatch,GX,GY,GZ,AX,AY,AZ,timestamp\n")
                    
                    start_time = time.time()
                    max_time = 120  # 2 minutes max
                    
                    # Keep reading data until we've collected enough or timed out
                    try:
                        while count < 500 and (time.time() - start_time) < max_time:
                            data = connection.recv(128).decode('utf-8').strip()
                            if data:
                                if data == "End":
                                    log("Received End signal from device")
                                    break
                                    
                                log(f'Data point {count+1}: {data}')
                                file.write(f"{data},{time.time()}\n")
                                file.flush()  # Ensure data is written immediately
                                count += 1
                                
                                # Give periodic status updates
                                if count % 50 == 0:
                                    log(f"Received {count} data points so far...")
                                    
                                # Shorter timeout once we're receiving data
                                connection.settimeout(5)
                            else:
                                log("Received empty data. Connection might be closed.")
                                break
                    except socket.timeout:
                        log("Timeout waiting for more data. Collection complete.")
                    except Exception as e:
                        log(f"Error during data collection: {e}")
                
                log(f'Received {count} data points, saved to {filename}')
                
                # Send end command
                log('Sending end command...')
                try:
                    connection.sendall(b'End\n')
                    log('End command sent')
                    
                    # Wait for acknowledgment
                    connection.settimeout(5)
                    try:
                        end_ack = connection.recv(16).decode('utf-8').strip()
                        log(f"Received acknowledgment: {end_ack}")
                    except:
                        log("No acknowledgment received")
                except:
                    log('Failed to send End command')
                    
            except socket.timeout:
                log('Timeout waiting for hand identifier or data')
            except Exception as e:
                log(f'Error handling connection: {e}')
                
        except socket.timeout:
            log('Timeout waiting for connection')
            log('No M5StickC Plus connected within the timeout period')
            log('Please check your M5StickC Plus configuration and WiFi connection')
        except Exception as e:
            log(f'Error accepting connection: {e}')
            
    except KeyboardInterrupt:
        log('Server interrupted by user')
    finally:
        # Clean up
        log('Closing server')
        if 'connection' in locals():
            connection.close()
        server_socket.close()
        log('Server closed')
        
        # Display summary if data was collected
        if 'count' in locals() and 'filename' in locals() and count > 0:
            log(f"\nData Collection Summary:")
            log(f"- Collected {count} data points")
            log(f"- Data saved to {filename}")
            log(f"- Next step: Use this data for your activity recognition project")

if __name__ == "__main__":
    log("Enhanced IMU Connection Test Server")
    run_server() 