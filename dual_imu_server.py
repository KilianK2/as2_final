import socket
import sys
import time
import os
from datetime import datetime
import subprocess
import signal
import threading

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
def log(message, hand=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if hand:
        print(f"[{timestamp}][{hand}] {message}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force output to be displayed immediately

# Thread to keep connection alive
def keepalive_thread(connection, stop_event):
    while not stop_event.is_set():
        try:
            # Send an empty ping every 5 seconds to keep connection alive
            connection.sendall(b"")
        except:
            # If connection is closed, exit thread
            break
        time.sleep(5)

# Thread to handle a single IMU device
def handle_imu(connection, client_address, hand, event_dict):
    """Handle a single IMU device connection"""
    log(f"Starting handler for {hand} hand", hand)
    
    # Set TCP_NODELAY to disable Nagle's algorithm for better real-time performance
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # Start keepalive thread
    stop_keepalive = threading.Event()
    keepalive = threading.Thread(target=keepalive_thread, args=(connection, stop_keepalive))
    keepalive.daemon = True
    keepalive.start()
    
    try:
        # Set initial timeout for data collection
        connection.settimeout(30)
        
        # Create output directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Open a file to save the data
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/imu_{hand}_{timestamp_str}.csv"
        with open(filename, 'w') as file:
            file.write("sync_flag,IMU_stopwatch,GX,GY,GZ,AX,AY,AZ,timestamp\n")
            
            # Wait for start event
            log(f"Waiting for start signal", hand)
            event_dict['ready_events'][hand].set()  # Signal that this IMU is ready
            event_dict['start_event'].wait()  # Wait for main thread to signal start
            
            # Send start command
            log('Sending start command "S\\n"...', hand)
            connection.sendall(b"S\n")
            
            # Receive IMU data
            log('Waiting for IMU data...', hand)
            count = 0
            
            start_time = time.time()
            max_time = 120  # 2 minutes max
            
            # Initialize variables for data rate calculation
            last_rate_check = time.time()
            points_since_check = 0
            
            # Keep reading data until we've collected enough or timed out
            try:
                while count < 500 and (time.time() - start_time) < max_time and not event_dict['stop_event'].is_set():
                    try:
                        data = connection.recv(128).decode('utf-8').strip()
                        if data:
                            if data == "End":
                                log("Received End signal from device", hand)
                                break
                            
                            log(f'Data point {count+1}: {data}', hand)
                            file.write(f"{data},{time.time()}\n")
                            file.flush()  # Ensure data is written immediately
                            count += 1
                            points_since_check += 1
                            
                            # Calculate and display data rate every 20 points
                            if count % 20 == 0:
                                now = time.time()
                                elapsed = now - last_rate_check
                                if elapsed > 0:
                                    rate = points_since_check / elapsed
                                    log(f"Data rate: {rate:.1f} points/second", hand)
                                    last_rate_check = now
                                    points_since_check = 0
                            
                            # Give periodic status updates
                            if count % 50 == 0:
                                log(f"Received {count} data points so far...", hand)
                                
                            # Shorter timeout once we're receiving data
                            connection.settimeout(5)
                        else:
                            log("Received empty data. Connection might be closed.", hand)
                            time.sleep(0.5)
                            # Try to send a ping to check connection
                            try:
                                connection.sendall(b"")
                            except:
                                log("Connection appears to be closed.", hand)
                                break
                    except socket.timeout:
                        log("Timeout waiting for data. Checking connection...", hand)
                        try:
                            connection.sendall(b"")
                            log("Connection still active. Continuing...", hand)
                        except:
                            log("Connection lost during timeout check.", hand)
                            break
            except socket.timeout:
                log("Timeout waiting for more data. Collection complete.", hand)
            except Exception as e:
                log(f"Error during data collection: {e}", hand)
            
            # Record the data count
            event_dict['data_counts'][hand] = count
            event_dict['filenames'][hand] = filename
            
            log(f'Received {count} data points, saved to {filename}', hand)
            
            # Wait for the stop event
            event_dict['stop_event'].wait()
            
            # Send end command
            log('Sending end command...', hand)
            try:
                for _ in range(3):  # Send multiple times to ensure delivery
                    connection.sendall(b"End\n")
                    time.sleep(0.1)
                log('End command sent', hand)
                
                # Wait for acknowledgment
                connection.settimeout(5)
                try:
                    end_ack = connection.recv(16).decode('utf-8').strip()
                    log(f"Received acknowledgment: {end_ack}", hand)
                except:
                    log("No acknowledgment received", hand)
            except:
                log('Failed to send End command', hand)
    except Exception as e:
        log(f"Error handling IMU: {e}", hand)
    finally:
        # Stop keepalive thread
        stop_keepalive.set()
        
        # Signal completion
        event_dict['done_events'][hand].set()
        
        # Close connection
        try:
            connection.close()
        except:
            pass
        
        log(f"Handler for {hand} hand completed", hand)

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
        log("Make sure both M5StickC Plus devices are configured with this IP address")
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
    server_socket.listen(2)  # Allow up to 2 connections
    server_socket.settimeout(300)  # 5 minute timeout for accepting connections

    # Initialize shared events
    event_dict = {
        'start_event': threading.Event(),  # Signal to start data collection
        'stop_event': threading.Event(),   # Signal to stop data collection
        'ready_events': {                 # Signal when each hand is ready
            'L': threading.Event(),
            'R': threading.Event()
        },
        'done_events': {                  # Signal when each hand is done
            'L': threading.Event(),
            'R': threading.Event()
        },
        'data_counts': {                  # Store data counts
            'L': 0,
            'R': 0
        },
        'filenames': {                    # Store filenames
            'L': "",
            'R': ""
        }
    }

    # Dictionary to store connections
    connections = {}
    threads = {}
    
    try:
        log('Waiting for two M5StickC Plus devices to connect (L and R)...')
        log('Make sure both M5StickC Plus devices are:')
        log('1. Powered on')
        log('2. Connected to the same WiFi network (S24)')
        log('3. Configured with the correct server IP and port')
        log('4. One configured with HAND="L" and one with HAND="R"')
        
        # Accept connections from both IMUs
        connected_devices = 0
        
        while connected_devices < 2 and not event_dict['stop_event'].is_set():
            try:
                # Accept connection with timeout
                connection, client_address = server_socket.accept()
                log(f'Connection from {client_address}')
                
                # Get hand identifier
                connection.settimeout(10)
                try:
                    hand = connection.recv(16).decode('utf-8').strip()
                    log(f'Received hand identifier: {hand}')
                    
                    if hand in ['L', 'R'] and hand not in connections:
                        connections[hand] = {
                            'connection': connection,
                            'address': client_address
                        }
                        connected_devices += 1
                        log(f'Device {hand} registered successfully ({connected_devices}/2 devices)')
                    else:
                        if hand in connections:
                            log(f'Warning: Duplicate hand identifier {hand}. Rejecting connection.')
                            connection.close()
                        else:
                            log(f'Warning: Invalid hand identifier {hand}. Expecting "L" or "R". Rejecting connection.')
                            connection.close()
                except socket.timeout:
                    log('Timeout waiting for hand identifier')
                    connection.close()
                except Exception as e:
                    log(f'Error reading hand identifier: {e}')
                    connection.close()
                    
            except socket.timeout:
                log('Timeout waiting for connections')
                log(f'Connected {connected_devices}/2 devices. Still waiting...')
            except Exception as e:
                log(f'Error accepting connection: {e}')

        # If both devices are connected, start data collection threads
        if connected_devices == 2:
            log('Both devices (L and R) connected successfully!')
            
            # Start threads for handling each device
            for hand, conn_info in connections.items():
                threads[hand] = threading.Thread(
                    target=handle_imu, 
                    args=(conn_info['connection'], conn_info['address'], hand, event_dict)
                )
                threads[hand].daemon = True
                threads[hand].start()
            
            # Wait for both devices to be ready
            log('Waiting for both devices to be ready...')
            ready_timeout = 30
            start_wait = time.time()
            
            while not (event_dict['ready_events']['L'].is_set() and 
                      event_dict['ready_events']['R'].is_set()):
                if (time.time() - start_wait) > ready_timeout:
                    log('Timeout waiting for devices to be ready')
                    event_dict['stop_event'].set()
                    break
                time.sleep(0.1)
            
            if not event_dict['stop_event'].is_set():
                # Both devices are ready, ask user to start data collection
                input("Press ENTER to start data collection from both devices...")
                
                # Signal start
                log('Starting data collection on both devices...')
                event_dict['start_event'].set()
                
                # Wait for user to stop data collection
                try:
                    input("Press ENTER to stop data collection...")
                except KeyboardInterrupt:
                    log("Interrupted by user.")
                
                # Signal stop
                log('Stopping data collection...')
                event_dict['stop_event'].set()
                
                # Wait for both threads to complete
                max_wait = 10  # seconds
                start_wait = time.time()
                
                while not (event_dict['done_events']['L'].is_set() and 
                          event_dict['done_events']['R'].is_set()):
                    if (time.time() - start_wait) > max_wait:
                        log('Timeout waiting for devices to complete')
                        break
                    time.sleep(0.1)
                
                # Wait for threads to finish
                for hand, thread in threads.items():
                    thread.join(timeout=2)
            
        else:
            log('Failed to connect both devices within the timeout period')
            
    except KeyboardInterrupt:
        log('Server interrupted by user')
    finally:
        # Clean up all connections
        for hand, conn_info in connections.items():
            try:
                conn_info['connection'].close()
            except:
                pass
        
        # Close server socket
        server_socket.close()
        log('Server closed')
        
        # Display summary if data was collected
        for hand in ['L', 'R']:
            if event_dict['data_counts'][hand] > 0:
                log(f"\nData Summary for {hand} hand:")
                log(f"- Collected {event_dict['data_counts'][hand]} data points")
                log(f"- Data saved to {event_dict['filenames'][hand]}")
        
        log("\nNext step: Use this data for your activity recognition project")

if __name__ == "__main__":
    log("Dual IMU Data Collection Server")
    log("This server connects to two M5StickC Plus devices (L and R) simultaneously")
    run_server() 