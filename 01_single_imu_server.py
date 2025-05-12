import socket
import sys
import time
import os
from datetime import datetime
import subprocess
import threading
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import msvcrt

# Global variables for real-time visualization
visualization_active = False
fig = None
axes = []
lines = {}
data_buffers = {
    'R': {
        'time': deque(maxlen=500),
        'GX': deque(maxlen=500),
        'GY': deque(maxlen=500),
        'GZ': deque(maxlen=500),
        'AX': deque(maxlen=500),
        'AY': deque(maxlen=500),
        'AZ': deque(maxlen=500)
    }
}
data_lock = threading.Lock()

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

# Function to setup visualization
def setup_visualization():
    global fig, axes, lines, visualization_active
    
    # Set the style for better visualization
    plt.style.use('dark_background')
    
    # Create a figure with 2 subplots (Gyroscope, Accelerometer)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Real-time IMU Data Stream (Right Hand)', fontsize=16)
    
    # Set titles
    axes[0].set_title('Right Hand (R) - Gyroscope', fontsize=14)
    axes[1].set_title('Right Hand (R) - Accelerometer', fontsize=14)
    
    # Data keys for each type of sensor
    gyro_keys = ['GX', 'GY', 'GZ']
    accel_keys = ['AX', 'AY', 'AZ']
    
    # Colors for each axis
    colors = {
        'GX': '#FF5733',  # X - Red
        'GY': '#33FF57',  # Y - Green
        'GZ': '#3357FF',  # Z - Blue
        'AX': '#FF5733',  # X - Red
        'AY': '#33FF57',  # Y - Green
        'AZ': '#3357FF'   # Z - Blue
    }
    
    # Initialize empty lines for each sensor
    lines = {'R': {}}
    
    # Gyroscope lines
    for key in gyro_keys:
        line, = axes[0].plot([], [], lw=1.5, color=colors[key], label=key)
        lines['R'][key] = line
    
    # Accelerometer lines
    for key in accel_keys:
        line, = axes[1].plot([], [], lw=1.5, color=colors[key], label=key)
        lines['R'][key] = line
    
    # Add legends
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    
    # Add grids for better readability
    for row in range(2):
        axes[row].grid(True, linestyle='--', alpha=0.6)
        axes[row].set_facecolor('#121212')  # Dark background
        
    # Set initial y-limits
    axes[0].set_ylim(-100, 100)  # Gyroscope
    axes[1].set_ylim(-2, 2)      # Accelerometer
    
    # Set common x-label for the bottom plot
    axes[1].set_xlabel('Time (seconds)')
    
    # Set common y-labels
    axes[0].set_ylabel('Angular Velocity (°/s)')
    axes[1].set_ylabel('Acceleration (g)')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Make the plot interactive and non-blocking
    plt.ion()
    plt.show(block=False)
    
    visualization_active = True
    return fig, axes

# Function to add data to the visualization buffers
def add_to_visualization(timestamp, data_parts):
    global data_buffers, data_lock
    
    # Extract the data values
    try:
        # Handle malformed data more robustly
        if len(data_parts) < 8:
            return
            
        # Remove any newlines or extra text from values
        clean_parts = []
        for part in data_parts:
            # Split on newline and take the first part
            if '\n' in part:
                part = part.split('\n')[0]
            clean_parts.append(part)
            
        if len(clean_parts) < 8:
            return
            
        # Convert string values to float - Format: S,millis(),GX,GY,GZ,AX,AY,AZ
        gx = float(clean_parts[2])
        gy = float(clean_parts[3])
        gz = float(clean_parts[4])
        ax = float(clean_parts[5])
        ay = float(clean_parts[6])
        az = float(clean_parts[7])
        
        # Use a lock to prevent race conditions when updating the data buffers
        with data_lock:
            # Add relative timestamp (seconds since start)
            if not data_buffers['R']['time']:
                # If this is the first data point, set the reference time
                data_buffers['R']['time'].append(0)
            else:
                # Calculate seconds since first data point
                elapsed = timestamp - data_buffers['R']['first_timestamp']
                data_buffers['R']['time'].append(elapsed)
            
            # Store the first timestamp if not already set
            if 'first_timestamp' not in data_buffers['R']:
                data_buffers['R']['first_timestamp'] = timestamp
                
            # Add the sensor data
            data_buffers['R']['GX'].append(gx)
            data_buffers['R']['GY'].append(gy)
            data_buffers['R']['GZ'].append(gz)
            data_buffers['R']['AX'].append(ax)
            data_buffers['R']['AY'].append(ay)
            data_buffers['R']['AZ'].append(az)
    except (IndexError, ValueError) as e:
        # If there's an error parsing the data, log it but don't crash
        log(f"Error adding data to visualization: {e}")

# Function to update visualization (for main thread)
def update_visualization():
    global lines, data_buffers, data_lock, fig
    
    if not visualization_active:
        return
    
    with data_lock:
        if data_buffers['R']['time']:  # Only update if there's data
            # Get time data
            times = list(data_buffers['R']['time'])
            if not times:
                return
                
            # Update gyroscope data (row 0)
            for key in ['GX', 'GY', 'GZ']:
                values = list(data_buffers['R'][key])
                if values:
                    # Update the line data
                    lines['R'][key].set_data(times, values)
            
            # Update accelerometer data (row 1)
            for key in ['AX', 'AY', 'AZ']:
                values = list(data_buffers['R'][key])
                if values:
                    # Update the line data
                    lines['R'][key].set_data(times, values)
            
            # Automatically adjust x-axis limits for both rows
            for row in range(2):
                # Update x-axis limits to show all data with some padding
                max_time = max(times)
                min_time = max(min(times), max_time - 10)  # Show last 10 seconds or all data
                axes[row].set_xlim(min_time, max_time + 0.5)
            
            # Dynamically adjust y-axis limits for better visualization
            # Gyroscope row
            gyro_data = []
            for key in ['GX', 'GY', 'GZ']:
                if data_buffers['R'][key]:
                    recent_values = list(data_buffers['R'][key])[-30:]  # Last 30 points
                    gyro_data.extend(recent_values)
            
            if gyro_data:
                min_val = min(gyro_data)
                max_val = max(gyro_data)
                margin = max(5, (max_val - min_val) * 0.1)
                axes[0].set_ylim(min_val - margin, max_val + margin)
            
            # Accelerometer row
            accel_data = []
            for key in ['AX', 'AY', 'AZ']:
                if data_buffers['R'][key]:
                    recent_values = list(data_buffers['R'][key])[-30:]  # Last 30 points
                    accel_data.extend(recent_values)
            
            if accel_data:
                min_val = min(accel_data)
                max_val = max(accel_data)
                margin = max(0.1, (max_val - min_val) * 0.1)
                axes[1].set_ylim(min_val - margin, max_val + margin)
    
    # Draw the updated plot
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception as e:
        # In case of visualization errors, just continue without crashing
        log(f"Visualization update error: {e}")

# Helper function to process a single data point
def process_data_point(data, timestamp, output_file, visualization_active):
    """Process a single data point and return True if successful, False otherwise"""
    try:
        data_parts = data.split(',')
        if len(data_parts) >= 8:  # Original format: S,millis(),GX,GY,GZ,AX,AY,AZ
            # Extract the relevant parts
            gx = float(data_parts[2])
            gy = float(data_parts[3])
            gz = float(data_parts[4])
            ax = float(data_parts[5])
            ay = float(data_parts[6])
            az = float(data_parts[7])
            
            # Write data to file
            data_line = f"{gx:.6f},{gy:.6f},{gz:.6f},{ax:.6f},{ay:.6f},{az:.6f},{timestamp:.6f}\n"
            output_file.write(data_line)
            output_file.flush()
            
            # Add data to visualization if active
            if visualization_active:
                add_to_visualization(timestamp, data_parts)
                
            return True
        else:
            log(f"Skipping malformed data point: {data}", 'R')
            return False
    except ValueError:
        log(f"Skipping non-numeric data point: {data}", 'R')
        return False
    except Exception as e:
        log(f"Error processing data point: {data}, Error: {e}", 'R')
        return False

# Thread to handle the IMU device
def handle_imu(connection, client_address, event_dict):
    """Handle the IMU device connection"""
    hand = 'R'  # We're only working with the Right hand
    log(f"Starting handler for Right hand", hand)
    
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
        
        # Construct the path using the folder name from event_dict
        data_path = event_dict['data_path']
        
        # Ensure the directory exists
        os.makedirs(data_path, exist_ok=True)
        
        # Open a file to save the data
        fixed_filename = f"{data_path}/IMU_R_data.csv"
        
        with open(fixed_filename, 'w') as fixed_file:
            # Write headers to the file
            header = "GX,GY,GZ,AX,AY,AZ,timestamp\n"
            fixed_file.write(header)
            
            # Signal that this IMU is ready
            log(f"Device ready, waiting for start signal", hand)
            event_dict['ready_event'].set()
            
            # Wait for start event
            event_dict['start_event'].wait()
            
            # Send start command
            log('Sending start command "S\\n"...', hand)
            connection.sendall(b"S\n")
            
            # Receive IMU data
            log('Waiting for IMU data...', hand)
            count = 0
            
            # Initialize variables for data rate calculation
            last_rate_check = time.time()
            points_since_check = 0
            last_status_update = time.time()
            
            # Keep reading data until stopped
            try:
                while not event_dict['stop_event'].is_set():
                    try:
                        data = connection.recv(128).decode('utf-8').strip()
                        if data:
                            if data == "End":
                                log("Received End signal from device", hand)
                                break
                            
                            # Split concatenated data points if multiple 'S,' patterns are found
                            if data.count('S,') > 1:
                                # Split the data by 'S,' and add 'S,' back to each part (except the first one)
                                data_points = data.split('S,')
                                if data_points[0] == '':  # If the data starts with 'S,'
                                    data_points.pop(0)  # Remove empty first element
                                    data_points = ['S,' + dp for dp in data_points]  # Add 'S,' back to each part
                                else:
                                    # First part likely doesn't start with 'S,' due to partial data
                                    first_part = data_points.pop(0)
                                    data_points = ['S,' + dp for dp in data_points]  # Add 'S,' back to each part
                                    if first_part:  # Only add non-empty first part
                                        data_points.insert(0, first_part)
                                
                                log(f"Split {len(data_points)} concatenated data points", hand)
                                
                                # Process each data point separately
                                current_timestamp = time.time()
                                for dp in data_points:
                                    process_data_point(dp, current_timestamp, fixed_file, visualization_active)
                                    count += 1
                                    points_since_check += 1
                                
                                # Update rate calculation if needed
                                if count % 50 == 0:
                                    now = time.time()
                                    elapsed = now - last_rate_check
                                    if elapsed > 0:
                                        rate = points_since_check / elapsed
                                        log(f"Data rate: {rate:.1f} points/second, Total: {count} points", hand)
                                        last_rate_check = now
                                        points_since_check = 0
                            else:
                                # Only log data point periodically to reduce console spam
                                current_time = time.time()
                                if current_time - last_status_update >= 5.0:  # Log every 5 seconds
                                    log(f'Data point {count+1}: {data}', hand)
                                    last_status_update = current_time
                                
                                # Get current timestamp for this data point
                                current_timestamp = time.time()
                                
                                # Process this single data point
                                if process_data_point(data, current_timestamp, fixed_file, visualization_active):
                                    count += 1
                                    points_since_check += 1
                                    
                                    # Calculate and display data rate every 50 points
                                    if count % 50 == 0:
                                        now = time.time()
                                        elapsed = now - last_rate_check
                                        if elapsed > 0:
                                            rate = points_since_check / elapsed
                                            log(f"Data rate: {rate:.1f} points/second, Total: {count} points", hand)
                                            last_rate_check = now
                                            points_since_check = 0
                            
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
            event_dict['data_count'] = count
            event_dict['filename'] = fixed_filename
            
            log(f'Received {count} data points, saved to {fixed_filename}', hand)
            
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
        event_dict['done_event'].set()
        
        # Close connection
        try:
            connection.close()
        except:
            pass
        
        log(f"Handler for Right hand completed", hand)

# Add this function near the top, after imports
def get_network_info():
    """Get detailed network information including all interfaces and IP addresses"""
    network_info = []
    try:
        # Use socket module without redefining it
        hostname = socket.gethostname()
        network_info.append(f"Hostname: {hostname}")
        
        # Try to get all IP addresses
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                result = subprocess.run(['ipconfig'], capture_output=True, text=True)
                network_info.append("Network interfaces:")
                network_info.append(result.stdout[:1000])  # Limit output size
            else:  # Linux/Mac
                result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                network_info.append("Network interfaces:")
                network_info.append(result.stdout[:1000])  # Limit output size
        except Exception as e:
            network_info.append(f"Error getting detailed network info: {e}")
        
        # Get local IP using socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            network_info.append(f"Primary IP (via socket): {local_ip}")
        except Exception as e:
            network_info.append(f"Error getting primary IP: {e}")
            
    except Exception as e:
        network_info.append(f"Error gathering network information: {e}")
    
    return "\n".join(network_info)

# Main server function
def run_server():
    global visualization_active
    
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
    
    # Ask for a folder name to save data
    folder_name = input("Enter a folder name for saving data (e.g., 'hammering', 'sawing'): ")
    
    # Remove any invalid characters from folder name
    folder_name = ''.join(c for c in folder_name if c.isalnum() or c in ' _-')
    if not folder_name:
        folder_name = "default"
    
    # Create the data path
    data_path = f"02_data/{folder_name}"
    os.makedirs(data_path, exist_ok=True)
    
    log(f"Data will be saved to: {data_path}")
    
    # Always enable visualization
    log("Starting real-time visualization...")
    try:
        setup_visualization()
    except Exception as e:
        log(f"Failed to start visualization: {e}")
        visualization_active = False
    
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Get detailed network information
    log("Getting detailed network information...")
    network_info = get_network_info()
    log(network_info)
    
    # Get IP addresses for user information
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
        log(f"Your computer hostname: {hostname}")
        log(f"Your IP address: {local_ip}")
        
        # Add all available IP addresses for better diagnosis
        all_ips = []
        try:
            # Get all IP addresses - use imported socket module directly
            addrs = socket.getaddrinfo(socket.gethostname(), None)
            for addr in addrs:
                if addr[0] == socket.AF_INET:  # Only IPv4
                    ip = addr[4][0]
                    if ip not in all_ips and not ip.startswith('127.'):
                        all_ips.append(ip)
            
            if all_ips:
                log(f"All available IPv4 addresses: {', '.join(all_ips)}")
                log("If connection fails with one IP, try others from this list")
        except Exception as e:
            log(f"Error getting all IP addresses: {e}")
        
        log("Make sure the M5StickC Plus device is configured with this IP address")
        log("If device cannot connect, try using one of the IP addresses listed above")
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
    server_socket.listen(1)  # Allow only 1 connection
    server_socket.settimeout(300)  # 5 minute timeout for accepting connections

    # Initialize shared events
    event_dict = {
        'start_event': threading.Event(),  # Signal to start data collection
        'stop_event': threading.Event(),   # Signal to stop data collection
        'ready_event': threading.Event(),  # Signal when the device is ready
        'done_event': threading.Event(),   # Signal when the device is done
        'data_count': 0,                   # Store data count
        'filename': "",                    # Store filename
        'data_path': data_path             # Path to save data
    }

    # Variable to store the connection
    connection = None
    client_address = None
    imu_thread = None
    
    try:
        log('Waiting for the M5StickC Plus device (Right hand / R) to connect...')
        log('Make sure the M5StickC Plus device is:')
        log('1. Powered on')
        log('2. Connected to the same WiFi network')
        log('3. Configured with the correct server IP and port')
        log('4. Configured with HAND="R"')
        
        # Add more debug information about the waiting state
        log('Server is now listening on port 5005 - waiting for connection...')
        log('Press Ctrl+C to stop the server if the device does not connect')
        
        # Accept connection from the IMU
        device_connected = False
        
        max_wait_time = 300  # 5 minutes total to wait for the device
        start_wait_time = time.time()
        
        while not device_connected and not event_dict['stop_event'].is_set():
            # Check if we've waited too long
            if time.time() - start_wait_time > max_wait_time:
                log('Maximum wait time exceeded. Giving up on connection attempts.')
                break
                
            try:
                # Accept connection with timeout
                log('Waiting for connection... (timeout: 10 seconds)')
                server_socket.settimeout(10)  # Increased timeout for more reliable connections
                connection, client_address = server_socket.accept()
                log(f'Connection from {client_address}')
                
                # Get hand identifier
                connection.settimeout(15)  # Increased timeout for reading identifier
                try:
                    log(f'Waiting for hand identifier from {client_address}...')
                    hand = connection.recv(128).decode('utf-8').strip()  # Increased buffer size
                    log(f'Received identifier data: "{hand}"')
                    
                    # Check if this is the Right hand device
                    if 'R' in hand:
                        log('Confirmed as Right hand (R) device')
                        device_connected = True
                        
                        # Start the handler thread
                        imu_thread = threading.Thread(
                            target=handle_imu, 
                            args=(connection, client_address, event_dict)
                        )
                        imu_thread.daemon = True
                        imu_thread.start()
                    else:
                        log(f'Warning: Not a Right hand device. Received: "{hand}". Rejecting connection.')
                        connection.close()
                        connection = None
                except socket.timeout:
                    log(f'Timeout waiting for hand identifier from {client_address}')
                    # Try sending a request for identifier
                    try:
                        log(f'Sending request for identifier to {client_address}...')
                        connection.sendall(b"ID?\n")
                        connection.settimeout(10)
                        hand = connection.recv(128).decode('utf-8').strip()
                        log(f'Received identifier after prompting: "{hand}"')
                        
                        # Check if this is the Right hand device
                        if 'R' in hand:
                            log('Confirmed as Right hand (R) device after prompting')
                            device_connected = True
                            
                            # Start the handler thread
                            imu_thread = threading.Thread(
                                target=handle_imu, 
                                args=(connection, client_address, event_dict)
                            )
                            imu_thread.daemon = True
                            imu_thread.start()
                        else:
                            log(f'Warning: Not a Right hand device after prompting. Received: "{hand}". Rejecting connection.')
                            connection.close()
                            connection = None
                    except Exception as e:
                        log(f'Failed to get identifier after prompting: {e}')
                        connection.close()
                        connection = None
                except Exception as e:
                    log(f'Error reading hand identifier: {e}')
                    connection.close()
                    connection = None
                    
                # Update visualization if active
                if visualization_active:
                    try:
                        update_visualization()
                    except Exception as e:
                        log(f"Visualization update error: {e}")
            
            except socket.timeout:
                log('Still waiting for connection...')
            except Exception as e:
                log(f'Error accepting connection: {e}')

        # If device is connected, wait for it to be ready
        if device_connected:
            log('Right hand (R) device connected successfully!')
            
            # Wait for device to be ready
            log('Waiting for device to be ready...')
            ready_timeout = 30
            start_wait = time.time()
            
            while not event_dict['ready_event'].is_set():
                if (time.time() - start_wait) > ready_timeout:
                    log('Timeout waiting for device to be ready')
                    event_dict['stop_event'].set()
                    break
                time.sleep(0.1)
            
            if not event_dict['stop_event'].is_set():
                # Device is ready, ask user to start data collection
                input("Press ENTER to start data collection...")
                
                # Signal start
                log('Starting data collection...')
                event_dict['start_event'].set()
                
                # Set data collection timer for 180 seconds (3 minutes)
                collection_start_time = time.time()
                collection_timeout = 80  # 3 minutes in seconds
                log(f'Data collection will automatically stop after {collection_timeout} seconds')
                
                # Wait for user to stop data collection or timeout
                try:
                    while not event_dict['stop_event'].is_set():
                        # Update visualization
                        if visualization_active:
                            update_visualization()
                        
                        # Check if Enter was pressed (non-blocking)
                        if msvcrt.kbhit() and msvcrt.getch() in [b'\r', b'\n']:
                            log('Data collection stopped by user')
                            break
                        
                        # Check if timeout reached
                        elapsed_time = time.time() - collection_start_time
                        if elapsed_time >= collection_timeout:
                            log(f'Reached maximum collection time of {collection_timeout} seconds')
                            break
                            
                        # Print time remaining every 30 seconds
                        if int(elapsed_time) % 30 == 0 and int(elapsed_time) > 0:
                            time_remaining = collection_timeout - elapsed_time
                            log(f'Time remaining: {time_remaining:.0f} seconds')
                        
                        # Short sleep to avoid excessive CPU usage
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    log("Interrupted by user.")
                
                # Signal stop
                log('Stopping data collection...')
                event_dict['stop_event'].set()
                
                # Wait for thread to complete
                max_wait = 10  # seconds
                start_wait = time.time()
                
                while not event_dict['done_event'].is_set():
                    if (time.time() - start_wait) > max_wait:
                        log('Timeout waiting for device to complete')
                        break
                    time.sleep(0.1)
                
                # Wait for thread to finish
                if imu_thread:
                    imu_thread.join(timeout=2)
                
                # Keep visualization open until user presses Enter if it's active
                if visualization_active:
                    log("Press ENTER to close visualization and exit...")
                    try:
                        while True:
                            # Update visualization
                            update_visualization()
                            
                            # Check if Enter was pressed (non-blocking)
                            if msvcrt.kbhit() and msvcrt.getch() in [b'\r', b'\n']:
                                break
                            
                            # Short sleep to avoid excessive CPU usage
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        pass
                    
                    # Clean up matplotlib
                    plt.close(fig)
            
        else:
            log('Failed to connect to Right hand device within the timeout period')
            
    except KeyboardInterrupt:
        log('Server interrupted by user')
    finally:
        # Clean up the connection
        if connection:
            try:
                connection.close()
            except:
                pass
        
        # Close server socket
        server_socket.close()
        log('Server closed')
        
        # Display summary if data was collected
        if event_dict['data_count'] > 0:
            log("\nData Summary for Right hand (R):")
            log(f"- Collected {event_dict['data_count']} data points")
            log(f"- Data saved to {event_dict['filename']}")
        
        log(f"\nAll data saved to folder: {event_dict['data_path']}")
        log("\nNext step: Use this data for your activity recognition project")

        # Make sure to close the visualization if it's open
        if visualization_active:
            try:
                plt.close(fig)
            except:
                pass

if __name__ == "__main__":
    log("Single IMU Data Collection Server (Right Hand Only)")
    log("This server connects to a single M5StickC Plus device (Right hand / R)")
    run_server() 