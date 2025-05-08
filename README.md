# IMU-Based Activity Recognition

This project uses data from two IMUs to predict if a person is hammering, sawing, or doing something else.

## Project Structure

- **Data/**: Data collection and processing (**Place Dataset here**)
- **Prediction/**: ML models for activity recognition
- **Visualization/**: Data visualization components
- **imu_data_collection_code/**: Code for collecting IMU data from M5StickC Plus devices

## Features

- IMU data collection and processing
- Feature extraction from raw sensor data
- Machine learning model for activity classification
- Real-time activity recognition 

## IMU Setup Instructions

### Setting up the M5StickC Plus Devices

1. **Hardware Requirements**:
   - Two M5StickC Plus devices (one for each hand)
   - Computer with Python installed
   - Both devices connected to the same WiFi network as your computer

2. **Upload Code to M5StickC Plus**:
   - Install Arduino IDE and M5StickC Plus library
   - Open `imu_data_collection_code/watch_code/working_watch_code.ino` in Arduino IDE
   - Configure the following parameters in the code:
     ```cpp
     // WiFi credentials
     const char *ssid = "S24";              // Change to your WiFi SSID
     const char *password = "Internet";     // Change to your WiFi password
     
     // Server settings
     const char *serverIP = "192.168.203.11"; // Change to your computer's IP address
     const int serverPort = 5005;
     
     // Hand identifier - change to "R" for right hand, "L" for left hand
     const String HAND = "L"; 
     ```
   - Upload the code to both M5StickC Plus devices:
     - Set `HAND = "L"` for the left-hand device
     - Set `HAND = "R"` for the right-hand device

### Collecting Data with the Server

1. **Run the Dual IMU Server**:
   ```
   python dual_imu_server.py
   ```

2. **Server Operation**:
   - The server will start and display your computer's IP address
   - Make sure this IP matches the `serverIP` in the M5StickC Plus code
   - The server will wait for both devices (L and R) to connect
   - When both devices are connected, press ENTER to start data collection
   - Press ENTER again to stop data collection
   - Data will be saved to the `data/` directory with filenames including the hand identifier and timestamp

3. **Troubleshooting**:
   - If the devices don't connect, check WiFi settings and IP address configuration
   - Ensure both devices are powered on and running the correct code
   - The server automatically checks for port conflicts and offers to resolve them
   - Look at the M5StickC Plus display for connection status information

4. **Data Format**:
   - Collected data is saved in CSV format with the following columns:
   - `sync_flag,IMU_stopwatch,GX,GY,GZ,AX,AY,AZ,timestamp`
   - GX/GY/GZ: Gyroscope data (angular velocity)
   - AX/AY/AZ: Accelerometer data (linear acceleration)

## Data Processing & Activity Recognition

After collecting data, use the provided data processing and machine learning components to identify activities 