#include <Wire.h>
#include <WiFi.h>
#include <M5StickCPlus.h>

// WiFi credentials
const char *ssid = "S24";
const char *password = "Internet";

// Server settings
const char *serverIP = "192.168.203.11"; // Your computer's IP address
const int serverPort = 5005;

// Hand identifier - change to "R" for right hand
const String HAND = "L";

// Connection state
bool isConnected = false;
bool isSending = false;
WiFiClient client;

// Smaller data buffer to prevent memory issues
char dataBuffer[96];

// Debug settings
bool debugMode = true;
unsigned long lastDisplayUpdate = 0;
const unsigned long DISPLAY_UPDATE_INTERVAL = 500; // Update display every 500ms

// Function declarations
void connectToWiFi();
void connectToServer();
void sendIMUData();
void displayStatus(String status, bool clear = false);

void setup()
{
    // Initialize M5StickC Plus with lower power consumption
    M5.begin();
    M5.Lcd.setRotation(3);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setCursor(0, 0);

    // Set lower brightness
    M5.Axp.ScreenBreath(10);

    // Initialize IMU with retry
    bool imuInitialized = false;
    for (int i = 0; i < 3; i++)
    {
        if (M5.IMU.Init() == 0)
        {
            imuInitialized = true;
            break;
        }
        delay(100);
    }

    if (!imuInitialized)
    {
        displayStatus("IMU init failed!", true);
        delay(2000);
    }

    displayStatus("Starting...", true);
    displayStatus("Hand: " + HAND);

    // Connect to WiFi
    connectToWiFi();
}

void loop()
{
    M5.update();

    // Handle button press for reset
    if (M5.BtnA.wasPressed())
    {
        displayStatus("Restarting...", true);
        delay(500);
        ESP.restart();
    }

    // Check if connected to WiFi
    if (WiFi.status() != WL_CONNECTED)
    {
        isConnected = false;
        displayStatus("WiFi disconnected", true);
        displayStatus("Reconnecting...");
        connectToWiFi();
        return;
    }

    // Check if connected to server
    if (!isConnected)
    {
        connectToServer();
        delay(100);
        return;
    }

    // Check if client is still connected
    if (!client.connected())
    {
        isConnected = false;
        isSending = false;
        displayStatus("Server disconnected", true);
        delay(1000);
        connectToServer();
        return;
    }

    // Check for incoming commands
    if (client.available())
    {
        String command = client.readStringUntil('\n');
        command.trim();

        if (command == "S")
        {
            displayStatus("Received START cmd", true);
            delay(200); // Short delay to stabilize
            isSending = true;
            sendIMUData();
        }
        else if (command == "End")
        {
            displayStatus("Received END cmd", true);
            client.print("End\n"); // Acknowledge with newline
            isSending = false;
        }
    }

    // Small delay to prevent excessive CPU usage
    delay(10);
}

void connectToWiFi()
{
    displayStatus("Connecting WiFi", true);
    displayStatus("SSID: " + String(ssid));

    WiFi.disconnect();
    delay(100);
    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20)
    {
        delay(500);
        M5.Lcd.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        displayStatus("WiFi connected!", true);
        displayStatus("IP: " + WiFi.localIP().toString());
        delay(500);
    }
    else
    {
        displayStatus("WiFi failed!", true);
        displayStatus("Check credentials");
        displayStatus("Press button to restart");
    }
}

void connectToServer()
{
    displayStatus("Connecting server", true);
    displayStatus(String(serverIP) + ":" + String(serverPort));

    // Close any existing connection
    if (client.connected())
    {
        client.stop();
        delay(100);
    }

    // Try to connect with timeout
    client.setTimeout(5000);
    if (client.connect(serverIP, serverPort))
    {
        isConnected = true;
        displayStatus("Server connected!", true);

        // Send hand identifier with newline
        displayStatus("Sending ID: " + HAND);
        client.print(HAND + "\n");

        // Wait for a moment to ensure the identifier is sent
        delay(200);

        displayStatus("Waiting for cmds");
    }
    else
    {
        isConnected = false;
        displayStatus("Server conn failed!", true);
        displayStatus("Will retry soon...");
    }
}

void sendIMUData()
{
    float gyroX, gyroY, gyroZ;
    float accX, accY, accZ;
    int sendCount = 0;
    unsigned long startTime = millis();

    displayStatus("Sending data...", true);
    displayStatus("Hand: " + HAND);

    // Set a longer timeout for data sending
    client.setTimeout(10000);

    // Loop until disconnected or stopped
    while (client.connected() && isSending)
    {
        // Get IMU data with error checking
        try
        {
            M5.IMU.getGyroData(&gyroX, &gyroY, &gyroZ);
            M5.IMU.getAccelData(&accX, &accY, &accZ);
        }
        catch (...)
        {
            // If reading IMU fails, use zeros
            gyroX = gyroY = gyroZ = 0;
            accX = accY = accZ = 0;
        }

        // Format data string - use simpler formatting to reduce CPU load
        sprintf(dataBuffer, "S,%lu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                millis(), gyroX, gyroY, gyroZ, accX, accY, accZ);

        // Send data with explicit newline in the buffer
        client.print(dataBuffer);
        sendCount++;

        // Update display periodically
        if (millis() - lastDisplayUpdate > DISPLAY_UPDATE_INTERVAL)
        {
            displayStatus("Sending data...", true);
            displayStatus("Count: " + String(sendCount));

            // Simple visualization of connection
            for (int i = 0; i < (sendCount % 10); i++)
            {
                M5.Lcd.print(">");
            }
            lastDisplayUpdate = millis();
        }

        // Check for End command with timeout
        client.setTimeout(10);
        if (client.available())
        {
            String command = client.readStringUntil('\n');
            command.trim();
            if (command == "End")
            {
                displayStatus("Received END cmd", true);
                displayStatus("Sent " + String(sendCount) + " points");
                client.print("End\n");
                isSending = false;
                break;
            }
        }
        client.setTimeout(10000);

        // Delay between data points to improve stability
        delay(100); // 10 samples per second - more stable
    }

    if (!client.connected())
    {
        isConnected = false;
        displayStatus("Connection lost", true);
        displayStatus("Sent " + String(sendCount) + " points");
    }
}

void displayStatus(String status, bool clear)
{
    if (clear)
    {
        M5.Lcd.fillScreen(BLACK);
        M5.Lcd.setCursor(0, 0);
    }
    M5.Lcd.println(status);
}