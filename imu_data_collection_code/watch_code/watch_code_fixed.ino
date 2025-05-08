#include <Wire.h>
#include <WiFi.h>
#include <M5StickCPlus.h>

// WiFi credentials
const char *ssid = "S24";
const char *password = "Internet";

// Server settings
const char *serverIP = "192.168.203.11"; // Your computer's IP address
const int serverPort = 5005;

WiFiClient client;

// Hand identifier
const String HAND = "L"; // Change to "R" for right hand

// Data buffer
char dataBuffer[128];

// Status indicator
bool isConnected = false;
bool isSending = false;
unsigned long reconnectTime = 0;
const unsigned long RECONNECT_INTERVAL = 5000; // Try reconnecting every 5 seconds

void setup()
{
    // Initialize M5StickC Plus
    M5.begin();
    M5.IMU.Init();
    M5.Lcd.setRotation(3);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setCursor(0, 0);

    // Set low brightness to save battery
    M5.Axp.ScreenBreath(10); // 10/255 brightness

    // Connect to WiFi
    connectToWiFi();
}

void loop()
{
    M5.update(); // Update button state

    // Handle button press to reset device
    if (M5.BtnA.wasPressed())
    {
        M5.Lcd.fillScreen(BLACK);
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.println("Restarting...");
        delay(500);
        ESP.restart();
    }

    // Check connection status
    if (!isConnected)
    {
        // Try to reconnect at intervals
        unsigned long currentTime = millis();
        if (currentTime - reconnectTime >= RECONNECT_INTERVAL)
        {
            reconnectTime = currentTime;
            connectToServer();
        }
    }
    else
    {
        // Check if still connected
        if (!client.connected())
        {
            M5.Lcd.println("Connection lost");
            isConnected = false;
            isSending = false;
            return;
        }

        // Check for incoming commands
        if (client.available())
        {
            String command = client.readStringUntil('\n');
            command.trim();

            if (command == "S")
            {
                M5.Lcd.fillScreen(BLACK);
                M5.Lcd.setCursor(0, 0);
                M5.Lcd.println("Start command received");
                M5.Lcd.println("Sending data...");
                isSending = true;
                sendIMUData();
            }
            else if (command == "End")
            {
                M5.Lcd.println("End command received");
                client.println("End");
                isSending = false;
            }
        }
    }

    // Small delay to prevent excessive CPU usage
    delay(10);
}

void connectToWiFi()
{
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("Connecting to WiFi...");
    M5.Lcd.print("SSID: ");
    M5.Lcd.println(ssid);

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
        M5.Lcd.println("\nWiFi connected!");
        M5.Lcd.print("IP: ");
        M5.Lcd.println(WiFi.localIP());

        // Now connect to server
        connectToServer();
    }
    else
    {
        M5.Lcd.println("\nWiFi connection failed");
        M5.Lcd.println("Check credentials or");
        M5.Lcd.println("network availability");
        M5.Lcd.println("Press button to restart");
    }
}

void connectToServer()
{
    M5.Lcd.println("\nConnecting to server:");
    M5.Lcd.print(serverIP);
    M5.Lcd.print(":");
    M5.Lcd.println(serverPort);

    if (client.connect(serverIP, serverPort))
    {
        M5.Lcd.println("Connected to server!");
        M5.Lcd.println("Sending identifier: " + HAND);

        // Send hand identifier
        client.println(HAND);

        M5.Lcd.println("Waiting for commands...");
        isConnected = true;
    }
    else
    {
        M5.Lcd.println("Server connection failed");
        M5.Lcd.println("Will retry in 5 seconds");
        isConnected = false;
    }
}

void sendIMUData()
{
    float gyroX, gyroY, gyroZ;
    float accX, accY, accZ;
    int sendCount = 0;
    unsigned long startTime = millis();

    // Show sending indicator
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println("Sending IMU data...");
    M5.Lcd.println("Hand: " + HAND);

    while (client.connected() && isSending)
    {
        // Get IMU data
        M5.IMU.getGyroData(&gyroX, &gyroY, &gyroZ);
        M5.IMU.getAccelData(&accX, &accY, &accZ);

        // Format data string
        sprintf(dataBuffer, "S,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                millis(), gyroX, gyroY, gyroZ, accX, accY, accZ);

        // Send data
        client.println(dataBuffer);
        sendCount++;

        // Update display occasionally
        if (sendCount % 20 == 0)
        {
            M5.Lcd.fillScreen(BLACK);
            M5.Lcd.setCursor(0, 0);
            M5.Lcd.println("Sending IMU data...");
            M5.Lcd.print("Count: ");
            M5.Lcd.println(sendCount);
            M5.Lcd.println(dataBuffer);
        }

        // Check for End command
        if (client.available())
        {
            String command = client.readStringUntil('\n');
            command.trim();
            if (command == "End")
            {
                M5.Lcd.fillScreen(BLACK);
                M5.Lcd.setCursor(0, 0);
                M5.Lcd.println("End command received");
                M5.Lcd.print("Sent ");
                M5.Lcd.print(sendCount);
                M5.Lcd.println(" data points");
                client.println("End");
                isSending = false;
                break;
            }
        }

        // Small delay to control data rate
        delay(50); // 20 samples per second
    }

    if (!client.connected())
    {
        M5.Lcd.fillScreen(BLACK);
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.println("Connection lost");
        M5.Lcd.print("Sent ");
        M5.Lcd.print(sendCount);
        M5.Lcd.println(" data points");
        isConnected = false;
    }
}