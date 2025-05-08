#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <NTPClient.h>
#include <M5StickCPlus.h>

// Update with your current WiFi network credentials
const char* ssid = "S24";      // Change to your WiFi network name
const char* password = "Internet";  // Change to your WiFi password

// Update with your computer's IP address (from the test script output)
const char* serverIP = "192.168.203.11";  // Change to your computer's IP address
const int serverPort = 5005;

WiFiClient client;

// Change this to "L" or "R" depending on which hand the M5Stick is worn
const String HAND = "L";

// Preallocate buffer for data string
char dataBuffer[128];

void setup() {
  M5.begin();
  M5.IMU.Init();
  M5.Lcd.setRotation(3);
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextSize(1);
  
  // Set low brightness (0-255)
  M5.Axp.ScreenBreath(7);
  
  M5.Lcd.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    M5.Lcd.print(".");
  }
  
  M5.Lcd.println("\nWiFi connected");
  M5.Lcd.print("IP: ");
  M5.Lcd.println(WiFi.localIP());
  
  M5.Lcd.println("Connecting to server...");
  if (!client.connect(serverIP, serverPort)) {
    M5.Lcd.println("Connection failed");
    return;
  }
  
  M5.Lcd.println("Connected to server");
  M5.Lcd.println("Sending hand identifier...");
  
  // Send hand identifier to server
  client.println(HAND);
  
  M5.Lcd.println("Ready for commands");
}

void loop() {
  if (!client.connected()) {
    M5.Lcd.println("Disconnected from server");
    return;
  }
  
  if (client.available()) {
    String command = client.readStringUntil('\n');
    command.trim();
    
    if (command == "S") {
      M5.Lcd.println("Start command received");
      M5.Lcd.println("Sending data...");
      sendIMUData();
    } else if (command == "End") {
      M5.Lcd.println("End command received");
      client.stop();
      M5.Lcd.println("Disconnected");
    }
  }
}

void sendIMUData() {
  float gyroX, gyroY, gyroZ;
  float accX, accY, accZ;
  
  while (client.connected()) {
    M5.IMU.getGyroData(&gyroX, &gyroY, &gyroZ);
    M5.IMU.getAccelData(&accX, &accY, &accZ);
    
    // Use sprintf for faster string formatting
    sprintf(dataBuffer, "S,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f", 
            millis(), gyroX, gyroY, gyroZ, accX, accY, accZ);
    
    client.println(dataBuffer);
    
    if (client.available()) {
      String command = client.readStringUntil('\n');
      command.trim();
      if (command == "End") {
        M5.Lcd.println("End command received");
        client.println("End");
        return;
      }
    }
  }
}