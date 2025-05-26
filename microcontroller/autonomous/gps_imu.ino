#include <TinyGPS++.h> // GPS
#include <Wire.h>    // IMU
#include <I2Cdev.h>
#include <MPU6050.h>


// Define software serial for GPS communication
TinyGPSPlus gps;
#define RXD2 16 // RX pin for GPS
#define TXD2 17 // TX pin for GPS

float lat, lon;

// IMU setup
MPU6050 mpu;

double senser_data[3] = {1, 1, 0};


void setup() {
  Serial.begin(115200);

  Serial3.begin(9600, SERIAL_8N1);  // Adjust to match your GPS module's baud rate

  Wire.begin();
  mpu.initialize();
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);


  
  delay(100);
}

void loop() {
  // Read IMU and GPS data
  readIMUData();
  readGPSData();

  // Publish combined data
  Serial.print("Auto:, ");
  Serial.print(senser_data[0],6);
  Serial.print(", ");
  Serial.print(senser_data[1],6);
  Serial.print(", ");
  Serial.println(senser_data[2],6);
  delay(100);  
}

void readIMUData() {
  int16_t gx, gy, gz;
  mpu.getRotation(&gx, &gy, &gz);
  float sensitivity = 131.0;
  float gz_dps = gz / sensitivity;
  senser_data[2] = gz_dps;
}


void readGPSData() {
  while (Serial3.available() > 0) {
    char c = Serial3.read();
    if (gps.encode(c)) {
      getGPSData();
      return;
    }
  }
 
}



void getGPSData() {
  if (gps.location.isValid()) {
    senser_data[0] = gps.location.lat();
    senser_data[1] = gps.location.lng();
   }
}
