#include <TinyGPS++.h> // GPS
#include <I2Cdev.h>
#include <Adafruit_AHT10.h>
#include <Adafruit_AHTX0.h>
#include <Wire.h>
#include <SPI.h>
#include <MQUnifiedsensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/Joy.h>

#define dirPin 49
#define stepPin 47
#define latch 12
#define light 13

const float pulse_count = 800.0;


// Define software serial for GPS communication
TinyGPSPlus gps;
#define RXD2 16 // RX pin for GPS
#define TXD2 17 // TX pin for GPS

float lat, lon;



void rotate(bool dir, float deg) {
  digitalWrite(dirPin, dir);
  float n = (deg / 360.0) * pulse_count;
  for (int i = 0; i < n; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(2500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(2500);
  }
}


void messageCb(const sensor_msgs::Joy& joy_msg) {
  //stepper movements
  if (joy_msg.buttons[11]) {//aaaaaah
    rotate(1, 120);
  } 
  else if (joy_msg.buttons[12]) {//oooooh
    rotate(0, 120);
  }
  else if(joy_msg.buttons[5]){//aa
    rotate(1, 10);
  }
  else if(joy_msg.buttons[6]){//oo
    rotate(0, 10);
  }

  else if(joy_msg.buttons[13]){ //patt
    digitalWrite(latch, 0);
    delay(1000);
    digitalWrite(latch, 1);
  }

  else if(joy_msg.buttons[7]){ //jhilmil
    digitalWrite(light, 0);
  }
  else if(joy_msg.buttons[8]){ //unjhilmil
    digitalWrite(light, 1);
  }
}

Adafruit_AHTX0 aht;
ros::NodeHandle nh;
std_msgs::Float32MultiArray sensor_data;
ros::Publisher science("sensor_data",&sensor_data);
ros::Subscriber<sensor_msgs::Joy> sub("j2", messageCb);


#define placa "Arduino UNO"
#define Voltage_Resolution 5
#define ADC_Bit_Resolution 10 // For Arduino UNO/MEGA/NANO

// MQ131 (Ozone Sensor)
#define MQ131_PIN A4
#define MQ131_TYPE "MQ-131"
#define RatioMQ131CleanAir 15

// MQ135 (NH4 Sensor)
#define MQ135_PIN A3
#define MQ135_TYPE "MQ-135"
#define RatioMQ135CleanAir 3.6 // RS / R0 = 3.6 ppm

// MQ4 (Methane Sensor)
#define MQ4_PIN A2
#define MQ4_TYPE "MQ-4"
#define RatioMQ4CleanAir 4.4 // RS / R0 = 60 ppm

// MQ7 (Carbon Monoxide Sensor)
#define MQ7_PIN A1

// UV Sensor Pin
#define UV_PIN A0

// Dallas Temperature sensor pins
#define ONE_WIRE_BUS 51 // Pin for Dallas Temperature sensor

// pH Sensor Pin
#define SensorPinPh A15

// Soil Moisture Sensor Pin
#define SOIL_MOISTURE_PIN A5
#define AIR_VALUE 540
#define WATER_VALUE 300

// Sensor objects
MQUnifiedsensor MQ131(placa, Voltage_Resolution, ADC_Bit_Resolution, MQ131_PIN, MQ131_TYPE);
MQUnifiedsensor MQ135(placa, Voltage_Resolution, ADC_Bit_Resolution, MQ135_PIN, MQ135_TYPE);
//MQUnifiedsensor MQ4(placa, Voltage_Resolution, ADC_Bit_Resolution, MQ4_PIN, MQ4_TYPE);

Adafruit_Sensor *aht_humidity, *aht_temp;

// Dallas temperature sensor
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature dallasSensors(&oneWire);

// Variables for pH sensor
int buf[10];
int avgValue = 0;



void setup() {
    nh.getHardware()->setBaud(115200);
    nh.initNode();
    nh.advertise(science);
    Serial3.begin(9600, SERIAL_8N1);  // Adjust to match your GPS module's baud rate

    Wire.begin();
    
    sensor_data.data_length = 10;
    sensor_data.data = (float*)malloc(sensor_data.data_length * sizeof(float));
    // Initialize MQ131 sensor (Ozone)
    MQ131.setRegressionMethod(1); //_PPM = a * ratio ^ b
    MQ131.setA(23.943); 
    MQ131.setB(-1.11); 
    MQ131.init();
    calibrateSensor(MQ131, RatioMQ131CleanAir);

    // Initialize MQ135 sensor (NH4)
    MQ135.setRegressionMethod(1); //_PPM = a * ratio ^ b
    MQ135.setA(102.2); 
    MQ135.setB(-2.473); 
    MQ135.init();
    calibrateSensor(MQ135, RatioMQ135CleanAir);

    // Initialize MQ4 sensor (Methane)
    // MQ4.setRegressionMethod(1); //_PPM = a * ratio ^ b
    // MQ4.setA(1012.7); 
    // MQ4.setB(-2.786); 
    // MQ4.init();
    // calibrateSensor(MQ4, RatioMQ4CleanAir);
    delay(2000);
    // Initialize AHT10 sensor (Temperature and Humidity)
    if (!aht.begin()) {
        while (1) delay(10);
    }
    aht_temp = aht.getTemperatureSensor();
    aht_humidity = aht.getHumiditySensor();

    // Initialize Dallas temperature sensor
    dallasSensors.begin();

    // Initialize Soil Moisture Sensor
    pinMode(SOIL_MOISTURE_PIN, INPUT);
    pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(latch, OUTPUT);
  pinMode(light, OUTPUT);

  digitalWrite(latch, 1);
  digitalWrite(light, 1);
  
  nh.subscribe(sub); // 0 means laser off
}

// Define intervals for each task
unsigned long lastSensorReadTime = 0;
unsigned long lastPublishTime = 0;
const unsigned long sensorReadInterval = 3000; // Read sensors every 1 second
const unsigned long publishInterval = 3000; // Publish data every 1 second

void loop() {
    unsigned long currentTime = millis();

    // Check if it's time to read sensors
    if (currentTime - lastSensorReadTime >= sensorReadInterval) {
        lastSensorReadTime = currentTime;

        // Read sensor values
        sensors_event_t humidity, temp;
        readTemperatureHumidity(humidity, temp);
        float uvVoltage = readUVSensor(UV_PIN);
        int mq7Value = readMQ7(MQ7_PIN);
        float phValue = readPHSensor(SensorPinPh);
        float ozoneValue = readOzoneSensor();
        float nh4Value = readMQ135Sensor();
        // float methaneValue = readMQ4Sensor();
        float dallasTempC = readDallasTemperature();
        int soilMoistureValue = readSoilMoistureSensor(SOIL_MOISTURE_PIN);

        // Store the values in the sensor_data array
        sensor_data.data[0] = temp.temperature;
        sensor_data.data[1] = humidity.relative_humidity;
        sensor_data.data[2] = uvVoltage;
        sensor_data.data[3] = mq7Value;
        sensor_data.data[4] = phValue;
        sensor_data.data[5] = ozoneValue;
        sensor_data.data[6] = dallasTempC;
        sensor_data.data[7] = soilMoistureValue;
        sensor_data.data[8] = nh4Value;
        sensor_data.data[9] = lat;
        sensor_data.data[10] = lon;

    }

    // Check if it's time to publish data
    if (currentTime - lastPublishTime >= publishInterval) {
        lastPublishTime = currentTime;

        // Publish the sensor data
        science.publish(&sensor_data);
    }

    // Call nh.spinOnce() to process any incoming messages
    nh.spinOnce();
}

// Function to calibrate MQ sensors
void calibrateSensor(MQUnifiedsensor &sensor, float cleanAirRatio) {
    // Serial.print("Calibrating sensor, please wait.");
    float calcR0 = 0;
    for (int i = 1; i <= 10; i++) {
        sensor.update();
        calcR0 += sensor.calibrate(cleanAirRatio);
        delay(500);
    }
    sensor.setR0(calcR0 / 10);
    // Serial.println(" done!");

    if (isinf(calcR0)) {
        // Serial.println("Warning: Connection issue (Open circuit). Please check wiring.");
        while (1);
    } else if (calcR0 == 0) {
        // Serial.println("Warning: Connection issue (Short circuit). Please check wiring.");
        while (1);
    }
    sensor.serialDebug(true);
}

// Function to read Temperature and Humidity (AHT10)
void readTemperatureHumidity(sensors_event_t &humidity, sensors_event_t &temp) {
    aht_humidity->getEvent(&humidity);
    aht_temp->getEvent(&temp);
}

// Function to read UV Sensor
float readUVSensor(int pin) {
    float sensorValue = analogRead(pin);
    return sensorValue / 1024 * 5.0;
}

// Function to read MQ7 Sensor
int readMQ7(int pin) {
    return analogRead(pin);
}

// Function to read pH Sensor
float readPHSensor(int pin) {
    int buf[10];  // Array to store sensor readings
    for (int i = 0; i < 10; i++) {
        buf[i] = analogRead(pin);
        delay(10);  // Delay to stabilize the sensor
    }

    int minValue = buf[0];
    int maxValue = buf[0];
    int sum = 0;

    // // Find min, max, and sum
    for (int i = 0; i < 10; i++) {
        if (buf[i] < minValue) minValue = buf[i];
        if (buf[i] > maxValue) maxValue = buf[i];
        sum += buf[i];
    }

    sum -= (minValue + maxValue); // Exclude min and max
    float voltage = (float)sum / 8 * 5.0 / 1024;  // Convert to voltage
    // float voltage= (float) analogRead(pin) * 5.0 / 1024; 

    // Adjust this formula based on the sensor's voltage-to-pH relation
    // For example, many pH sensors give ~2.5V at pH 7, with ~0.5V change per pH unit
    float phValue = (voltage - 2.5) * 3.0;  // This is a basic conversion, may need calibration

    return phValue;  // Return the calculated pH value
}


// Function to read Ozone Sensor (MQ131)
float readOzoneSensor() {
    MQ131.update();
    MQ131.readSensorR0Rs();
    return MQ131.readSensor(); // Change this if necessary
}

// Function to read MQ135 Sensor (NH4)
float readMQ135Sensor() {
    MQ135.update();
    return MQ135.readSensor(); // Change this if necessary
}

// Function to read MQ4 Sensor (Methane)
// float readMQ4Sensor() {
//     MQ4.update();
//     return MQ4.readSensor(); // Change this if necessary
// }

// Function to read Dallas Temperature Sensor
float readDallasTemperature() {
    dallasSensors.requestTemperatures(); // Request temperature from all devices
    return dallasSensors.getTempCByIndex(0); // Assuming only one device
}

// Function to read Soil Moisture Sensor
// Function to read Soil Moisture Sensor
int readSoilMoistureSensor(int pin) {
    int sensorValue = analogRead(pin);

    // Debug print to check raw sensor value
    //Serial.print("Raw Soil Moisture Sensor Value: ");
    //Serial.println(sensorValue);

    // Constrain sensor value within expected limits
    sensorValue = constrain(sensorValue, WATER_VALUE, AIR_VALUE);

    // Map the constrained value to percentage
    int moisturePercentage = map(sensorValue, WATER_VALUE, AIR_VALUE, 100, 0);

    return moisturePercentage;

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
    lat = gps.location.lat();
    lon = gps.location.lng();
   }
}