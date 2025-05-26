#include <Adafruit_AHT10.h>
#include <Adafruit_AHTX0.h>
#include <Wire.h>
#include <SPI.h>
#include <MQUnifiedsensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>


Adafruit_AHTX0 aht;


// Define constants
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
MQUnifiedsensor MQ4(placa, Voltage_Resolution, ADC_Bit_Resolution, MQ4_PIN, MQ4_TYPE);


Adafruit_Sensor *aht_humidity, *aht_temp;

// Dallas temperature sensor
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature dallasSensors(&oneWire);

// Variables for pH sensor
int buf[10];
int avgValue = 0;

const int laserPin = 11;

void setup() {
    Serial.begin(9600);

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
    MQ4.setRegressionMethod(1); //_PPM = a * ratio ^ b
    MQ4.setA(1012.7); 
    MQ4.setB(-2.786); 
    MQ4.init();
    calibrateSensor(MQ4, RatioMQ4CleanAir);
    delay(2000);
    // Initialize AHT10 sensor (Temperature and Humidity)
    if (!aht.begin()) {
        Serial.println("Failed to find AHT10 chip");
        while (1) delay(10);
    }
    aht_temp = aht.getTemperatureSensor();
    aht_humidity = aht.getHumiditySensor();

    // Initialize Dallas temperature sensor
    dallasSensors.begin();

    pinMode(13, OUTPUT);
    
    // Initialize Soil Moisture Sensor
    pinMode(SOIL_MOISTURE_PIN, INPUT);
    pinMode(laserPin, OUTPUT);
  
  // Set the laser to OFF initially
  analogWrite(laserPin, 0);  // 0 means laser off
}

void loop() {
    // Read sensor values
    sensors_event_t humidity, temp;
    readTemperatureHumidity(humidity, temp);
    float uvVoltage = readUVSensor(UV_PIN);
    int mq7Value = readMQ7(MQ7_PIN);
    float phValue = readPHSensor(SensorPinPh);
    float ozoneValue = readOzoneSensor();
    float nh4Value = readMQ135Sensor();
    float methaneValue = readMQ4Sensor();
    float dallasTempC = readDallasTemperature(); // Reading Dallas Temperature sensor
    int soilMoistureValue = readSoilMoistureSensor(SOIL_MOISTURE_PIN); // Reading Soil Moisture

   /* // Display values
    Serial.print("\t\tTemperature: ");
    Serial.print(temp.temperature);
    Serial.println(" deg C");

    Serial.print("\t\tHumidity: ");
    Serial.print(humidity.relative_humidity);
    Serial.println(" % rH");

    Serial.println("\t\tUV Sensor Output:");
    Serial.print("\t\tsensor voltage = ");
    Serial.println(uvVoltage);

    Serial.print("\t\tMQ7 Value: ");
    Serial.println(mq7Value);

    Serial.print("\t\tpH: ");
    Serial.println(phValue, 2);

    Serial.println("\t\tOzone Sensor:");
    MQ131.serialDebug();  // Ozone sensor debug info

    Serial.println("\t\tNH4 Sensor:");
    MQ135.serialDebug();  // NH4 sensor debug info

    Serial.print("Methane (CH4) Sensor: ");
    MQ4.serialDebug();  // Methane sensor debug info

    Serial.print("Dallas Temperature (Celsius): ");
    Serial.println(dallasTempC);

    // Display Soil Moisture
    Serial.print("\t\tSoil Moisture: ");
    Serial.println(soilMoistureValue);
    */
    // Data logging output
    Serial.print("Science:,");
    Serial.print(temp.temperature); Serial.print(",");
    Serial.print(humidity.relative_humidity); Serial.print(",");
    Serial.print(uvVoltage); Serial.print(",");
    Serial.print(mq7Value); Serial.print(",");
    Serial.print(phValue); Serial.print(",");
    Serial.print(ozoneValue); Serial.print(",");
    Serial.print(methaneValue); Serial.print(",");
    Serial.print(dallasTempC); Serial.print(",");
    Serial.println(soilMoistureValue);
    

   // delay(1000);
   // analogWrite(laserPin, 0);
    // Laser ON for 2 seconds
  
  // Turn off the laser (0 means off)
    // Laser OFF for 2 seconds
}

// Function to calibrate MQ sensors
void calibrateSensor(MQUnifiedsensor &sensor, float cleanAirRatio) {
    Serial.print("Calibrating sensor, please wait.");
    float calcR0 = 0;
    for (int i = 1; i <= 10; i++) {
        sensor.update();
        calcR0 += sensor.calibrate(cleanAirRatio);
        Serial.print(".");
        delay(500);
    }
    sensor.setR0(calcR0 / 10);
    Serial.println(" done!");

    if (isinf(calcR0)) {
        Serial.println("Warning: Connection issue (Open circuit). Please check wiring.");
        while (1);
    } else if (calcR0 == 0) {
        Serial.println("Warning: Connection issue (Short circuit). Please check wiring.");
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

    // Find min, max, and sum
    for (int i = 0; i < 10; i++) {
        if (buf[i] < minValue) minValue = buf[i];
        if (buf[i] > maxValue) maxValue = buf[i];
        sum += buf[i];
    }

    sum -= (minValue + maxValue); // Exclude min and max
    float voltage = (float)sum / 8 * 5.0 / 1024;  // Convert to voltage

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
float readMQ4Sensor() {
    MQ4.update();
    return MQ4.readSensor(); // Change this if necessary
}

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