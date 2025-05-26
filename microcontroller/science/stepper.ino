#include <ros.h>
#include <sensor_msgs/Joy.h>

#define dirPin 49
#define stepPin 47
#define latch 12
#define light 13


const float pulse_count = 800.0;

ros::NodeHandle nh;

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




void setup() {
  Serial.begin(57500);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  
}

void loop() {
  rotate(0, 120);
  delay(500);
  rotate(1, 120);
  delay(500);

}
