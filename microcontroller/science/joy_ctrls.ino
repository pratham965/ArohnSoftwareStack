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

void messageCb(const sensor_msgs::Joy& joy_msg) {
  // Serial.println("hi");
  if (joy_msg.axes[0] != 0) {
    rotate(1, 120);
  } else if (joy_msg.axes[2] != 0) {
    rotate(0, 120);
  }

  if(joy_msg.axes[1]<0){
    digitalWrite(latch, 1);
  }
  else if(joy_msg.axes[1]>0){
    digitalWrite(latch, 0);
  }
  if(joy_msg.axes[3]<0){
    digitalWrite(light, 1);
  }
  else if(joy_msg.axes[3]>0){
    digitalWrite(light, 0);
  }
}

ros::Subscriber<sensor_msgs::Joy> sub("joy", messageCb);

void setup() {
  Serial.begin(57600);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(latch, OUTPUT);

  digitalWrite(latch, 1);
  digitalWrite(light, 1);

  nh.initNode();
  nh.subscribe(sub);
}

void loop() {
  nh.spinOnce();
  delay(1);
}
