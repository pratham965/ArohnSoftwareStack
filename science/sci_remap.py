#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy

# msg = Joy()
previous_msg = Joy()

def joyCallback(msg):
    previous_msg.buttons = msg.buttons
    if(previous_msg.buttons[12]):
        rospy.loginfo("Stepper +120 rotated (ooooo)")
    elif(previous_msg.buttons[11]):
        rospy.loginfo("Stepper -120 rotated (aaaaa)")
    elif(previous_msg.buttons[6]):
        rospy.loginfo("Stepper +10 rotated (oo)")
    elif(previous_msg.buttons[5]):
        rospy.loginfo("Stepper -10 rotated (aa)")
    elif(previous_msg.buttons[13]):
        rospy.loginfo("Latch OPEN (patt)")
    elif(previous_msg.buttons[7]):
        rospy.loginfo("LED OPEN (jhilmil)")
    elif(previous_msg.buttons[8]):
        rospy.loginfo("LED CLOSE (un-jhilmil)")

rospy.init_node("remap_joy",anonymous=True)
joy_sub = rospy.Subscriber('j1',Joy, joyCallback)
pub = rospy.Publisher('j2', Joy, queue_size=10)
def publisher():
    rate = rospy.Rate(20) 
    rospy.loginfo("SCI REMAP STARTED")

    while not rospy.is_shutdown():
        
        if len(previous_msg.buttons) and any(previous_msg.buttons):
            
            pub.publish(previous_msg)

        rate.sleep()


publisher()
