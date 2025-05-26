#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

def cmd_vel_callback(cmd_vel_msg):
    # Create a Joy message
    joy_msg = Joy()

    # Initialize axes and buttons (ensure the list has the required size)
    joy_msg.axes = [0.0] * 2  # At least two axes for this example
    joy_msg.buttons = []  # Empty if no buttons are needed

    # Map cmd_vel values to Joy axes
    if cmd_vel_msg.angular.z > 0:
        joy_msg.axes[0] = 1.0
    elif cmd_vel_msg.angular.z < 0:
        joy_msg.axes[0] = -1.0
    elif cmd_vel_msg.linear.x > 0:
        joy_msg.axes[1] = 1.0
    elif cmd_vel_msg.linear.x < 0:
        joy_msg.axes[1] = -1.0
    else:
        joy_msg.axes[1] = 0.0
        joy_msg.axes[0] = 0.0
    # joy_msg.axes[1] = 1.0 if cmd_vel_msg.linear.x!=0 else 0.0  # Linear x -> axes[1]
    # joy_msg.axes[0] = 1.0 if cmd_vel_msg.angular.z!=0 else 0.0  # Angular z -> axes[0]

    # Publish the Joy message
    joy_pub.publish(joy_msg)

if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("cmd_vel_to_joy")

    # Publisher for the Joy message
    joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    # Subscriber to the cmd_vel topic
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)

    # Spin to keep the node running
    rospy.spin()
