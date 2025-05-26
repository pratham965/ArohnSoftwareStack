#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
import ctrl

armObj = ctrl.arm("/dev/ttyUSB0", 115200)

def callback(data):
    status = armObj.setState(round(data.axes[0]*255), round(data.axes[1]*255), round(data.axes[2]*255), data.buttons)
    rospy.loginfo("%s", status)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('joy', Joy, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
