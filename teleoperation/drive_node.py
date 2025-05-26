#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
import ctrl

PORT = ctrl.SerialPortChecker(115200, 2).find_port("drive")

driveObj = ctrl.drive(PORT, 115200)
driveObj.connect()

def callback(data):
    status = driveObj.setState(-round(data.axes[0]*255), -round(data.axes[1]*255))
    driveObj.serialWrite()
    rospy.loginfo("%s", status)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('j0', Joy, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
