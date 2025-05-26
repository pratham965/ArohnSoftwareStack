#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
import ctrl

SERIAL_PORT = ctrl.SerialPortChecker(115200, 2).find_port("arm")
armObj = ctrl.arm(SERIAL_PORT, 115200)
armObj.connect()

def callback(data):
    status = armObj.setState(round(data.axes[0]*255), round(data.axes[1]*255), round(data.axes[2]*255), data.buttons)
    armObj.serialWrite()
    rospy.loginfo("%s", status)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('j1', Joy, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
