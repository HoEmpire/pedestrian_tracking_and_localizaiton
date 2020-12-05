#!/usr/bin/env python

import rospy
from std_msgs.msg import UInt16MultiArray


def talker():
    pub = rospy.Publisher('chatter', UInt16MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        a = UInt16MultiArray()
        a.data.append(0)
        a.data.append(1)
        pub.publish(a)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
