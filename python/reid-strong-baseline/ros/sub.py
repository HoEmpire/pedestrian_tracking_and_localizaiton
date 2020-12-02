#!/usr/bin/env python

import rospy
import cv2
# from std_msgs.msg import String
from pedestrain_tracking_and_localizaiton.msg import ImageBlock
from cv_bridge import CvBridge


class listener():
    id = 0

    def callback(self, data):
        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

        bridge = CvBridge()
        cvimg = bridge.imgmsg_to_cv2(data.img, "bgr8")
        # rospy.loginfo(cvimg.rol)
        # rospy.loginfo(type(data.img))
        # rospy.loginfo(type(cvimg))

        for b in data.bboxs:
            rospy.loginfo("bbox ratio = %f", 1.0 * b.data[3] / b.data[2])
            if 1 <= 1.0 * b.data[3] / b.data[2] <= 3:
                rospy.loginfo("saving imgs")
                path = "/home/tim/test/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/rgbd"
                image_path = path + "/" + str(self.id) + ".jpg"
                self.id = self.id + 1
                cv2.imwrite(
                    image_path, cvimg[b.data[1]:b.data[1] + b.data[3],
                                      b.data[0]:b.data[0] + b.data[2]])
            # rospy.loginfo(b.data)
            # cv2.imshow(
            #     'image', cvimg[b.data[1]:b.data[1] + b.data[3],
            #                    b.data[0]:b.data[0] + b.data[2]])
            # rospy.loginfo("y from %d to %d", b.data[1], b.data[1] + b.data[3])
            # rospy.loginfo("x from %d to %d", b.data[0], b.data[0] + b.data[2])
            # cv2.waitKey(0)

    def main(self):

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)

        # rospy.Subscriber('chatter', String, callback)
        rospy.Subscriber('/detector_test/image_blocks',
                         ImageBlock,
                         self.callback,
                         None,
                         queue_size=100)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    main = listener()
    main.main()
