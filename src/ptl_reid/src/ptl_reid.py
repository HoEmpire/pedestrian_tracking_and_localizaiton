#!/usr/bin/env python
import time
import numpy as np
import torch
from cv_bridge import CvBridge

from ptl_msgs.msg import DeadTracker
from ptl_msgs.msg import ReidInfo
from ptl_msgs.msg import ImageBlock
from ptl_msgs.msg import FrontEndInterface
from std_msgs.msg import Int16
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import rospy
from cv2 import cv2 as cv2
from collections import Counter
import time
import reid_database
from inference import cal_dis
import model
import utils


def init_marker():
    marker = Marker()
    marker.header.frame_id = "map"
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.5
    marker.color.r = 1.0
    marker.color.a = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    return marker


class ReIDNode():
    def __init__(self):
        start = time.time()
        self.model = model.build_model()
        self.database = reid_database.ReIDDatabase()
        self.markers = MarkerArray()
        self.id_marker = init_marker()
        self.id_marker.ns = "id"
        self.id_marker.type = Marker.TEXT_VIEW_FACING
        self.pos_marker = init_marker()
        self.pos_marker.ns = "pos"
        self.pos_marker.type = Marker.CUBE
        self.markers.markers.append(self.id_marker)
        self.markers.markers.append(self.pos_marker)
        self.front_end_interface_info = FrontEndInterface()
        rospy.init_node('ptl_reid', anonymous=True)
        rospy.Subscriber('/ptl_tracker/tracker_to_reid',
                         DeadTracker,
                         self.tracker_loginfo_callback,
                         None,
                         queue_size=10)
        rospy.Subscriber('/ptl_detector/detector_to_reid',
                         ImageBlock,
                         self.detector_callback,
                         None,
                         queue_size=10)
        self.tracker_pub = rospy.Publisher('/ptl_reid/reid_to_tracker',
                                           ReidInfo,
                                           queue_size=1)
        self.reid_vis_pub = rospy.Publisher('/ptl_reid/reid_result',
                                            Image,
                                            queue_size=1)
        self.position_vis_pub = rospy.Publisher('/ptl_reid/position_vis',
                                                MarkerArray,
                                                queue_size=1)
        self.detector_reid_to_tracker_pub = rospy.Publisher(
            '/ptl_reid/detector_to_reid_to_tracker', ImageBlock, queue_size=1)
        self.front_end_interface_pub = rospy.Publisher(
            '/ptl_reid/front_end_interface_pub',
            FrontEndInterface,
            queue_size=1)

        rospy.loginfo("Load ReID net successfully!")
        rospy.loginfo("Init takes %f seconds", time.time() - start)
        rospy.spin()

    def tracker_loginfo_callback(self, data):
        rospy.loginfo("**********into call back***********")
        bridge = CvBridge()
        query_img_list = []
        example_block_orig = bridge.imgmsg_to_cv2(data.img_blocks[0], "rgb8")
        # example_block = utils.image_block_preprocess(example_block)
        example_block = cv2.resize(example_block_orig, (128, 256),
                                   interpolation=cv2.INTER_CUBIC)
        for img in data.img_blocks:
            img_block = bridge.imgmsg_to_cv2(img, "rgb8")
            img_block = utils.image_block_preprocess(img_block)
            img_block = cv2.resize(img_block, (128, 256),
                                   interpolation=cv2.INTER_CUBIC)
            img_block = img_block.transpose(2, 0, 1)

            # subtracting 0.485, 0.456, 0.406 and dividing by 0.229, 0.224, 0.225 to normailize the data
            img_block = img_block / 255.0
            img_block[0] = (img_block[0] - 0.485) / 0.229
            img_block[1] = (img_block[1] - 0.456) / 0.224
            img_block[2] = (img_block[2] - 0.406) / 0.225

            query_img_list.append(img_block)

        pub_msg = ReidInfo()
        # process img for query
        if len(query_img_list) != 0:
            start = time.time()
            feats_query = self.cal_feat(query_img_list)
            rospy.loginfo("Calculating features takes %f seconds",
                          time.time() - start)
            start = time.time()
            is_query_new, pub_msg.last_query_id = self.query(feats_query)
            pub_msg.total_num = self.database.object_num
            position = data.position

            if is_query_new:
                self.id_marker.id = pub_msg.last_query_id
                self.id_marker.text = str(pub_msg.last_query_id)
                self.id_marker.pose.position.x = position.x + 0.2
                self.id_marker.pose.position.y = position.y + 0.2
                self.id_marker.pose.position.z = position.z + 0.2

                self.pos_marker.id = pub_msg.last_query_id
                self.pos_marker.text = str(pub_msg.last_query_id)
                self.pos_marker.pose.position = position

                self.database.object_list[-1].img = example_block

                # update interface data
                self.front_end_interface_info.id.append(pub_msg.last_query_id)
                example_block_msg = bridge.cv2_to_imgmsg(example_block, "rgb8")
                self.front_end_interface_info.img_blocks.append(
                    example_block_msg)
                self.front_end_interface_info.position.append(position)

                vis = np.concatenate((example_block, example_block), axis=1)
                # cv2.putText(img,'Hello World!',
                #     bottomLeftCornerOfText,
                #     font,
                #     fontScale,
                #     fontColor,
                #     lineType)
                vis = cv2.resize(vis, (512, 512),
                                 interpolation=cv2.INTER_CUBIC)
                cv2.putText(vis, 'Add new one!', (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 255), 3)
            else:
                self.id_marker.id = pub_msg.last_query_id
                self.id_marker.text = str(pub_msg.last_query_id)
                self.id_marker.pose.position.x = position.x + 0.2
                self.id_marker.pose.position.y = position.y + 0.2
                self.id_marker.pose.position.z = position.z + 0.2

                self.pos_marker.id = pub_msg.last_query_id
                self.pos_marker.text = str(pub_msg.last_query_id)
                self.pos_marker.pose.position = position

                # update interface data
                self.front_end_interface_info.position[
                    pub_msg.last_query_id].position

                vis = np.concatenate(
                    (example_block,
                     self.database.object_list[pub_msg.last_query_id].img),
                    axis=1)
                vis = cv2.resize(vis, (512, 512),
                                 interpolation=cv2.INTER_CUBIC)
                cv2.putText(vis, 'Query:', (10, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1.5, (255, 0, 0), 3)
                cv2.putText(vis, 'Gallery:', (266, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 3)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            image_message = bridge.cv2_to_imgmsg(vis, encoding="passthrough")
            self.reid_vis_pub.publish(image_message)
            self.position_vis_pub.publish(self.markers)
            rospy.loginfo("Query takes %f seconds", time.time() - start)

        # summary
        rospy.loginfo("********summary*********")
        rospy.loginfo("Database object num: %d", self.database.object_num)
        for (i, ob) in enumerate(self.database.object_list):
            rospy.loginfo("object id: %d | feat num: %d", i, ob.feats.shape[0])
        rospy.loginfo("***********************")
        rospy.loginfo(" ")

        self.tracker_pub.publish(pub_msg)
        self.front_end_interface_pub.publish(self.front_end_interface_info)

    def cal_feat(self, img_block):
        # rospy.loginfo(type(img_block))
        # rospy.loginfo(len(img_block))
        for (i, ib) in enumerate(img_block):
            if ('input_tensor' not in dir()):
                input_tensor = torch.from_numpy(ib).unsqueeze(0).float()
            else:
                input_tensor = torch.cat(
                    (input_tensor, torch.from_numpy(ib).unsqueeze(0).float()),
                    0)
            if ((i + 1) % self.database.cfg.query_batch_size
                    == 0) | (i + 1 == len(img_block)):
                with torch.no_grad():
                    input_tensor_cuda = input_tensor.cuda()
                    if ('feats' not in dir()):
                        feats = self.model(input_tensor_cuda).cpu()
                    else:
                        feats = torch.cat(
                            (feats, self.model(input_tensor_cuda).cpu()), 0)
                del input_tensor
        return feats

    def query(self, feats):
        is_query_new = True
        if self.database.object_num == 0:
            id = 0
            self.database.init_database(feats[0])
            self.database.add_new_feat(feats[1:], id)
        else:
            distmat = cal_dis(feats, self.database.feat_all)
            rank = np.argsort(distmat, axis=1)
            ids = []
            for (r, dis) in zip(rank, distmat):
                rospy.loginfo("In query:")
                rospy.loginfo(dis[r[0:5]])
                if dis[r[0]] < self.database.cfg.same_id_threshold:
                    ids.append(self.database.feat_id_list[r[0]])
                else:
                    ids.append(-1)

            # get the id of this batch
            id_most_common = Counter(ids).most_common()[0][0]
            id_most_common_frequency = Counter(ids).most_common()[0][1]

            rospy.loginfo("Most common id: %d, ratio %d/%d=%f", id_most_common,
                          id_most_common_frequency, len(ids),
                          id_most_common_frequency * 1.0 / len(ids))
            # old object
            if (id_most_common_frequency >= len(ids) *
                    self.database.cfg.batch_ratio) & (id_most_common != -1):
                id = id_most_common
                is_query_new = False
                self.database.add_new_feat(feats, id)
            # new object
            else:
                id = self.database.add_new_object(feats[0:1])
                self.database.add_new_feat(feats[1:], id)
            rospy.loginfo("Query result id: %d", id)
        return is_query_new, id

    def detector_callback(self, data):
        # rospy.loginfo("**********into detector call back***********")
        start = time.time()
        bridge = CvBridge()
        query_img_list = []
        cvimg = bridge.imgmsg_to_cv2(data.img, "rgb8")
        for bbox in data.bboxes:
            img_block = cvimg[bbox.data[1]:bbox.data[1] + bbox.data[3],
                              bbox.data[0]:bbox.data[0] + bbox.data[2]]
            img_block = cv2.resize(img_block, (128, 256),
                                   interpolation=cv2.INTER_CUBIC)
            img_block = img_block.transpose(2, 0, 1)

            # subtracting 0.485, 0.456, 0.406 and dividing by 0.229, 0.224, 0.225 to normailize the data
            img_block = img_block / 255.0
            img_block[0] = (img_block[0] - 0.485) / 0.229
            img_block[1] = (img_block[1] - 0.456) / 0.224
            img_block[2] = (img_block[2] - 0.406) / 0.225

            query_img_list.append(img_block)
        rospy.loginfo("Data preprocessing takes %f seconds",
                      time.time() - start)
        start = time.time()
        feats = self.cal_feat(query_img_list)
        rospy.loginfo("Query takes %f seconds", time.time() - start)
        return_msgs = data
        for f in feats:
            feat = Float32MultiArray()
            for i in f:
                feat.data.append(float(i))
            return_msgs.features.append(feat)
        start = time.time()
        rospy.loginfo("Others takes %f seconds", time.time() - start)
        self.detector_reid_to_tracker_pub.publish(return_msgs)
        # rospy.loginfo(feats.shape)
        # rospy.loginfo("**********out of detector call back***********")


if __name__ == '__main__':
    ReIDNode()
