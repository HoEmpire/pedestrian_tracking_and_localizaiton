import os
import sys

sys.path.append('.')
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from cv_bridge import CvBridge
from modeling import baseline
from PIL import Image
from ptl_msgs.msg import DeadTracker
from std_msgs.msg import Int16
from torchvision import transforms
import rospy
from cv2 import cv2 as cv2
from collections import Counter

SIM_TEST_THRESHOLD = 50.0
SAME_OBJECT_THRESHOLD = 600.0
BLUR_DETECTION_THRESHOLD = 160.0
BATCH_RATIO = 0.5


def cal_dis(qf, gf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    return distmat


def build_model():
    num_classes = 1
    last_stride = 1
    pretrain_path = "/home/tim/market_resnet50_model_120_rank1_945.pth"
    # pretrain_path = '/home/tim/duke_resnet50_model_120_rank1_864.pth'
    model_neck = 'bnneck'
    neck_feat = "after"
    model_name = "resnet50"
    pretrain_choice = "self"
    model = baseline.Baseline(num_classes, last_stride, pretrain_path,
                              model_neck, neck_feat, model_name,
                              pretrain_choice)
    model.load_param(pretrain_path)
    model.eval()
    model.cuda()
    return model


def blur_detection(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=1)
    score = res.var()
    if score < BLUR_DETECTION_THRESHOLD:
        rospy.loginfo(
            "Image Blur Detected, discard this image for database maintaince!")
        return True
    else:
        rospy.loginfo("blur score: %f", score)
        return False


class Object():
    def __init__(self, id, feat):
        self.id = id
        self.feats = feat

    def add_feat(self, feat):
        feat.feats = torch.cat((self.feats, feat), 0)

    def is_full(self):
        if self.feats.shape[0] > 20:
            return True
        else:
            return False


class ReIDDatabase():
    def __init__(self):
        self.object_list = []
        self.feat_id_list = []
        self.feat_all = torch.tensor([]).cuda()
        self.object_num = 0

    def init_database(self, feat):
        self.add_new_object(feat.unsqueeze(0))

    def add_new_object(self, feat):
        new_object = Object(self.object_num, feat)
        self.object_list.append(new_object)
        self.feat_id_list.append(new_object.id)
        if self.feat_all.shape[0] == 0:
            self.feat_all = feat
        else:
            self.feat_all = torch.cat((self.feat_all, feat), 0)
        rospy.loginfo("adding new object with id:%d", self.object_num)
        self.object_num = self.object_num + 1
        return self.object_num - 1

    def add_new_feat(self, feats, id):
        for f in feats:
            distmat = cal_dis(f.unsqueeze(0), self.object_list[id].feats)
            rank = np.argsort(distmat, axis=1).squeeze(0)
            # rospy.loginfo("In add new features of id: %d", id)
            # rospy.loginfo(distmat[0][rank])
            if (SIM_TEST_THRESHOLD < distmat[0][rank[0]] <
                    SAME_OBJECT_THRESHOLD) & (
                        not self.object_list[id].is_full()):
                rospy.loginfo("adding new feature to id:%d", id)
                # update global database
                self.feat_all = torch.cat((self.feat_all, f.unsqueeze(0)), 0)
                self.feat_id_list.append(id)

                #update local database
                self.object_list[id].feats = torch.cat(
                    (self.object_list[id].feats, f.unsqueeze(0)), 0)


'''
Database management strategies:
    before query: 
        - do nothing when the image is blur
        - the image with id != -1
            - remove the one with the height/width ratio is not good 
            - remove the one that database has been already full
        - the image with id = -1
            - do nothing
    
    after query:
        - the image with id != -1
            - remove the one too similar to the database
        - the image with id = 1
            - new image:
                - do nothing
            - old image:
                - remove the one with the height/width ratio is not good 
                - remove the one that database has been already full
                - remove the one too similar to the database

# Publisher
# need to publish the object with id=-1 back with new assigned id
'''


class ReIDNode():
    def __init__(self):
        self.model = build_model()
        self.database = ReIDDatabase()
        rospy.init_node('ReID', anonymous=True)
        rospy.Subscriber('/ptl_tracker/tracker_to_reid',
                         DeadTracker,
                         self.tracker_loginfo_callback,
                         None,
                         queue_size=10)
        rospy.loginfo("Load ReID net successfully!")
        rospy.spin()

    def tracker_loginfo_callback(self, data):
        rospy.loginfo("**********into call back***********")
        bridge = CvBridge()
        query_img_list = []
        for img in data.img_blocks:
            img_block = bridge.imgmsg_to_cv2(img, "rgb8")
            img_block = cv2.resize(img_block, (128, 256),
                                   interpolation=cv2.INTER_CUBIC)
            img_block = img_block.transpose(2, 0, 1)

            # subtracting 0.485, 0.456, 0.406 and dividing by 0.229, 0.224, 0.225 to normailize the data
            img_block = img_block / 255.0
            img_block[0] = (img_block[0] - 0.485) / 0.229
            img_block[1] = (img_block[1] - 0.456) / 0.224
            img_block[2] = (img_block[2] - 0.406) / 0.225

            query_img_list.append(img_block)

            # process img for query
            if len(query_img_list) != 0:
                feats_query = self.cal_feat(query_img_list)
                self.query(feats_query)

            # summary
            rospy.loginfo("********summary*********")
            rospy.loginfo("Database object num: %d", self.database.object_num)
            for (i, ob) in enumerate(self.database.object_list):
                rospy.loginfo("object id: %d | feat num: %d", i,
                              ob.feats.shape[0])
            rospy.loginfo("***********************")
            rospy.loginfo(" ")

    def cal_feat(self, img_block):
        initial_flag = False
        for ib in img_block:
            if initial_flag == False:
                input_tensor = torch.from_numpy(ib).unsqueeze(0).float()
                initial_flag = True
            else:
                input_tensor = torch.cat(
                    (input_tensor, torch.from_numpy(ib).unsqueeze(0).float()),
                    0)
        with torch.no_grad():
            input_tensor_cuda = input_tensor.cuda()
            feats = self.model(input_tensor_cuda)
        return feats

    def query(self, feats):
        if self.database.object_num == 0:
            id = 0
            self.database.init_database(feats[0])
            self.database.add_new_feat(feats[1:], id)
        else:
            distmat = cal_dis(feats, self.database.feat_all)
            rank = np.argsort(distmat, axis=1)
            ids = []
            for (r, dis) in zip(rank, distmat):
                # rospy.loginfo("In query:")
                # rospy.loginfo(dis[r])
                if dis[r[0]] < SAME_OBJECT_THRESHOLD:
                    ids.append(self.database.feat_id_list[r[0]])
                else:
                    ids.append(-1)

            # get the id of this batch
            id_most_common = Counter(ids).most_common()[0][0]
            id_most_common_frequency = Counter(ids).most_common()[0][1]
            # old object
            if (id_most_common_frequency >
                    len(ids) * BATCH_RATIO) & (id_most_common != -1):
                id = id_most_common
                self.database.add_new_feat(feats, id)
            # new object
            else:
                id = self.database.add_new_object(feats[0:1])
                self.database.add_new_feat(feats[1:], id)
            rospy.loginfo("Query result id: %d", id)
        return id


if __name__ == '__main__':
    ReIDNode()
