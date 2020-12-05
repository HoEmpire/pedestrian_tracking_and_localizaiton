import os
import sys

sys.path.append('.')
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import cfg
from cv_bridge import CvBridge
from data import make_data_loader
from engine.inference import inference
from modeling import baseline
from PIL import Image
from ptl_msgs.msg import ImageBlock
from torchvision import transforms
from utils import reid_metric


def build_model():
    num_classes = 1
    last_stride = 1
    pretrain_path = "/home/tim/market_resnet50_model_120_rank1_945.pth"
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

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=1)
    score = res.var()
    if score < 100.0:
        rospy.info(
            "Image Blur Detected, discard this image for database maintaince!")
        return True
    else:
        return False


class Object():
    def __init__(self):
        self.id = id
        self.feats = feat


    def add_feat(self, feat):
        feat.feats = torch.cat((self.feats, feat), 0)


class ReIDDatabase():
    def __init__(self):
        self.object_list = []
        self.feat_id_list = []
        self.feat_all = torch.tensor([])
        self.object_num = 0

    def init_database(self, feats):
        for f in feats:
            new_object = Object(self.object_num, f)
            self.object_list.append(new_object)
            self.feat_id_list.append(new_object.id)
            if self.feat_all.shape[0] == 0:
                self.feat_all = f
            else:
                self.feat_all = torch.cat((self.feat_all, f), 0)
            self.object_num = self.object_num++
            

    def query_feat(self, feats):
        if self.id_max == 0:
            self.init_database(feats)
        else:
            distmat = reid_metric.re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.5)
            print(distmat.shape)
            rank = np.argsort(distmat, axis=1).squeeze()

            for i in range(10):
                print(gallery_list[rank[i]])
                print(distmat.squeeze(0)[rank[i]])
                plt.figure(0)
                plt.imshow(image[rank[i]].numpy().transpose(1, 2, 0))
                plt.show()
             


class ReIDNode():
    def __init__(self):
        self.model = build_model()
        rospy.info("Load ReID net successfully!")

    def tracker_info_callback(self, data):
        bridge = CvBridge()
        cvimg = bridge.imgmsg_to_cv2(data.img, "bgr8")
    
    def re_identification(self, data)


def main():
    model = build_model()

    gallery_root = "/home/tim/ptl_project/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/gallery2"
    query_root = "/home/tim/ptl_project/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/query"
    gallery_list = os.listdir(gallery_root)
    print(gallery_list)
    query_list = os.listdir(query_root)
    for i, g in enumerate(gallery_list):
        gallery_list[i] = os.path.join(gallery_root, g)

    for i, q in enumerate(query_list):
        query_list[i] = os.path.join(query_root, q)

    path_all = gallery_list + query_list
    # print(path_all)
    image = []
    for p in path_all:
        image.append(
            transforms.Resize(
                (256, 128))(transforms.ToTensor()(Image.open(p))))
    feat = []
    model.eval()
    start = time.time()
    for img in image:
        with torch.no_grad():
            image_tmp = img.unsqueeze(0).cuda()
            feat.append(model(image_tmp))
    gf = feat[0]
    for f in feat[1:-1]:
        gf = torch.cat((gf, f), 0)
    qf = feat[-1]
    print("it takes {} seconds".format(
        (time.time() - start) / image.__len__()))
    print(gf.shape)
    print(qf.shape)
    print(gf)
    print(qf)
    # m, n = qf.shape[0], gf.shape[0]
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    print("Enter reranking")
    distmat = reid_metric.re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.5)
    print(distmat.shape)
    rank = np.argsort(distmat, axis=1).squeeze()

    for i in range(10):
        print(gallery_list[rank[i]])
        print(distmat.squeeze(0)[rank[i]])
        plt.figure(0)
        plt.imshow(image[rank[i]].numpy().transpose(1, 2, 0))
        plt.show()


if __name__ == '__main__':
    main()
