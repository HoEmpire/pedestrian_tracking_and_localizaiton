import os
import sys
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import baseline

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import time
import torch
from utils import reid_metric
import numpy as np


def main():
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