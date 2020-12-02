# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import rospy
import argparse
import os
import sys
from os import mkdir
import matplotlib.pyplot as plt
from torchvision import transforms

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from PIL import Image
import time
import numpy as np
import pynvml


def re_ranking(probFea,
               galFea,
               k1,
               k2,
               lambda_value,
               local_distmat=None,
               only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1, -2, feat, feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 -
                                 lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument("--config_file",
                        default="",
                        help="path to config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    model = build_model(cfg, 1)
    model.load_param(cfg.TEST.WEIGHT)

    model.eval()
    model.cuda()
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo.used)

    gallery_root = "/home/tim/test/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/gallery"
    query_root = "/home/tim/test/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/query"
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
    distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.5)
    print(distmat.shape)
    rank = np.argsort(distmat, axis=1).squeeze()

    for i in range(20):
        print(gallery_list[rank[i]])
        plt.figure(0)
        plt.imshow(image[rank[i]].numpy().transpose(1, 2, 0))
        plt.show()

    # print(distmat)

    # print(type(image[0]))
    # plt.figure(0)
    # plt.imshow(image[1].numpy().transpose(1, 2, 0))
    # plt.figure(1)
    # plt.imshow(image[1].numpy())
    # plt.show()


if __name__ == '__main__':
    main()
