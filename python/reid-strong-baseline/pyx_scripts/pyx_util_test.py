import os
from cv2 import cv2 as cv2
import numpy as np
import time

gallery_root = "/home/tim/ptl_project/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/test"
gallery_list = os.listdir(gallery_root)
for i, g in enumerate(gallery_list):
    gallery_list[i] = os.path.join(gallery_root, g)

id = 0
gallery_new_root = "/home/tim/ptl_project/src/pedestrain_tracking_and_localizaiton/python/reid-strong-baseline/data/test_data/gallery3/"
for img_path in gallery_list:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    start = time.time()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)
    # img_gray.resize(256, 256)
    # print(img_gray.shape)
    res = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=1)
    score = res.var()

    # if score > 50:
    print("***************************")
    print(img_path)
    print('score:', score)
    print('process time:', time.time() - start)
    # cv2.imwrite(img_path.replace("gallery", "gallery3"), img)
    id = id + 1
