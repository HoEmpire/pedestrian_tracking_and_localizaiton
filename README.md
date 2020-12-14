# pedestrian tracking and localizaiton

<div align=center><img width="480" height="360" src="img/demo.gif"/></div>

## Content

- [Introduction](#Introduction)
- [Dependence](#Dependence)
- [Usage](#Usage)
- [Reference](#Reference)

## Introduction

A package for pedestrian detection, tracking and re-identification. This package is designed for the job of counting the total numbers of pedestrians in an area, and showing their locations on a map.

Just imagining such a condition, you have a wheeled robot equipping with a lidar and a camera (or an RGBD camera). You ask it to go into a building and explore the building thoroughly. When the robot is exploring the building, it will detect and track the pedestrian in its sight, and report the locations of the pedestrians on a map. And after it finishes the exploration, it will tell you how many people are in the building, and the location where the robot see them most recently. This is what this project aims for.

## Prerequisite

- Ubuntu 18.04
- ROS Melodic
- OpenCV (two versions of OpenCV are needed at the same time)
  - version > 4.1.0 (ptl_detector)
  - version = 3.2.0 (ptl_tracker)
- CUDA > 10.0
- cuDNN > 7.6.5
- TensorRT > 7.0.0
- GLOG
- [OpenTracker](https://github.com/rockkingjy/OpenTracker)
- pytorch >= 0.4
- torchvision
- ignite = 0.1.2
- yacs

## Usage

### 1. Build

1. Install the prerequisites

   - Install the other required packages
   - Install the [OpenTracker](https://github.com/rockkingjy/OpenTracker) according to its guildline

1. Build a workspace

   ```shell
   mkdir -p ptl_ws/src
   ```

1. Clone this repository in `/ws/src`

   ```shell
   cd ptl_ws/src
   git clone https://github.com/HoEmpire/pedestrian_tracking_and_localizaiton.git
   ```

1. Build the files

   ```shell
   catkin_make
   ```

### 2. Config

The config files of each package can be found in `${PROJECT_NAME}/config/config.yaml`

- ptl_detector

  ```yaml
  data_topic:
    camera_topic: /camera/color/image_raw/compressed

  cam:
    cam_net_type: "YOLOV4_TINY" #net type
    cam_file_model_cfg: "/asset/yolov4-tiny.cfg" #config file path
    cam_file_model_weights: "/asset/yolov4-tiny.weights" #weight file path
    cam_inference_precison: "FP32" #float precision
    cam_n_max_batch: 1 #number of batch
    cam_prob_threshold: 0.5 # the threshold of the detection probability
    cam_min_width: 0 # the min/max width/height of an object in detection
    cam_max_width: 1440
    cam_min_height: 0
    cam_max_height: 1080
  ```

- ptl_tracker

  ```yaml
  data_topic:
    lidar_topic: /livox/lidar
    camera_topic: /camera/color/image_raw/compressed
    depth_topic: /livox/lidar

  tracker:
    track_fail_timeout_tick: 3
    bbox_overlap_ratio: 0.5

  local_database:
    height_width_ratio_min: 1.0
    height_width_ratio_max: 3.0
    blur_detection_threshold: 200.0
    record_interval: 0.2
    batch_num_min: 3
  ```

  - **track_fail_timeout_tick**: if the tracker fails for `track_fail_timeout_tick` frames, we consider this tracker fails and remove it from the list
  - **bbox_overlap_ratio**: if the overlapping area ratio of the bounding box from the detector and the tracker is higher than this value, we match these two bounding boxes.

    - Overlaping area ratio is calculated by

      $$ratio = min(\frac{Area_{overlap}}{Area_{detector}},\frac{Area_{overlap}}{Area_{detector}})$$

  - **height_width_ratio_min/max**: only the image block with height/width fails in the range of (height_width_ratio_min, height_width_ratio_max) will be added into the local database
  - **blur_detection_threshold**: if the image blur detection score is lower than this value, we will discard this image
  - **record_interval**: the unit is second. It defines the minimum time interval between two images in a local database
  - **batch_num_min**: if the number of images in the local database is smaller than this value, we will not perform re-identification of this object (remove false-positive in detection)

- ptl_reid

  ```yaml
  similarity_test_threshold: 100.0
  same_id_threshold: 550.0
  batch_ratio: 0.5
  object_img_num: 30
  weights_path: "/home/tim/market_resnet50_model_120_rank1_945.pth"
  ```

  - **similarity_test_threshold**: when updating the database, if the minimal distance score between the query image and gallery images is smaller than this value, we will not add this query image to the database (to maintain adequate differnce in the database of an object)
  - **same_id_threshold**: when querying the database, if the minimal distance score between the query image and gallery images is bigger than this value, we will consider this query image comes from a new object; Or else we consider it belongs to one of the objects in the database
  - **batch_ratio**: in an image batch from an object, if the number of images assigned with the same id after re-identification, is bigger than the batch_ratio times the total number of images in this batch, we consider this object is the object already in the database. Otherwise, we will take it as a new object.
  - **object_img_num**: the maximal number of images stored in the database for one object
  - **weights_path**: the path of the weights of the re-identification net

### 3. Run

```shell
cd ptl_ws
source devel/setup.zsh
sudo chmod +x src/pedestrain_tracking_and_localizaiton/src/ptl_reid/src/ptl_reid.py
roslaunch ptl_tracker tracker
roslaunch ptl_detector detector
roslaunch ptl_reid reid.py
```

- **tips**
  - Make sure the reid.py is run in the environment of python3. You can check [here](https://blog.csdn.net/weixin_42675603/article/details/107785376)(in Chinese) for some guide.

## Reference

- [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)
- [OpenTracker](https://github.com/rockkingjy/OpenTracker)
