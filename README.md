# pedestrian tracking and localizaiton

<div align=center><img width="958" height="540" src="https://github.com/HoEmpire/demo-images-and-gifs/blob/main/pedestrian_tracking_and_localization/perfect.gif"/></div>
<div align=center>visualization</div>

## Content

- [Introduction](#Introduction)
- [Dependence](#Dependence)
- [Usage](#Usage)
- [Reference](#Reference)

## Introduction

A ROS package based on C++ for pedestrian detection, tracking and re-identification (The python package ptl_reid is deprecated, for reference, it remains in this project.). This package is designed for the task of counting the total numbers of pedestrians in an area, and showing their locations on a map.

Just imagining such a condition, you have a wheeled robot equipping with a lidar and a camera (or an RGBD camera). You ask it to go into a building and explore the building thoroughly. When the robot is exploring the building, it will detect and track the pedestrian in its sight, and report the locations of the pedestrians on a map. And after it finishes the exploration, it will tell you how many people are in the building, and the location where the robot see them most recently. This is what this project aims for.

## Prerequisite

- Ubuntu 18.04
- ROS Melodic
- OpenCV > 4.1
- CUDA > 10.0
- cuDNN > 7.6.5
- TensorRT > 7.0.0
- GLOG
- [faiss](https://github.com/facebookresearch/faiss)

## Usage

### 1. Build

1. Install the prerequisites

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
  basic:
    use_compressed_image: true #using compressed image
    use_lidar: true #enable using lidar
    enable_pcp_vis: true #enable filtered pointcloud visualization
    lidar_topic: /rslidar_points
    camera_topic: /camera2/color/image_raw/compressed
    map_frame: map
    lidar_frame: rslidar
    camera_frame: camera2_link

  tracker:
    track_fail_timeout_tick: 30 #if the tracker fails for track_fail_timeout_tick frames, we consider this tracker fails and remove it from the list.
    bbox_overlap_ratio: 0.6
    detector_update_timeout_tick: 30 #if the ticks after last update by detector is too long, we consider that we lose track of this target
    detector_bbox_padding: 80 #  to ensure overlap between detector and tracker, we pad the bounding box of the detector to enlarge it
    reid_match_threshold: 3.0 #maximum feature distance to consider a match between a detected and a tracking object
    reid_match_bbox_dis: 80 #maximum bbox center distance to consider a match between a detected and a tracking object
    reid_match_bbox_size_diff: 80 #maximum bbox size distance to consider a match between a detected and a tracking object
    stop_opt_timeout: 6

  local_database:
    height_width_ratio_min: 0.85 #only the image block with height/width falls in the range of (height_width_ratio_min, height_width_ratio_max) will be added into the local database.
    height_width_ratio_max: 4.0
    record_interval: 0.1 # the minimum time interval between two recorded images in a local database (Unit: s).
    batch_num_min: 8
    feature_smooth_ratio: 0.7

  pc_processor:
    resample_size: 0.1 # point cloud resample size(Unit:m)
    x_min: 0.0 # point cloud conditional filter(Unit:m)
    x_max: 15.0
    z_min: 0.0
    z_max: 4.04
    std_dev_thres: 0.1 # statistial filter param
    mean_k: 20 # statistial filter param
    cluster_tolerance: 0.5
    cluster_size_min: 20
    cluster_size_max: 10000
    match_centroid_padding: 20 # padding of bbox for robust reprojection matching between 2d bbox and 3d centroids

  camera_intrinsic:
    fx: 613.783
    fy: 612.895
    cx: 416.969
    cy: 240.223

  kalman_filter:
    q_xy: 100 # bbox center position state variance (Unit: Pixel^2)
    q_wh: 25 # bbox size state variance (Unit: Pixel^2)
    p_xy_pos: 100 # bbox center position initial variance (Unit: Pixel^2)
    p_xy_dp: 10000 # bbox center position velocity initial variance (Unit: Pixel^2)
    p_wh_size: 25 # bbox size initial variance (Unit: (Pixel/s)^2)
    p_wh_ds: 25 # bbox size velocity initial variance (Unit: (Pixel/s)^2)
    r_theta: 0.08 # observation variance
    r_f: 0.04
    r_tx: 4
    r_ty: 4
    residual_threshold: 16 # if the residual is higher than this param, this observation will be rejected

  kalman_filter_3d:
    q_factor: 100 # position state variance (Unit (m/s^2)^2)
    r_factor: 0.25 # position observation variance (Unit m^2)
    p_pos: 100 # position initial variance (Unit m^2)
    p_vel: 4 # position initial variance (Unit (m/s)^2)
    start_predict_only_timeout: 10 #using the detector update tick as the timeout count, if this tick is higher than this param, will only use state predict result to tracker the position to avoid degeneration of false 2d tracking
    stop_track_timeout: 15 #using the detector update tick as the timeout count, if this tick is higher than this param, will stop 3d tracking to avoid drift
    outlier_threshold: 4.0 # if the residual ratio is higher than this param, this observation will be rejected

  optical_flow:
    min_keypoints_to_track: 40 # minimal keypoints to track for one object in keypoints_num_factor_area (e.g. 40 keypoints in 8000 pixel^2)
    keypoints_num_factor_area: 8000
    corner_detector_max_num: 100 # maximum keypoints to track for one object
    corner_detector_quality_level: 0.0001 # corner detetion params...
    corner_detector_min_distance: 3
    corner_detector_block_size: 3
    corner_detector_use_harris: true
    corner_detector_k: 0.03
    min_keypoints_to_cal_H_mat: 10 # minimal number of keypoints to calculate transformation matrix of an object. If the number of successfully tracked keypoints is less than this, will consider calculation fails
    min_keypoints_for_motion_estimation: 50 # minimal number of keypoints to calculate transformation matrix of the motion of the platfrom. If the number of successfully tracked keypoints is less than this, will consider calculation fails
    min_pixel_dis_square_for_scene_point: 2 # we use this param to remove scene point in the tracking bbox of an object
    use_resize: true # using resize in optical flow tracking to speedup
    resize_factor: 2 # resize ratio
  ```

  - **bbox_overlap_ratio**: if the overlapping area ratio of the bounding box(bbox) from the detector and the tracker is higher than this value, we match these two bounding boxes, and use the detector bbox to reinitialized the matched tracker.

    - Overlaping area ratio is calculated by

      $$ratio = min(\frac{Area_{overlap}}{Area_{detector}},\frac{Area_{overlap}}{Area_{detector}})$$

  - **track_fail_timeout_tick**: if the tracker fails for track_fail_timeout_tick frames, we consider this tracker fails and remove it from the list.
  - **detector_update_timeout_tick**: if the ticks after last update by detector is too long, we consider that we lose track of this target
  - **detector_bbox_padding**: to ensure overlap between detector and tracker, we pad the bounding box of the detector to enlarge it
  - **reid_match_threshold**: maximum feature distance to consider a match between a detected and a tracking object
  - **reid_match_bbox_dis**: maximum bbox center distance to consider a match between a detected and a tracking object
  - **reid_match_bbox_size_diff**: maximum bbox size distance to consider a match between a detected and a tracking object
  - **stop_opt_timeout**: when the ticks after last update by detector is larger than this param, we stop updating the tracker by optical flow, but only update the tracker by its state. The purpose is to prevent degeneration of performance when occlussion happens.
  - **height_width_ratio_min/max**: only the image block with height/width falls in the range of (height_width_ratio_min, height_width_ratio_max) will be added into the local database.
    record_interval: 0.1 # the minimum time interval between two recorded images in a local database (Unit: s).
  - **batch_num_min**: deprecated, check the `min_offline_query_data_size` in `ptl_reid_cpp/config/config.yaml`
  - **feature_smooth_ratio**: the current feature of a tracking object is calculated by:
    $$feature_{current} = ratio * feature_{previous} + (1- ratio) * feature_{new}$$
  - **resample_size**: point cloud resample size(Unit:m)
  - **x_min/x_max/z_min/z_max**: point cloud conditional filter(Unit:m)
  - **std_dev_thres/mean_k**: statistial filter param
  - **match_centroid_padding**: padding of bbox for robust reprojection matching between 2d bbox and 3d centroids
  - **q_xy**: bbox center position state variance (Unit: Pixel^2)
  - **q_wh**: bbox size state variance (Unit: Pixel^2)
  - **p_xy_pos**: bbox center position initial variance (Unit: Pixel^2)
  - **p_xy_dp**: # bbox center position velocity initial variance (Unit: Pixel^2)
  - **p_wh_size**: # bbox size initial variance (Unit: (Pixel/s)^2)
  - **p_wh_ds**: 25 # bbox size velocity initial variance (Unit: (Pixel/s)^2)
  - **r_theta/r_f/r_tx/r_ty**: observation variance
  - **residual_threshold**: if the residual is higher than this param, this observation will be rejected
  - **q_factor**: position state variance (Unit (m/s^2)^2)
  - **r_factor**: position observation variance (Unit m^2)
  - **p_pos**: position initial variance (Unit m^2)
  - **p_vel**: position initial variance (Unit (m/s)^2)
  - **start_predict_only_timeout**: using the detector update tick as the timeout count, if this tick is higher than this param, will only use state predict result to tracker the position to avoid degeneration of false 2d tracking
  - **stop_track_timeout**: using the detector update tick as the timeout count, if this tick is higher than this param, will stop 3d tracking to avoid drift
  - **outlier_threshold**: if the residual ratio is higher than this param, this observation will be rejected
  - **min_keypoints_to_track/keypoints_num_factor_area**: minimal keypoints to track for one object in keypoints_num_factor_area (e.g. 40 keypoints in 8000 pixel^2)
  - **corner_detector_max_num**: maximum keypoints to track for one object
  - **corner_detector_quality_level/corner_detector_min_distance/corner_detector_block_size/corner_detector_use_harris/corner_detector_k**: corner detetion params...
  - **min_keypoints_to_cal_H_mat**: minimal number of keypoints to calculate transformation matrix of an object. If the number of successfully tracked keypoints is less than this, will consider calculation fails
  - **min_keypoints_for_motion_estimation**: minimal number of keypoints to calculate transformation matrix of the motion of the platfrom. If the number of successfully tracked keypoints is less than this, will consider calculation fails
  - **min_pixel_dis_square_for_scene_point**: we use this param to remove scene point in the tracking bbox of an object
  - **use_resize**: using resize in optical flow tracking to speedup
  - **resize_factor**: resize ratio

- ptl_reid_cpp

  ```yaml
  reid_db:
    similarity_test_threshold: 1.0
    same_id_threshold: 1.6
    batch_ratio: 0.5
    max_feat_num_one_object: 50
    use_inverted_file_db_threshold: 2500
    feat_dimension: 2048
    find_first_k: 2
    nlist_ratio: 50
    sim_check_start_threshold: 5

  reid_inference:
    engine_file_name: "reid_engine.engine"
    onnx_file_name: "reid.onnx"
    inference_offline_batch_size: 1
    inference_real_time_batch_size: 1
  ```

  - **similarity_test_threshold**: when updating the database, if the minimal distance score between the query image and gallery images is smaller than this value, we will not add this query image to the database (to maintain adequate differnce in the database of an object)
  - **same_id_threshold**: when querying the database, if the minimal distance score between the query image and gallery images is bigger than this value, we will consider this query image comes from a new object; Or else we consider it belongs to one of the objects in the database
  - **batch_ratio**: in a image batch from a query object, if the number of images assigned with the same id after re-identification, is bigger than the batch_ratio times the total number of images in this batch, we consider this object is the object already in the database. Otherwise, we will take it as a new object.
  - **use_inverted_file_db_threshold**: if the number of object in the database is bigger than this value, the database will switch to inverted file to speedup search
  - **feat_dimension**: the dimension of the feature
  - **find_first_k**: find the first k closest object in query
  - **nlist_ratio**: sub cell ratio in inverted file
  - **sim_check_start_threshold**: when the number of the image of an object is bigger than this value, we will start the similarity test (to ensure wrong image will have little effect in building database)

- ptl_node
  ```yaml
  detect_every_k_frame: 5 # perform detection in everyt k frame to reduce GPU load
  lidar_topic: "/rslidar_points"
  camera_topic: "/camera2/color/image_raw/compressed"
  min_offline_query_data_size: 20 # the minimal data size (feature size + image size) to query a dead tracking object. This param is to make sure that we will not query the wrong detected object
  ```

### 3. Run

1. Copy the `.weight` file of yolo to `ptl_ws/src/pedestrain_tracking_and_localizaiton/src/ptl_detector/asset`. Copy the `.onnx` file of re-identification model (which can be obtained from [fast-reid model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md). You can also train your own model using [fast-reid](https://github.com/JDAI-CV/fast-reid).) to `ptl_ws/src/pedestrain_tracking_and_localizaiton/src/ptl_reid_cpp/asset`

2. Launch the node

   ```shell
   cd ptl_ws
   source devel/setup.zsh
   roslaunch ptl_node ptl_node
   ```

3. visualize the result

   ```
   rosrun rviz rviz -d full_vis_2.rviz
   ```

## Reference

- [fast-reid](https://github.com/JDAI-CV/fast-reid)
- [FastMOT](https://github.com/GeekAlexis/FastMOT)
- [faiss](https://github.com/facebookresearch/faiss)
