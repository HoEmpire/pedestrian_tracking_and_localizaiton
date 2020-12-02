#pragma once
#include <iostream>

#include <ros/package.h>
#include <ros/ros.h>

#include "usfs_inference/component/object_detection_component.h"
#include "usfs_inference/detector/yolo_object_detector.h"

using namespace std;
// config util
struct ConfigSetting
{
    std::unordered_map<std::string, ModelType> net_type_table = {{"YOLOV3", ModelType::YOLOV3}, {"YOLOV3_TINY", ModelType::YOLOV3_TINY}, {"YOLOV4", ModelType::YOLOV4}, {"YOLOV4_TINY", ModelType::YOLOV4_TINY}, {"YOLOV4", ModelType::YOLOV4}, {"YOLOV5", ModelType::YOLOV5}};
    std::unordered_map<std::string, Precision> precision_table = {{"INT8", Precision::INT8}, {"FP16", Precision::FP16}, {"FP32", Precision::FP32}};

    string lidar_topic, camera_topic, depth_topic;

    string cam_net_type = "YOLOV4_TINY";
    string cam_file_model_cfg;
    string cam_file_model_weights;
    string cam_inference_precison;
    int cam_n_max_batch = 1;
    float cam_min_width = 0;
    float cam_max_width = 640;
    float cam_prob_threshold;
    float cam_min_height = 0;
    float cam_max_height = 480;

} config;

void loadConfig(ros::NodeHandle n)
{
    n.getParam("/data_topic/lidar_topic", config.lidar_topic);
    n.getParam("/data_topic/camera_topic", config.camera_topic);
    n.getParam("/data_topic/depth_topic", config.depth_topic);

    n.getParam("/cam/cam_net_type", config.cam_net_type);
    n.getParam("/cam/cam_file_model_cfg", config.cam_file_model_cfg);
    n.getParam("/cam/cam_file_model_weights", config.cam_file_model_weights);
    n.getParam("/cam/cam_inference_precison", config.cam_inference_precison);
    n.getParam("/cam/cam_n_max_batch", config.cam_file_model_cfg);
    n.getParam("/cam/cam_prob_threshold", config.cam_prob_threshold);
    n.getParam("/cam/cam_min_width", config.cam_min_width);
    n.getParam("/cam/cam_max_width", config.cam_max_width);
    n.getParam("/cam/cam_min_height", config.cam_min_height);
    n.getParam("/cam/cam_max_height", config.cam_max_height);
}