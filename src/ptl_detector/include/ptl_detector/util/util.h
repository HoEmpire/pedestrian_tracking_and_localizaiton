#pragma once
#include <iostream>

#include <ros/package.h>
#include <ros/ros.h>
#include "ptl_detector/detector/yolo_object_detector.h"

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
    int detect_every_k_frames = 5;

} config;

void loadConfig(ros::NodeHandle n)
{
    n.getParam("/basic/lidar_topic", config.lidar_topic);
    n.getParam("/basic/camera_topic", config.camera_topic);
    n.getParam("/basic/depth_topic", config.depth_topic);
    n.getParam("/basic/detect_every_k_frames", config.detect_every_k_frames);

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

// Integral type equal
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type Equal(
    const T &lhs, const T &rhs)
{
    return lhs == rhs;
}

// Floating point type equal
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type Equal(
    const T &lhs, const T &rhs)
{
    return std::fabs(lhs - rhs) < std::numeric_limits<T>::epsilon();
}

/** Template function that generates a comma separated string from the contents
 * of a vector. Elements are separated by a comma and a space for readability.
 */
template <typename T>
std::string Vector2Csv(const std::vector<T> &vec)
{
    std::string s;
    for (typename std::vector<T>::const_iterator it = vec.begin();
         it != vec.end(); ++it)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << *it;
        s += ss.str();
        s += ", ";
    }
    if (s.size() >= 2)
    { // clear the trailing comma, space
        s.erase(s.size() - 2);
    }
    return s;
}