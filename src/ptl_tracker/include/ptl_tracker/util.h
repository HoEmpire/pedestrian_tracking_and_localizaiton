#pragma once

#include <ros/ros.h>
#include <string>
#include <opencv/cv.h>
#include <Eigen/Dense>
#include <std_msgs/Float32MultiArray.h>
template <class T>
void GPARAM(ros::NodeHandle *n, std::string param_path, T &param)
{
    if (!n->getParam(param_path, param))
        ROS_ERROR_STREAM("Load param from " << param_path << " failed...");
}

struct ReidInfo
{
    int total_num = 0;
    int last_query_id = -1;
};

struct CameraIntrinsic
{
    float fx, fy, cx, cy;
};

inline cv::Rect2d BboxPadding(cv::Rect2d bbox_to_pad, cv::Rect2d bbox_max, int padding_pixel)
{
    return (cv::Rect2d(bbox_to_pad.x - padding_pixel,
                       bbox_to_pad.y - padding_pixel,
                       bbox_to_pad.width + 2 * padding_pixel,
                       bbox_to_pad.height + 2 * padding_pixel) &
            bbox_max);
}

inline cv::Rect2d BboxPadding(cv::Rect2d bbox_to_pad, int padding_pixel)
{
    return (cv::Rect2d(bbox_to_pad.x - padding_pixel,
                       bbox_to_pad.y - padding_pixel,
                       bbox_to_pad.width + 2 * padding_pixel,
                       bbox_to_pad.height + 2 * padding_pixel));
}

inline Eigen::VectorXf feature_ros_to_eigen(std_msgs::Float32MultiArray feats_ros)
{
    Eigen::VectorXf feats_eigen(feats_ros.data.size());
    for (int i = 0; i < feats_ros.data.size(); i++)
        feats_eigen[i] = feats_ros.data[i];
    return feats_eigen;
}

inline std::vector<Eigen::VectorXf> feature_vector_to_eigen(std::vector<float> feats_vec, int feat_dimension = 2048)
{

    std::vector<Eigen::VectorXf> feats_eigen;
    for (int i = 0; i < feats_vec.size() / feat_dimension; i++)
        feats_eigen.push_back(Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(feats_vec.data() + i * feat_dimension, feat_dimension));
    return feats_eigen;
}

inline void print_bbox(cv::Rect2d bbox)
{
    ROS_INFO_STREAM("Bbox Info: " << bbox.x << ", " << bbox.y << ", "
                                  << bbox.width << ", " << bbox.height);
}

inline double cal_bbox_overlap_ratio(cv::Rect2d track_bbox, cv::Rect2d detect_bbox)
{
    return std::max((track_bbox & detect_bbox).area() / track_bbox.area(),
                    (track_bbox & detect_bbox).area() / detect_bbox.area());
}

inline double cal_bbox_distance(cv::Rect2d track_bbox, cv::Rect2d detect_bbox)
{
    return std::sqrt(std::pow(track_bbox.x - detect_bbox.x, 2) + std::pow(track_bbox.y - detect_bbox.y, 2));
}

inline double cal_bbox_size_diff(cv::Rect2d track_bbox, cv::Rect2d detect_bbox)
{
    return std::sqrt(std::pow(track_bbox.width - detect_bbox.width, 2) + std::pow(track_bbox.height - detect_bbox.height, 2));
}

inline double cal_bbox_match_score(cv::Rect2d track_bbox, cv::Rect2d detect_bbox)
{
    return std::sqrt(std::pow(track_bbox.x - detect_bbox.x, 2) + std::pow(track_bbox.y - detect_bbox.y, 2) +
                     std::pow(track_bbox.width - detect_bbox.width, 2) + std::pow(track_bbox.height - detect_bbox.height, 2));
}
