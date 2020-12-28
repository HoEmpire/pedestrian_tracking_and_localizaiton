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

struct TrackerParam
{
    float tracker_success_threshold = 0.2;
    float interp_factor = 0.005;
    float sigma = 0.4;
    float lambda = 0.0001;
    int cell_size = 4;
    float padding = 2.5;
    float output_sigma_factor = 0.1;
    int template_size = 96;
    float scale_step = 1.05;
    float scale_weight = 0.95;
};

inline cv::Rect2d BboxPadding(cv::Rect2d bbox_to_pad, cv::Rect2d bbox_max, int padding_pixel)
{
    return (cv::Rect2d(bbox_to_pad.x - padding_pixel,
                       bbox_to_pad.y - padding_pixel,
                       bbox_to_pad.width + 2 * padding_pixel,
                       bbox_to_pad.height + 2 * padding_pixel) &
            bbox_max);
}

inline Eigen::VectorXf feature_ros_to_eigen(std_msgs::Float32MultiArray feats_ros)
{
    Eigen::VectorXf feats_eigen(feats_ros.data.size());
    for (int i = 0; i < feats_ros.data.size(); i++)
        feats_eigen[i] = feats_ros.data[i];
    return feats_eigen;
}

inline void print_bbox(cv::Rect2d bbox)
{
    ROS_INFO_STREAM("Bbox Info: " << bbox.x << ", " << bbox.y << ", "
                                  << bbox.width << ", " << bbox.height);
}

class AssociationType
{
public:
    AssociationType(int id_, float score_)
    {
        id = id_;
        score = score_;
    }
    int id;
    float score;
};

class AssociationVector
{
public:
    AssociationVector() = default;

    void add_new_ass(AssociationType ass_object)
    {
        if (ass_vector.empty())
            ass_vector.push_back(ass_object);
        else
        {
            for (auto iter = ass_vector.begin(); iter < ass_vector.end(); iter++)
            {
                if (ass_object.score < iter->score)
                {
                    ass_vector.insert(iter, ass_object);
                    return;
                }
            }
        }
    }

    void report()
    {
        ROS_INFO("Data Association Report:");
        for (auto ass : ass_vector)
            ROS_INFO_STREAM("id: " << ass.id << " | score: " << ass.score);
    }

    std::vector<AssociationType> ass_vector;
};