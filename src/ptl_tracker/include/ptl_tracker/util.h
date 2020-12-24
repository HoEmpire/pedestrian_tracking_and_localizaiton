#pragma once

#include <ros/ros.h>
#include <string>
#include <opencv/cv.h>
template <class T>
void GPARAM(ros::NodeHandle *n, std::string param_path, T &param)
{
    if (!n->getParam(param_path, param))
        ROS_ERROR_STREAM("Load param from " << param_path << " failed...");
}

struct ReidInfo
{
public:
    int total_num = 0;
    int last_query_id = -1;
};

inline cv::Rect2d BboxPadding(cv::Rect2d bbox_to_pad, cv::Rect2d bbox_max, int padding_pixel)
{
    return (cv::Rect2d(bbox_to_pad.x - padding_pixel,
                       bbox_to_pad.y - padding_pixel,
                       bbox_to_pad.width + 2 * padding_pixel,
                       bbox_to_pad.height + 2 * padding_pixel) &
            bbox_max);
}