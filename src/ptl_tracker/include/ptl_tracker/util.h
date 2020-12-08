#pragma once
#include <iostream>

#include <ros/package.h>
#include <ros/ros.h>

using std::string;
// config util
namespace ptl_tracker
{
    struct ConfigSetting
    {
        string lidar_topic, camera_topic, depth_topic;

        int track_fail_timeout_tick = 30;
        double bbox_overlap_ratio = 0.5;
        int track_to_reid_bbox_margin = 10;
        float height_width_ratio_min = 1.0;
        float height_width_ratio_max = 3.0;

    } config;

    void loadConfig(ros::NodeHandle n)
    {
        n.getParam("/data_topic/lidar_topic", config.lidar_topic);
        n.getParam("/data_topic/camera_topic", config.camera_topic);
        n.getParam("/data_topic/depth_topic", config.depth_topic);

        n.getParam("/tracker/track_fail_timeout_tick", config.track_fail_timeout_tick);
        n.getParam("/tracker/bbox_overlap_ratio", config.bbox_overlap_ratio);
        n.getParam("/tracker/track_to_reid_bbox_margin", config.track_to_reid_bbox_margin);

        n.getParam("/local_database/height_width_ratio_min", config.height_width_ratio_min);
        n.getParam("/local_database/height_width_ratio_max", config.height_width_ratio_max);
    }
} // namespace ptl_tracker
