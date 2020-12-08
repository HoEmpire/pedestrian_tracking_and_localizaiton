#pragma once
#include <iostream>

#include <ros/package.h>
#include <ros/ros.h>

using namespace std;
// config util
struct ConfigSetting
{
    string lidar_topic, camera_topic, depth_topic;

    int track_fail_timeout_tick = 30;
    double bbox_overlap_ratio = 0.5;
    int track_to_reid_bbox_margin = 10;

} config;

void loadConfig(ros::NodeHandle n)
{
    n.getParam("/data_topic/lidar_topic", config.lidar_topic);
    n.getParam("/data_topic/camera_topic", config.camera_topic);
    n.getParam("/data_topic/depth_topic", config.depth_topic);

    n.getParam("/tracker/track_fail_timeout_tick", config.track_fail_timeout_tick);
    n.getParam("/tracker/bbox_overlap_ratio", config.bbox_overlap_ratio);
    n.getParam("/tracker/track_to_reid_bbox_margin", config.track_to_reid_bbox_margin);
}