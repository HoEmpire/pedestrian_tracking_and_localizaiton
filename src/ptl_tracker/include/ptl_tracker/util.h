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
    int bbox_match_pixel_dis = 30;

} config;

void loadConfig(ros::NodeHandle n)
{
    n.getParam("/data_topic/lidar_topic", config.lidar_topic);
    n.getParam("/data_topic/camera_topic", config.camera_topic);
    n.getParam("/data_topic/depth_topic", config.depth_topic);

    n.getParam("/tracker/track_fail_timeout_tick", config.track_fail_timeout_tick);
    n.getParam("/tracker/bbox_match_pixel_dis", config.bbox_match_pixel_dis);
}