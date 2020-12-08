#pragma once
#include <iostream>
#include <mutex>
#include <string>

#include "ros/ros.h"
#include "std_msgs/UInt16MultiArray.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include "cv_bridge/cv_bridge.h"
#include <image_transport/image_transport.h>

#include "opentracker/kcf/kcftracker.hpp"

#include "ptl_tracker/local_object.h"
#include "ptl_msgs/ImageBlock.h"

namespace ptl_tracker
{

    class TrackerInterface
    {
    public:
        TrackerInterface(ros::NodeHandle *n);

    private:
        void detector_result_callback(const ptl_msgs::ImageBlockPtr &msg);
        void data_callback(const sensor_msgs::CompressedImageConstPtr &msg);
        bool bbox_matching(cv::Rect2d track_bbox, cv::Rect2d detect_bbox);
        void load_config(ros::NodeHandle *n);

    public:
        std::vector<LocalObject> local_objects_list;

    private:
        int id;
        ros::NodeHandle *nh_;
        ros::Publisher m_track_vis_pub, m_track_to_reid_pub;
        ros::Subscriber m_detector_sub, m_data_sub;
        std::mutex mtx;

        std::string lidar_topic, camera_topic, depth_topic;
        int track_fail_timeout_tick = 30;
        double bbox_overlap_ratio = 0.5;
        int track_to_reid_bbox_margin = 10;
        float height_width_ratio_min = 1.0;
        float height_width_ratio_max = 3.0;
    };

} // namespace ptl_tracker
