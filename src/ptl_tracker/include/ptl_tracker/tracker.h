#pragma once
#include <iostream>
#include <mutex>
#include <string>

//ros
#include "ros/ros.h"
#include <tf2_ros/transform_listener.h>
#include "cv_bridge/cv_bridge.h"
#include <image_transport/image_transport.h>

//ros msg
#include "std_msgs/UInt16MultiArray.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

//opentracker
#include "opentracker/kcf/kcftracker.hpp"

#include "ptl_tracker/local_object.h"
#include "ptl_tracker/association_type.hpp"
#include "ptl_msgs/ImageBlock.h"
#include "ptl_msgs/ReidInfo.h"

namespace ptl
{
    namespace tracker
    {
        class TrackerInterface
        {
        public:
            TrackerInterface(ros::NodeHandle *n);

        private:
            void detector_result_callback(const ptl_msgs::ImageBlockPtr &msg);
            void data_callback(const sensor_msgs::CompressedImageConstPtr &msg);
            void data_callback(const sensor_msgs::CompressedImageConstPtr &msg_img, const sensor_msgs::PointCloud2ConstPtr &msg_pc);
            void reid_callback(const ptl_msgs::ReidInfo &msg);
            void load_config(ros::NodeHandle *n);
            bool update_local_database(LocalObject local_object, const cv::Mat img_block);
            bool update_local_database(std::vector<LocalObject>::iterator local_object, const cv::Mat img_block);

        public:
            std::vector<LocalObject> local_objects_list;

        private:
            bool blur_detection(cv::Mat img);

            int id;
            ros::NodeHandle *nh_;
            ros::Publisher m_track_vis_pub, m_track_to_reid_pub;
            ros::Subscriber m_detector_sub, m_data_sub, m_reid_sub;
            std::mutex mtx;
            struct ReidInfo reid_infos;
            tf2_ros::Buffer tf_buffer;
            tf2_ros::TransformListener *tf_listener;

            std::string lidar_topic, camera_topic, depth_topic;
            int track_fail_timeout_tick = 30;
            double bbox_overlap_ratio_threshold = 0.5;
            int track_to_reid_bbox_margin = 10;
            float height_width_ratio_min = 1.0;
            float height_width_ratio_max = 3.0;
            float blur_detection_threshold = 160.0;
            float record_interval = 0.1;
            int batch_num_min = 3;

            int detector_update_timeout_tick = 10;
            int detector_bbox_padding = 10;
            float reid_match_threshold = 200;
            double reid_match_bbox_dis = 30;
            double reid_match_bbox_size_diff = 30;

            struct TrackerParam tracker_param;
        };
    } // namespace tracker

} // namespace ptl
