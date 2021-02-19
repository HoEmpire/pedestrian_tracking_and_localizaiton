#pragma once
#include "ptl_tracker/tracker.h"
#include "ptl_reid_cpp/reid.h"

namespace ptl
{
    namespace node
    {
        struct NodeParam
        {
            int detect_every_k_frame = 5;
            std::string lidar_topic = "/rslidar_points";
            std::string camera_topic = "/camera2/color/image_raw/compressed";
            int min_offline_query_data_size = 10;
        };

        class Node
        {
        public:
            Node() = default;
            Node(const ros::NodeHandle &n);

            // //load config, init ptl_reid, register two callback function with the subscriber
            // void init();

        private:
            void load_config();

            void camera_callback(const sensor_msgs::CompressedImageConstPtr &image);

            void lidar_callback(const sensor_msgs::PointCloud2ConstPtr &point_cloud);

            void reid_real_time(const cv::Mat &image, const ros::Time &time_now);

            ros::NodeHandle nh_;
            reid::Reid ptl_reid;
            tracker::TrackerInterface ptl_tracker;
            NodeParam node_param;

            ros::Subscriber lidar_sub, camera_sub;

            std::thread *reid_real_time_thread = nullptr;
            long int frame_count = 0;
        };
    } // namespace node
} // namespace ptl
