#pragma once
#include <iostream>
#include <mutex>
#include <string>

//ros
#include "ros/ros.h"
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "cv_bridge/cv_bridge.h"
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>

//ros msg
#include "std_msgs/UInt16MultiArray.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include "ptl_tracker/optical_flow.h"
#include "ptl_tracker/association_type.hpp"
#include "ptl_tracker/point_cloud_processor.h"
#include "ptl_msgs/ImageBlock.h"
#include "ptl_msgs/ReidInfo.h"

namespace ptl
{
    namespace tracker
    {
        class TrackerInterface
        {
        public:
            TrackerInterface() = default;
            TrackerInterface(const ros::NodeHandle &n) : nh_(n) {}

            void init(bool register_subscriber = true);

            //udpate bbox by optical tracker and return the dead tracker
            std::vector<LocalObject> update_bbox_by_tracker(const cv::Mat &img, const ros::Time &update_time);
            void update_bbox_by_detector(const cv::Mat &img,
                                         const std::vector<cv::Rect2d> &bboxes,
                                         const std::vector<float> feature,
                                         const ros::Time &update_time);

            void lidar_tracker_callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc);

        private:
            void detector_result_callback(const ptl_msgs::ImageBlockPtr &msg);

            void image_tracker_callback(const sensor_msgs::ImageConstPtr &msg);
            void image_tracker_callback_compressed_img(const sensor_msgs::CompressedImageConstPtr &msg);

            void reid_callback(const ptl_msgs::ReidInfo &msg);
            void load_config(ros::NodeHandle *n);

            bool update_local_database(LocalObject &local_object, const cv::Mat &img_block);

            void match_between_2d_and_3d(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const ros::Time &ros_pc_time);
            void get_tf();
            void update_tracker_pos_marker_visualization();
            void update_overlap_flag();

            //bbox update by optical flow tracker
            void track_bbox_by_optical_flow(const cv::Mat &img, const ros::Time &update_time, bool update_database);
            std::vector<LocalObject> remove_dead_trackers();
            void report_local_object();
            void visualize_tracking(cv::Mat &img);

            //do segementation by reprojection
            pcl::PointCloud<pcl::PointXYZI> point_cloud_segementation(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const cv::Rect2d &bbox);

            //associate the detected results with local tracking objects, make sure one detected object matches only 0 or 1 tracking object
            void detector_and_tracker_association(const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
                                                  const std::vector<Eigen::VectorXf> &features, std::vector<AssociationVector> &all_detected_bbox_ass_vec);
            void manage_local_objects_list_by_detector(const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
                                                       const std::vector<Eigen::VectorXf> &features, const cv::Mat &img,
                                                       const ros::Time &update_time,
                                                       const std::vector<AssociationVector> &all_detected_bbox_ass_vec);

            void manage_local_objects_list_by_reid_detector(const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
                                                            const std::vector<Eigen::VectorXf> &feat_eigen, const std::vector<float> &feat_vector,
                                                            const cv::Mat &img, const ros::Time &update_time, const std::vector<AssociationVector> &all_detected_bbox_ass_vec);

            std::vector<cv::Rect2d> bbox_ros_to_opencv(const std::vector<std_msgs::UInt16MultiArray> &bbox_ros);
            void bbox_rect(const cv::Rect2d &bbox_max);

        public:
            std::vector<LocalObject> local_objects_list;

        private:
            ros::NodeHandle nh_;
            ros::Publisher m_track_vis_pub, m_track_to_reid_pub, m_track_marker_pub;
            ros::Publisher m_pc_filtered_debug, m_pc_cluster_debug;
            ros::Subscriber m_detector_sub, m_image_sub, m_reid_sub, m_lidar_sub;

            //tf
            tf2_ros::Buffer tf_buffer;
            tf2_ros::TransformListener *tf_listener = new tf2_ros::TransformListener(tf_buffer);
            geometry_msgs::TransformStamped lidar2camera, lidar2map, camera2map;

            //lock
            std::mutex mtx;
            ReidInfo reid_infos;

            OpticalFlow opt_tracker;
            int local_id_not_assigned = 0;

            //params
            std::string lidar_topic, camera_topic;
            std::string map_frame, lidar_frame, camera_frame;
            bool use_compressed_image = true;
            bool use_lidar = false;
            bool enable_pcp_vis = true;
            int track_fail_timeout_tick = 30;
            double bbox_overlap_ratio_threshold = 0.5;
            int track_to_reid_bbox_margin = 10;
            float height_width_ratio_min = 1.0;
            float height_width_ratio_max = 3.0;
            float record_interval = 0.1;
            int batch_num_min = 3;

            int detector_update_timeout_tick = 10;
            int stop_opt_timeout = 5;
            int detector_bbox_padding = 10;
            float reid_match_threshold = 200;
            double reid_match_bbox_dis = 30;
            double reid_match_bbox_size_diff = 30;
            int match_centroid_padding = 20;
            float feature_smooth_ratio = 0.8;

            //params
            PointCloudProcessorParam pcp_param;
            OpticalFlowParam opt_param;
            KalmanFilterParam kf_param;
            KalmanFilter3dParam kf3d_param;
            CameraIntrinsic camera_intrinsic;
        };
    } // namespace tracker

} // namespace ptl
