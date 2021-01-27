#pragma once
#include <opencv2/opencv.hpp>
#include "opentracker/kcf/kcftracker.hpp"
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <Eigen/Dense>

#include "ptl_tracker/kalman_filter.h"
#include "ptl_tracker/kalman_filter_3d.h"
#include "ptl_tracker/timer.hpp"
#include "ptl_tracker/util.h"

namespace ptl
{
    namespace tracker
    {
        class LocalObject
        {
        public:
            LocalObject(const int id_init, const cv::Rect2d &bbox_init, const cv::Mat &frame,
                        const Eigen::VectorXf &feat, const TrackerParam &track_param_init, const KalmanFilterParam &kf_param_init,
                        const KalmanFilter3dParam &kf3d_param_init, const ros::Time &time_now);

            //update bbox by optical flow
            void track_bbox_by_optical_flow(const ros::Time &time_now);

            //update bbox by detector
            //general before track by detector, we will perform track by optical flow first
            void track_bbox_by_detector(const cv::Rect2d &bbox_detector, const ros::Time &update_time);

            void update_3d_tracker(const geometry_msgs::Point &measurement, const ros::Time &time_now);
            void update_3d_tracker(const ros::Time &time_now);

            float find_min_query_score(const Eigen::VectorXf &query);

            int id;
            cv::Rect2d bbox, bbox_optical_flow;
            int tracking_fail_count;
            int detector_update_count;
            int overlap_count;
            cv::Scalar color;
            bool is_track_succeed;
            std::vector<cv::Mat> img_blocks;
            std::vector<Eigen::VectorXf> features;
            geometry_msgs::Point position;
            ros::Time bbox_last_update_time;
            ros::Time ros_time_pc_last;
            bool is_overlap = false;

            //control the time interval between adding two image blocks into this local object database
            timer database_update_timer;

            std::vector<cv::Point2f> keykoints_pre, keypoints_curr;
            cv::Rect2d bbox_measurement;

        private:
            TrackerParam tracker_param;
            KalmanFilter *kf;
            KalmanFilter3d *kf_3d;
        };
    } // namespace tracker

} // namespace ptl
