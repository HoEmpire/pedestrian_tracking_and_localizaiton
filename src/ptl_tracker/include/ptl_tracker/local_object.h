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
            LocalObject(int id_init, cv::Rect2d bbox_init, cv::Mat frame,
                        Eigen::VectorXf feat, TrackerParam track_param_init, KalmanFilterParam kf_param_init, KalmanFilter3dParam kf3d_param_init, ros::Time time_now);
            void update_tracker(cv::Mat frame, ros::Time update_time);
            void update_3d_tracker(geometry_msgs::Point measurement, ros::Time time_now);
            void update_3d_tracker(ros::Time time_now);
            void reinit(cv::Rect2d bbox_init, cv::Mat frame, ros::Time update_time);
            float find_min_query_score(Eigen::VectorXf);

            int id;
            cv::Rect2d bbox;
            int tracking_fail_count;
            int detector_update_count;
            int overlap_count;
            cv::Scalar color;
            bool is_track_succeed;
            std::vector<cv::Mat> img_blocks;
            std::vector<Eigen::VectorXf> features;
            timer time;
            geometry_msgs::Point position;
            ros::Time ros_time_image_last;
            ros::Time ros_time_pc_last;
            bool is_overlap = false;

        private:
            bool HOG = true;
            bool FIXEDWINDOW = true;
            bool MULTISCALE = true;
            bool LAB = true;
            bool DSST = false;
            TrackerParam tracker_param;
            KalmanFilter *kf;
            KalmanFilter3d *kf_3d;

            kcf::KCFTracker *dssttracker;
        };
    } // namespace tracker

} // namespace ptl
