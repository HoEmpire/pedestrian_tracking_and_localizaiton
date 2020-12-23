#pragma once
#include <opencv2/opencv.hpp>
#include "opentracker/kcf/kcftracker.hpp"
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>

#include "ptl_tracker/timer.hpp"

namespace ptl
{
    namespace tracker
    {
        class LocalObject
        {
        public:
            LocalObject(int id_init, cv::Rect2d bbox_init, cv::Mat frame, float tracker_success_param = 0.5);
            void update_tracker(cv::Mat frame);
            void reinit(cv::Rect2d bbox_init, cv::Mat frame);

            int id;
            cv::Rect2d bbox;
            int tracking_fail_count;
            int overlap_count;
            geometry_msgs::Point position_local;
            cv::Scalar color;
            bool is_track_succeed;
            std::vector<cv::Mat> img_blocks;
            timer time;

        private:
            bool HOG = true;
            bool FIXEDWINDOW = true;
            bool MULTISCALE = true;
            bool LAB = false;
            bool DSST = false;
            float tracker_success_threshold;

            kcf::KCFTracker *dssttracker;
        };
    } // namespace tracker

} // namespace ptl
