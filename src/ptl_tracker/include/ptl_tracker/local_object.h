#pragma once
#include <opencv2/opencv.hpp>
#include "opentracker/kcf/kcftracker.hpp"
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <Eigen/Dense>

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
                        Eigen::VectorXf feat, struct TrackerParam track_param_init);
            void update_tracker(cv::Mat frame);
            void reinit(cv::Rect2d bbox_init, cv::Mat frame);
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

        private:
            bool HOG = true;
            bool FIXEDWINDOW = true;
            bool MULTISCALE = true;
            bool LAB = true;
            bool DSST = false;
            struct TrackerParam tracker_param;

            kcf::KCFTracker *dssttracker;
        };
    } // namespace tracker

} // namespace ptl
