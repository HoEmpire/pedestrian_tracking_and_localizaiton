#pragma once
#include <deque>
#include <thread>
#include <ros/ros.h>
#include <geometry_msgs/Point.h>

#include "ptl_reid_cpp/reid_database.h"
#include "ptl_reid_cpp/reid_inference.h"
#include "ptl_reid_cpp/util.h"
#include "ptl_detector/yolo_detector.hpp"

namespace ptl
{
    namespace reid
    {
        struct ReidOfflineType
        {
            ReidOfflineType(const cv::Mat &example_image_init, const std::vector<cv::Mat> &images_init,
                            const geometry_msgs::Point &position_init, const std::vector<float> &feat_init) : example_image(example_image_init), image(images_init),
                                                                                                              position(position_init), feat_all(feat_init) {}
            cv::Mat example_image;
            std::vector<cv::Mat> image;
            geometry_msgs::Point position;
            std::vector<float> feat_all;
        };

        class Reid
        {
        public:
            Reid() = default;

            Reid(const ros::NodeHandle &n) : nh_(n), reid_detector(n) {}

            // initialize reid engine, create offline reid thread
            void init();

            void reid_realtime(const cv::Mat &image, std::vector<cv::Rect2d> &bboxes, std::vector<float> &feat);

            ReidDatabase reid_db;
            ReidInference reid_inferencer;
            detector::YoloPedestrainDetector reid_detector;
            std::deque<ReidOfflineType> reid_offline_buffer;

        private:
            // load parameters fros
            void load_config();

            void reid_offline();

            ros::NodeHandle nh_;
            DataBaseParam db_param;
            InferenceParam inference_param;
            std::thread reid_offline_thread;
        };
    } // namespace reid
} // namespace ptl