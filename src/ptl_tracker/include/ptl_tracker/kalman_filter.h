#pragma once
#include <opencv/cv.hpp>
#include <Eigen/Dense>
#include <iostream>
namespace ptl
{
    namespace tracker
    {
        struct KalmanFilterParam
        {

            double p_init_pos;
            double p_init_size;

            double q_pos;
            double r_pos_tracker;
            double r_pos_detector;

            double q_size;
            double r_size_tracker;
            double r_size_detector;
        };

        class KalmanFilter
        {
        public:
            KalmanFilter(KalmanFilterParam kf_param);
            void init(cv::Rect2d bbox);
            void update_bbox();
            cv::Rect2d predict(double time);
            cv::Rect2d update(cv::Rect2d measurement, bool is_tracker);

            Eigen::Vector4d x_pos;
            Eigen::Vector4d x_size;

        private:
            void init_state();
            void get_measurement(cv::Rect2d bbox);
            inline cv::Rect2d point_to_bbox(Eigen::Vector4d point);

            Eigen::Matrix4d F, P_pos, P_size;
            Eigen::MatrixXd H;
            Eigen::Vector2d z_pos, z_size;
            KalmanFilterParam _kf_param;
            cv::Rect2d _bbox;
        };

    } // namespace tracker
} // namespace ptl