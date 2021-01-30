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
            double q_xy;
            double q_wh;
            double p_xy_pos;
            double p_xy_dp;
            double p_wh_size;
            double p_wh_ds;
            double r_theta;
            double r_f;
            double r_tx;
            double r_ty;
            double residual_threshold;
        };

        class KalmanFilter
        {
        public:
            KalmanFilter(const KalmanFilterParam &kf_param);
            void init(const cv::Rect2d &bbox);
            void update_bbox(const cv::Rect2d &bbox);
            cv::Rect2d predict(const double time);
            cv::Rect2d update(const cv::Mat &T_mat, const double dt);
            cv::Rect2d predict_only(const double time);

            // ensure the bbox height and width is the latest one
            // especially in estimate step of tracker
            Eigen::Vector4d x_xy, x_wh;

        private:
            Eigen::Vector4d predict_measurement_xy();
            Eigen::Vector4d predict_measurement_wh();

            Eigen::Matrix4d P_wh, P_xy, F, R_xy;
            Eigen::Matrix2d R_wh;
            cv::Rect2d _bbox;
            KalmanFilterParam kf_param_;
        };

    } // namespace tracker
} // namespace ptl