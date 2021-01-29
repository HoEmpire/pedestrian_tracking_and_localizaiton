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
            double p_xy;
            double p_wh;
            double r_yolo_xy;
            double r_yolo_wh;
            double r_opt_xy;
            double r_opt_wh;
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
            Eigen::Matrix4d Q_wh, P_wh, Q_xy, P_xy, F;
            Eigen::Matrix2d R_wh, Q_xy;
            Eigen::MatrixXd H;
            cv::Rect2d _bbox;
            KalmanFilterParam kf_param_;
        };

    } // namespace tracker
} // namespace ptl