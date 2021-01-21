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
            double Q_factor;
            double R_factor;
            double P_factor;
        };

        class KalmanFilter
        {
        public:
            KalmanFilter(const KalmanFilterParam &kf_param);
            void init(const cv::Rect2d &bbox);
            void update_bbox(const cv::Rect2d &bbox);
            cv::Rect2d estimate(const double time);
            cv::Rect2d update(const cv::Rect2d &measurement);

            // ensure the bbox height and width is the latest one
            // especially in estimate step of tracker
            Eigen::Vector4d x;

        private:
            inline Eigen::Vector4d bbox_to_state(const cv::Rect2d &bbox);
            inline Eigen::Vector2d bbox_to_measurement(const cv::Rect2d &bbox);
            inline cv::Rect2d point_to_bbox(const Eigen::Vector4d &point);

            Eigen::Matrix4d Q, P, F;
            Eigen::Matrix2d R;
            Eigen::MatrixXd H;
            cv::Rect2d _bbox;
            KalmanFilterParam _kf_param;
        };

    } // namespace tracker
} // namespace ptl