#pragma once
#include <geometry_msgs/Point.h>
#include <Eigen/Dense>
#include <iostream>
namespace ptl
{
    namespace tracker
    {
        struct KalmanFilter3dParam
        {
            double Q_factor;
            double R_factor;
            double P_factor;
            int start_predict_only_timeout; //using detector update tick as signal
            int stop_track_timeout;         //using detector
            double outlier_threshold;       //calculated by the probability of gaussian distribution
        };

        class KalmanFilter3d
        {
        public:
            KalmanFilter3d(const KalmanFilter3dParam &kf_param);
            void init(const geometry_msgs::Point &x_init);
            void estimate(const double time);
            void update(const geometry_msgs::Point &measurement);
            geometry_msgs::Point get_pos();

            Eigen::VectorXd x;
            bool is_init = false;

        private:
            Eigen::Vector3d get_measurement(const geometry_msgs::Point &measurement);

            Eigen::MatrixXd Q, P, F;
            Eigen::Matrix3d R;
            Eigen::MatrixXd H;
            KalmanFilter3dParam _kf_param;
        };

    } // namespace tracker
} // namespace ptl