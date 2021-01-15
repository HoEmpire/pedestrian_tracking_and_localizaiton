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
            int tracker_fail_timeout; //using detector update tick as signal
            double outlier_threshold; //calculated by the probability of gaussian distribution
        };

        class KalmanFilter3d
        {
        public:
            KalmanFilter3d(KalmanFilter3dParam kf_param);
            void init(geometry_msgs::Point x_init);
            void estimate(double time);
            void update(geometry_msgs::Point measurement);
            geometry_msgs::Point get_pos();

            Eigen::VectorXd x;
            bool is_init = false;

        private:
            Eigen::Vector3d get_measurement(geometry_msgs::Point measurement);

            Eigen::MatrixXd Q, P, F;
            Eigen::Matrix3d R;
            Eigen::MatrixXd H;
            KalmanFilter3dParam _kf_param;
        };

    } // namespace tracker
} // namespace ptl