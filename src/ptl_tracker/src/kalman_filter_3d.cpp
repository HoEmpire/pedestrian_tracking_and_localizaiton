#include "ptl_tracker/kalman_filter_3d.h"
namespace ptl
{
    namespace tracker
    {
        KalmanFilter3d::KalmanFilter3d(const KalmanFilter3dParam &kf_param)
        {
            _kf_param = kf_param;
            // Q = Eigen::Matrix4d::Identity() * kf_param.Q_factor;
            P = Eigen::MatrixXd::Identity(6, 6);
            P.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3) * kf_param.P_pos;
            P.block<3, 3>(3, 3) = Eigen::MatrixXd::Identity(3, 3) * kf_param.P_vel;
            R = Eigen::Matrix3d::Identity() * kf_param.R_factor;

            F = Eigen::MatrixXd::Identity(6, 6);
            H = Eigen::MatrixXd::Identity(3, 6);
        }

        void KalmanFilter3d::init(const geometry_msgs::Point &x_init)
        {
            x = Eigen::VectorXd(6, 1);
            x << x_init.x, x_init.y, x_init.z, 0, 0, 0;
            is_init = true;
        }

        void KalmanFilter3d::estimate(const double dt)
        {
            F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
            x = F * x;
            Q = Eigen::MatrixXd::Zero(6, 6);
            //random acceleration model
            Q(0, 0) = Q(1, 1) = Q(2, 2) = pow(dt, 4) * 0.25;
            Q(3, 3) = Q(4, 4) = Q(5, 5) = pow(dt, 2);
            Q(0, 3) = Q(3, 0) = Q(1, 4) = Q(4, 1) = Q(2, 5) = Q(5, 2) = pow(dt, 3) * 0.5;

            Q *= _kf_param.Q_factor;
            P = F * P * F.transpose() + Q;
            // std::cout << "estimate: dt = " << dt << std::endl;
            // std::cout << "estimate: F = \n"
            //           << F << std::endl;
            // std::cout << "estimate: x = \n"
            //           << x << std::endl;
            // std::cout << "estimate: P = \n"
            //           << P << std::endl;
        }

        void KalmanFilter3d::update(const geometry_msgs::Point &measurement)
        {

            Eigen::Vector3d z = get_measurement(measurement);

            //update
            Eigen::Matrix3d S = H * P * H.transpose() + R;
            const double score = 5.0; // correspond to 100% in gaussian distribution
            if (z(0) / sqrt(S(0, 0)) > score || z(1) / sqrt(S(1, 1)) > score || z(2) / sqrt(S(2, 2)) > score)
            {
                std::cerr << "WARNING:KalamnFilter3d::update():This measurement is a outlier, it will be discarded..." << std::endl;
                return;
            }

            Eigen::MatrixXd K = P * H.transpose() * S.inverse();

            x = x + K * (z - H * x);                           //update x
            P = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P; //update P

            // std::cout << "update: z = \n"
            //           << z << std::endl;
            // std::cout << "update: K = \n"
            //           << K << std::endl;
            // std::cout << "update: x = \n"
            //           << x << std::endl;
        }

        geometry_msgs::Point KalmanFilter3d::get_pos()
        {
            geometry_msgs::Point p;
            p.x = x(0);
            p.y = x(1);
            p.z = x(2);
            return p;
        }

        Eigen::Vector3d KalmanFilter3d::get_measurement(const geometry_msgs::Point &measurement)
        {
            Eigen::Vector3d z;
            z << measurement.x, measurement.y, measurement.z;
            return z;
        }
    } // namespace tracker
} // namespace ptl