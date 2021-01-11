#include "ptl_tracker/kalman_filter.h"
namespace ptl
{
    namespace tracker
    {
        inline Eigen::Vector4d KalmanFilter::bbox_to_state(cv::Rect2d bbox)
        {
            return Eigen::Vector4d(bbox.x + 0.5 * bbox.width, bbox.y + 0.5 * bbox.height, 0, 0);
        }

        inline Eigen::Vector2d KalmanFilter::bbox_to_measurement(cv::Rect2d bbox)
        {
            return Eigen::Vector2d(bbox.x + 0.5 * bbox.width, bbox.y + 0.5 * bbox.height);
        }

        inline cv::Rect2d KalmanFilter::point_to_bbox(Eigen::Vector4d point)
        {
            return cv::Rect2d(x(0) - 0.5 * _bbox.width, x(1) - 0.5 * _bbox.height, _bbox.width, _bbox.height);
        }

        KalmanFilter::KalmanFilter(KalmanFilterParam kf_param)
        {
            Q = Eigen::Matrix4d::Identity() * kf_param.Q_factor;
            P = Eigen::Matrix4d::Identity() * kf_param.P_factor;
            R = Eigen::Matrix2d::Identity() * kf_param.R_factor;

            F = Eigen::Matrix4d::Identity();
            H = Eigen::MatrixXd::Identity(2, 4);
        }

        void KalmanFilter::init(cv::Rect2d bbox)
        {
            _bbox = bbox;
            x = bbox_to_state(bbox);
        }

        void KalmanFilter::update_bbox(cv::Rect2d bbox)
        {
            _bbox = bbox;
        }

        cv::Rect2d KalmanFilter::estimate(double dt)
        {
            F.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity() * dt;
            std::cout << "estimate: dt = " << dt << std::endl;
            std::cout << "estimate: F = \n"
                      << F << std::endl;
            x = F * x;
            std::cout << "estimate: x = \n"
                      << x << std::endl;
            Q(0, 0) = 0.5 * dt * dt;
            Q(1, 1) = dt;
            P = F * P * F.transpose() + Q;
            std::cout << "estimate: P = \n"
                      << P << std::endl;
            return point_to_bbox(x);
        }

        cv::Rect2d KalmanFilter::update(cv::Rect2d measurement)
        {

            Eigen::Vector2d z = bbox_to_measurement(measurement);
            std::cout << "update: z = \n"
                      << z << std::endl;
            //update
            Eigen::Matrix2d S = H * P * H.transpose() + R;
            Eigen::MatrixXd K = P * H.transpose() * S.inverse();
            std::cout << "update: K = \n"
                      << K << std::endl;
            x = x + K * (z - H * x); //update x
            std::cout << "update: x = \n"
                      << x << std::endl;
            P = (Eigen::Matrix4d::Identity() - K * H) * P; //update P
            //finally update bbox
            update_bbox(measurement);
            return point_to_bbox(x);
        }
    } // namespace tracker
} // namespace ptl