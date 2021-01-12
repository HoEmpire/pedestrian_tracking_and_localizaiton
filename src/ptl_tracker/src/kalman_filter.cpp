#include "ptl_tracker/kalman_filter.h"
namespace ptl
{
    namespace tracker
    {

        KalmanFilter::KalmanFilter(KalmanFilterParam kf_param)
        {
            _kf_param = kf_param;
            P_pos = Eigen::Matrix4d::Identity() * _kf_param.p_init_pos;
            P_size = Eigen::Matrix4d::Identity() * _kf_param.p_init_size;

            F = Eigen::Matrix4d::Identity();
            H = Eigen::MatrixXd::Identity(2, 4);
        }

        void KalmanFilter::init(cv::Rect2d bbox)
        {
            _bbox = bbox;
            init_state();
        }

        void KalmanFilter::update_bbox()
        {
            _bbox = cv::Rect2d(x_pos(0), x_pos(1), x_size(0), x_size(1));
        }

        cv::Rect2d KalmanFilter::predict(double dt)
        {
            //get state matrix (same one)
            F.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity() * dt;
            Eigen::Matrix4d Q_pos = Eigen::Matrix4d::Identity(),
                            Q_size = Eigen::Matrix4d::Identity();
            Q_pos(0, 0) = 0.5 * dt * dt;
            Q_pos(1, 1) = dt;
            Q_size(0, 0) = 0.5 * dt * dt;
            Q_size(1, 1) = dt;
            Q_pos *= _kf_param.q_pos;
            Q_size *= _kf_param.q_size;

            //predict x_pos
            x_pos = F * x_pos;
            P_pos = F * P_pos * F.transpose() + Q_pos;

            //predict x_size
            x_size = F * x_size;
            P_size = F * P_size * F.transpose() + Q_size;

            std::cout << "estimate: dt = " << dt << std::endl;
            std::cout << "estimate: F = \n"
                      << F << std::endl;

            std::cout << "estimate: x_pos = \n"
                      << x_pos << std::endl;
            std::cout << "estimate: x_size = \n"
                      << x_size << std::endl;

            std::cout << "estimate: P_pos = \n"
                      << P_pos << std::endl;
            std::cout << "estimate: P_size = \n"
                      << P_size << std::endl;

            update_bbox();
            return _bbox;
        }

        cv::Rect2d KalmanFilter::update(cv::Rect2d measurement, bool is_tracker)
        {
            Eigen::Matrix2d R_pos = Eigen::Matrix2d::Identity(),
                            R_size = Eigen::Matrix2d::Identity();
            if (is_tracker)
            {
                R_pos *= _kf_param.r_pos_tracker;
                R_size *= _kf_param.r_size_tracker;
            }
            else
            {
                R_pos *= _kf_param.r_pos_detector;
                R_size *= _kf_param.r_size_detector;
            }

            get_measurement(measurement);

            //update
            Eigen::MatrixXd K_pos = P_pos * H.transpose() * (H * P_pos * H.transpose() + R_pos).inverse();
            Eigen::MatrixXd K_size = P_size * H.transpose() * (H * P_size * H.transpose() + R_size).inverse();

            x_pos = x_pos + K_pos * (z_pos - H * x_pos);                  //update x_pos
            x_size = x_size + K_size * (z_size - H * x_size);             //update x_size
            P_pos = (Eigen::Matrix4d::Identity() - K_pos * H) * P_pos;    //update P_pos
            P_size = (Eigen::Matrix4d::Identity() - K_size * H) * P_size; //update P

            //finally update bbox
            update_bbox();
            std::cout << "update: z_pos = \n"
                      << z_pos << std::endl;
            std::cout << "update: z_size = \n"
                      << z_size << std::endl;
            std::cout << "update: K_pos = \n"
                      << K_pos << std::endl;
            std::cout << "update: K_size = \n"
                      << K_size << std::endl;
            std::cout << "update: x_pos = \n"
                      << x_pos << std::endl;
            std::cout << "update: x_size = \n"
                      << x_size << std::endl;
            return _bbox;
        }

        void KalmanFilter::init_state()
        {
            x_pos = Eigen::Vector4d(_bbox.x, _bbox.y, 0, 0);
            x_size = Eigen::Vector4d(_bbox.width, _bbox.height, 0, 0);
        }

        void KalmanFilter::get_measurement(cv::Rect2d bbox)
        {
            z_pos = Eigen::Vector2d(bbox.x, bbox.y);
            z_size = Eigen::Vector2d(bbox.width, bbox.height);
        }
    } // namespace tracker
} // namespace ptl