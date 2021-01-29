#include "ptl_tracker/kalman_filter.h"
namespace ptl
{
    namespace tracker
    {

        KalmanFilter::KalmanFilter(const KalmanFilterParam &kf_param) : kf_param_(kf_param)
        {
            F = Eigen::Matrix4d::Identity();
            H = Eigen::MatrixXd::Identity(2, 4);
        }

        void KalmanFilter::init(const cv::Rect2d &bbox)
        {
            // initilize covariance matrix
            P_xy = Eigen::Matrix4d::Identity() * kf_param_.p_xy;
            P_wh = Eigen::Matrix4d::Identity() * kf_param_.p_wh;

            //initialize state
            x_xy = Eigen::Vector4d(bbox.x + 0.5 * bbox.width, bbox.y + 0.5 * bbox.height, 0, 0);
            x_wh = Eigen::Vector4d(bbox.width, bbox.height, 0, 0);
        }

        cv::Rect2d KalmanFilter::predict(const double dt)
        {
            F.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity() * dt;
            // std::cout << "before estimate: x = \n"
            //           << x << std::endl;

            //update state
            x_xy = F * x_xy;
            x_wh = F * x_wh;

            Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();

            //random acceleration model
            Q(0, 0) = Q(1, 1) = pow(dt, 4) * 0.25;
            Q(2, 2) = Q(3, 3) = pow(dt, 2);
            Q(0, 2) = Q(2, 0) = Q(1, 3) = Q(3, 1) = pow(dt, 3) * 0.5;

            //update covariance
            P_xy = F * P_xy * F.transpose() + Q * kf_param_.q_xy;
            P_wh = F * P_wh * F.transpose() + Q * kf_param_.q_xy;

            // std::cout << "estimate: dt = " << dt << std::endl;
            // std::cout << "estimate: F = \n"
            //           << F << std::endl;
            // std::cout << "estimate: x = \n"
            //           << x << std::endl;
            // std::cout << "estimate: P = \n"
            //           << P << std::endl;
            return cv::Rect2d(x_xy(0), x_xy(1), x_wh(0), x_wh(1));
        }

        cv::Rect2d KalmanFilter::update(const cv::Mat &T_mat, const double dt)
        {
            //T_mat = [ f * cos(theta) , -f * sin(theta), tx
            //          f * sin(theta) ,  f * cos(theta), ty]
            double f = sqrt(pow(T_mat.at<double>(0, 0), 2) + pow(T_mat.at<double>(0, 1), 2));
            double theta = atan(T_mat.at<double>(1, 0) / T_mat.at<double>(1, 1));
            double tx = T_mat.at<double>(0, 2);
            double ty = T_mat.at<double>(1, 2);
            double sin_theta = T_mat.at<double>(1, 0) / f;
            double cos_theta = T_mat.at<double>(1, 1) / f;

            // get measurement
            Eigen::Vector4d z_xy = Eigen::Vector4d(f, theta, tx, ty); // z = [f, theat, tx, ty]
            Eigen::Vector2d z_wh = Eigen::Vector2d(f, theta);         // z = [f, theat]

            // get measurement matrix by calculating jacobian
            Eigen::MatrixXd H_xy, H_wh;
            H_xy = Eigen::MatrixXd::Zero(4, 4);
            H_wh = Eigen::MatrixXd::Zero(2, 4);

            //for clarity, we use the variable with the same name instead of the Eigen index
            double x = x_xy(0);
            double y = x_xy(1);
            double vx = x_xy(2);
            double vy = x_xy(3);
            double w = x_wh(0);
            double h = x_wh(1);
            double vw = x_wh(2);
            double vh = x_wh(3);

            // H_xy
            // 1 row
            H_xy(0, 0) = -(vx * dt * cos_theta + (y + vy * dt) * sin_theta - tx * pow(cos_theta, 2) - ty * pow(sin_theta, 2)) / pow(x, 2); //d(f)/d(x)
            H_xy(0, 1) = sin_theta / x;                                                                                                    // d(f)/d(y)
            H_xy(0, 2) = cos_theta * dt / x;                                                                                               // d(f)/d(vx)
            H_xy(0, 3) = sin_theta * dt / x;                                                                                               // d(f)/d(vy)

            
            Eigen::MatrixXd K;
            //update xy

            K = P_xy * H.transpose() * (H * P_xy * H.transpose() +).inverse();
            x_xy = x_xy + K * (z_xy - H * x_xy);                 //update x
            P_xy = (Eigen::Matrix4d::Identity() - K * H) * P_xy; //update P

            // std::cout << "update: z = \n"
            //           << z << std::endl;
            // std::cout << "update: K = \n"
            //           << K << std::endl;
            // std::cout << "update: x = \n"
            //           << x << std::endl;
            return cv::Rect2d(x_xy(0), x_xy(1), x_wh(0), x_wh(1));
        }

        cv::Rect2d KalmanFilter::predict_only(const double dt)
        {
            return cv::Rect2d(x_xy(0) + x_xy(2) * dt, x_xy(1) + x_xy(3) * dt,
                              x_wh(0) + x_wh(2) * dt, x_wh(1) + x_wh(3) * dt)
        }
    } // namespace tracker
} // namespace ptl