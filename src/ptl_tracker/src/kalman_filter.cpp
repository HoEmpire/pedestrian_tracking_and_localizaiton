#include "ptl_tracker/kalman_filter.h"
namespace ptl
{
    namespace tracker
    {

        KalmanFilter::KalmanFilter(const KalmanFilterParam &kf_param) : kf_param_(kf_param)
        {
            F = Eigen::Matrix4d::Identity();
            R_xy = Eigen::Matrix4d::Zero();
            R_xy(0, 0) = kf_param_.r_f;
            R_xy(1, 1) = kf_param_.r_theta;
            R_xy(2, 2) = kf_param_.r_tx;
            R_xy(3, 3) = kf_param_.r_ty;

            R_wh = Eigen::Matrix2d::Zero();
            R_wh(0, 0) = kf_param_.r_f;
            R_wh(1, 1) = kf_param_.r_theta;
        }

        void KalmanFilter::init(const cv::Rect2d &bbox)
        {
            // initilize covariance matrix
            P_xy = Eigen::Matrix4d::Zero();
            P_xy.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity(2, 2) * kf_param_.p_xy_pos;
            P_xy.block<2, 2>(2, 2) = Eigen::Matrix2d::Identity(2, 2) * kf_param_.p_xy_dp;

            P_wh = Eigen::Matrix4d::Zero();
            P_wh.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity(2, 2) * kf_param_.p_wh_size;
            P_wh.block<2, 2>(2, 2) = Eigen::Matrix2d::Identity(2, 2) * kf_param_.p_wh_ds;

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
            return cv::Rect2d(x_xy(0) - x_wh(0) * 0.5, x_xy(1) - x_wh(1) * 0.5, x_wh(0), x_wh(1));
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
            std::cout << "z_xy: " << z_xy << std::endl;
            std::cout << "z_wh: " << z_wh << std::endl;

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
            // first row
            H_xy(0, 0) = -(vx * dt * cos_theta + (y + vy * dt) * sin_theta - tx * pow(cos_theta, 2) - ty * pow(sin_theta, 2)) / pow(x, 2); //d(f)/d(x)
            H_xy(0, 1) = sin_theta / x;                                                                                                    // d(f)/d(y)
            H_xy(0, 2) = cos_theta * dt / x;                                                                                               // d(f)/d(vx)
            H_xy(0, 3) = sin_theta * dt / x;                                                                                               // d(f)/d(vy)

            // second row
            double a_xy = (pow(x, 2) + pow(y, 2)) * (x * vy * dt - vx * y * dt - x * ty + y * tx);
            double b_xy = (pow(x, 2) - pow(y, 2)) * (pow(x, 2) + x * vx * dt - x * tx + pow(y, 2) + y * vy * dt - y * ty) + 1e-6;
            double alpha = a_xy / b_xy;
            double dalpha = 1 / (1 + pow(alpha, 2)); //d(arctan(alpha))/d(alpha)
            //d(theta)/d(x)
            double da_xy_dx = 3 * pow(x, 2) * vy * dt - 2 * x * vx * y * dt - 3 * pow(x, 2) * ty +
                              2 * x * y * tx + pow(y, 2) * vy * dt - pow(y, 2) * ty;
            double db_xy_dx = 4 * pow(x, 3) + 3 * pow(x, 2) * vx * dt - 3 * pow(x, 2) * tx + 2 * x * y * vy * dt -
                              2 * x * y * ty - pow(y, 2) * vx * dt + pow(y, 2) * tx;
            H_xy(1, 0) = dalpha * (da_xy_dx * b_xy - db_xy_dx * a_xy) / pow(b_xy, 2);
            //d(theta)/d(y)
            double da_xy_dy = -pow(x, 2) * vx * dt + pow(x, 2) * tx + 2 * y * x * vy * dt -
                              3 * vx * pow(y, 2) * dt - 2 * y * x * ty + 3 * pow(y, 2) * tx;
            double db_xy_dy = pow(x, 2) * vy * dt - pow(x, 2) * ty - 2 * x * y * vx * dt +
                              2 * x * y * tx - 4 * pow(y, 3) - 3 * pow(y, 2) * vy * dt - 3 * pow(y, 2) * ty;
            H_xy(1, 1) = dalpha * (da_xy_dy * b_xy - db_xy_dy * a_xy) / pow(b_xy, 2);
            //d(theta)/d(vx)
            double da_xy_dvx = -pow(x, 2) * y * dt - pow(y, 3) * dt;
            double db_xy_dvx = pow(x, 3) * dt - x * pow(y, 2) * dt;
            H_xy(1, 2) = dalpha * (da_xy_dvx * b_xy - db_xy_dvx * a_xy) / pow(b_xy, 2);
            //d(theta)/d(vy)
            double da_xy_dvy = pow(x, 3) * dt + pow(y, 2) * x * dt;
            double db_xy_dvy = pow(x, 2) * y * dt - pow(y, 3) * dt;
            H_xy(1, 3) = dalpha * (da_xy_dvy * b_xy - db_xy_dvy * a_xy) / pow(b_xy, 2);

            //third row
            H_xy(2, 0) = 1 - f * cos_theta; //d(tx)/d(x)
            H_xy(2, 1) = f * sin_theta;     //d(tx)/d(y)
            H_xy(2, 2) = dt;                //d(tx)/d(vx)
            H_xy(2, 3) = 0;                 //d(tx)/d(vy)

            //forth row
            H_xy(3, 0) = -f * sin_theta;    //d(ty)/d(x)
            H_xy(3, 1) = 1 - f * cos_theta; //d(ty)/d(y)
            H_xy(3, 2) = 0;                 //d(ty)/d(vx)
            H_xy(3, 3) = dt;                //d(ty)/d(vy)

            //H_wh
            // first row
            H_wh(0, 0) = -(vw * dt * cos_theta + (h + vh * dt) * sin_theta) / pow(w, 2); // d(f)/d(w)
            H_wh(0, 1) = sin_theta / w;                                                  // d(f)/d(h)
            H_wh(0, 2) = cos_theta * dt / w;                                             // d(f)/d(vw)
            H_wh(0, 3) = sin_theta * dt / w;                                             // d(f)/d(vh)

            // second row
            double a_wh = (pow(w, 2) + pow(h, 2)) * (w * vh * dt - vw * h * dt);
            double b_wh = (pow(w, 2) - pow(h, 2)) * (pow(w, 2) + w * vw * dt + pow(h, 2) + h * vh * dt);
            double gama = a_wh / b_wh;
            double dgama = 1 / (1 + pow(gama, 2)); //d(arctan(gama))/d(gama)
            //d(theta)/d(w)
            double da_wh_dw = 3 * pow(w, 2) * vh * dt - 2 * w * vw * h * dt + pow(h, 2) * vh * dt;
            double db_wh_dw = 4 * pow(w, 3) + 3 * pow(w, 2) * vw * dt + 2 * w * h * vh * dt - 2 * x * y * ty - pow(h, 2) * vw * dt;
            H_wh(1, 0) = dgama * (da_wh_dw * b_wh - db_wh_dw * a_wh) / pow(b_wh, 2);
            //d(theta)/d(h)
            double da_wh_dh = -pow(w, 2) * vw * dt + 2 * h * w * vh * dt - 3 * vw * pow(h, 2) * dt;
            double db_wh_dh = pow(w, 2) * vh * dt - 2 * w * h * vw * dt - 4 * pow(h, 3) - 3 * pow(h, 2) * vh * dt;
            H_wh(1, 1) = dgama * (da_wh_dh * b_wh - db_wh_dh * a_wh) / pow(b_wh, 2);
            //d(theta)/d(vw)
            double da_wh_dvw = -pow(w, 2) * h * dt - pow(h, 3) * dt;
            double db_wh_dvw = pow(w, 3) * dt - w * pow(h, 2) * dt;
            H_wh(1, 2) = dgama * (da_wh_dvw * b_wh - db_wh_dvw * a_wh) / pow(b_wh, 2);
            //d(theta)/d(vh)
            double da_wh_dvy = pow(w, 3) * dt + pow(h, 2) * w * dt;
            double db_wh_dvy = pow(w, 2) * h * dt - pow(h, 3) * dt;
            H_wh(1, 3) = dgama * (da_wh_dvy * b_wh - db_wh_dvy * a_wh) / pow(b_wh, 2);

            std::cout << "H_xy: " << H_xy << std::endl;
            std::cout << "H_wh: " << H_wh << std::endl;

            //calculate K
            Eigen::MatrixXd K_xy, K_wh;
            K_xy = P_xy * H_xy.transpose() * (H_xy * P_xy * H_xy.transpose() + R_xy).inverse();
            K_wh = P_wh * H_wh.transpose() * (H_wh * P_wh * H_wh.transpose() + R_wh).inverse();

            std::cout << "K_xy: " << K_xy << std::endl;
            std::cout << "K_wh： " << K_wh << std::endl;

            //calculate predicted measurement
            double f_predicted_xy = cos_theta + (vx * dt * cos_theta + (y + vy * dt) * sin_theta - tx * pow(cos_theta, 2) - ty * pow(sin_theta, 2)) / x;
            double theta_predicted_xy = atan(alpha);
            double tx_predicted_xy = x + vx * dt - f * cos_theta * x + f * sin_theta * y;
            double ty_predicted_xy = y + vy * dt - f * sin_theta * x - f * cos_theta * y;

            double f_predicted_wh = cos_theta + (vw * dt * cos_theta + (h + vh * dt) * sin_theta) / w;
            double theta_predicted_wh = atan(gama);

            //update x_xy, x_wh
            std::cout << "x_xy before: " << x_xy << std::endl;
            std::cout << "x_wh before: " << x_wh << std::endl;
            Eigen::Vector4d resiual_xy = z_xy - Eigen::Vector4d(f_predicted_xy, theta_predicted_xy, tx_predicted_xy, ty_predicted_xy);
            Eigen::Vector2d resiual_wh = z_wh - Eigen::Vector2d(f_predicted_wh, theta_predicted_wh);
            std::cout << "residual_xy: " << resiual_xy << std::endl;
            std::cout << "residual_wh: " << resiual_wh << std::endl;

            // wrong measurement
            if (sqrt(pow(resiual_xy(2), 2) + pow(resiual_xy(3), 2)) > kf_param_.residual_threshold)
            {
                std::cout << "wrong measurement!!" << std::endl;
                return cv::Rect2d(x_xy(0) - x_wh(0) * 0.5, x_xy(1) - x_wh(1) * 0.5, x_wh(0), x_wh(1));
            }

            x_xy = x_xy + K_xy * resiual_xy;
            x_wh = x_wh + K_wh * resiual_wh;

            std::cout << "x_xy after: " << x_xy << std::endl;
            std::cout << "x_wh after: " << x_wh << std::endl;

            //update covariance matrix
            P_xy = (Eigen::Matrix4d::Identity() - K_xy * H_xy) * P_xy;
            P_wh = (Eigen::Matrix4d::Identity() - K_wh * H_wh) * P_wh;
            std::cout << "P_xy after: " << P_xy << std::endl;
            std::cout << "P_wh after: " << P_wh << std::endl;

            // std::cout << "update: z = \n"
            //           << z << std::endl;
            // std::cout << "update: K = \n"
            //           << K << std::endl;
            // std::cout << "update: x = \n"
            //           << x << std::endl;
            return cv::Rect2d(x_xy(0) - x_wh(0) * 0.5, x_xy(1) - x_wh(1) * 0.5, x_wh(0), x_wh(1));
        }

        cv::Rect2d KalmanFilter::predict_only(const double dt)
        {
            return cv::Rect2d(x_xy(0) + x_xy(2) * dt + (x_wh(0) + x_wh(2) * dt) / 2,
                              x_xy(1) + x_xy(3) * dt + (x_wh(1) + x_wh(3) * dt) / 2,
                              x_wh(0) + x_wh(2) * dt, x_wh(1) + x_wh(3) * dt);
        }

    } // namespace tracker
} // namespace ptl