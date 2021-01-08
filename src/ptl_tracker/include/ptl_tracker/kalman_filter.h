#include <opencv/cv.hpp>
#include <Eigen/Dense>
namespace ptl
{
    namespace tracker
    {
        struct KalamFilterParam
        {
            double Q_factor;
            double R_factor;
            double P_factor;
        };

        class KalmanFilter
        {
        public:
            KalmanFilter(double Q_factor, double R_factor, double P_factor);
            void init(cv::Rect2d bbox);
            void update_bbox(cv::Rect2d bbox);
            cv::Rect2d estimate(double time);
            cv::Rect2d update(cv::Rect2d measurement);

            // ensure the bbox height and width is the latest one
            // especially in estimate step of tracker
            Eigen::Vector4d x;

        private:
            inline Eigen::Vector4d bbox_to_state(cv::Rect2d bbox);
            inline Eigen::Vector2d bbox_to_measurement(cv::Rect2d bbox);
            inline cv::Rect2d point_to_bbox(Eigen::Vector4d point);

            Eigen::Matrix4d Q, P, F;
            Eigen::Matrix2d R;
            Eigen::MatrixXd H;
            cv::Rect2d _bbox;
        };

    } // namespace tracker
} // namespace ptl