#include <math.h>

#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <ptl_tracker/local_object.h>

namespace ptl
{
    namespace tracker
    {
        struct OpticalFlowParam
        {
            //an pixel area of 100 should have more than 30 keypoints
            int min_keypoints_to_track = 30;
            double keypoints_num_factor_area = 8000;

            //corner detector params
            int corner_detector_max_num = 100;
            double corner_detector_quality_level = 0.0001;
            double corner_detector_min_distance = 5;
            int corner_detector_block_size = 3;
            bool corner_detector_use_harris = true;
            double corner_detector_k = 0.03; //harris param

            int min_keypoints_to_cal_H_mat = 10;

            int min_keypoints_for_motion_estimation = 50;
            int min_pixel_dis_square_for_scene_point = 4; //pixel dis = 2
        };

        class OpticalFlow
        {
        public:
            OpticalFlow() = default;
            OpticalFlow(const OpticalFlowParam &optical_flow_param) : optical_flow_param_(optical_flow_param) {}
            void update(const cv::Mat &frame_curr, std::vector<LocalObject> &local_objects);

        private:
            void detect_enough_keypoints(std::vector<LocalObject> &local_objects, std::vector<cv::Point2f> &keypoints_all);
            void camera_motion_compensate(std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status);
            void update_local_objects_curr_kp(std::vector<LocalObject> &local_objects, std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status);
            void calculate_measurement(std::vector<LocalObject> &local_objects);

            inline double min_keypoints_num_factor(const cv::Rect2d &bbox);
            inline cv::Rect2d transform_bbox(const cv::Mat &H, const cv::Rect2d &bbox_pre);
            inline bool is_scene_points(const cv::Point2f &kp_pre, const cv::Point2f &kp_curr);

            cv::Mat frame_pre_;
            OpticalFlowParam optical_flow_param_;

            std::vector<cv::Point2f> keypoints_vo_pre, keypoints_vo_curr;
            cv::Mat H_motion;

            bool is_motion_estimation_succeeed = false;
        };

    } // namespace tracker
} // namespace ptl