#include <ptl_tracker/optical_flow.h>
namespace ptl
{
    namespace tracker
    {
        //TODO add resize
        void OpticalFlow::update(const cv::Mat &frame_curr_bgr, std::vector<LocalObject> &local_objects)
        {
            cv::Mat frame_curr;
            cv::cvtColor(frame_curr_bgr, frame_curr, cv::COLOR_BGR2GRAY);
            //add first frame
            if (frame_pre_.empty() || local_objects.empty())
            {

                frame_pre_ = frame_curr;
                for (auto &lo : local_objects)
                {
                    lo.is_track_succeed = false;
                }
                return;
            }

            //detect enought keypoints for each tracking object,
            // and forms a keypoints vector that conatins all the keypoints
            std::vector<cv::Point2f> keypoints_pre_all, keypoints_curr_all;
            ROS_INFO("FUCK0");
            detect_enough_keypoints(local_objects, keypoints_pre_all);
            ROS_INFO("FUCK1");

            //track keypoints by optical flow
            std::vector<uchar> status; // tracking succeed or not
            std::vector<float> errors; // tracking error
            if (!keypoints_pre_all.empty())
            {
                cv::calcOpticalFlowPyrLK(frame_pre_, frame_curr, keypoints_pre_all, keypoints_curr_all, status, errors);
                // for (int i = 0; i < keypoints_curr_all.size(); i++)
                // {
                //     std::cout << keypoints_pre_all[i] << std::endl;
                //     std::cout << keypoints_curr_all[i] << std::endl;
                // }
                ROS_INFO("FUCK2");

                //calculate homography matrix to compensate camera motion
                camera_motion_compensate(keypoints_curr_all, status);

                //remove the keypoints that fails to track,
                //and update the current keypoints of the local object
                update_local_objects_curr_kp(local_objects, keypoints_curr_all, status);
                ROS_INFO("FUCK3");
                //calculate the transform matrix and remove the outliers
                calculate_measurement(local_objects);
                ROS_INFO("FUCK4");
                //update variable
                frame_pre_ = frame_curr;
                for (auto &lo : local_objects)
                {
                    lo.keypoints_pre = lo.keypoints_curr;
                }
            }
            else
            {
                frame_pre_ = frame_curr;
                for (auto &lo : local_objects)
                {
                    lo.is_track_succeed = false;
                }
                return;
            }
        }

        void OpticalFlow::detect_enough_keypoints(std::vector<LocalObject> &local_objects, std::vector<cv::Point2f> &keypoints_all)
        {
            for (auto &lo : local_objects)
            {

                ROS_INFO("FUCK0.1");
                // detect fast keypoints for the objects with too few succefully tracked keypoints
                if (lo.keypoints_pre.size() < optical_flow_param_.min_keypoints_to_track * min_keypoints_num_factor(lo.bbox))
                {
                    ROS_INFO("FUCK0.2");
                    cv::goodFeaturesToTrack(frame_pre_(lo.bbox), lo.keypoints_pre, optical_flow_param_.corner_detector_max_num,
                                            optical_flow_param_.corner_detector_quality_level, optical_flow_param_.corner_detector_min_distance,
                                            cv::noArray(), optical_flow_param_.corner_detector_block_size,
                                            optical_flow_param_.corner_detector_use_harris, optical_flow_param_.corner_detector_k);
                    //add  offset
                    cv::Point2f tmp(lo.bbox.x, lo.bbox.y);
                    for (auto &p : lo.keypoints_pre)
                    {
                        p = p + tmp;
                    }

                    ROS_INFO("FUCK0.3");
                }
                keypoints_all.insert(keypoints_all.end(), lo.keypoints_pre.begin(), lo.keypoints_pre.end());
            }

            //detect the keypoints for tracking the motion of the camera
            if (keypoints_vo_pre.size() < optical_flow_param_.min_keypoints_for_motion_estimation)
            {
                cv::goodFeaturesToTrack(frame_pre_, keypoints_vo_pre, optical_flow_param_.corner_detector_max_num,
                                        optical_flow_param_.corner_detector_quality_level, optical_flow_param_.corner_detector_min_distance,
                                        cv::noArray(), optical_flow_param_.corner_detector_block_size,
                                        optical_flow_param_.corner_detector_use_harris, optical_flow_param_.corner_detector_k);
            }
            keypoints_all.insert(keypoints_all.end(), keypoints_vo_pre.begin(), keypoints_vo_pre.end());
            std::cout << keypoints_all.size() << std::endl;
        }

        void OpticalFlow::camera_motion_compensate(std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status)
        {
            //deal with keypoints used to calculate the Homography matrix between two frame
            keypoints_vo_curr.clear();
            for (int i = keypoints_all.size() - keypoints_vo_pre.size(); i < keypoints_all.size(); i++)
            {
                if (int(status[i]) == 1)
                {
                    keypoints_vo_curr.push_back(keypoints_all[i]);
                }
                else
                {
                    keypoints_vo_pre.erase(keypoints_vo_pre.begin() + keypoints_vo_curr.size());
                }
            }

            std::vector<uchar> inliers;
            // if the successfully tracked keypoints is too few, we takes it as a failure in tracking
            if (keypoints_vo_curr.size() < optical_flow_param_.min_keypoints_to_cal_H_mat)
            {
                std::cout << "Too few points, estimate homography matrix fails..." << std::endl;
                is_motion_estimation_succeeed = false;
                keypoints_vo_curr.clear();
                return;
            }
            else
            {
                H_motion = cv::findHomography(keypoints_vo_pre, keypoints_vo_curr, inliers, cv::RANSAC);
                //empty matrix means fail to estimate the transform matrix
                if (H_motion.empty())
                {
                    std::cout << "Estimate homography matrix fails..." << std::endl;
                    is_motion_estimation_succeeed = false;
                    return;
                }
                else
                { //estimate succeed
                    is_motion_estimation_succeeed = true;
                    //get transformed bbox
                    std::cout << "Motion compensation matrix" << H_motion << std::endl;

                    //remove outliers
                    int i = 0;
                    for (auto kpc = keypoints_vo_curr.begin(); kpc != keypoints_vo_curr.end();)
                    {
                        //outlier detected
                        if (int(inliers[i]) == 0)
                        {
                            kpc = keypoints_vo_curr.erase(kpc); //remove outlier
                        }
                        else
                        {
                            kpc++;
                        }
                        i++;
                    }

                    //update the motion estimation keypoints
                    keypoints_vo_pre = keypoints_vo_curr;
                }
            }
        }

        void OpticalFlow::update_local_objects_curr_kp(std::vector<LocalObject> &local_objects, std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status)
        {
            int start = 0, end = 0, i;
            for (auto &lo : local_objects)
            {
                lo.keypoints_curr.clear();
                end += lo.keypoints_pre.size();
                for (i = start; i < end; i++)
                {
                    if (int(status[i]) == 1)
                    {
                        if (is_motion_estimation_succeeed)
                        {
                            if (!is_scene_points(lo.keypoints_pre[lo.keypoints_curr.size()], keypoints_all[i]))
                            {
                                lo.keypoints_curr.push_back(keypoints_all[i]);
                            }
                            else
                            {
                                lo.keypoints_pre.erase(lo.keypoints_pre.begin() + lo.keypoints_curr.size());
                            }
                        }
                        else
                        {
                            lo.keypoints_curr.push_back(keypoints_all[i]);
                        }
                    }
                    else
                    {
                        lo.keypoints_pre.erase(lo.keypoints_pre.begin() + lo.keypoints_curr.size());
                    }
                }
                start = end;
            }
        }

        void OpticalFlow::calculate_measurement(std::vector<LocalObject> &local_objects)
        {
            for (auto &lo : local_objects)
            {
                std::vector<uchar> inliers;
                // if the successfully tracked keypoints is too few, we takes it as a failure in tracking
                if (lo.keypoints_curr.size() < optical_flow_param_.min_keypoints_to_cal_H_mat)
                {
                    std::cout << "Too few points, estimate affine partial matrix fails..." << std::endl;
                    lo.is_track_succeed = false;
                    lo.keypoints_curr.clear(); //clear all the points
                    continue;
                }
                else
                {
                    ROS_INFO("FUCK3.0");
                    std::cout << lo.keypoints_pre.size() << std::endl;
                    std::cout << lo.keypoints_curr.size() << std::endl;
                    cv::Mat H = cv::estimateAffinePartial2D(lo.keypoints_pre, lo.keypoints_curr, inliers, cv::RANSAC);
                    ROS_INFO("FUCK3.1");
                    //empty matrix means fail to estimate the transform matrix
                    if (H.empty())
                    {
                        std::cout << "Estimate affine partial matrix fails..." << std::endl;
                        lo.is_track_succeed = false;
                        continue;
                    }
                    //estimate succeed
                    else
                    {
                        lo.is_track_succeed = true;
                        //get transformed bbox
                        std::cout << H << std::endl;
                        lo.T_measurement = H;

                        //remove outliers
                        int i = 0;
                        for (auto lo_kpc = lo.keypoints_curr.begin(); lo_kpc != lo.keypoints_curr.end();)
                        {
                            //outlier detected
                            if (int(inliers[i]) == 0)
                            {
                                lo_kpc = lo.keypoints_curr.erase(lo_kpc); //remove outlier
                            }
                            else
                            {
                                lo_kpc++;
                            }
                            i++;
                        }
                    }
                }
            }
        }

        inline double OpticalFlow::min_keypoints_num_factor(const cv::Rect2d &bbox)
        {
            return (bbox.area() / optical_flow_param_.keypoints_num_factor_area);
        }

        inline cv::Rect2d OpticalFlow::transform_bbox(const cv::Mat &H, const cv::Rect2d &bbox_pre)
        {
            double x1, y1, x2, y2;
            x1 = bbox_pre.tl().x * H.at<double>(0, 0) + bbox_pre.tl().y * H.at<double>(0, 1) + H.at<double>(0, 2);
            x2 = bbox_pre.br().x * H.at<double>(0, 0) + bbox_pre.br().y * H.at<double>(0, 1) + H.at<double>(0, 2);
            y1 = bbox_pre.tl().x * H.at<double>(1, 0) + bbox_pre.tl().y * H.at<double>(1, 1) + H.at<double>(1, 2);
            y2 = bbox_pre.br().x * H.at<double>(1, 0) + bbox_pre.br().y * H.at<double>(1, 1) + H.at<double>(1, 2);
            return cv::Rect2d(cv::Point2d(x1, y1), cv::Point2d(x2, y2));
        }

        inline bool OpticalFlow::is_scene_points(const cv::Point2f &kp_pre, const cv::Point2f &kp_curr)
        {
            double result = pow(kp_pre.x * H_motion.at<double>(0, 0) + kp_pre.y * H_motion.at<double>(0, 1) + H_motion.at<double>(0, 2) - kp_curr.x, 2) +
                            pow(kp_pre.x * H_motion.at<double>(1, 0) + kp_pre.y * H_motion.at<double>(1, 1) + H_motion.at<double>(1, 2) - kp_curr.y, 2);
            // std::cout << result << std::endl;
            // std::cout << H_motion << std::endl;
            // std::cout << kp_pre << std::endl;
            // std::cout << kp_curr << std::endl;
            if (result < optical_flow_param_.min_pixel_dis_square_for_scene_point)
                return true;
            else
                return false;
        }
    } // namespace tracker

} // namespace ptl