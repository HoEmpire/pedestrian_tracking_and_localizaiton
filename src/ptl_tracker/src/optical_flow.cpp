#include <ptl_tracker/optical_flow.h>
namespace ptl
{
    namespace tracker
    {
        OpticalFlow::OpticalFlow(const OpticalFlowParam &optical_flow_param)
        {
            optical_flow_param_ = optical_flow_param;
        }

        void OpticalFlow::update(const cv::Mat &frame_curr, std::vector<LocalObject> &local_objects)
        {
            //add first frame
            if (frame_pre_.empty())
            {
                frame_pre_ = frame_curr;
                return;
            }

            //detect enought keypoints for each tracking object,
            // and forms a keypoints vector that conatins all the keypoints
            std::vector<cv::Point2f> keypoints_pre_all, keypoints_curr_all;
            detect_enough_keypoints(frame_curr, local_objects, keypoints_curr_all);

            //track keypoints by optical flow
            std::vector<uchar> status; // tracking succeed or not
            std::vector<float> errors; // tracking error
            cv::calcOpticalFlowPyrLK(frame_pre_, frame_curr, keypoints_pre_all, keypoints_curr_all, status, errors);

            //remove the keypoints that fails to track,
            //and update the current keypoints of the local object
            update_local_objects_curr_kp(local_objects, keypoints_pre_all, status);

            //calculate the transform matrix and remove the outliers
            calculate_measurement(local_objects);

            //update variable
            frame_pre_ = frame_curr;
            for (auto &lo : local_objects)
            {
                lo.keykoints_pre = lo.keypoints_curr;
            }
        }

        void OpticalFlow::detect_enough_keypoints(const cv::Mat &frame_curr, std::vector<LocalObject> &local_objects, std::vector<cv::Point2f> &keypoints_all)
        {
            std::vector<cv::KeyPoint> keypoints_tmp;
            for (auto &lo : local_objects)
            {
                // detect fast keypoints for the objects with too few succefully tracked keypoints
                if (lo.keykoints_pre.size() > optical_flow_param_.min_keypoints_to_track * min_keypoints_num_factor(lo.bbox))
                {
                    cv::goodFeaturesToTrack(frame_curr(lo.bbox), lo.keykoints_pre, optical_flow_param_.corner_detector_max_num,
                                            optical_flow_param_.corner_detector_quality_level, optical_flow_param_.corner_detector_min_distance,
                                            cv::noArray(), optical_flow_param_.corner_detector_block_size,
                                            optical_flow_param_.corner_detector_use_harris, optical_flow_param_.corner_detector_k);
                    cv::KeyPoint::convert(keypoints_tmp, lo.keykoints_pre);
                }
                keypoints_all.insert(keypoints_all.end(), lo.keykoints_pre.begin(), lo.keykoints_pre.end());
            }
        }

        void OpticalFlow::update_local_objects_curr_kp(std::vector<LocalObject> &local_objects, std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status)
        {
            int i = 0;
            for (auto kp = keypoints_all.begin(); kp != keypoints_all.end();)
            {
                for (auto &lo : local_objects)
                {
                    // process one local object
                    auto kp_start_for_this_local_object = kp;
                    for (auto lo_kp = lo.keykoints_pre.begin(); lo_kp != lo.keykoints_pre.end(); lo_kp++)
                    {
                        //tracking fails
                        if (status[i] == 0)
                        {
                            lo_kp = lo.keykoints_pre.erase(lo_kp);
                            kp = keypoints_all.erase(kp);
                        }
                        else
                        {
                            lo_kp++; //track succeed, just skip to next keypoints
                            kp++;
                        }
                        i++;
                    }

                    //update the current keypoints of the local object
                    lo.keypoints_curr.insert(lo.keypoints_curr.begin(), kp_start_for_this_local_object,
                                             kp_start_for_this_local_object + lo.keykoints_pre.size());
                }
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
                    return;
                }
                else
                {
                    cv::Mat H = cv::estimateAffinePartial2D(lo.keykoints_pre, lo.keypoints_curr, inliers, cv::RANSAC);
                    //empty matrix means fail to estimate the transform matrix
                    if (H.empty())
                    {
                        std::cout << "Estimate affine partial matrix fails..." << std::endl;
                        lo.is_track_succeed = false;
                        return;
                    }
                    //estimate succeed
                    else
                    {
                        lo.is_track_succeed = true;
                        //get transformed bbox
                        lo.bbox_optical_flow = transform_bbox(H, lo.bbox);

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
    } // namespace tracker

} // namespace ptl