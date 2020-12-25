#include "ptl_tracker/local_object.h"
namespace ptl
{
    namespace tracker
    {
        inline float cal_reid_score(Eigen::VectorXf query, Eigen::VectorXf gallery)
        {
            return (query - gallery).squaredNorm();
        }

        LocalObject::LocalObject(int id_init, cv::Rect2d bbox_init, cv::Mat frame,
                                 Eigen::VectorXf feat, float tracker_success_param)
        {
            id = id_init;
            bbox = bbox_init;
            features.push_back(feat);
            tracker_success_threshold = tracker_success_param;
            tracking_fail_count = 0;
            overlap_count = 0;
            dssttracker = new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            if (DSST)
            {
                dssttracker->detect_thresh_dsst = tracker_success_threshold;
                dssttracker->scale_step = 1;
            }
            else
            {
                dssttracker->detect_thresh_kcf = tracker_success_threshold;
                dssttracker->padding = 2.5;
            }

            dssttracker->init(frame, bbox);
            cv::RNG rng(std::time(0));
            color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            time.tic();
        }

        void LocalObject::update_tracker(cv::Mat frame)
        {
            is_track_succeed = dssttracker->update(frame, bbox);
            if (is_track_succeed)
            {
                tracking_fail_count = 0;
            }
            else
            {
                tracking_fail_count++;
                ROS_INFO_STREAM("Object " << id << " tracking failure detected!");
            }
        }

        void LocalObject::reinit(cv::Rect2d bbox_init, cv::Mat frame)
        {
            bbox = bbox_init;
            tracking_fail_count = 0;
            dssttracker = new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            if (DSST)
            {
                dssttracker->detect_thresh_dsst = tracker_success_threshold;
                dssttracker->scale_step = 1;
            }
            else
            {
                dssttracker->detect_thresh_kcf = tracker_success_threshold;
                dssttracker->padding = 2.5;
            }
            dssttracker->init(frame, bbox);
        }

        float LocalObject::find_min_query_score(Eigen::VectorXf query)
        {
            float min_score = 100000.0;
            for (auto feat : features)
            {
                float score = cal_reid_score(query, feat);
                if (score < min_score)
                {
                    min_score = score;
                }
            }
            return min_score;
        }
    } // namespace tracker

} // namespace ptl
