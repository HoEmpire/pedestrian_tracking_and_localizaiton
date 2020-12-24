#include "ptl_tracker/local_object.h"
namespace ptl
{
    namespace tracker
    {
        LocalObject::LocalObject(int id_init, cv::Rect2d bbox_init, cv::Mat frame, float tracker_success_param)
        {
            id = id_init;
            bbox = bbox_init;
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

    } // namespace tracker

} // namespace ptl
