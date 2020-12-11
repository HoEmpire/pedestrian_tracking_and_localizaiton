#include "ptl_tracker/local_object.h"
namespace ptl
{
    namespace tracker
    {
        LocalObject::LocalObject(int id_init, cv::Rect2d bbox_init, cv::Mat frame)
        {
            id = id_init;
            bbox = bbox_init;
            tracking_fail_count = 0;
            dssttracker = new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
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
            dssttracker->init(frame, bbox);
        }

    } // namespace tracker

} // namespace ptl
