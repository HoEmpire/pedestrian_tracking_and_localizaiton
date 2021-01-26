#include "ptl_tracker/local_object.h"
namespace ptl
{
    namespace tracker
    {
        inline float cal_reid_score(const Eigen::VectorXf &query, const Eigen::VectorXf &gallery)
        {
            return (query - gallery).squaredNorm();
        }

        LocalObject::LocalObject(const int id_init, const cv::Rect2d &bbox_init, const cv::Mat &frame,
                                 const Eigen::VectorXf &feat, const TrackerParam &track_param_init, const KalmanFilterParam &kf_param_init,
                                 const KalmanFilter3dParam &kf3d_param_init, const ros::Time &time_now)
        {
            id = id_init;
            bbox = bbox_init;
            features.push_back(feat);
            tracker_param = track_param_init;

            tracking_fail_count = 0;
            detector_update_count = 0;

            overlap_count = 0;

            kf = new KalmanFilter(kf_param_init);
            kf_3d = new KalmanFilter3d(kf3d_param_init);
            kf->init(bbox_init);
            bbox_last_update_time = time_now;

            cv::RNG rng(std::time(0));
            color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }

        void LocalObject::track_bbox_by_optical_flow(const ros::Time &time_now)
        {
            //predict the bbox at current timestamp by kalman filter
            bbox = kf->estimate((time_now - bbox_last_update_time).toSec());

            // if track succeed, use kalman filter to update the
            if (is_track_succeed)
            {
                bbox = kf->update(bbox_optical_flow);
                tracking_fail_count = 0;
            }
            else
            {
                tracking_fail_count++;
                ROS_INFO_STREAM("Object " << id << " tracking failure detected!");
            }

            //update the bbox last updated time
            bbox_last_update_time = time_now;

            //increase the ticks after last update by detector
            detector_update_count++;
        }

        void LocalObject::track_bbox_by_detector(const cv::Rect2d &bbox_detector, const ros::Time &update_time)
        {
            //re-initialized the important state and data
            tracking_fail_count = 0;
            detector_update_count = 0;
            keykoints_pre.clear();

            //update by kalman filter to the timestamp of the detector
            bbox = kf->update(bbox_detector);
        }

        void LocalObject::update_3d_tracker(const geometry_msgs::Point &measurement, const ros::Time &time_now)
        {
            if (!kf_3d->is_init)
            {
                kf_3d->init(measurement);
                position = measurement;
                ros_time_pc_last = time_now; //update time
                return;
            }

            kf_3d->estimate((time_now - ros_time_pc_last).toSec());
            ros_time_pc_last = time_now; //update time
            kf_3d->update(measurement);
            position = kf_3d->get_pos();
        }

        void LocalObject::update_3d_tracker(const ros::Time &time_now)
        {
            if (!kf_3d->is_init)
            {
                ROS_WARN("Update 3d tracker fails!! 3d tracker is not initialized!!");
                return;
            }

            kf_3d->estimate((time_now - ros_time_pc_last).toSec());
            ros_time_pc_last = time_now;
            position = kf_3d->get_pos();
        }

        float LocalObject::find_min_query_score(const Eigen::VectorXf &query)
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
