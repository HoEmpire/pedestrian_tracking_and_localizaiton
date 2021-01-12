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
                                 Eigen::VectorXf feat, TrackerParam tracker_param_init,
                                 KalmanFilterParam kf_param_init, ros::Time time_now)
        {
            id = id_init;
            bbox = bbox_init;
            features.push_back(feat);
            tracker_param = tracker_param_init;
            tracking_fail_count = 0;
            detector_update_count = 0;
            overlap_count = 0;
            dssttracker = new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            kf_param = kf_param_init;
            kf = new KalmanFilter(kf_param_init);
            kf->init(bbox_init);
            ros_time = time_now;
            if (DSST)
            {
                dssttracker->detect_thresh_dsst = tracker_param.tracker_success_threshold;
                dssttracker->scale_step = 1;
            }
            else
            {
                dssttracker->detect_thresh_kcf = tracker_param.tracker_success_threshold;
                dssttracker->padding = tracker_param.padding;
                dssttracker->interp_factor = tracker_param.interp_factor;
                dssttracker->sigma = tracker_param.sigma;
                dssttracker->lambda = tracker_param.lambda;
                dssttracker->cell_size = tracker_param.cell_size;
                dssttracker->padding = tracker_param.padding;
                dssttracker->output_sigma_factor = tracker_param.output_sigma_factor;
                dssttracker->template_size = tracker_param.template_size;
                dssttracker->scale_step = tracker_param.scale_step;
                dssttracker->scale_weight = tracker_param.scale_weight;
            }

            dssttracker->init(frame, bbox);
            cv::RNG rng(std::time(0));
            color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            time.tic();
        }

        void LocalObject::update_tracker(cv::Mat frame, ros::Time update_time)
        {
            bbox = kf->estimate((update_time - ros_time).toSec());
            // std::cout << kf->x << std::endl;
            ros_time = update_time;
            // ROS_INFO_STREAM("FUCK " << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height);
            is_track_succeed = dssttracker->update(frame, bbox);
            detector_update_count++;
            if (is_track_succeed)
            {
                bbox = kf->update(bbox);
                // std::cout << kf->x << std::endl;
                tracking_fail_count = 0;
            }
            else
            {
                tracking_fail_count++;
                ROS_INFO_STREAM("Object " << id << " tracking failure detected!");
            }
        }

        void LocalObject::reinit(cv::Rect2d bbox_init, cv::Mat frame, ros::Time update_time)
        {
            tracking_fail_count = 0;
            detector_update_count = 0;

            // std::cout << kf->x << std::endl;
            dssttracker = new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            if (DSST)
            {
                dssttracker->detect_thresh_dsst = tracker_param.tracker_success_threshold;
                dssttracker->scale_step = 1;
            }
            else
            {
                dssttracker->detect_thresh_kcf = tracker_param.tracker_success_threshold;
                dssttracker->padding = tracker_param.padding;
                dssttracker->interp_factor = tracker_param.interp_factor;
                dssttracker->sigma = tracker_param.sigma;
                dssttracker->lambda = tracker_param.lambda;
                dssttracker->cell_size = tracker_param.cell_size;
                dssttracker->padding = tracker_param.padding;
                dssttracker->output_sigma_factor = tracker_param.output_sigma_factor;
                dssttracker->template_size = tracker_param.template_size;
                dssttracker->scale_step = tracker_param.scale_step;
                dssttracker->scale_weight = tracker_param.scale_weight;
            }
            dssttracker->init(frame, bbox_init);
            kf->estimate((update_time - ros_time).toSec());
            ros_time = update_time;
            bbox = kf->update(bbox_init);
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
