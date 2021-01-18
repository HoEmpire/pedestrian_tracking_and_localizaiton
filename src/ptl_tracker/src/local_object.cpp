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
                                 KalmanFilterParam kf_param_init, KalmanFilter3dParam kf3d_param_init, ros::Time time_now)
        {
            id = id_init;
            bbox = bbox_init;
            features.push_back(feat);
            tracker_param = tracker_param_init;
            tracking_fail_count = 0;
            detector_update_count = 0;
            overlap_count = 0;
            dssttracker = new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            kf = new KalmanFilter(kf_param_init);
            kf_3d = new KalmanFilter3d(kf3d_param_init);
            kf->init(bbox_init);
            ros_time_image_last = time_now;
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
            bbox = kf->estimate((update_time - ros_time_image_last).toSec());
            // std::cout << kf->x << std::endl;
            ros_time_image_last = update_time;
            // ROS_INFO_STREAM("FUCK " << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height);
            timer kfc_time_count;
            is_track_succeed = dssttracker->update(frame, bbox);
            ROS_INFO_STREAM("KCF update takes " << kfc_time_count.toc() << "seconds");
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

        void LocalObject::update_3d_tracker(geometry_msgs::Point measurement, ros::Time time_now)
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

        void LocalObject::update_3d_tracker(ros::Time time_now)
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
            timer kfc_time_count;
            dssttracker->init(frame, bbox_init);
            ROS_INFO_STREAM("KCF reinit takes " << kfc_time_count.toc() << "seconds");

            kf->estimate((update_time - ros_time_image_last).toSec());
            ros_time_image_last = update_time;
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
