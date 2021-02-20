#include <algorithm>

#include "ptl_tracker/tracker.h"
#include "ptl_msgs/DeadTracker.h"

using cv::Mat;
using cv::Rect2d;
using std::lock_guard;
using std::mutex;
using std::vector;

namespace ptl
{
    namespace tracker
    {
        void TrackerInterface::init(bool register_subscriber)
        {
            load_config(&nh_);
            opt_tracker = OpticalFlow(opt_param);

            //publisher
            m_track_vis_pub = nh_.advertise<sensor_msgs::Image>("tracker_results", 1);
            m_track_to_reid_pub = nh_.advertise<ptl_msgs::DeadTracker>("tracker_to_reid", 1);
            m_track_marker_pub = nh_.advertise<visualization_msgs::Marker>("marker_tracking", 1);
            m_pc_filtered_debug = nh_.advertise<sensor_msgs::PointCloud2>("point_cloud_tracking", 1);

            if (register_subscriber)
            {
                //subscirbe
                m_detector_sub = nh_.subscribe("/ptl_reid/detector_to_reid_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
                m_reid_sub = nh_.subscribe("/ptl_reid/reid_to_tracker", 1, &TrackerInterface::reid_callback, this);

                if (use_compressed_image)
                {
                    m_image_sub = nh_.subscribe(camera_topic, 1, &TrackerInterface::image_tracker_callback_compressed_img, this);
                }
                else
                {
                    m_image_sub = nh_.subscribe(camera_topic, 1, &TrackerInterface::image_tracker_callback, this);
                }

                if (use_lidar)
                {
                    m_lidar_sub = nh_.subscribe(lidar_topic, 1, &TrackerInterface::lidar_tracker_callback, this);
                }
            }
        }

        void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
        {
            ROS_INFO_STREAM("******Into Detector Callback******");
            timer t_spent;
            if (msg->ids.empty())
                return;
            vector<Eigen::VectorXf> features;
            vector<cv::Rect2d> bboxes;
            for (int i = 0; i < msg->features.size(); i++)
            {
                bboxes.push_back(cv::Rect2d(msg->bboxes[i].data[0], msg->bboxes[i].data[1], msg->bboxes[i].data[2], msg->bboxes[i].data[3]));
                features.push_back(feature_ros_to_eigen(msg->features[i]));
            }
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);
            //maximum block size, to perform augumentation and rectification of the tracker block
            cv::Rect2d block_max(0, 0, cv_ptr->image.cols, cv_ptr->image.rows);

            //update by optical flow first
            track_bbox_by_optical_flow(cv_ptr->image, msg->img.header.stamp, false);

            //associate the detected bboxes with tracking bboxes
            vector<AssociationVector> all_detected_bbox_ass_vec;
            detector_and_tracker_association(bboxes, block_max, features, all_detected_bbox_ass_vec);

            //local object list management
            manage_local_objects_list_by_detector(bboxes, block_max, features, cv_ptr->image, msg->img.header.stamp, all_detected_bbox_ass_vec);
            //summary
            report_local_object();
            ROS_INFO_STREAM("detector update:" << t_spent.toc() * 1000 << " ms");
            ROS_INFO_STREAM("******Out of Detector Callback******");
            std::cout << std::endl;
        }

        std::vector<LocalObject> TrackerInterface::update_bbox_by_tracker(const cv::Mat &img, const ros::Time &update_time)
        {
            //update the tracker and the database of each tracking object
            timer efficiency_clock;
            track_bbox_by_optical_flow(img, update_time, true);
            ROS_INFO_STREAM("update tracker:" << efficiency_clock.toc() * 1000 << " ms");

            //remove the tracker that loses track and also check whether enable opt(to avoid degeneration under occlusion)
            efficiency_clock.tic();
            std::vector<LocalObject> dead_tracker = remove_dead_trackers();
            ROS_INFO_STREAM("remove dead tracker:" << efficiency_clock.toc() * 1000 << " ms");

            //udpate overlap flag
            //TODO might remove this part
            efficiency_clock.tic();
            update_overlap_flag();
            ROS_INFO_STREAM("udpate overlap flag:" << efficiency_clock.toc() * 1000 << " ms");

            //for visualization
            efficiency_clock.tic();
            cv::Mat img_vis = img.clone();
            visualize_tracking(img_vis);
            ROS_INFO_STREAM("visualization:" << efficiency_clock.toc() * 1000 << " ms");

            //summary
            efficiency_clock.tic();
            report_local_object();
            ROS_INFO_STREAM("report:" << efficiency_clock.toc() * 1000 << " ms");

            return dead_tracker;
        }

        void TrackerInterface::update_bbox_by_detector(const cv::Mat &img,
                                                       const std::vector<cv::Rect2d> &bboxes,
                                                       const std::vector<float> features_detector,
                                                       const ros::Time &update_time)
        {
            ROS_INFO_STREAM("******Update Bbox by Detector******");
            timer t_spent;

            //maximum block size, to perform augumentation and rectification of the tracker block
            cv::Rect2d block_max(0, 0, img.cols, img.rows);

            //update by optical flow first
            track_bbox_by_optical_flow(img, update_time, false);

            //associate the detected bboxes with tracking bboxes
            vector<AssociationVector> all_detected_bbox_ass_vec;
            std::vector<Eigen::VectorXf> feats_eigen = feature_vector_to_eigen(features_detector);

            detector_and_tracker_association(bboxes, block_max, feats_eigen, all_detected_bbox_ass_vec);

            //local object list management
            manage_local_objects_list_by_reid_detector(bboxes, block_max, feats_eigen, features_detector,
                                                       img, update_time, all_detected_bbox_ass_vec);

            //summary
            report_local_object();
            ROS_INFO_STREAM("detector update:" << t_spent.toc() * 1000 << " ms");
            ROS_INFO_STREAM("**********************************");
            std::cout << std::endl;
        }

        void TrackerInterface::image_tracker_callback(const sensor_msgs::ImageConstPtr &msg)
        {
            ROS_INFO("******Into Tracker Callback******");
            timer img_proc_timer;
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            ROS_INFO_STREAM("Data preprocess takes: " << img_proc_timer.toc() * 1000 << " ms");
            update_bbox_by_tracker(cv_ptr->image, cv_ptr->header.stamp);
            ROS_INFO_STREAM("optical flow tracking takes: " << img_proc_timer.toc() * 1000 << " ms");
            ROS_INFO("******Out of Tracker Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::image_tracker_callback_compressed_img(const sensor_msgs::CompressedImageConstPtr &msg)
        {
            ROS_INFO("******Into Tracker Callback******");
            timer img_proc_timer;
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            update_bbox_by_tracker(cv_ptr->image, cv_ptr->header.stamp);
            ROS_INFO_STREAM("optical flow tracking takes: " << img_proc_timer.toc() * 1000 << " ms");
            ROS_INFO("******Out of Tracker Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::lidar_tracker_callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc)
        {
            ROS_INFO_STREAM("******Into Localization Callback******");
            timer efficency_timer;
            //get tf of the current timestamp
            get_tf();

            //clear the marker when there is no trakcing object
            if (local_objects_list.empty())
            {
                update_tracker_pos_marker_visualization();
                ROS_INFO_STREAM("******Out of Localization Callback******");
                std::cout << std::endl;
                return;
            }

            //data transform
            pcl::PointCloud<pcl::PointXYZI> point_cloud;
            pcl::fromROSMsg(*msg_pc, point_cloud);
            ROS_INFO_STREAM("Original point cloud size: " << point_cloud.size());

            //Resample and conditional filter the point cloud to reduce computation cost
            //in the following steps
            PointCloudProcessor pcp(point_cloud, pcp_param);
            pcp.compute(true, true, false, false, false);
            // ROS_INFO_STREAM("After resampled, point cloud size: " << pcp.pc_resample.size());
            ROS_INFO_STREAM("After preprocessed, point cloud size: " << pcp.pc_conditional_filtered.size());

            //match 2d bbox and 3d point cloud centroids
            match_between_2d_and_3d(pcp.pc_conditional_filtered.makeShared(), msg_pc->header.stamp);

            //update visualization
            update_tracker_pos_marker_visualization();
            ROS_INFO_STREAM("In track_and_locate_callback: point cloud processing takes " << efficency_timer.toc());
            ROS_INFO_STREAM("******Out of Localization Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::reid_callback(const ptl_msgs::ReidInfo &msg)
        {
            reid_infos.total_num = msg.total_num;
            reid_infos.last_query_id = msg.last_query_id;
        }

        void TrackerInterface::load_config(ros::NodeHandle *n)
        {
            GPARAM(n, "/basic/use_compressed_image", use_compressed_image);
            GPARAM(n, "/basic/use_lidar", use_lidar);
            GPARAM(n, "/basic/enable_pcp_vis", enable_pcp_vis);
            GPARAM(n, "/basic/lidar_topic", lidar_topic);
            GPARAM(n, "/basic/camera_topic", camera_topic);
            GPARAM(n, "/basic/map_frame", map_frame);
            GPARAM(n, "/basic/lidar_frame", lidar_frame);
            GPARAM(n, "/basic/camera_frame", camera_frame);

            GPARAM(n, "/tracker/track_fail_timeout_tick", track_fail_timeout_tick);
            GPARAM(n, "/tracker/bbox_overlap_ratio", bbox_overlap_ratio_threshold);
            GPARAM(n, "/tracker/detector_update_timeout_tick", detector_update_timeout_tick);
            GPARAM(n, "/tracker/stop_opt_timeout", stop_opt_timeout);
            GPARAM(n, "/tracker/detector_bbox_padding", detector_bbox_padding);
            GPARAM(n, "/tracker/reid_match_threshold", reid_match_threshold);
            GPARAM(n, "/tracker/reid_match_bbox_dis", reid_match_bbox_dis);
            GPARAM(n, "/tracker/reid_match_bbox_size_diff", reid_match_bbox_size_diff);

            GPARAM(n, "/local_database/height_width_ratio_min", height_width_ratio_min);
            GPARAM(n, "/local_database/height_width_ratio_max", height_width_ratio_max);
            GPARAM(n, "/local_database/blur_detection_threshold", blur_detection_threshold);
            GPARAM(n, "/local_database/record_interval", record_interval);
            GPARAM(n, "/local_database/batch_num_min", batch_num_min);
            GPARAM(n, "/local_database/feature_smooth_ratio", feature_smooth_ratio);

            //point_cloud_processor
            GPARAM(n, "/pc_processor/resample_size", pcp_param.resample_size);
            GPARAM(n, "/pc_processor/x_min", pcp_param.x_min);
            GPARAM(n, "/pc_processor/x_max", pcp_param.x_max);
            GPARAM(n, "/pc_processor/z_min", pcp_param.z_min);
            GPARAM(n, "/pc_processor/z_max", pcp_param.z_max);
            GPARAM(n, "/pc_processor/std_dev_thres", pcp_param.std_dev_thres);
            GPARAM(n, "/pc_processor/mean_k", pcp_param.mean_k);
            GPARAM(n, "/pc_processor/cluster_tolerance", pcp_param.cluster_tolerance);
            GPARAM(n, "/pc_processor/cluster_size_min", pcp_param.cluster_size_min);
            GPARAM(n, "/pc_processor/cluster_size_max", pcp_param.cluster_size_max);
            GPARAM(n, "/pc_processor/match_centroid_padding", match_centroid_padding);

            //camera intrinsic
            GPARAM(n, "/camera_intrinsic/fx", camera_intrinsic.fx);
            GPARAM(n, "/camera_intrinsic/fy", camera_intrinsic.fy);
            GPARAM(n, "/camera_intrinsic/cx", camera_intrinsic.cx);
            GPARAM(n, "/camera_intrinsic/cy", camera_intrinsic.cy);

            //kalman filter
            GPARAM(n, "/kalman_filter/q_xy", kf_param.q_xy);
            GPARAM(n, "/kalman_filter/q_wh", kf_param.q_wh);
            GPARAM(n, "/kalman_filter/p_xy_pos", kf_param.p_xy_pos);
            GPARAM(n, "/kalman_filter/p_xy_dp", kf_param.p_xy_dp);
            GPARAM(n, "/kalman_filter/p_wh_size", kf_param.p_wh_size);
            GPARAM(n, "/kalman_filter/p_wh_ds", kf_param.p_wh_ds);
            GPARAM(n, "/kalman_filter/r_theta", kf_param.r_theta);
            GPARAM(n, "/kalman_filter/r_f", kf_param.r_f);
            GPARAM(n, "/kalman_filter/r_tx", kf_param.r_tx);
            GPARAM(n, "/kalman_filter/r_ty", kf_param.r_ty);
            GPARAM(n, "/kalman_filter/residual_threshold", kf_param.residual_threshold); //TODO better usage of residual

            GPARAM(n, "/kalman_filter_3d/q_factor", kf3d_param.Q_factor);
            GPARAM(n, "/kalman_filter_3d/r_factor", kf3d_param.R_factor);
            GPARAM(n, "/kalman_filter_3d/p_pos", kf3d_param.P_pos);
            GPARAM(n, "/kalman_filter_3d/p_vel", kf3d_param.P_vel);
            GPARAM(n, "/kalman_filter_3d/start_predict_only_timeout", kf3d_param.start_predict_only_timeout);
            GPARAM(n, "/kalman_filter_3d/stop_track_timeout", kf3d_param.stop_track_timeout);
            GPARAM(n, "/kalman_filter_3d/outlier_threshold", kf3d_param.outlier_threshold);

            //optical tracker
            GPARAM(n, "/optical_flow/min_keypoints_to_track", opt_param.min_keypoints_to_track);
            GPARAM(n, "/optical_flow/keypoints_num_factor_area", opt_param.keypoints_num_factor_area);
            GPARAM(n, "/optical_flow/corner_detector_max_num", opt_param.corner_detector_max_num);
            GPARAM(n, "/optical_flow/corner_detector_quality_level", opt_param.corner_detector_quality_level);
            GPARAM(n, "/optical_flow/corner_detector_min_distance", opt_param.corner_detector_min_distance);
            GPARAM(n, "/optical_flow/corner_detector_block_size", opt_param.corner_detector_block_size);
            GPARAM(n, "/optical_flow/corner_detector_use_harris", opt_param.corner_detector_use_harris);
            GPARAM(n, "/optical_flow/corner_detector_k", opt_param.corner_detector_k);
            GPARAM(n, "/optical_flow/min_keypoints_to_cal_H_mat", opt_param.min_keypoints_to_cal_H_mat);
            GPARAM(n, "/optical_flow/min_keypoints_for_motion_estimation", opt_param.min_keypoints_for_motion_estimation);
            GPARAM(n, "/optical_flow/min_pixel_dis_square_for_scene_point", opt_param.min_pixel_dis_square_for_scene_point);
            GPARAM(n, "/optical_flow/use_resize", opt_param.use_resize);
            GPARAM(n, "/optical_flow/resize_factor", opt_param.resize_factor);
        }

        bool TrackerInterface::update_local_database(LocalObject &local_object, const cv::Mat &img_block)
        {
            // two criterion to manage local database:
            // 1. appropriate width/height ratio
            // 2. fulfill the minimum time interval
            if (1.0 * img_block.rows / img_block.cols > height_width_ratio_min &&
                1.0 * img_block.rows / img_block.cols < height_width_ratio_max &&
                local_object.database_update_timer.toc() > record_interval)
            {
                local_object.img_blocks.push_back(img_block);
                local_object.database_update_timer.tic();
                ROS_INFO_STREAM("Adding an image to the datebase id: " << local_object.id);
                return true;
            }
            else
            {
                return false;
            }
        }

        void TrackerInterface::match_between_2d_and_3d(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const ros::Time &ros_pc_time)
        {
            pcl::PointCloud<pcl::PointXYZI> pc_tracking;
            for (auto &lo : local_objects_list)
            {
                //stop 3d tracking when the detector fails to update this object for certain ticks
                if (lo.detector_update_count > kf3d_param.stop_track_timeout)
                {
                    continue;
                }

                // only update this tracking object by kalman filter when the detector fails to update this object
                // for certain ticks, or overlap of two objects occur
                if (lo.detector_update_count > kf3d_param.start_predict_only_timeout || lo.is_overlap)
                {
                    lo.update_3d_tracker(ros_pc_time);
                    continue;
                }

                // get the point cloud that might belong to this trackign object by reproject the point cloud to the image frame
                cv::Rect2d bbox_now = lo.bbox_of_lidar_time(ros_pc_time);
                // std::cout << "bbox_now: " << bbox_now << std::endl;
                pcl::PointCloud<pcl::PointXYZI> pc_seg = point_cloud_segementation(pc, bbox_now);

                if (pc_seg.empty())
                    continue;

                //cluster the point cloud
                PointCloudProcessor pcp(pc_seg, pcp_param);
                pcp.compute(false, false, true, true, true);
                if (pcp.centroids.empty())
                    continue;
                pc_tracking += *(pcp.pc_final);
                pcl::PointXYZ p = pcp.get_centroid_closest(); // take the closest cluster as tehe measurement

                geometry_msgs::Point p_camera_frame, p_map_frame;
                p_camera_frame.x = p.x;
                p_camera_frame.y = p.y;
                p_camera_frame.z = p.z;

                //transform the measurement to the map frame
                tf2::doTransform(p_camera_frame, p_map_frame, camera2map);

                //update the 3d pos of this tracking object by kalman filter
                lo.update_3d_tracker(p_map_frame, ros_pc_time);
            }

            //pcl to ros for debug
            sensor_msgs::PointCloud2 pc_tracking_msg;
            pcl::toROSMsg(pc_tracking, pc_tracking_msg);
            tf2::doTransform(pc_tracking_msg, pc_tracking_msg, camera2map);
            pc_tracking_msg.header.frame_id = map_frame;
            m_pc_filtered_debug.publish(pc_tracking_msg);
        }

        void TrackerInterface::get_tf()
        {
            //get tf transforms
            try
            {
                lidar2camera = tf_buffer.lookupTransform(camera_frame, lidar_frame,
                                                         ros::Time(0), ros::Duration(0.05));
                lidar2map = tf_buffer.lookupTransform(map_frame, lidar_frame, ros::Time(0), ros::Duration(0.01));
                camera2map = tf_buffer.lookupTransform(map_frame, camera_frame, ros::Time(0), ros::Duration(0.01));
            }
            catch (tf2::TransformException &ex)
            {
                ROS_WARN("%s", ex.what());
            }
        }

        pcl::PointCloud<pcl::PointXYZI> TrackerInterface::point_cloud_segementation(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const cv::Rect2d &bbox)
        {
            pcl::PointCloud<pcl::PointXYZI> pc_new, pc_camera_frame;
            Eigen::Quaterniond q(lidar2camera.transform.rotation.w, lidar2camera.transform.rotation.x,
                                 lidar2camera.transform.rotation.y, lidar2camera.transform.rotation.z);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3, 3>(0, 0) = q.matrix();
            T(0, 3) = lidar2camera.transform.translation.x;
            T(1, 3) = lidar2camera.transform.translation.y;
            T(2, 3) = lidar2camera.transform.translation.z;
            // std::cout << T << std::endl;

            pcl::transformPointCloud(*pc, pc_camera_frame, T);
            for (auto p : pc_camera_frame)
            {
                int u = int(-p.y / p.x * camera_intrinsic.fx + camera_intrinsic.cx);
                int v = int(-p.z / p.x * camera_intrinsic.fy + camera_intrinsic.cy);
                // ROS_INFO_STREAM("u = " << u << ", v = " << v);

                // ROS_INFO_STREAM("umin = " << bbox.x << ", umax = " << bbox.x + bbox.width);
                // ROS_INFO_STREAM("vmin = " << bbox.y << ", vmax = " << bbox.y + bbox.height);
                if (u < bbox.x + bbox.width + match_centroid_padding &&
                    u > bbox.x - match_centroid_padding &&
                    v < bbox.y + bbox.height + match_centroid_padding &&
                    v > bbox.y - match_centroid_padding)
                {
                    pc_new.points.push_back(p);
                }
            }
            ROS_INFO_STREAM("After reprojection " << pc_new.size() << " points remain.");
            return pc_new;
        }

        void TrackerInterface::update_tracker_pos_marker_visualization()
        {
            //visualizaztion
            visualization_msgs::Marker markers;
            markers.header.frame_id = map_frame;
            markers.header.stamp = ros::Time::now();
            markers.id = 0;
            markers.ns = "tracking_objects_position";
            markers.action = visualization_msgs::Marker::ADD;
            markers.pose.orientation.w = 1.0;
            markers.scale.x = 0.2;
            markers.scale.y = 0.2;
            markers.color.r = 0.0;
            markers.color.a = 1.0;
            markers.color.g = 1.0;
            markers.color.b = 0.0;
            markers.type = visualization_msgs::Marker::POINTS;
            for (auto lo : local_objects_list)
            {
                markers.points.push_back(lo.position);
            }
            m_track_marker_pub.publish(markers);
        }

        void TrackerInterface::update_overlap_flag()
        {
            lock_guard<mutex> lk(mtx); //lock the thread
            for (auto &lo : local_objects_list)
            {
                lo.is_overlap = false;
            }

            for (auto &lo : local_objects_list)
            {
                if (lo.is_overlap)
                {
                    continue;
                }

                for (auto lo2 : local_objects_list)
                {
                    if (lo.id == lo2.id)
                    {
                        continue;
                    }
                    if ((BboxPadding(lo.bbox, match_centroid_padding) & BboxPadding(lo2.bbox, match_centroid_padding)).area() > 1e-3) //TODO hard code in here
                    {
                        lo.is_overlap = true;
                        lo2.is_overlap = true;
                    }
                }
            }
        }

        void TrackerInterface::track_bbox_by_optical_flow(const cv::Mat &img, const ros::Time &update_time, bool update_database)
        {
            cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(img.cols, img.rows));
            lock_guard<mutex> lk(mtx); //lock the thread
            // get the bbox measurement by optical flow
            opt_tracker.update(img, local_objects_list);

            // update each tracking object in tracking list by kalman filter
            for (auto &lo : local_objects_list)
            {
                lo.track_bbox_by_optical_flow(update_time);
                bbox_rect(block_max);
                std::cout << lo.bbox << std::endl;
                //update database
                if (lo.is_track_succeed & update_database)
                {
                    update_local_database(lo, img(lo.bbox));
                }
            }
        }

        std::vector<LocalObject> TrackerInterface::remove_dead_trackers()
        {
            lock_guard<mutex> lk(mtx);                     //lock the thread
            std::vector<LocalObject> dead_tracking_object; //TOOD inefficient implementation in here
            for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();)
            {
                // two criterion to determine whether tracking failure occurs:
                // 1. too long from the last update by detector
                // 2. continuous tracking failure in optical flow tracking
                if (lo->tracking_fail_count >= track_fail_timeout_tick || lo->detector_update_count >= detector_update_timeout_tick)
                {
                    ptl_msgs::DeadTracker msg_pub; // publish the dead tracker to reid
                    for (auto ib : lo->img_blocks)
                    {
                        msg_pub.img_blocks.push_back(*cv_bridge::CvImage(std_msgs::Header(), "bgr8", ib).toImageMsg());
                    }
                    msg_pub.position = lo->position;
                    if (msg_pub.img_blocks.size() > batch_num_min)
                        m_track_to_reid_pub.publish(msg_pub);
                    dead_tracking_object.push_back(*lo);
                    lo = local_objects_list.erase(lo);
                    continue;
                }
                else
                {
                    // also disable opt when the occulusion occurs
                    if (lo->detector_update_count >= stop_opt_timeout)
                    {
                        lo->is_opt_enable = false;
                    }
                    lo++;
                }
            }
            return dead_tracking_object;
        }

        void TrackerInterface::report_local_object()
        {
            // lock_guard<mutex> lk(mtx); //lock the thread
            ROS_INFO("------Local Object List Summary------");
            ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
            for (auto lo : local_objects_list)
            {
                ROS_INFO_STREAM("id: " << lo.id << "| database images num: " << lo.img_blocks.size());
                std::cout << lo.bbox << std::endl;
            }
            ROS_INFO("------Summary End------");
        }

        void TrackerInterface::visualize_tracking(cv::Mat &img)
        {
            lock_guard<mutex> lk(mtx); //lock the thread
            for (auto lo : local_objects_list)
            {
                if (lo.is_opt_enable)
                {
                    cv::rectangle(img, lo.bbox, lo.color, 4.0);
                    cv::rectangle(img, cv::Rect2d(lo.bbox.x, lo.bbox.y, 40, 15), lo.color, -1);
                    cv::putText(img, "id:" + std::to_string(lo.id), cv::Point(lo.bbox.x, lo.bbox.y + 15), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                }

                // for (auto kp : lo.keypoints_pre)
                //     cv::circle(img, kp, 2, cv::Scalar(255, 0, 0), 2);
            }
            m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg());
        }

        void TrackerInterface::detector_and_tracker_association(const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
                                                                const std::vector<Eigen::VectorXf> &features,
                                                                std::vector<AssociationVector> &all_detected_bbox_ass_vec)
        {
            lock_guard<mutex> lk(mtx); //lock
            if (!local_objects_list.empty())
            {
                ROS_INFO_STREAM("SUMMARY:" << bboxes.size() << " bboxes detected!");
                //perform matching from detector to tracker
                for (int i = 0; i < bboxes.size(); i++)
                {
                    AssociationVector one_detected_object_ass_vec;
                    cv::Rect2d detector_bbox = BboxPadding(bboxes[i], block_max, detector_bbox_padding);
                    std::cout << "Detector" << i << " bbox: " << bboxes[i] << std::endl;
                    /*Data Association part: 
                      - two critertion:
                        - augemented bboxes(bbox with padding) have enough overlap
                        - Reid score is higher than a certain value
                    */
                    for (int j = 0; j < local_objects_list.size(); j++)
                    {
                        double bbox_overlap_ratio = cal_bbox_overlap_ratio(local_objects_list[j].bbox, detector_bbox);
                        ROS_INFO_STREAM("Bbox overlap ratio: " << bbox_overlap_ratio);
                        std::cout << "Tracker " << local_objects_list[j].id << " bbox: " << local_objects_list[j].bbox << std::endl;
                        if (bbox_overlap_ratio > bbox_overlap_ratio_threshold)
                        {
                            //TODO might speed up in here
                            float min_query_score = local_objects_list[j].find_min_query_score(features[i]);

                            //find a match, add it to association vector to construct the association graph
                            if (min_query_score < reid_match_threshold)
                                one_detected_object_ass_vec.add(AssociationType(j, min_query_score, cal_bbox_match_score(bboxes[i], local_objects_list[j].bbox)));
                        }
                    }
                    if (one_detected_object_ass_vec.ass_vector.size() > 1)
                        one_detected_object_ass_vec.reranking();
                    one_detected_object_ass_vec.report();
                    ROS_INFO("---------------------------------");
                    all_detected_bbox_ass_vec.push_back(one_detected_object_ass_vec);
                }
                uniquify_detector_association_vectors(all_detected_bbox_ass_vec, local_objects_list.size());

                ROS_INFO("---Report after uniquification---");
                for (auto ass : all_detected_bbox_ass_vec)
                {
                    ass.report();
                }
                ROS_INFO("---Report finished---");
            }
            else
            {
                //create empty association vectors to indicate all the detected objects are new
                all_detected_bbox_ass_vec = vector<AssociationVector>(bboxes.size(), AssociationVector());
            }
        }

        void TrackerInterface::manage_local_objects_list_by_detector(const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
                                                                     const std::vector<Eigen::VectorXf> &features, const cv::Mat &img,
                                                                     const ros::Time &update_time,
                                                                     const std::vector<AssociationVector> &all_detected_bbox_ass_vec)
        {
            lock_guard<mutex> lk(mtx); //lock the thread
            for (int i = 0; i < all_detected_bbox_ass_vec.size(); i++)
            {

                if (all_detected_bbox_ass_vec[i].ass_vector.empty())
                {
                    //this detected object is a new object
                    ROS_INFO_STREAM("Adding Tracking Object with ID:" << local_id_not_assigned);
                    cv::Mat example_img;
                    cv::resize(img(bboxes[i]), example_img, cv::Size(128, 256)); //hard code in here
                    LocalObject new_object(local_id_not_assigned, bboxes[i], features[i],
                                           kf_param, kf3d_param, update_time, example_img);
                    local_id_not_assigned++;
                    //update database
                    update_local_database(new_object, img(new_object.bbox));
                    local_objects_list.push_back(new_object);
                }
                else
                {
                    //re-detect a previous tracking object
                    int matched_id = all_detected_bbox_ass_vec[i].ass_vector[0].id;
                    ROS_INFO_STREAM("Object " << local_objects_list[matched_id].id << " re-detected!");

                    local_objects_list[matched_id].track_bbox_by_detector(bboxes[i], update_time);
                    local_objects_list[matched_id].features.push_back(features[i]);
                    local_objects_list[matched_id].update_feat(features[i], feature_smooth_ratio);

                    //update database
                    //TODO this part can be removed later
                    update_local_database(local_objects_list[matched_id], img(local_objects_list[matched_id].bbox & block_max));
                }
            }

            //rectify the bbox
            bbox_rect(block_max);
        }

        void TrackerInterface::manage_local_objects_list_by_reid_detector(const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
                                                                          const std::vector<Eigen::VectorXf> &feat_eigen, const std::vector<float> &feat_vector,
                                                                          const cv::Mat &img, const ros::Time &update_time, const std::vector<AssociationVector> &all_detected_bbox_ass_vec)
        {
            lock_guard<mutex> lk(mtx); //lock the thread
            const int feat_dimension = feat_vector.size() / feat_eigen.size();

            for (int i = 0; i < all_detected_bbox_ass_vec.size(); i++)
            {
                if (all_detected_bbox_ass_vec[i].ass_vector.empty())
                {
                    //this detected object is a new object
                    ROS_INFO_STREAM("Adding Tracking Object with ID:" << local_id_not_assigned);
                    cv::Mat example_img;
                    cv::resize(img(bboxes[i]), example_img, cv::Size(128, 256)); //hard code in here
                    LocalObject new_object(local_id_not_assigned, bboxes[i], feat_eigen[i],
                                           kf_param, kf3d_param, update_time, example_img);
                    local_id_not_assigned++;
                    //insert the 2048d feature vector
                    new_object.features_vector.insert(new_object.features_vector.end(), feat_vector.begin(), feat_vector.end());

                    local_objects_list.push_back(new_object);
                }
                else
                {
                    //re-detect a previous tracking object
                    int matched_id = all_detected_bbox_ass_vec[i].ass_vector[0].id;
                    ROS_INFO_STREAM("Object " << local_objects_list[matched_id].id << " re-detected!");

                    local_objects_list[matched_id].track_bbox_by_detector(bboxes[i], update_time);
                    local_objects_list[matched_id].update_feat(feat_eigen[i], feature_smooth_ratio);
                    //insert the 2048d feature vector
                    local_objects_list[matched_id].features_vector.insert(local_objects_list[matched_id].features_vector.end(),
                                                                          feat_vector.begin() + i * feat_dimension,
                                                                          feat_vector.begin() + (i + 1) * feat_dimension);
                }
            }

            //rectify the bbox
            bbox_rect(block_max);
        }

        std::vector<cv::Rect2d> TrackerInterface::bbox_ros_to_opencv(const std::vector<std_msgs::UInt16MultiArray> &bbox_ros)
        {
            std::vector<cv::Rect2d> bbox_opencv;
            for (auto b : bbox_ros)
            {
                bbox_opencv.push_back(Rect2d(b.data[0], b.data[1], b.data[2], b.data[3]));
            }
            return bbox_opencv;
        }

        void TrackerInterface::bbox_rect(const cv::Rect2d &bbox_max)
        {
            for (auto &lo : local_objects_list)
            {
                // std::cout << lo.bbox << std::endl;
                // std::cout << bbox_max << std::endl;
                lo.bbox = lo.bbox & bbox_max;
                // std::cout << lo.bbox << std::endl;
            }
        }
    } // namespace tracker

} // namespace ptl
