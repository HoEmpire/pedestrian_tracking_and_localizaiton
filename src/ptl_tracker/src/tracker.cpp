#include <algorithm>

#include "ptl_tracker/tracker.h"
#include "ptl_msgs/DeadTracker.h"

using cv::Mat;
using cv::Rect2d;
using kcf::KCFTracker;
using std::lock_guard;
using std::mutex;
using std::vector;

namespace ptl
{
    namespace tracker
    {
        TrackerInterface::TrackerInterface(ros::NodeHandle *n)
        {
            id = 0;
            nh_ = n;
            load_config(nh_);
            tf_listener = new tf2_ros::TransformListener(tf_buffer);

            //publisher
            m_track_vis_pub = n->advertise<sensor_msgs::Image>("tracker_results", 1);
            m_track_to_reid_pub = n->advertise<ptl_msgs::DeadTracker>("tracker_to_reid", 1);
            m_track_marker_pub = n->advertise<visualization_msgs::Marker>("marker_vis", 1);
            m_pc_filtered_debug = n->advertise<sensor_msgs::PointCloud2>("pc_filtered", 1);
            // m_pc_cluster_debug = n->advertise<visualization_msgs::Marker>("pc_cluster", 1);

            //subscirbe
            m_detector_sub = n->subscribe("/ptl_reid/detector_to_reid_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
            m_reid_sub = n->subscribe("/ptl_reid/reid_to_tracker", 1, &TrackerInterface::reid_callback, this);

            if (use_lidar)
            {
                m_lidar_sub.subscribe(*n, lidar_topic, 1);
                if (use_compressed_image)
                {
                    m_compressed_image_sub.subscribe(*n, camera_topic, 1);
                    sync_compressed = new message_filters::Synchronizer<MySyncPolicyCompressed>(MySyncPolicyCompressed(10), m_compressed_image_sub, m_lidar_sub);
                    sync_compressed->registerCallback(boost::bind(&TrackerInterface::track_and_locate_callback_compressed, this, _1, _2));
                }
                else
                {
                    m_image_sub.subscribe(*n, camera_topic, 1);
                    sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), m_image_sub, m_lidar_sub);
                    sync->registerCallback(boost::bind(&TrackerInterface::track_and_locate_callback, this, _1, _2));
                }
            }
            else
            {
                if (use_compressed_image)
                {
                    m_data_sub = n->subscribe(camera_topic, 1, &TrackerInterface::tracker_callback_compressed, this);
                }
                else
                {
                    m_data_sub = n->subscribe(camera_topic, 1, &TrackerInterface::tracker_callback, this);
                }
            }
        }

        void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
        {
            ROS_INFO_STREAM("******Into Detector Callback******");
            ros::Duration time = ros::Time::now() - msg->header.stamp;
            ROS_INFO_STREAM("Time: detector->reid->tracekr end:" << time.toSec() * 1000 << "ms");
            ROS_INFO_STREAM("Time: detector->reid->tracekr start:" << time.sec);
            ROS_INFO_STREAM("Time: detector->reid->tracekr start:" << time.nsec);
            ROS_INFO_STREAM("Time: now:" << ros::Time::now().sec);
            ROS_INFO_STREAM("Time: now:" << ros::Time::now().nsec);
            ROS_INFO_STREAM("Time: image:" << msg->header.stamp.sec);
            ROS_INFO_STREAM("Time: image:" << msg->header.stamp.nsec);
            if (msg->ids.empty())
                return;
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat image_detection_result;
            cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);

            //maximum block size, to perform and operation to rectify the tracker block
            cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(cv_ptr->image.cols - 1, cv_ptr->image.rows - 1));

            //match the previous one
            lock_guard<mutex> lk(mtx); //加锁pub

            // bool is_blur = blur_detection(cv_ptr->image);

            /*Data Association part: 
                - two critertion:
                    - augemented bboxes have enough overlap
                    - Reid score is higher than a certain value
            */
            vector<int> matched_ids;
            vector<AssociationVector> detector_bbox_ass_vec;
            if (!local_objects_list.empty())
            {
                ROS_INFO_STREAM(msg->bboxes.size() << " bboxes detected!");
                for (int i = 0; i < msg->bboxes.size(); i++)
                {
                    AssociationVector ass_vec;
                    ROS_INFO("Deal with bboxes...");
                    cv::Rect2d detector_bbox_origin = Rect2d(msg->bboxes[i].data[0], msg->bboxes[i].data[1],
                                                             msg->bboxes[i].data[2], msg->bboxes[i].data[3]);
                    cv::Rect2d detector_bbox = BboxPadding(detector_bbox_origin, block_max, detector_bbox_padding);
                    print_bbox(detector_bbox_origin);
                    // ROS_INFO_STREAM("Detector bbox:" << msg->bboxes[i].data[0] << ", " << msg->bboxes[i].data[1] << ", "
                    //  << msg->bboxes[i].data[2] << ", " << msg->bboxes[i].data[3]);
                    for (int j = 0; j < local_objects_list.size(); j++)
                    {
                        double bbox_overlap_ratio_score = cal_bbox_overlap_ratio(local_objects_list[j].bbox, detector_bbox);
                        ROS_INFO_STREAM("Bbox overlap ratio: " << bbox_overlap_ratio_score);
                        print_bbox(local_objects_list[j].bbox);
                        if (bbox_overlap_ratio_score > bbox_overlap_ratio_threshold)
                        {
                            float min_query_score = local_objects_list[j].find_min_query_score(feature_ros_to_eigen(msg->features[i]));
                            if (min_query_score < reid_match_threshold)
                                ass_vec.add_new_ass(AssociationType(j, min_query_score, cal_bbox_match_score(detector_bbox_origin, local_objects_list[j].bbox)));
                        }
                    }
                    if (ass_vec.ass_vector.size() > 1)
                        ass_vec.reranking();
                    ass_vec.report();
                    ROS_INFO("---------------------------------");
                    detector_bbox_ass_vec.push_back(ass_vec);
                }
                uniquify_detector_association_vectors(detector_bbox_ass_vec, local_objects_list.size());
                ROS_INFO("---Report after uniquification---");
                for (auto ass : detector_bbox_ass_vec)
                {
                    ass.report();
                }
            }
            else
            {
                detector_bbox_ass_vec = vector<AssociationVector>(msg->bboxes.size(), AssociationVector());
            }

            //local object list management
            for (int i = 0; i < detector_bbox_ass_vec.size(); i++)
            {
                if (detector_bbox_ass_vec[i].ass_vector.empty())
                {
                    ROS_INFO_STREAM("Adding Tracking Object with ID:" << id);
                    LocalObject new_object(id, Rect2d(msg->bboxes[i].data[0], msg->bboxes[i].data[1], msg->bboxes[i].data[2], msg->bboxes[i].data[3]),
                                           cv_ptr->image, feature_ros_to_eigen(msg->features[i]), tracker_param, kf_param, kf3d_param, msg->img.header.stamp);
                    id++;
                    //update database
                    cv::Mat image_block = cv_ptr->image(new_object.bbox);
                    update_local_database(new_object, image_block);
                    local_objects_list.push_back(new_object);
                }
                else
                {

                    int matched_id = detector_bbox_ass_vec[i].ass_vector[0].id;
                    ROS_INFO_STREAM("Object " << local_objects_list[matched_id].id << " re-detected!");

                    local_objects_list[matched_id].reinit(Rect2d(msg->bboxes[i].data[0], msg->bboxes[i].data[1],
                                                                 msg->bboxes[i].data[2], msg->bboxes[i].data[3]),
                                                          cv_ptr->image, msg->img.header.stamp);

                    local_objects_list[matched_id].features.push_back(feature_ros_to_eigen(msg->features[i]));

                    //update database
                    // ROS_WARN("Update database in detector callback");
                    cv::Mat image_block = cv_ptr->image(local_objects_list[matched_id].bbox & block_max);
                    update_local_database(local_objects_list[matched_id], image_block);
                }
            }

            //summary
            ROS_INFO("------Local Object List Summary------");
            ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
            for (auto lo : local_objects_list)
            {
                ROS_INFO_STREAM("id: " << lo.id << "| db imgs: " << lo.img_blocks.size() << "| overlap: " << lo.overlap_count);
            }
            ROS_INFO("------Summary End------");

            time = ros::Time::now() - msg->header.stamp;
            ROS_INFO_STREAM("Time: detector->reid->tracekr end:" << time.toSec() * 1000 << "ms");
            std::cout << std::endl;
        }

        void TrackerInterface::tracker_callback(const sensor_msgs::ImageConstPtr &msg)
        {
            ROS_INFO("******Into Tracker Callback******");
            timer efficiency_clock;
            // ROS_ERROR("Into data callback");
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat image_detection_result;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            ROS_INFO_STREAM("Data preprocess:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            // bool is_blur = blur_detection(cv_ptr->image);

            ROS_INFO_STREAM("Blur detection:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(cv_ptr->image.cols - 1, cv_ptr->image.rows - 1));
            lock_guard<mutex> lk(mtx); //加锁

            //update the tracker and update the local database
            for (auto lo = local_objects_list.begin(); lo < local_objects_list.end(); lo++)
            {
                lo->update_tracker(cv_ptr->image, msg->header.stamp);

                //update database
                if (lo->is_track_succeed)
                {
                    cv::Mat image_block = cv_ptr->image(lo->bbox & block_max);
                    update_local_database(lo, image_block);
                }
            }

            ROS_INFO_STREAM("update tracker:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            //remove the tracker that loses track
            for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();)
            {
                if (lo->tracking_fail_count >= track_fail_timeout_tick || lo->detector_update_count >= detector_update_timeout_tick)
                {
                    ptl_msgs::DeadTracker msg_pub;
                    for (auto ib : lo->img_blocks)
                    {
                        sensor_msgs::ImagePtr img_tmp = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ib).toImageMsg();
                        msg_pub.img_blocks.push_back(*img_tmp);
                    }
                    msg_pub.position = lo->position;
                    if (msg_pub.img_blocks.size() > batch_num_min)
                        m_track_to_reid_pub.publish(msg_pub);
                    lo = local_objects_list.erase(lo);
                    continue;
                }
                else
                {
                    lo++;
                }
            }

            ROS_INFO_STREAM("remove dead tracker:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            //for visualization
            Mat track_vis = cv_ptr->image.clone();
            for (auto lo : local_objects_list)
            {
                // if (lo.is_track_succeed)
                // {
                // std::string text;
                // text = "id: " + std::to_string(lo.id);
                cv::rectangle(track_vis, lo.bbox, lo.color, 4.0);
                // cv::putText(track_vis, text, cv::Point(lo.bbox.x, lo.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, lo.color, 4.0);
                // }
            }
            std::string reid_infos_text;
            reid_infos_text = "Total: " + std::to_string(reid_infos.total_num) +
                              "  Last Id: " + std::to_string(reid_infos.last_query_id);
            cv::putText(track_vis, reid_infos_text, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3.0);
            m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_vis).toImageMsg());

            ROS_INFO_STREAM("visualization:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            //summary
            ROS_INFO("------Local Object List Summary------");
            ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
            for (auto lo : local_objects_list)
            {
                ROS_INFO_STREAM("id: " << lo.id << "| database images num: " << lo.img_blocks.size());
            }
            ROS_INFO("------Summary End------");

            ROS_INFO_STREAM("report:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();
            ROS_INFO("******Out of Data Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::track_and_locate_callback(const sensor_msgs::ImageConstPtr &msg_img, const sensor_msgs::PointCloud2ConstPtr &msg_pc)
        {
            timer efficency_timer;
            tracker_callback(msg_img);
            update_overlap_flag();
            ROS_INFO_STREAM("In track_and_locate_callback: image processing takes " << efficency_timer.toc());
            efficency_timer.tic();
            get_tf();
            if (local_objects_list.empty())
            {
                update_tracker_pos_marker_visualization();
                return;
            }
            pcl::PointCloud<pcl::PointXYZI> point_cloud;
            pcl::fromROSMsg(*msg_pc, point_cloud);

            //Resample and conditional filter the point cloud to reduce computation cost
            //in the following steps
            ROS_INFO_STREAM("Original point cloud size: " << point_cloud.size());
            PointCloudProcessor pcp(point_cloud, pcp_param);
            pcp.compute(true, true, false, false, false);
            // ROS_INFO_STREAM("After resampled, point cloud size: " << pcp.pc_resample.size());
            ROS_INFO_STREAM("After preprocessed, point cloud size: " << pcp.pc_conditional_filtered.size());
            match_between_2d_and_3d(pcp.pc_conditional_filtered.makeShared(), msg_pc->header.stamp);
            update_tracker_pos_marker_visualization();
            ROS_INFO_STREAM("In track_and_locate_callback: point cloud processing takes " << efficency_timer.toc());
        }

        void TrackerInterface::tracker_callback_compressed(const sensor_msgs::CompressedImageConstPtr &msg)
        {
            ROS_INFO("******Into Tracker Callback******");
            timer efficiency_clock;
            // ROS_ERROR("Into data callback");
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat image_detection_result;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            ROS_INFO_STREAM("Data preprocess:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            // bool is_blur = blur_detection(cv_ptr->image);

            ROS_INFO_STREAM("Blur detection:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(cv_ptr->image.cols - 1, cv_ptr->image.rows - 1));
            lock_guard<mutex> lk(mtx); //加锁

            //update the tracker and update the local database
            for (auto lo = local_objects_list.begin(); lo < local_objects_list.end(); lo++)
            {
                lo->update_tracker(cv_ptr->image, msg->header.stamp);

                //update database
                if (lo->is_track_succeed)
                {
                    update_local_database(lo, cv_ptr->image(lo->bbox & block_max));
                }
            }

            ROS_INFO_STREAM("update tracker:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            //remove the tracker that loses track
            for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();)
            {
                if (lo->tracking_fail_count >= track_fail_timeout_tick || lo->detector_update_count >= detector_update_timeout_tick)
                {
                    ptl_msgs::DeadTracker msg_pub;
                    for (auto ib : lo->img_blocks)
                    {
                        sensor_msgs::ImagePtr img_tmp = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ib).toImageMsg();
                        msg_pub.img_blocks.push_back(*img_tmp);
                    }
                    msg_pub.position = lo->position;
                    if (msg_pub.img_blocks.size() > batch_num_min)
                        m_track_to_reid_pub.publish(msg_pub);
                    lo = local_objects_list.erase(lo);
                    continue;
                }
                else
                {
                    lo++;
                }
            }

            ROS_INFO_STREAM("remove dead tracker:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            //for visualization
            Mat track_vis = cv_ptr->image.clone();
            for (auto lo : local_objects_list)
            {
                // if (lo.is_track_succeed)
                // {
                // std::string text;
                // text = "id: " + std::to_string(lo.id);
                cv::rectangle(track_vis, lo.bbox, lo.color, 4.0);
                // cv::putText(track_vis, text, cv::Point(lo.bbox.x, lo.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, lo.color, 4.0);
                // }
            }
            std::string reid_infos_text;
            reid_infos_text = "Total: " + std::to_string(reid_infos.total_num) +
                              "  Last Id: " + std::to_string(reid_infos.last_query_id);
            cv::putText(track_vis, reid_infos_text, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3.0);
            m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_vis).toImageMsg());

            ROS_INFO_STREAM("visualization:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();

            //summary
            ROS_INFO("------Local Object List Summary------");
            ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
            for (auto lo : local_objects_list)
            {
                ROS_INFO_STREAM("id: " << lo.id << "| database images num: " << lo.img_blocks.size());
            }
            ROS_INFO("------Summary End------");

            ROS_INFO_STREAM("report:" << efficiency_clock.toc() << " s");
            efficiency_clock.tic();
            ROS_INFO("******Out of Data Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::track_and_locate_callback_compressed(const sensor_msgs::CompressedImageConstPtr &msg_img, const sensor_msgs::PointCloud2ConstPtr &msg_pc)
        {
            timer efficency_timer;
            tracker_callback_compressed(msg_img);
            update_overlap_flag();
            ROS_INFO_STREAM("In track_and_locate_callback: image processing takes " << efficency_timer.toc());
            efficency_timer.tic();
            get_tf();
            if (local_objects_list.empty())
            {
                update_tracker_pos_marker_visualization();
                return;
            }
            pcl::PointCloud<pcl::PointXYZI> point_cloud;
            pcl::fromROSMsg(*msg_pc, point_cloud);

            //Resample and conditional filter the point cloud to reduce computation cost
            //in the following steps
            ROS_INFO_STREAM("Original point cloud size: " << point_cloud.size());
            PointCloudProcessor pcp(point_cloud, pcp_param);
            pcp.compute(true, true, false, false, false);
            // ROS_INFO_STREAM("After resampled, point cloud size: " << pcp.pc_resample.size());
            ROS_INFO_STREAM("After preprocessed, point cloud size: " << pcp.pc_conditional_filtered.size());
            match_between_2d_and_3d(pcp.pc_conditional_filtered.makeShared(), msg_pc->header.stamp);
            update_tracker_pos_marker_visualization();
            ROS_INFO_STREAM("In track_and_locate_callback: point cloud processing takes " << efficency_timer.toc());
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
            GPARAM(n, "/tracker/detector_bbox_padding", detector_bbox_padding);
            GPARAM(n, "/tracker/reid_match_threshold", reid_match_threshold);
            GPARAM(n, "/tracker/reid_match_bbox_dis", reid_match_bbox_dis);
            GPARAM(n, "/tracker/reid_match_bbox_size_diff", reid_match_bbox_size_diff);

            GPARAM(n, "/local_database/height_width_ratio_min", height_width_ratio_min);
            GPARAM(n, "/local_database/height_width_ratio_max", height_width_ratio_max);
            GPARAM(n, "/local_database/blur_detection_threshold", blur_detection_threshold);
            GPARAM(n, "/local_database/record_interval", record_interval);
            GPARAM(n, "/local_database/batch_num_min", batch_num_min);

            //kcf
            GPARAM(n, "/kcf/tracker_success_threshold", tracker_param.tracker_success_threshold);
            GPARAM(n, "/kcf/interp_factor", tracker_param.interp_factor);
            GPARAM(n, "/kcf/sigma", tracker_param.sigma);
            GPARAM(n, "/kcf/lambda", tracker_param.lambda);
            GPARAM(n, "/kcf/cell_size", tracker_param.cell_size);
            GPARAM(n, "/kcf/padding", tracker_param.padding);
            GPARAM(n, "/kcf/output_sigma_factor", tracker_param.output_sigma_factor);
            GPARAM(n, "/kcf/template_size", tracker_param.template_size);
            GPARAM(n, "/kcf/scale_step", tracker_param.scale_step);
            GPARAM(n, "/kcf/scale_weight", tracker_param.scale_weight);

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
            GPARAM(n, "/kalman_filter/q_factor", kf_param.Q_factor);
            GPARAM(n, "/kalman_filter/r_factor", kf_param.R_factor);
            GPARAM(n, "/kalman_filter/p_factor", kf_param.P_factor);
            GPARAM(n, "/kalman_filter_3d/q_factor", kf3d_param.Q_factor);
            GPARAM(n, "/kalman_filter_3d/r_factor", kf3d_param.R_factor);
            GPARAM(n, "/kalman_filter_3d/p_factor", kf3d_param.P_factor);
            GPARAM(n, "/kalman_filter_3d/start_predict_only_timeout", kf3d_param.start_predict_only_timeout);
            GPARAM(n, "/kalman_filter_3d/stop_track_timeout", kf3d_param.stop_track_timeout);
            GPARAM(n, "/kalman_filter_3d/outlier_threshold", kf3d_param.outlier_threshold);
        }

        bool TrackerInterface::blur_detection(cv::Mat img)
        {
            cv::Mat img_gray;
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
            cv::Mat lap;
            cv::Laplacian(img_gray, lap, CV_64F);

            cv::Scalar mu, sigma;
            cv::meanStdDev(lap, mu, sigma);
            ROS_INFO_STREAM("Blur detection score: " << sigma.val[0] * sigma.val[0]);
            return sigma.val[0] * sigma.val[0] < blur_detection_threshold;
        }

        bool TrackerInterface::update_local_database(LocalObject &local_object, const cv::Mat &img_block)
        {
            if (1.0 * img_block.rows / img_block.cols > height_width_ratio_min && 1.0 * img_block.rows / img_block.cols < height_width_ratio_max && local_object.time.toc() > record_interval)
            {
                local_object.img_blocks.push_back(img_block);
                local_object.time.tic();
                ROS_INFO_STREAM("Adding an image to the datebase id: " << local_object.id);
                return true;
            }
            else
            {
                return false;
            }
        }

        bool TrackerInterface::update_local_database(std::vector<LocalObject>::iterator local_object, const cv::Mat &img_block)
        {
            if (1.0 * img_block.rows / img_block.cols > height_width_ratio_min && 1.0 * img_block.rows / img_block.cols < height_width_ratio_max && local_object->time.toc() > record_interval)
            {
                local_object->img_blocks.push_back(img_block);
                local_object->time.tic();
                ROS_INFO_STREAM("Adding an image to the datebase id: " << local_object->id);
                return true;
            }
            else
            {
                return false;
            }
        }

        void TrackerInterface::match_between_2d_and_3d(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc, const ros::Time &ros_pc_time)
        {
            pcl::PointCloud<pcl::PointXYZI> pc_filtered;
            for (auto &lo : local_objects_list)
            {
                //stop 3d tracking
                if (lo.detector_update_count > kf3d_param.stop_track_timeout)
                {
                    continue;
                }

                if (lo.detector_update_count > kf3d_param.start_predict_only_timeout || lo.is_overlap) //TODO hard code in here
                {
                    lo.update_3d_tracker(ros_pc_time);
                    continue;
                }

                pcl::PointCloud<pcl::PointXYZI> pc_seg = point_cloud_segementation(pc, lo.bbox);
                if (pc_seg.empty())
                    continue;
                PointCloudProcessor pcp(pc_seg, pcp_param);
                pcp.compute(false, false, true, true, true); //TODO shit code
                if (pcp.centroids.empty())
                    continue;
                pc_filtered += *(pcp.pc_final);
                pcl::PointXYZ p = pcp.get_centroid_closest();

                geometry_msgs::Point p_camera_frame, p_map_frame;
                p_camera_frame.x = p.x;
                p_camera_frame.y = p.y;
                p_camera_frame.z = p.z;

                tf2::doTransform(p_camera_frame, p_map_frame, camera2map);
                lo.update_3d_tracker(p_map_frame, ros_pc_time);
            }

            //pcl to ros for debug
            sensor_msgs::PointCloud2 pc_filtered_msg;
            pcl::toROSMsg(pc_filtered, pc_filtered_msg);
            tf2::doTransform(pc_filtered_msg, pc_filtered_msg, camera2map);
            pc_filtered_msg.header.frame_id = map_frame;
            m_pc_filtered_debug.publish(pc_filtered_msg);
        }

        void TrackerInterface::get_tf()
        {
            //get tf transforms
            try
            {
                lidar2camera = tf_buffer.lookupTransform(camera_frame, lidar_frame,
                                                         ros::Time(0), ros::Duration(0.05));
                lidar2map = tf_buffer.lookupTransform(map_frame, lidar_frame, ros::Time(0), ros::Duration(0.05));
                camera2map = tf_buffer.lookupTransform(map_frame, camera_frame, ros::Time(0), ros::Duration(0.05));
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
                    if ((BboxPadding(lo.bbox, match_centroid_padding) & BboxPadding(lo2.bbox, match_centroid_padding)).area() > 1e-3)
                    {
                        lo.is_overlap = true;
                        lo2.is_overlap = true;
                    }
                }
            }
        }
    } // namespace tracker

} // namespace ptl
