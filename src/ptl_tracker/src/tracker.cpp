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
            m_track_vis_pub = n->advertise<sensor_msgs::Image>("tracker_results", 1);
            m_track_to_reid_pub = n->advertise<ptl_msgs::DeadTracker>("tracker_to_reid", 1);
            m_detector_sub = n->subscribe("/ptl_reid/detector_to_reid_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
            m_data_sub = n->subscribe(camera_topic, 1, &TrackerInterface::data_callback, this);
            m_reid_sub = n->subscribe("/ptl_reid/reid_to_tracker", 1, &TrackerInterface::reid_callback, this);
        }

        void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
        {
            ROS_INFO_STREAM("******Into Detector Callback******");
            if (msg->ids.empty())
                return;
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat image_detection_result;
            cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);

            //maximum block size, to perform and operation to rectify the tracker block
            cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(cv_ptr->image.cols - 1, cv_ptr->image.rows - 1));

            //match the previous one
            lock_guard<mutex> lk(mtx); //加锁pub

            bool is_blur = blur_detection(cv_ptr->image);

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
                                           cv_ptr->image, feature_ros_to_eigen(msg->features[i]), tracker_param);
                    id++;
                    //update database
                    if (!is_blur)
                    {
                        cv::Mat image_block = cv_ptr->image(new_object.bbox);
                        update_local_database(new_object, image_block);
                    }
                    local_objects_list.push_back(new_object);
                }
                else
                {

                    int matched_id = detector_bbox_ass_vec[i].ass_vector[0].id;
                    ROS_INFO_STREAM("Object " << local_objects_list[matched_id].id << " re-detected!");

                    local_objects_list[matched_id].reinit(Rect2d(msg->bboxes[i].data[0], msg->bboxes[i].data[1],
                                                                 msg->bboxes[i].data[2], msg->bboxes[i].data[3]),
                                                          cv_ptr->image);
                    local_objects_list[matched_id].features.push_back(feature_ros_to_eigen(msg->features[i]));

                    //update database
                    if (!is_blur)
                    {
                        // ROS_WARN("Update database in detector callback");
                        cv::Mat image_block = cv_ptr->image(local_objects_list[matched_id].bbox & block_max);
                        update_local_database(local_objects_list[matched_id], image_block);
                    }
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

            ROS_INFO("******Out of Detctor Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::data_callback(const sensor_msgs::CompressedImageConstPtr &msg)
        {
            ROS_INFO("******Into Data Callback******");
            // ROS_ERROR("Into data callback");
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat image_detection_result;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            bool is_blur = blur_detection(cv_ptr->image);
            cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(cv_ptr->image.cols - 1, cv_ptr->image.rows - 1));
            lock_guard<mutex> lk(mtx); //加锁

            //update the tracker and update the local database
            for (auto lo = local_objects_list.begin(); lo < local_objects_list.end(); lo++)
            {
                lo->update_tracker(cv_ptr->image);

                //update database
                if (!is_blur && lo->is_track_succeed)
                {
                    cv::Mat image_block = cv_ptr->image(lo->bbox & block_max);
                    update_local_database(lo, image_block);
                }
            }

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

            //for visualization
            Mat track_vis = cv_ptr->image.clone();
            for (auto lo : local_objects_list)
            {
                if (lo.is_track_succeed)
                {
                    // std::string text;
                    // text = "id: " + std::to_string(lo.id);
                    cv::rectangle(track_vis, lo.bbox, lo.color, 4.0);
                    // cv::putText(track_vis, text, cv::Point(lo.bbox.x, lo.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, lo.color, 4.0);
                }
            }
            std::string reid_infos_text;
            reid_infos_text = "Total: " + std::to_string(reid_infos.total_num) +
                              "  Last Id: " + std::to_string(reid_infos.last_query_id);
            cv::putText(track_vis, reid_infos_text, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3.0);
            m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_vis).toImageMsg());

            //summary
            ROS_INFO("------Local Object List Summary------");
            ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
            for (auto lo : local_objects_list)
            {
                ROS_INFO_STREAM("id: " << lo.id << "| database images num: " << lo.img_blocks.size());
            }
            ROS_INFO("------Summary End------");

            ROS_INFO("******Out of Data Callback******");
            std::cout << std::endl;
        }

        void TrackerInterface::reid_callback(const ptl_msgs::ReidInfo &msg)
        {
            reid_infos.total_num = msg.total_num;
            reid_infos.last_query_id = msg.last_query_id;
        }

        void TrackerInterface::load_config(ros::NodeHandle *n)
        {
            GPARAM(n, "/data_topic/lidar_topic", lidar_topic);
            GPARAM(n, "/data_topic/camera_topic", camera_topic);
            GPARAM(n, "/data_topic/depth_topic", depth_topic);

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

        bool TrackerInterface::update_local_database(LocalObject local_object, const cv::Mat img_block)
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

        bool TrackerInterface::update_local_database(std::vector<LocalObject>::iterator local_object, const cv::Mat img_block)
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
    } // namespace tracker

} // namespace ptl
