#include "ptl_tracker/tracker.h"
#include "ptl_tracker/util.h"
#include "ptl_msgs/DeadTracker.h"

using cv::Mat;
using cv::Rect2d;
using kcf::KCFTracker;
using std::lock_guard;
using std::mutex;
using std::vector;

namespace ptl_tracker
{
    TrackerInterface::TrackerInterface(ros::NodeHandle *n)
    {
        id = 0;
        nh_ = n;
        load_config(nh_);
        m_track_vis_pub = n->advertise<sensor_msgs::Image>("tracker_results", 1);
        m_track_to_reid_pub = n->advertise<ptl_msgs::DeadTracker>("tracker_to_reid", 1);
        m_detector_sub = n->subscribe("/ptl_detector/detector_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
        m_data_sub = n->subscribe(camera_topic, 1, &TrackerInterface::data_callback, this);
    }

    void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
    {
        ROS_INFO_STREAM("******Into Detctor Callback******");
        if (msg->ids.empty())
            return;
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat image_detection_result;
        cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);

        //match the previous one
        lock_guard<mutex> lk(mtx); //加锁pub
        vector<std_msgs::UInt16MultiArray> bboxs = msg->bboxs;

        bool erase_flag = false;
        bool is_blur = blur_detection(cv_ptr->image);
        if (!local_objects_list.empty())
        {
            for (auto b = bboxs.begin(); b != bboxs.end();)
            {
                for (auto lo = local_objects_list.begin(); lo != local_objects_list.end(); lo++)
                {
                    if (bbox_matching(lo->bbox, Rect2d(b->data[0], b->data[1], b->data[2], b->data[3])))
                    {
                        ROS_INFO_STREAM("Object " << lo->id << " re-detected!");
                        lo->reinit(Rect2d(b->data[0], b->data[1], b->data[2], b->data[3]), cv_ptr->image); //TODO might try to improve efficiency in here
                        b = bboxs.erase(b);                                                                //删除该对象
                        erase_flag = true;

                        //update database
                        if (!is_blur)
                        {
                            // ROS_WARN("Update database in detector callback");
                            cv::Mat image_block = cv_ptr->image(lo->bbox);
                            update_local_database(lo, image_block);
                        }

                        break;
                    }
                }
                if (erase_flag == false)
                {
                    b++;
                }
                else
                {
                    erase_flag = false;
                }
            }
        }

        //add the new ones
        if (!bboxs.empty())
        {
            for (auto b : bboxs)
            {
                ROS_INFO_STREAM("Adding Tracking Object with ID:" << id);
                LocalObject new_object(id++, Rect2d(b.data[0], b.data[1], b.data[2], b.data[3]), cv_ptr->image);
                local_objects_list.push_back(new_object);
                //update database
                if (!is_blur)
                {
                    vector<LocalObject>::iterator lo = local_objects_list.end() - 1;
                    cv::Mat image_block = cv_ptr->image(lo->bbox);
                    update_local_database(lo, image_block);
                }
            }
        }

        //summary
        ROS_INFO("------Local Object List Summary------");
        ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
        for (auto lo : local_objects_list)
        {
            ROS_INFO_STREAM("id: " << lo.id << "| database iamges num: " << lo.img_blocks.size());
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
        for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();)
        {
            if (lo->tracking_fail_count >= track_fail_timeout_tick)
            {
                vector<sensor_msgs::Image> msg_pub;
                for (auto ib : lo->img_blocks)
                {
                    sensor_msgs::ImagePtr img_tmp = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ib).toImageMsg();
                    msg_pub.push_back(*img_tmp);
                }
                m_track_to_reid_pub.publish(msg_pub);
                lo = local_objects_list.erase(lo);
                continue;
            }
            lo->update_tracker(cv_ptr->image);

            //update database
            if (!is_blur)
            {
                // ROS_WARN("Update database in data callback");
                cv::Mat image_block = cv_ptr->image(lo->bbox & block_max);
                update_local_database(lo, image_block);
            }
            lo++;
        }

        //for visualization
        Mat track_vis = cv_ptr->image.clone();
        for (auto lo : local_objects_list)
        {
            if (lo.is_track_succeed)
            {
                std::string text;
                text = "id: " + std::to_string(lo.id);
                cv::rectangle(track_vis, lo.bbox, lo.color, 4.0);
                cv::putText(track_vis, text, cv::Point(lo.bbox.x, lo.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, lo.color, 5.0);
            }
        }
        m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_vis).toImageMsg());

        //summary
        ROS_INFO("------Local Object List Summary------");
        ROS_INFO_STREAM("Local Object Num: " << local_objects_list.size());
        for (auto lo : local_objects_list)
        {
            ROS_INFO_STREAM("id: " << lo.id << "| database iamges num: " << lo.img_blocks.size());
        }
        ROS_INFO("------Summary End------");

        ROS_INFO("******Out of Data Callback******");
        std::cout << std::endl;
    }

    //TODO might use better matching strategies
    bool TrackerInterface::bbox_matching(Rect2d track_bbox, Rect2d detect_bbox)
    {
        return ((track_bbox & detect_bbox).area() / track_bbox.area() > bbox_overlap_ratio) ||
               ((track_bbox & detect_bbox).area() / detect_bbox.area() > bbox_overlap_ratio);
    }

    void TrackerInterface::load_config(ros::NodeHandle *n)
    {
        GPARAM(n, "/data_topic/lidar_topic", lidar_topic);
        GPARAM(n, "/data_topic/camera_topic", camera_topic);
        GPARAM(n, "/data_topic/depth_topic", depth_topic);

        GPARAM(n, "/tracker/track_fail_timeout_tick", track_fail_timeout_tick);
        GPARAM(n, "/tracker/bbox_overlap_ratio", bbox_overlap_ratio);
        GPARAM(n, "/tracker/track_to_reid_bbox_margin", track_to_reid_bbox_margin);

        GPARAM(n, "/local_database/height_width_ratio_min", height_width_ratio_min);
        GPARAM(n, "/local_database/height_width_ratio_max", height_width_ratio_max);
        GPARAM(n, "/local_database/blur_detection_threshold", blur_detection_threshold);
        GPARAM(n, "/local_database/record_interval", record_interval);
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

    bool TrackerInterface::update_local_database(vector<LocalObject>::iterator local_object, const cv::Mat img_block)
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
} // namespace ptl_tracker
