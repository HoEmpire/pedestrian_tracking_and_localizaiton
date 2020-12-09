#include "ptl_tracker/tracker.h"

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
        m_track_to_reid_pub = n->advertise<ptl_msgs::ImageBlock>("tracker_to_reid", 1);
        m_detector_sub = n->subscribe("/ptl_detector/detector_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
        m_data_sub = n->subscribe(camera_topic, 1, &TrackerInterface::data_callback, this);
    }

    void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
    {
        if (msg->ids.empty())
            return;
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat image_detection_result;
        cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);

        //match the previous one
        lock_guard<mutex> lk(mtx); //加锁pub
        vector<std_msgs::UInt16MultiArray> bboxs = msg->bboxs;

        bool erase_flag = false;
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
            }
        }
    }

    void TrackerInterface::data_callback(const sensor_msgs::CompressedImageConstPtr &msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat image_detection_result;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        bool is_blur = blur_detection(cv_ptr->image);
        lock_guard<mutex> lk(mtx); //加锁
        for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();)
        {
            if (lo->tracking_fail_count >= track_fail_timeout_tick)
            {
                lo = local_objects_list.erase(lo);
                continue;
            }
            lo->update_tracker(cv_ptr->image);
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
    }

    //TODO might use better matching strategies
    bool TrackerInterface::bbox_matching(Rect2d track_bbox, Rect2d detect_bbox)
    {
        return ((track_bbox & detect_bbox).area() / track_bbox.area() > bbox_overlap_ratio) ||
               ((track_bbox & detect_bbox).area() / detect_bbox.area() > bbox_overlap_ratio);
    }

    void TrackerInterface::load_config(ros::NodeHandle *n)
    {
        n->getParam("/data_topic/lidar_topic", lidar_topic);
        n->getParam("/data_topic/camera_topic", camera_topic);
        n->getParam("/data_topic/depth_topic", depth_topic);

        n->getParam("/tracker/track_fail_timeout_tick", track_fail_timeout_tick);
        n->getParam("/tracker/bbox_overlap_ratio", bbox_overlap_ratio);
        n->getParam("/tracker/track_to_reid_bbox_margin", track_to_reid_bbox_margin);

        n->getParam("/local_database/height_width_ratio_min", height_width_ratio_min);
        n->getParam("/local_database/height_width_ratio_max", height_width_ratio_max);
        n->getParam("/local_database/blur_detection_threshold", blur_detection_threshold);
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
} // namespace ptl_tracker
