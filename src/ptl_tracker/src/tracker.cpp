#include <iostream>
#include <mutex>

#include "ros/ros.h"
#include "std_msgs/UInt16MultiArray.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include "cv_bridge/cv_bridge.h"
#include <image_transport/image_transport.h>

#include "opentracker/kcf/kcftracker.hpp"

#include "ptl_tracker/type.h"
#include "ptl_tracker/util.h"
#include "ptl_msgs/ImageBlock.h"

using namespace cv;
using namespace std;
using namespace kcf;

class TrackerInterface
{
public:
    TrackerInterface(ros::NodeHandle *n);

private:
    void detector_result_callback(const ptl_msgs::ImageBlockPtr &msg);
    void data_callback(const sensor_msgs::CompressedImageConstPtr &msg);
    bool bbox_matching(Rect2d track_bbox, Rect2d detect_bbox);

public:
    vector<LocalObject> local_objects_list;
    GlobalObjectInfo global_objects;

private:
    ros::NodeHandle *nh_;
    ros::Publisher m_track_vis_pub, m_track_to_reid_pub;
    ros::Subscriber m_detector_sub, m_data_sub;
    mutex mtx;
};

TrackerInterface::TrackerInterface(ros::NodeHandle *n)
{
    nh_ = n;
    m_track_vis_pub = n->advertise<sensor_msgs::Image>("tracker_results", 1);
    m_track_to_reid_pub = n->advertise<ptl_msgs::ImageBlock>("tracker_to_reid", 1);
    m_detector_sub = n->subscribe("/ptl_detector/detector_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
    m_data_sub = n->subscribe(config.camera_topic, 1, &TrackerInterface::data_callback, this);
}

void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
{
    if (msg->ids.empty())
        return;
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat image_detection_result;
    cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);

    ptl_msgs::ImageBlock pub_msg = *msg;
    //match the previous one
    lock_guard<mutex> lk(mtx); //加锁pub
    if (!local_objects_list.empty())
    {
        for (int i = 0; i < pub_msg.bboxs.size(); i++)
        {
            for (auto lo = local_objects_list.begin(); lo != local_objects_list.end(); lo++)
            {
                if (bbox_matching(lo->bbox, Rect2d(pub_msg.bboxs[i].data[0], pub_msg.bboxs[i].data[1],
                                                   pub_msg.bboxs[i].data[2], pub_msg.bboxs[i].data[3])))
                {
                    ROS_INFO_STREAM("Object " << lo->id << " re-detected!");
                    lo->reinit(Rect2d(pub_msg.bboxs[i].data[0], pub_msg.bboxs[i].data[1],
                                      pub_msg.bboxs[i].data[2], pub_msg.bboxs[i].data[3]),
                               cv_ptr->image); //TODO might try to improve efficiency in here                                                                                //删除该对象
                    pub_msg.ids[i].data = lo->id;
                    break;
                }
            }
        }
    }
    m_track_to_reid_pub.publish(pub_msg);

    // bool erase_flag = false;
    // if (!local_objects_list.empty())
    // {
    //     for (auto b = bboxs.begin(); b != bboxs.end();)
    //     {
    //         for (auto lo = local_objects_list.begin(); lo != local_objects_list.end(); lo++)
    //         {
    //             if (bbox_matching(lo->bbox, Rect2d(b->data[0], b->data[1], b->data[2], b->data[3])))
    //             {
    //                 ROS_INFO_STREAM("Object " << lo->id << " re-detected!");
    //                 lo->reinit(Rect2d(b->data[0], b->data[1], b->data[2], b->data[3]), cv_ptr->image);//TODO might try to improve efficiency in here
    //                 b = bboxs.erase(b); //删除该对象
    //                 erase_flag = true;
    //                 break;
    //             }
    //         }
    //         if (erase_flag == false)
    //         {
    //             b++;
    //         }
    //         else
    //         {
    //             erase_flag = false;
    //         }
    //     }
    // }

    // //add the new ones
    // if (!bboxs.empty())
    // {
    //     for (auto b : bboxs)
    //     {
    //         LocalObject new_object;
    //         ROS_INFO_STREAM("Adding Tracking Object with ID:" << global_objects.object_num);
    //         new_object.init(global_objects.object_num++, Rect2d(b.data[0], b.data[1], b.data[2], b.data[3]), cv_ptr->image);
    //         local_objects_list.push_back(new_object);
    //     }
    // }
}

void TrackerInterface::data_callback(const sensor_msgs::CompressedImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat image_detection_result;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    lock_guard<mutex> lk(mtx); //加锁
    for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();)
    {
        if (lo->tracking_fail_count >= config.track_fail_timeout_tick)
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
            string text;
            text = "id: " + to_string(lo.id);
            cv::rectangle(track_vis, lo.bbox, lo.color, 5.0);
            cv::putText(track_vis, text, cv::Point(lo.bbox.x, lo.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, lo.color, 5.0);
        }
    }
    m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_vis).toImageMsg());
}

//TODO might use better matching strategies
bool TrackerInterface::bbox_matching(Rect2d track_bbox, Rect2d detect_bbox)
{

    // return sqrt(pow(track_bbox.x + track_bbox.width / 2 - detect_bbox.x + detect_bbox.width / 2, 2) +
    //             pow(track_bbox.y + track_bbox.height / 2 - detect_bbox.y + detect_bbox.height / 2, 2)) < config.bbox_match_pixel_dis;
    return sqrt(pow(track_bbox.x - detect_bbox.x, 2) +
                pow(track_bbox.y - detect_bbox.y, 2)) < config.bbox_match_pixel_dis;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "ptl_tracker");
    ros::NodeHandle n("ptl_tracker");
    loadConfig(n);
    TrackerInterface tracker(&n);
    ros::spin();
    return 0;
}
