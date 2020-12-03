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
  ros::Publisher m_track_vis_pub;
  ros::Subscriber m_detector_sub, m_data_sub;
  mutex mtx;
};

TrackerInterface::TrackerInterface(ros::NodeHandle *n)
{
  nh_ = n;
  m_track_vis_pub = n->advertise<sensor_msgs::Image>("tracker_results", 1);
  m_detector_sub = n->subscribe("/ptl_detector/detector_to_tracker", 1, &TrackerInterface::detector_result_callback, this);
  m_data_sub = n->subscribe(config.camera_topic, 1, &TrackerInterface::data_callback, this);
}

void TrackerInterface::detector_result_callback(const ptl_msgs::ImageBlockPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat image_detection_result;
  cv_ptr = cv_bridge::toCvCopy(msg->img, sensor_msgs::image_encodings::BGR8);
  vector<std_msgs::UInt16MultiArray> bboxs = msg->bboxs;
  //match the previous one
  if (!local_objects_list.empty())
  {
    for (auto b = bboxs.begin(); b != bboxs.end(); b++)
    {

      for (auto lo : local_objects_list)
      {
        if (bbox_matching(lo.bbox, Rect2d(b->data[0], b->data[1], b->data[2], b->data[3])))
        {
          bboxs.erase(b);            //删除该对象
          lock_guard<mutex> lk(mtx); //加锁
          lo.bbox = Rect2d(b->data[0], b->data[1], b->data[2], b->data[3]);
          break;
        }
      }
    }
  }

  //add the new ones
  if (!bboxs.empty())
  {
    for (auto b : bboxs)
    {
      LocalObject new_object;
      new_object.init(global_objects.object_num++, Rect2d(b.data[0], b.data[1], b.data[2], b.data[3]), cv_ptr->image);
    }
  }
}

void TrackerInterface::data_callback(const sensor_msgs::CompressedImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat image_detection_result;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  for (auto lo = local_objects_list.begin(); lo < local_objects_list.end(); lo++)
  {
    if (lo->tracking_fail_count >= config.track_fail_timeout_tick)
    {
      local_objects_list.erase(lo);
      continue;
    }

    lock_guard<mutex> lk(mtx); //加锁
    lo->update_tracker(cv_ptr->image);
  }

  //for visualization
  Mat track_vis = cv_ptr->image.clone();
  for (auto lo : local_objects_list)
  {
    string text;
    text = "id: " + to_string(lo.id);
    cv::rectangle(track_vis, lo.bbox, lo.color, 5.0);
    cv::putText(track_vis, text, cv::Point(lo.bbox.x, lo.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, lo.color, 5.0);
  }
  m_track_vis_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_vis).toImageMsg());
}

//TODO might use better matching strategies
bool TrackerInterface::bbox_matching(Rect2d track_bbox, Rect2d detect_bbox)
{
  return sqrt((track_bbox.x - detect_bbox.x) * (track_bbox.x - detect_bbox.x) +
              (track_bbox.y - detect_bbox.y) * (track_bbox.y - detect_bbox.y)) < config.bbox_match_pixel_dis;
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
