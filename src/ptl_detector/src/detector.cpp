#include "ptl_detector/detector.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include "ptl_msgs/ImageBlock.h"

class DataProcessHub
{
public:
    DataProcessHub(ros::NodeHandle *n);
    void image_callback(const sensor_msgs::CompressedImageConstPtr &msg_img);
    YoloPedestrainDetector m_detector;

private:
    ros::NodeHandle *nh_;
    ros::Publisher m_detected_result_pub, m_image_block_pub;
    ros::Subscriber m_image_sub;
};

DataProcessHub::DataProcessHub(ros::NodeHandle *n)
{
    nh_ = n;
    m_detected_result_pub = nh_->advertise<sensor_msgs::Image>("detection_result", 1);
    m_image_block_pub = nh_->advertise<ptl_msgs::ImageBlock>("image_blocks", 1);
    m_image_sub = nh_->subscribe(config.camera_topic, 1, &DataProcessHub::image_callback, this);
}

void DataProcessHub::image_callback(const sensor_msgs::CompressedImageConstPtr &msg_img)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat image_detection_result;
    cv_ptr = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);
    m_detector.detect_pedestrain(cv_ptr->image);

    //publish detection results for visualization
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", m_detector.result_vis).toImageMsg();
    m_detected_result_pub.publish(image_msg);

    //publish bbox and image info for re-identificaiton
    ptl_msgs::ImageBlock image_block_msg;
    sensor_msgs::ImagePtr image_origin_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
    image_block_msg.img = *image_origin_msg;
    for (auto r : m_detector.results)
    {
        if (r.type == 0)
        {
            std_msgs::UInt16MultiArray bbox;
            bbox.data.push_back(r.bbox.x);
            bbox.data.push_back(r.bbox.y);
            bbox.data.push_back(r.bbox.width);
            bbox.data.push_back(r.bbox.height);
            image_block_msg.bboxs.push_back(bbox);
        }
    }
    m_image_block_pub.publish(image_block_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ptl_detector");

    ros::NodeHandle n("ptl_detector");
    loadConfig(n);
    DataProcessHub data_process_hub(&n);
    ros::spin();
}