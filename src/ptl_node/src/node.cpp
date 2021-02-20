#include "ptl_node/node.h"
namespace ptl
{
    namespace node
    {
        template <class T>
        void GPARAM(const ros::NodeHandle &n, std::string param_path, T &param)
        {
            if (!n.getParam(param_path, param))
                ROS_ERROR_STREAM("Load param from " << param_path << " failed...");
        }

        Node::Node(const ros::NodeHandle &n) : nh_(n), ptl_reid(n), ptl_tracker(n)
        {
            load_config();
            ptl_reid.init();
            ptl_tracker.init(false);
            lidar_sub = nh_.subscribe(node_param.lidar_topic, 1, &Node::lidar_callback, this);
            camera_sub = nh_.subscribe(node_param.camera_topic, 1, &Node::camera_callback, this);
        }

        void Node::load_config()
        {
            GPARAM(nh_, "/node/lidar_topic", node_param.lidar_topic);
            GPARAM(nh_, "/node/camera_topic", node_param.camera_topic);
            GPARAM(nh_, "/node/detect_every_k_frame", node_param.detect_every_k_frame);
            GPARAM(nh_, "/node/min_offline_query_data_size", node_param.min_offline_query_data_size);
        }

        void Node::camera_callback(const sensor_msgs::CompressedImageConstPtr &image)
        {
            ROS_INFO("******Into camera callback******");
            timer t;
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

            //create a real-time reid thread for every k frames to detect objects in the image
            if (frame_count % node_param.detect_every_k_frame == 0)
            {
                ROS_INFO("Create real-time reid thread!");
                reid_real_time_thread = new std::thread(&Node::reid_real_time, this, cv_ptr->image, cv_ptr->header.stamp);
            }
            frame_count++;

            // track by optical flow tracker
            std::vector<tracker::LocalObject> dead_object = ptl_tracker.update_bbox_by_tracker(cv_ptr->image, cv_ptr->header.stamp);

            // push dead object to the offlin reid buffer
            if (!dead_object.empty())
            {
                for (auto dio : dead_object)
                {
                    if (dio.img_blocks.size() + dio.features_vector.size() / 2048 > node_param.min_offline_query_data_size) //TODO hardcode in here
                    {
                        std::lock_guard<std::mutex> lk(ptl_reid.mtx);
                        ptl_reid.reid_offline_buffer.emplace_back(dio.example_image, dio.img_blocks, dio.position, dio.features_vector);
                    }
                }
            }
            ROS_INFO_STREAM("Camera callback takes " << t.toc() * 1000 << " ms.");
            ROS_INFO("******Out of camera callback******");
            std::cout << std::endl;
        }

        void Node::lidar_callback(const sensor_msgs::PointCloud2ConstPtr &point_cloud)
        {
            ptl_tracker.lidar_tracker_callback(point_cloud);
        }

        void Node::reid_real_time(const cv::Mat &image, const ros::Time &time_now)
        {
            timer t;
            //do real time reid first
            std::vector<cv::Rect2d> bboxes;
            std::vector<float> feat;
            ptl_reid.reid_realtime(image, bboxes, feat);
            ROS_INFO_STREAM("(0x0): Real-time Reid takes " << t.toc() * 1000 << " ms.");

            //then update the tracker
            t.tic();
            if (!feat.empty())
            {
                ptl_tracker.update_bbox_by_detector(image, bboxes, feat, time_now);
            }
            ROS_INFO_STREAM("(0x0): Update tracker takes " << t.toc() * 1000 << " ms.");
        }
    } // namespace node
} // namespace ptl
