#include <iostream>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波器头文件
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/transforms.h>

#include <visualization_msgs/Marker.h>
#include <ptl_tracker/timer.hpp>
#include "ptl_tracker/point_cloud_processor.h"
using namespace std;

geometry_msgs::TransformStamped lidar2map;
pcl::PointCloud<pcl::PointXYZI> pc_filtered;
visualization_msgs::Marker markers_global;
void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc, ros::Publisher pub_vis, ros::Publisher pub_vis_global)
{
    timer a;
    pcl::PointCloud<pcl::PointXYZI> point_cloud;
    pcl::fromROSMsg(*msg_pc, point_cloud);
    ROS_INFO_STREAM("Conversion takes " << a.toc() << " seconds");
    a.tic();

    // vector<pcl::PointCloud<pcl::PointXYZI>> cloud_clustered;
    ptl::tracker::PointCloudProcessorParam pcp_param;
    ptl::tracker::PointCloudProcessor pcp(point_cloud, pcp_param);
    pcp.compute();
    pc_filtered = pcp.pc_statistical_filtered;
    ROS_INFO_STREAM("After resample, pc size = " << pcp.pc_resample.size());
    ROS_INFO_STREAM("After conditional filter, pc size = " << pcp.pc_conditional_filtered.size());
    ROS_INFO_STREAM("After statistical filter, pc size = " << pcp.pc_statistical_filtered.size());
    ROS_INFO_STREAM("Cluster size = " << pcp.pc_clustered.size());
    visualization_msgs::Marker markers;

    markers.header.frame_id = "rslidar";
    markers.header.stamp = ros::Time::now();
    markers.id = 0;
    markers.ns = "points_and_lines";
    markers.action = visualization_msgs::Marker::ADD;
    markers.pose.orientation.w = 1.0;
    markers.scale.x = 0.2;
    markers.scale.y = 0.2;
    markers.color.r = 0.0;
    markers.color.a = 1.0;
    markers.color.g = 1.0;
    markers.color.b = 0.0;
    markers.type = visualization_msgs::Marker::POINTS;
    for (auto centroid : pcp.centroids)
    {

        geometry_msgs::Point p;
        geometry_msgs::Point p_global;
        p.x = centroid.x;
        p.y = centroid.y;
        p.z = centroid.z;
        markers.points.push_back(p);

        tf2::doTransform(p, p_global, lidar2map);
        markers_global.points.push_back(p_global);
    }
    pub_vis.publish(markers);
    pub_vis_global.publish(markers_global);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_tf2_listener");

    ros::NodeHandle node("~");

    ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/pc_debug", 1);
    ros::Publisher pub_vis = node.advertise<visualization_msgs::Marker>("/pub_vis", 1);
    ros::Publisher pub_vis_global = node.advertise<visualization_msgs::Marker>("/pub_vis_global", 1);
    auto point_cloud_callback_bind = std::bind(point_cloud_callback, std::placeholders::_1, pub_vis, pub_vis_global);
    ros::Subscriber sub = node.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, point_cloud_callback_bind);

    markers_global.header.frame_id = "map";
    markers_global.header.stamp = ros::Time::now();
    markers_global.id = 0;
    markers_global.ns = "points_and_lines";
    markers_global.action = visualization_msgs::Marker::ADD;
    markers_global.pose.orientation.w = 1.0;
    markers_global.scale.x = 0.2;
    markers_global.scale.y = 0.2;
    markers_global.color.r = 1.0;
    markers_global.color.a = 1.0;
    markers_global.color.g = 0.0;
    markers_global.color.b = 0.0;
    markers_global.type = visualization_msgs::Marker::POINTS;

    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    ros::Rate rate(100.0);
    while (node.ok())
    {
        geometry_msgs::TransformStamped transformStamped;
        try
        {
            transformStamped = tfBuffer.lookupTransform("camera2_link", "rslidar",
                                                        ros::Time(0));
            lidar2map = tfBuffer.lookupTransform("map", "rslidar", ros::Time(0));
            // ROS_INFO_STREAM("translation = " << transformStamped.transform.translation);
            // ROS_INFO_STREAM("rotation = " << transformStamped.transform.rotation);
            // Eigen::Quaternionf rotation_quaternion(transformStamped.transform.rotation.w,
            //                                        transformStamped.transform.rotation.y,
            //                                        transformStamped.transform.rotation.z,
            //                                        transformStamped.transform.rotation.x);
            // Eigen::Matrix3f rotation_matrix(rotation_quaternion);
            // Eigen::Vector3f translation_vector(transformStamped.transform.translation.x,
            //                                    transformStamped.transform.translation.y,
            //                                    transformStamped.transform.translation.z);
            // Eigen::Matrix4f transformation_matrix;
            // transformation_matrix.block<3, 3>(0, 0) = rotation_matrix;
            // transformation_matrix.block<3, 1>(0, 3) = translation_vector;
            // cout << transformation_matrix << endl;
            // pcl::transformPointCloud(pc_filtered, pc_filtered, transformation_matrix);

            if (!pc_filtered.empty())
            {

                sensor_msgs::PointCloud2 pc_msgs_lidar;
                sensor_msgs::PointCloud2 pc_msgs_cam;
                pcl::toROSMsg(pc_filtered, pc_msgs_lidar);
                tf2::doTransform(pc_msgs_lidar, pc_msgs_cam, transformStamped);
                ROS_INFO_STREAM(pc_msgs_cam.header.frame_id);
                pc_msgs_cam.header.frame_id = "camera2_link";
                pub.publish(pc_msgs_cam);
                pc_filtered.clear();
            }
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }
        rate.sleep();
        ros::spinOnce();
    }
    return 0;
};