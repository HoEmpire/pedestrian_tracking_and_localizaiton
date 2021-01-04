#include <iostream>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>

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
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#include <visualization_msgs/Marker.h>
#include <ptl_tracker/timer.hpp>
using namespace std;

pcl::PointCloud<pcl::PointXYZI> pc_filtered;
void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc, ros::Publisher pub_vis)
{
    timer a;
    pcl::PointCloud<pcl::PointXYZI> point_cloud;
    pcl::fromROSMsg(*msg_pc, point_cloud);
    ROS_INFO_STREAM("Conversion takes " << a.toc() << " seconds");
    a.tic();

    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(point_cloud.makeShared());
    sor.setLeafSize(0.1, 0.1, 0.1);
    sor.filter(point_cloud);
    ROS_INFO_STREAM("Resample takes " << a.toc() << " seconds");
    a.tic();
    // filter
    // pcl::PassThrough<pcl::PointXYZI> pass;
    pcl::PointCloud<pcl::PointXYZI> cloud_filtered_pre;
    // build the condition
    pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZI>());
    // 设置Z轴的限制范围 [-1.5, 1.0]
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT, 0.0)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT, 10.0)));
    // 设置Y轴的限制范围 [-5.0, 5.0]
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::GT, 0.0)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::LT, 5.0)));

    pcl::ConditionalRemoval<pcl::PointXYZI> conditional_filter;
    conditional_filter.setCondition(range_cond);
    conditional_filter.setInputCloud(point_cloud.makeShared()); //设置输入点云
    conditional_filter.setKeepOrganized(false);
    conditional_filter.filter(cloud_filtered_pre); //执行滤波，保存过滤结果在cloud_filtered

    ROS_INFO_STREAM("Before filtering, pc has " << point_cloud.size() << " points");
    ROS_INFO_STREAM("After pass-through filtering, pc has " << cloud_filtered_pre.size() << " points");
    ROS_INFO_STREAM("Conditional filter takes " << a.toc() << " seconds");
    a.tic();
    if (cloud_filtered_pre.size() == 0)
        return;

    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> statistical_filter;
    statistical_filter.setStddevMulThresh(0.1);
    statistical_filter.setMeanK(30);
    // statistical_filter.setKeepOrganized(true);
    statistical_filter.setInputCloud(cloud_filtered_pre.makeShared());
    statistical_filter.filter(pc_filtered);
    ROS_INFO_STREAM("Statistical filter takes " << a.toc() << " seconds");
    a.tic();

    // clustering
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(pc_filtered.makeShared()); //创建点云索引向量，用于存储实际的点云信息
    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.5); //设置近邻搜索的搜索半径为2cm
    ec.setMinClusterSize(20);    //设置一个聚类需要的最少点数目为100
    ec.setMaxClusterSize(20000); //设置一个聚类需要的最大点数目为25000
    ec.setSearchMethod(tree);    //设置点云的搜索机制
    ec.setInputCloud(pc_filtered.makeShared());
    ec.extract(cluster_indices); //从点云中提取聚类，并将点云索引保存在cluster_indices中
    ROS_INFO_STREAM("Clustering takes " << a.toc() << " seconds");
    a.tic();

    vector<pcl::PointCloud<pcl::PointXYZI>> cloud_clustered;
    visualization_msgs::Marker markers;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZI> cloud_cluster;
        //创建新的点云数据集cloud_cluster，将所有当前聚类写入到点云数据集中
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            cloud_cluster.points.push_back(pc_filtered.points[*pit]);
            cloud_cluster.width = cloud_cluster.size();
            cloud_cluster.height = 1;
            cloud_cluster.is_dense = true;
        }

        cloud_clustered.push_back(cloud_cluster);
    }

    markers.header.frame_id = "camera2_link";
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
    for (uint i = 0; i < cloud_clustered.size(); i++)
    {

        geometry_msgs::Point p;
        pcl::PointXYZ centroid;
        pcl::computeCentroid(cloud_clustered[i], centroid);
        p.x = centroid.x;
        p.y = centroid.y;
        p.z = centroid.z;
        markers.points.push_back(p);
    }
    pub_vis.publish(markers);
    ROS_INFO_STREAM("After statistical filtering, pc has " << pc_filtered.size() << " points");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_tf2_listener");

    ros::NodeHandle node("~");

    ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/pc_debug", 1);
    ros::Publisher pub_vis = node.advertise<visualization_msgs::Marker>("/pub_vis", 1);
    auto point_cloud_callback_bind = std::bind(point_cloud_callback, std::placeholders::_1, pub_vis);
    ros::Subscriber sub = node.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, point_cloud_callback_bind);

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
            // ROS_INFO_STREAM("translation = " << transformStamped.transform.translation);
            // ROS_INFO_STREAM("rotation = " << transformStamped.transform.rotation);
            Eigen::Quaternionf rotation_quaternion(transformStamped.transform.rotation.w,
                                                   transformStamped.transform.rotation.y,
                                                   transformStamped.transform.rotation.z,
                                                   transformStamped.transform.rotation.x);
            Eigen::Matrix3f rotation_matrix(rotation_quaternion);
            Eigen::Vector3f translation_vector(transformStamped.transform.translation.x,
                                               transformStamped.transform.translation.y,
                                               transformStamped.transform.translation.z);
            Eigen::Matrix4f transformation_matrix;
            transformation_matrix.block<3, 3>(0, 0) = rotation_matrix;
            transformation_matrix.block<3, 1>(0, 3) = translation_vector;
            // cout << transformation_matrix << endl;
            pcl::transformPointCloud(pc_filtered, pc_filtered, transformation_matrix);
            sensor_msgs::PointCloud2 pc_msgs;
            if (!pc_filtered.empty())
            {
                pcl::toROSMsg(pc_filtered, pc_msgs);
                ROS_INFO_STREAM(pc_msgs.header.frame_id);
                pc_msgs.header.frame_id = "camera2_link";
                pub.publish(pc_msgs);
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