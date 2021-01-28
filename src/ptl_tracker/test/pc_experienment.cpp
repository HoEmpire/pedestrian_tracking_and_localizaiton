#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>

using namespace std;

void callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc)
{
    cout << "fuck" << endl;
    pcl::PointCloud<pcl::PointXYZI> point_cloud;
    pcl::fromROSMsg(*msg_pc, point_cloud);
    vector<double> range_class = {100, 200, 300, 350, 400, 450};
    vector<int> range_count = {0, 0, 0, 0, 0, 0};
    for (auto p : point_cloud.points)
    {
        int i = 0;
        for (auto rc : range_class)
        {
            if (p.x > rc)
            {
                range_count[i]++;
            }
            i++;
        }
    }

    //report
    for (int i = 0; i < 6; i++)
    {
        cout << "range " << range_class[i] << " has "
             << range_count[i] << " points." << endl;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "shit");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, callback);
    ros::spin();
}