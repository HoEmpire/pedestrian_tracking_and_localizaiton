#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>
using namespace std;
using namespace Eigen;

template <class T>
void GPARAM(ros::NodeHandle *n, std::string param_path, T &param)
{
    if (!n->getParam(param_path, param))
        ROS_ERROR_STREAM("Load param from " << param_path << " failed...");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tf_topic");
    string map_frame, lidar_frame, camera_frame;
    vector<double> rotation, translation;
    ros::NodeHandle n("~");
    GPARAM(&n, "/map_frame", map_frame);
    GPARAM(&n, "/lidar_frame", lidar_frame);
    GPARAM(&n, "/camera_frame", camera_frame);
    GPARAM(&n, "/rotation", rotation);
    GPARAM(&n, "/translation", translation);
    Quaterniond q(rotation[3], rotation[0], rotation[1], rotation[2]);

    geometry_msgs::TransformStamped map2lidar, lidar2camera;

    map2lidar.header.frame_id = map_frame;
    map2lidar.child_frame_id = lidar_frame;
    map2lidar.transform.translation.x = 0.0;
    map2lidar.transform.translation.y = 0.0;
    map2lidar.transform.translation.z = 0.0;
    map2lidar.transform.rotation.x = 0.0;
    map2lidar.transform.rotation.y = 0.0;
    map2lidar.transform.rotation.z = 0.0;
    map2lidar.transform.rotation.w = 1.0;

    lidar2camera.header.frame_id = lidar_frame;
    lidar2camera.child_frame_id = camera_frame;
    lidar2camera.transform.translation.x = translation[0];
    lidar2camera.transform.translation.y = translation[1];
    lidar2camera.transform.translation.z = translation[2];
    lidar2camera.transform.rotation.x = q.x();
    lidar2camera.transform.rotation.y = q.y();
    lidar2camera.transform.rotation.z = q.z();
    lidar2camera.transform.rotation.w = q.w();

    static tf2_ros::StaticTransformBroadcaster br;
    br.sendTransform(map2lidar);
    br.sendTransform(lidar2camera);
    ros::spin();
}