#include "ptl_node/node.h"
int main(int argc, char **argv)
{
    ros::init(argc, argv, "ptl_node");
    ros::NodeHandle n("~");
    ptl::node::Node ptl_node(n);
    ros::spin();
}