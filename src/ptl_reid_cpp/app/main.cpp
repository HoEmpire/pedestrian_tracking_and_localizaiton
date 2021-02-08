#include "ptl_reid_cpp/reid.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "ptl_reid");
    ros::NodeHandle n("~");
    ptl::reid::Reid reid(n);
    reid.init();
    ros::spin();
    return 0;
}
