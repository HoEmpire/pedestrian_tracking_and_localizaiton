#include "ptl_tracker/tracker.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "ptl_tracker");
    ros::NodeHandle n("~");
    ptl::tracker::TrackerInterface tracker(&n);
    ros::spin();
    return 0;
}
