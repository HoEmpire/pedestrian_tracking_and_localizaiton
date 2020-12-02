#include <opencv2/opencv.hpp>
#include "opentracker/kcf/kcftracker.hpp"
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#define HOG true
#define FIXEDWINDOW true
#define MULTISCALE true
#define LAB true
#define DSST true

using namespace cv;
using namespace kcf;
using namespace std;

struct LocalObject
{
    int id;
    Rect2d bbox;
    int tracking_fail_count;
    geometry_msgs::Point position_local;
    KCFTracker *dssttracker;

    void update_tracker(Mat frame)
    {
        dssttracker->update(frame, bbox);
    }

    void init(int id_init)
    {
        id = id_init;
        dssttracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    }
};

struct GlobalObject
{
    int id;
    bool is_tracking_now;
    geometry_msgs::Point position_global;
};

struct ImageDataBase
{
    vector<Mat> images;
    vector<int> object_id;
    vector<int> image_num_per_object;
};

struct GlobalObjectInfo
{
    vector<GlobalObject> global_object_list;
    visualization_msgs::MarkerArray marker_array;
    int object_num;
};
