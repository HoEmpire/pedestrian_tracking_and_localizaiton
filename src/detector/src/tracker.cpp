// #include <opencv2/opencv.hpp>
// #include <opencv2/tracking.hpp>
// #include <opencv2/tracking/tracker.hpp>
#include "opentracker/kcf/kcftracker.hpp"
#include <iostream>

using namespace cv;
using namespace std;
using namespace kcf;

void draw_rectangle(int event, int x, int y, int flags, void *);
Mat firstFrame;
Point previousPoint, currentPoint;
Rect2d bbox;
int main(int argc, char *argv[])
{
  VideoCapture capture;
  Mat frame;
  frame = capture.open("/home/tim/output.avi");
  if (!capture.isOpened())
  {
    printf("can not open ...\n");
    return -1;
  }
  //获取视频的第一帧,并框选目标
  capture.read(firstFrame);
  cout << "fuck1" << endl;
  if (!firstFrame.empty())
  {
    namedWindow("output", WINDOW_AUTOSIZE);
    imshow("output", firstFrame);
    setMouseCallback("output", draw_rectangle, 0);
    waitKey();
  }
  cout << "fuck2" << endl;
  //使用TrackerMIL跟踪
  bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
  KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
  // Ptr<Tracker> tracker = TrackerMIL::create();
  // Ptr<TrackerTLD> tracker = TrackerTLD::create();
  // cv::TrackerKCF::Params params;
  // params.detect_thresh = 0.3f;
  // Ptr<TrackerKCF> tracker = TrackerKCF::create();
  // Ptr<TrackerMedianFlow> tracker = TrackerMedianFlow::create();
  // Ptr<TrackerBoosting> tracker = TrackerBoosting::create();
  cout << "fuck3" << endl;
  capture.read(frame);
  cout << "frame cols: " << frame.cols << endl;
  cout << "frame rows: " << frame.rows << endl;
  cout << "fuck4" << endl;
  tracker.init(frame, bbox);
  cout << "fuck5" << endl;
  namedWindow("output", WINDOW_AUTOSIZE);
  imshow("output", frame);
  waitKey();
  cout << "fuck6" << endl;
  while (capture.read(frame))
  {
    cout << "fuck7" << endl;
    cout << "bbox.x: " << bbox.x << endl;
    cout << "bbox.y: " << bbox.y << endl;
    cout << "bbox.height: " << bbox.height << endl;
    cout << "bbox.width: " << bbox.width << endl;
    cout << "frame cols: " << frame.cols << endl;
    cout << "frame rows: " << frame.rows << endl;
    imshow("output", frame);
    waitKey();
    bool okdsst = tracker.update(frame, bbox);
    if (okdsst)
    {
      rectangle(frame, bbox, Scalar(0, 0, 255), 2, 1);
    }
    else
    {
      putText(frame, " tracking failure detected", cv::Point(10, 100), FONT_HERSHEY_SIMPLEX,
              0.75, Scalar(0, 0, 255), 2);
    }
    imshow("output", frame);
    if (waitKey(20) == 'q')
      return 0;
  }
  capture.release();
  destroyWindow("output");
  return 0;
}

//框选目标
void draw_rectangle(int event, int x, int y, int flags, void *)
{
  if (event == EVENT_LBUTTONDOWN)
  {
    previousPoint = Point(x, y);
  }
  else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
  {
    Mat tmp;
    firstFrame.copyTo(tmp);
    currentPoint = Point(x, y);
    rectangle(tmp, previousPoint, currentPoint, Scalar(0, 255, 0, 0), 1, 8, 0);
    imshow("output", tmp);
  }
  else if (event == EVENT_LBUTTONUP)
  {
    bbox.x = previousPoint.x;
    bbox.y = previousPoint.y;
    bbox.width = abs(previousPoint.x - currentPoint.x);
    bbox.height = abs(previousPoint.y - currentPoint.y);
  }
  else if (event == EVENT_RBUTTONUP)
  {
    destroyWindow("output");
  }
}