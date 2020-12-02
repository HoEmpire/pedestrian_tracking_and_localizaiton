#include "pedestrain_tracking_and_localizaiton/util.h"
#include "opencv2/opencv.hpp"

class YoloPedestrainDetector
{
public:
    YoloPedestrainDetector()
    {
        package_path = ros::package::getPath("usfs_inference");
        config_cam.net_type = config.net_type_table[config.cam_net_type];
        config_cam.file_model_cfg = package_path + config.cam_file_model_cfg;
        config_cam.file_model_weights = package_path + config.cam_file_model_weights;
        config_cam.inference_precison = config.precision_table[config.cam_inference_precison]; // use FP16 for Jetson Xavier NX
        config_cam.n_max_batch = config.cam_n_max_batch;
        config_cam.detect_thresh = config.cam_prob_threshold;
        config_cam.min_width = config.cam_min_width;
        config_cam.max_width = config.cam_max_width;
        config_cam.min_height = config.cam_min_height;
        config_cam.max_height = config.cam_max_height;
        detector.Init(config_cam);
    }

    void detect_pedestrain(cv::Mat image)
    {
        results.clear();
        detector.Detect(image, results);
        result_vis = image.clone();
        for (auto r : results)
        {
            cv::Scalar draw_color;
            string text;
            if (r.type == 0)
            {
                text = "pedestrain: " + to_string(r.prob);
                draw_color = cv::Scalar(0, 255, 0);
                cv::rectangle(result_vis, r.bbox, draw_color, 5.0);
                cv::putText(result_vis, text, cv::Point(r.bbox.x, r.bbox.y), cv::FONT_HERSHEY_COMPLEX, 1.5, draw_color, 6.0);
            }
        }
    }

public:
    cv::Mat result_vis;
    std::vector<usfs::inference::ObjectDetectionResult> results;

private:
    Config config_cam;
    std::string package_path;
    usfs::inference::YoloObjectDetector detector;
};