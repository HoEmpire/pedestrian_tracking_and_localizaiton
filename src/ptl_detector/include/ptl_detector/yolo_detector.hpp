#include "ptl_detector/util/util.h"
#include "opencv2/opencv.hpp"
namespace ptl
{
    namespace detector
    {
        struct YoloParam
        {
            bool use_compressed_image = true;
            std::unordered_map<std::string, ModelType> net_type_table = {{"YOLOV3", ModelType::YOLOV3}, {"YOLOV3_TINY", ModelType::YOLOV3_TINY}, {"YOLOV4", ModelType::YOLOV4}, {"YOLOV4_TINY", ModelType::YOLOV4_TINY}, {"YOLOV4", ModelType::YOLOV4}, {"YOLOV5", ModelType::YOLOV5}};
            std::unordered_map<std::string, Precision> precision_table = {{"INT8", Precision::INT8}, {"FP16", Precision::FP16}, {"FP32", Precision::FP32}};

            string lidar_topic, camera_topic, depth_topic;

            string cam_net_type = "YOLOV4_TINY";
            string cam_file_model_cfg;
            string cam_file_model_weights;
            string cam_inference_precison = "FP32";
            int cam_n_max_batch = 1;
            float cam_min_width = 0;
            float cam_max_width = 640;
            float cam_prob_threshold = 0.5;
            float cam_min_height = 0;
            float cam_max_height = 480;
        };

        class YoloPedestrainDetector
        {
        public:
            YoloPedestrainDetector(const ros::NodeHandle &n) : nh_(n)
            {
                loadConfig(n);
                package_path = ros::package::getPath("ptl_detector");
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

            void detect_pedestrain(const cv::Mat &image)
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

            void loadConfig(const ros::NodeHandle &n)
            {
                n.getParam("/cam/cam_net_type", config.cam_net_type);
                n.getParam("/cam/cam_file_model_cfg", config.cam_file_model_cfg);
                n.getParam("/cam/cam_file_model_weights", config.cam_file_model_weights);
                n.getParam("/cam/cam_inference_precison", config.cam_inference_precison);
                n.getParam("/cam/cam_n_max_batch", config.cam_file_model_cfg);
                n.getParam("/cam/cam_prob_threshold", config.cam_prob_threshold);
                n.getParam("/cam/cam_min_width", config.cam_min_width);
                n.getParam("/cam/cam_max_width", config.cam_max_width);
                n.getParam("/cam/cam_min_height", config.cam_min_height);
                n.getParam("/cam/cam_max_height", config.cam_max_height);
            }

        public:
            cv::Mat result_vis;
            std::vector<ObjectDetectionResult> results;

        private:
            ros::NodeHandle nh_;

            YoloParam config;
            Config config_cam;
            std::string package_path;
            YoloObjectDetector detector;
        };
    } // namespace detector
} // namespace ptl
