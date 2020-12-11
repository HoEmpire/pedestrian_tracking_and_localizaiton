#include "ptl_detector/detector/yolo_object_detector.h"

#include <ros/package.h>

#include <string>

#include "ptl_detector/util/ros_util.h"

namespace ptl
{
    namespace detector
    {

        bool YoloObjectDetector::Init()
        {
            // get params
            auto nh_ = RosNodeHandler::Instance()->GetNh();
            XmlRpc::XmlRpcValue params;
            GPARAM("/ptl_detector/object_detector", params);
            std::string package_path = ros::package::getPath("ptl_detector");

            auto net_type = static_cast<std::string>(params["net_type"]);
            if (net_type == "YOLOV4")
            {
                config_.net_type = YOLOV4;
            }
            else if (net_type == "YOLOV4_TINY")
            {
                config_.net_type = YOLOV4_TINY;
            }
            else
            {
                AERROR << "Unsupported net type: " << net_type;
                return false;
            }
            auto precision = static_cast<std::string>(params["inference_precision"]);
            if (precision == "FP32")
            {
                config_.inference_precison = FP32;
            }
            else if (precision == "FP16")
            {
                config_.inference_precison = FP16;
            }
            else if (precision == "INT8")
            {
                config_.inference_precison = INT8;
                config_.calibration_image_list_file_txt =
                    package_path +
                    static_cast<std::string>(params["calibration_image_list_file_txt"]);
            }
            else
            {
                AERROR << "Unsupported inference precision: " << precision;
                return false;
            }

            config_.file_model_cfg =
                package_path + static_cast<std::string>(params["model_cfg"]);
            config_.file_model_weights =
                package_path + static_cast<std::string>(params["model_weights"]);
            config_.gpu_id = static_cast<int>(params["gpu_id"]);
            config_.n_max_batch = static_cast<int>(params["n_max_batch"]);
            config_.detect_thresh =
                static_cast<float>(static_cast<double>(params["detect_thresh"]));
            config_.min_width =
                static_cast<float>(static_cast<int>(params["min_width"]));
            config_.max_width =
                static_cast<float>(static_cast<int>(params["max_width"]));
            config_.min_height =
                static_cast<float>(static_cast<int>(params["min_height"]));
            config_.max_height =
                static_cast<float>(static_cast<int>(params["max_height"]));

            // init detector
            detector_ = std::make_shared<Detector>();
            detector_->init(config_);

            return true;
        }

        bool YoloObjectDetector::Init(const Config &config)
        {
            // copy config
            config_ = config;
            // init detector
            detector_ = std::make_shared<Detector>();
            detector_->init(config_);
            return true;
        }

        bool YoloObjectDetector::Detect(const cv::Mat &frame,
                                        std::vector<ObjectDetectionResult> &results)
        {
            // FIXME note that the frame should be RGB order instead of BGR
            // construct inputs
            std::vector<cv::Mat> batch_frames;
            std::vector<BatchResult> batch_results_raw;
            batch_frames.push_back(frame);
            // detect
            detector_->detect(batch_frames, batch_results_raw);
            // post-process
            FilterResults(batch_results_raw[0]);
            for (const auto &res : batch_results_raw[0])
            {
                results.emplace_back(res.rect, res.id, res.prob);
            }
            return true;
        }

        bool YoloObjectDetector::Detect(
            const std::vector<cv::Mat> &batch_frames,
            std::vector<std::vector<ObjectDetectionResult>> &batch_results)
        {
            // detect
            std::vector<BatchResult> batch_results_raw;
            detector_->detect(batch_frames, batch_results_raw);
            // post-process
            for (auto &batch_result_raw : batch_results_raw)
            {
                FilterResults(batch_result_raw);
                std::vector<ObjectDetectionResult> batch_result;
                for (const auto &result : batch_result_raw)
                {
                    batch_result.emplace_back(result.rect, result.id, result.prob);
                }
                batch_results.push_back(batch_result);
            }
            return true;
        }

        void YoloObjectDetector::SetProbThresh(float m_prob_thresh)
        {
            detector_->setProbThresh(m_prob_thresh);
        }

        void YoloObjectDetector::SetWidthLimitation(float min_value, float max_value)
        {
            config_.min_width = min_value;
            config_.max_width = max_value;
        }

        void YoloObjectDetector::SetHeightLimitation(float min_value, float max_value)
        {
            config_.min_height = min_value;
            config_.max_height = max_value;
        }

        void YoloObjectDetector::FilterResults(BatchResult &results)
        {
            auto end =
                std::remove_if(results.begin(), results.end(), [&](const Result &result) {
                    bool is_width_valid = result.rect.width <= config_.max_width &&
                                          result.rect.width >= config_.min_width;
                    bool is_height_valid = result.rect.height <= config_.max_height &&
                                           result.rect.height >= config_.min_height;
                    // FIXME check label index when not using COCO models
                    //        bool is_person_class = result.id == 0;
                    return !(is_width_valid && is_height_valid);
                });
            results.erase(end, results.end());
        }

    } // namespace detector
} // namespace ptl
