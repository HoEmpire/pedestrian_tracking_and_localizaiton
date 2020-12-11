#pragma once

#include "class_detector.h"
#include "ptl_detector/interface/base_object_detector.h"

namespace ptl
{
    namespace detector
    {

        class YoloObjectDetector : public BaseObjectDetector
        {
        public:
            YoloObjectDetector() = default;
            ~YoloObjectDetector() override = default;

            bool Init() override;
            bool Init(const Config &config);
            bool Detect(const cv::Mat &frame,
                        std::vector<ObjectDetectionResult> &results) override;
            bool Detect(const std::vector<cv::Mat> &batch_frames,
                        std::vector<std::vector<ObjectDetectionResult>> &batch_results);
            void Infer() override {}

            void SetProbThresh(float m_prob_thresh);
            void SetWidthLimitation(float min_value, float max_value);
            void SetHeightLimitation(float min_value, float max_value);

        private:
            void FilterResults(BatchResult &results);

            Config config_;
            std::shared_ptr<Detector> detector_;
        };

    } // namespace detector
} // namespace ptl
