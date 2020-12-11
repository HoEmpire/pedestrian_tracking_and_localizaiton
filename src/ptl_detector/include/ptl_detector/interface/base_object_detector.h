#pragma once

#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "ptl_detector/geometry/box.h"
#include "ptl_detector/interface/base_inference.h"

namespace ptl
{
    namespace detector
    {

        struct ObjectDetectionResult
        {
            cv::Rect bbox;
            int type = -1;
            float prob = 0.0;
            ObjectDetectionResult(cv::Rect bbox, int type, float prob)
                : bbox(std::move(bbox)), type(type), prob(prob) {}
        };

        class BaseObjectDetector : public BaseInference
        {
        public:
            void Infer() override = 0;
            virtual bool Detect(const cv::Mat &frame,
                                std::vector<ObjectDetectionResult> &result) = 0;
        };

    } // namespace detector
} // namespace ptl