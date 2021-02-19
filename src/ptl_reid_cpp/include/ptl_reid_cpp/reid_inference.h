#pragma once
#include <iostream>
#include <fstream>
#include <memory>

#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>

#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <ros/package.h>
namespace ptl
{
    namespace reid
    {

        class Logger : public nvinfer1::ILogger
        {
        public:
            void log(Severity severity, const char *msg) override
            {
                // remove this 'if' if you need more logged info
                if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
                {
                    std::cout << msg << "\n";
                }
            }
        };

        // destroy TensorRT objects if something goes wrong
        struct TRTDestroy
        {
            template <class T>
            void operator()(T *obj) const
            {
                if (obj)
                {
                    obj->destroy();
                }
            }
        };

        template <class T>
        using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

        struct InferenceParam
        {
            int inference_real_time_batch_size = 4;
            int inference_offline_batch_size = 8;
            std::string engine_file_name = "reid_engine.engine";
            std::string onnx_file_name = "reid.onnx";
        };

        class ReidInference
        {
        public:
            ReidInference() = default;
            ReidInference(const InferenceParam &reid_param) : reid_param_(reid_param) {}

            //initialize the engine
            void init();

            //do inference for real time thread
            std::vector<float> do_inference_real_time(const cv::Mat &image, const std::vector<cv::Rect2d> &bboxes);

            //do inference for offline thread (back-end reid)
            std::vector<float> do_inference_offline(const std::vector<cv::Mat> &images);

        private:
            bool load_engine(const std::string &engine_path);
            bool parse_onnx_model(const std::string &model_path);
            void save_engine(const std::string &engine_path);

            void data_preprocesss(const cv::Mat &image, std::vector<cv::Rect2d>::const_iterator bboxes_begin,
                                  std::vector<cv::Rect2d>::const_iterator bboxes_end, float *gpu_input);
            void data_preprocesss(std::vector<cv::Mat>::const_iterator image_begin,
                                  std::vector<cv::Mat>::const_iterator image_end,
                                  float *gpu_input);

            void inference(std::vector<void *> &buffers, bool is_real_time);

            void result_postprocess(float *gpu_output, std::vector<float> &cpu_output);

            TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
            TRTUniquePtr<nvinfer1::IExecutionContext> context_real_time{nullptr};
            TRTUniquePtr<nvinfer1::IExecutionContext> context_offline{nullptr};
            Logger gLogger;

            InferenceParam reid_param_;

            std::vector<void *> buffers;
        };
    } // namespace reid
} // namespace ptl