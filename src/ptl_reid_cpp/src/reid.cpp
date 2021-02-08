#include "ptl_reid_cpp/reid.h"

namespace ptl
{
    namespace reid
    {
        void Reid::init()
        {
            load_config();
            reid_db = ReidDatabase(db_param);
            reid_inferencer = ReidInference(inference_param);
            reid_inferencer.init();
            reid_offline_thread = std::thread(&Reid::reid_offline, this);
        }

        void Reid::load_config()
        {
            GPARAM(nh_, "/reid_db/similarity_test_threshold", db_param.similarity_test_threshold);
            GPARAM(nh_, "/reid_db/same_id_threshold", db_param.same_id_threshold);
            GPARAM(nh_, "/reid_db/batch_ratio", db_param.batch_ratio);
            GPARAM(nh_, "/reid_db/max_feat_num_one_object", db_param.max_feat_num_one_object);
            GPARAM(nh_, "/reid_db/use_inverted_file_db_threshold", db_param.use_inverted_file_db_threshold);
            GPARAM(nh_, "/reid_db/feat_dimension", db_param.feat_dimension);
            GPARAM(nh_, "/reid_db/find_first_k", db_param.find_first_k);
            GPARAM(nh_, "/reid_db/nlist_ratio", db_param.nlist_ratio);
            GPARAM(nh_, "/reid_db/sim_check_start_threshold", db_param.sim_check_start_threshold);

            GPARAM(nh_, "/reid_inference/engine_file_name", inference_param.engine_file_name);
            GPARAM(nh_, "/reid_inference/onnx_file_name", inference_param.onnx_file_name);
            GPARAM(nh_, "/reid_inference/inference_offline_batch_size", inference_param.inference_offline_batch_size);
            GPARAM(nh_, "/reid_inference/inference_real_time_batch_size", inference_param.inference_real_time_batch_size);
        }

        void Reid::reid_realtime(const cv::Mat &image, std::vector<cv::Rect2d> &bboxes, std::vector<float> &feat)
        {
            reid_detector.detect_pedestrain(image);
            bboxes.clear();
            feat.clear();
            if (!reid_detector.results.empty())
            {
                for (auto r : reid_detector.results)
                {
                    if (r.type == 0) //TODO hard code in here
                    {
                        bboxes.push_back(r.bbox);
                        std::cout << "Orz::" << r.bbox << std::endl;
                    }
                }
            }

            // do reid inference
            if (!bboxes.empty())
            {
                feat = reid_inferencer.do_inference_real_time(image, bboxes);
            }
            //TODO dont forget visulize the detector
        }

        void Reid::reid_offline()
        {
            ros::Rate r(100);
            while (ros::ok())
            {
                if (!reid_offline_buffer.empty())
                {
                    ROS_INFO("fuck1");
                    //do offline inference
                    std::vector<float> feature_reid = reid_inferencer.do_inference_offline(reid_offline_buffer[0].image);
                    ROS_INFO("fuck2");
                    reid_offline_buffer[0].feat_all.insert(reid_offline_buffer[0].feat_all.end(), feature_reid.begin(), feature_reid.end());

                    //udpate database
                    reid_db.query_and_update(reid_offline_buffer[0].feat_all, reid_offline_buffer[0].example_image, reid_offline_buffer[0].position);

                    //remove the first object of the buffer
                    reid_offline_buffer.pop_front();
                }
                r.sleep();
            }
        }
    } // namespace reid
} // namespace ptl
