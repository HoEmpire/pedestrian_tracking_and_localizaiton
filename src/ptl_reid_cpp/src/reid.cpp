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

            reid_result_pub = nh_.advertise<sensor_msgs::Image>("reid_result", 1);
            detect_result_pub = nh_.advertise<sensor_msgs::Image>("detect_result", 1);
            marker_history_pub = nh_.advertise<visualization_msgs::MarkerArray>("marker_history", 1);

            id_marker = init_marker(true);
            pos_marker = init_marker(false);
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
                    }
                }
                detect_result_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", reid_detector.result_vis).toImageMsg());
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
                    //do offline inference
                    std::vector<float> feature_reid = reid_inferencer.do_inference_offline(reid_offline_buffer[0].image);
                    reid_offline_buffer[0].feat_all.insert(reid_offline_buffer[0].feat_all.end(), feature_reid.begin(), feature_reid.end());

                    //udpate database
                    int previous_max_id = reid_db.max_id;
                    int id = reid_db.query_and_update(reid_offline_buffer[0].feat_all, reid_offline_buffer[0].example_image, reid_offline_buffer[0].position);

                    //visualization
                    visualize_reid(id, previous_max_id != reid_db.max_id);

                    //remove the first object of the buffer
                    std::lock_guard<std::mutex> lk(mtx);
                    reid_offline_buffer.pop_front();
                }
                r.sleep();
            }
        }

        void Reid::visualize_reid(int reid_id, bool is_new_object)
        {
            cv::Mat result;
            std::vector<cv::Mat> example_imgs;

            //reid result visualization
            if (is_new_object)
            {
                example_imgs.push_back(reid_offline_buffer[0].example_image);
                example_imgs.push_back(reid_offline_buffer[0].example_image);
                cv::hconcat(example_imgs, result);
                cv::resize(result, result, cv::Size(512, 512));
                cv::putText(result, "Add new one!", cv::Point(10, 50),
                            cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 0, 255), 3);
            }
            else
            {
                example_imgs.push_back(reid_offline_buffer[0].example_image);
                example_imgs.push_back(reid_db.object_db[reid_id].example_image);
                cv::hconcat(example_imgs, result);
                cv::resize(result, result, cv::Size(512, 512));
                cv::putText(result, "Query:", cv::Point(10, 50),
                            cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 0, 0), 3);
                cv::putText(result, "Gallery:", cv::Point(266, 50),
                            cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            }
            cv::putText(result, "This id: " + std::to_string(reid_id), cv::Point(10, 460),
                        cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            cv::putText(result, "Total:   " + std::to_string(reid_db.max_id), cv::Point(10, 505),
                        cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            reid_result_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", result).toImageMsg());

            //history marker update
            id_marker.id = reid_id;
            id_marker.text = std::to_string(reid_id);
            id_marker.pose.position.x = reid_db.object_db[reid_id].pos.x + 0.2;
            id_marker.pose.position.y = reid_db.object_db[reid_id].pos.y + 0.2;
            id_marker.pose.position.z = reid_db.object_db[reid_id].pos.z + 0.2;

            pos_marker.id = reid_id;
            pos_marker.pose.position = reid_db.object_db[reid_id].pos;
            visualization_msgs::MarkerArray marker_history;
            marker_history.markers.push_back(id_marker);
            marker_history.markers.push_back(pos_marker);
            marker_history_pub.publish(marker_history);
        }

        visualization_msgs::Marker Reid::init_marker(bool is_id_marker)
        {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.5;
            marker.color.r = 1.0;
            marker.color.a = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            if (is_id_marker)
            {
                marker.ns = "id";
                marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            }
            else
            {
                marker.ns = "pos";
                marker.type = visualization_msgs::Marker::CUBE;
            }

            return marker;
        }
    } // namespace reid
} // namespace ptl
