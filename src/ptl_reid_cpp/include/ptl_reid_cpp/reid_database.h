#pragma once
#include <geometry_msgs/Point.h>
#include <opencv/cv.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

namespace ptl
{
    namespace reid
    {

        struct DataBaseParam
        {
            float similarity_test_threshold = 150.0;
            float same_id_threshold = 350.0;
            float batch_ratio = 0.55;
            float max_feat_num_one_object = 50;

            // if the total feature is larger than this, use Inverted file in feature database to increase search efficency
            int use_inverted_file_db_threshold = 2500;
            int feat_dimension = 2048;
            int find_first_k = 2;
            int nlist_ratio = 50;
            int sim_check_start_threshold = 5;
        };

        class ObjectType
        {

        public:
            ObjectType(const int id_init, const cv::Mat &img, const geometry_msgs::Point &pos_init)
                : id(id_init), example_image(img), pos(pos_init), feat_dimension(2048), db(2048) {}

            ObjectType(const int id_init, const cv::Mat &img, const geometry_msgs::Point &pos_init,
                       const int feat_dim)
                : id(id_init), example_image(img), pos(pos_init), feat_dimension(feat_dim), db(feat_dim) {}

            void update_pos(const geometry_msgs::Point &pos_new)
            {
                pos = pos_new;
            }

            faiss::IndexFlatL2 db;
            int feat_num = 0;
            int id = 0;
            cv::Mat example_image;
            geometry_msgs::Point pos;

        private:
            const int feat_dimension;
        };

        class ReidDatabase
        {
        public:
            ReidDatabase() = default;
            ReidDatabase(const DataBaseParam &db_param) : db_param_(db_param), db_small(db_param_.feat_dimension) {}

            //query a set of image features and update the database
            int query_and_update(const std::vector<float> &feat_query, const cv::Mat &example_image, const geometry_msgs::Point &position);
            int max_id = 0;
            std::vector<ObjectType> object_db;

        private:
            //query a set of features
            int query(const std::vector<float> &feat_query, bool need_report = true);

            //report the query result
            void report_query(const std::vector<faiss::Index::idx_t> &index, const std::vector<float> &distance);

            //update the database
            void update(const std::vector<float> &feat_query, int id, const cv::Mat &example_image, const geometry_msgs::Point &position);

            //update the object datbase
            void update_object_db(const std::vector<float> &feat_query, std::vector<float> &feat_update, const int id,
                                  const cv::Mat &example_image, const geometry_msgs::Point &position);

            //update the feature datbase
            void update_feature_db(const std::vector<float> &feat_update, const int id);

            void report_object_db();

            void convert_index(const std::vector<faiss::Index::idx_t> &index_fake, std::vector<faiss::Index::idx_t> &index);

            DataBaseParam db_param_;

            bool is_using_db_small = true;
            faiss::IndexFlatL2 db_small;
            faiss::IndexIVFFlat *db_large;

            faiss::Index::idx_t num_feat = 0;
            faiss::Index::idx_t num_feat_when_building_db = 0;
            std::vector<float> feat_all;
            std::vector<faiss::Index::idx_t> id_all;
        };

    } // namespace reid
} // namespace ptl