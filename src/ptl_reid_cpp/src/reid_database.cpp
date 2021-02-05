#include "ptl_reid_cpp/reid_database.h"

using idx_t = faiss::Index::idx_t;
namespace ptl
{
    namespace reid
    {
        void ReidDatabase::query_and_update(const std::vector<float> &feat_query, const cv::Mat &example_image, const geometry_msgs::Point &position)
        {
            int id = query(feat_query);
            update(feat_query, id, example_image, position);
        }

        int ReidDatabase::query(const std::vector<float> &feat_query, bool need_report)
        {
            if (object_db.empty())
            {
                return -1;
            }
            else
            {
                std::vector<idx_t> index(db_param_.find_first_k * feat_query.size() / db_param_.feat_dimension);
                std::vector<float> distance(db_param_.find_first_k * feat_query.size() / db_param_.feat_dimension);
                //find the k nearest neighbour first
                if (is_using_db_small)
                {
                    db_small.search(feat_query.size() / db_param_.feat_dimension, feat_query.data(),
                                    db_param_.find_first_k, distance.data(), index.data());
                }
                else
                {
                    db_large->search(feat_query.size() / db_param_.feat_dimension, feat_query.data(),
                                     db_param_.find_first_k, distance.data(), index.data());
                }
                report_query(index, distance);

                //get the id of this batch
                std::vector<int> match_count(object_db.size() + 1, 0);
                int max_id = -1;
                int max_count = 0;
                for (int i = 0; i < index.size() / db_param_.find_first_k; i++)
                {
                    for (int j = 0; j < db_param_.find_first_k; j++)
                    {
                        int matched_db_id_for_this_query;
                        if (distance[i * db_param_.find_first_k + j] > db_param_.same_id_threshold)
                        {
                            match_count[index[i * db_param_.find_first_k + j]]++;
                            matched_db_id_for_this_query = index[i * db_param_.find_first_k + j];
                        }
                        else
                        {
                            // no correct matched object found
                            match_count[object_db.size()]++;
                            matched_db_id_for_this_query = object_db.size();
                        }

                        if (match_count[matched_db_id_for_this_query] > max_count)
                        {
                            max_id = matched_db_id_for_this_query;
                            max_count = match_count[matched_db_id_for_this_query];
                        }
                    }
                }
                return max_id;
            }
        }

        void ReidDatabase::report_query(const std::vector<faiss::Index::idx_t> &index, const std::vector<float> &distance)
        {
            for (int i = 0; i < index.size() / db_param_.find_first_k; i++)
            {
                std::cout << "image " << i << " : ";
                for (int j = 0; j < db_param_.find_first_k; j++)
                {
                    std::cout << index[i * db_param_.find_first_k + j]
                              << "-" << distance[i * db_param_.find_first_k + j] << " | ";
                }
                std::cout << std::endl;
            }
        }

        void ReidDatabase::update(const std::vector<float> &feat_query, int id, const cv::Mat &example_image, const geometry_msgs::Point &position)
        {
            std::vector<float> feat_update;

            //update object database and get the feature that will be added to the feature database
            update_object_db(feat_query, feat_update, id, example_image, position);

            //updat feature database
            update_feature_db(feat_update, id);
        }

        void ReidDatabase::update_object_db(const std::vector<float> &feat_query, std::vector<float> &feat_update, const int id,
                                            const cv::Mat &example_image, const geometry_msgs::Point &position)
        {
            for (int i = 0; i < feat_query.size() / db_param_.feat_dimension; ++i)
            {
            }
        }

        void ReidDatabase::update_feature_db(const std::vector<float> &feat_update, const int id)
        {
            //save the features
            feat_all.insert(feat_all.end(), feat_update.begin(), feat_update.end());
            num_feat += feat_update.size() / db_param_.feat_dimension;

            // database mangement strategy
            // small feature database use Brute-force search
            // large feature database use Inverted file to store teh database
            // when the number of features became two times of the number when we build the inverted file, we rebuild the inverted file
            std::vector<idx_t> ids(feat_update.size() / db_param_.feat_dimension, id);
            if (is_using_db_small)
            {
                if (num_feat > db_param_.use_inverted_file_db_thres)
                {
                    //build a new larger database
                    db_large = new faiss::IndexIVFFlat(&db_small, db_param_.feat_dimension, size_t(num_feat / db_param_.nlist_ratio), faiss::METRIC_INNER_PRODUCT);
                    db_large->train(num_feat, feat_all.data()); //TODO this step might perform asychronously
                    db_large->add(num_feat, feat_all.data());
                }
                else
                {
                    db_small.add_with_ids(feat_update.size() / db_param_.feat_dimension, feat_update.data(), ids.data());
                }
            }
            else
            {
                if (num_feat / num_feat_when_building_db > 2)
                {
                    //rebuild a new larger database when the current inverted file become twice the size when it was built
                    db_large = new faiss::IndexIVFFlat(&db_small, db_param_.feat_dimension, size_t(num_feat / db_param_.nlist_ratio), faiss::METRIC_INNER_PRODUCT);
                    db_large->train(num_feat, feat_all.data()); //TODO this step might perform asychronously
                    db_large->add(num_feat, feat_all.data());
                }
                db_large->add_with_ids(feat_update.size() / db_param_.feat_dimension, feat_update.data(), ids.data());
            }
        }
    } // namespace reid
} // namespace ptl