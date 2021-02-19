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
            report_object_db();
        }

        int ReidDatabase::query(const std::vector<float> &feat_query, bool need_report)
        {
            if (object_db.empty())
            {
                return 0;
            }
            else
            {
                std::vector<idx_t> index;
                std::vector<idx_t> index_fake(db_param_.find_first_k * feat_query.size() / db_param_.feat_dimension);

                std::vector<float> distance(db_param_.find_first_k * feat_query.size() / db_param_.feat_dimension);
                //find the k nearest neighbour first
                if (is_using_db_small)
                {
                    db_small.search(feat_query.size() / db_param_.feat_dimension, feat_query.data(),
                                    db_param_.find_first_k, distance.data(), index_fake.data());
                }
                else
                {
                    db_large->search(feat_query.size() / db_param_.feat_dimension, feat_query.data(),
                                     db_param_.find_first_k, distance.data(), index_fake.data());
                }
                convert_index(index_fake, index);

                //get the id of this batch
                std::vector<int> match_count(object_db.size() + 1, 0);
                int match_id = -1;
                int max_count = 0;
                for (int i = 0; i < index.size() / db_param_.find_first_k; i++)
                {
                    int matched_db_id_for_this_query;
                    if (distance[i * db_param_.find_first_k] < db_param_.same_id_threshold)
                    {
                        match_count[index[i * db_param_.find_first_k]]++;
                        matched_db_id_for_this_query = index[i * db_param_.find_first_k];
                    }
                    else
                    {
                        // no correct matched object found
                        match_count[object_db.size()]++;
                        matched_db_id_for_this_query = object_db.size();
                    }

                    if (match_count[matched_db_id_for_this_query] > max_count)
                    {
                        match_id = matched_db_id_for_this_query;
                        max_count = match_count[matched_db_id_for_this_query];
                    }
                }

                if (need_report)
                {
                    report_query(index, distance);
                }

                std::cout << "\033[32m"
                          << "id: " << match_id << " , batch: " << max_count << "/"
                          << feat_query.size() / db_param_.feat_dimension << " = " << 1.0 * max_count / (feat_query.size() / db_param_.feat_dimension)
                          << "\033[0m" << std::endl;

                if (1.0 * max_count / (feat_query.size() / db_param_.feat_dimension) < db_param_.batch_ratio)
                {
                    std::cout << db_param_.batch_ratio << std::endl;
                    std::cout << "\033[31m"
                              << "Failed the batch test! This is an unknown object."
                              << "\033[0m" << std::endl;
                    match_id = object_db.size();
                }

                return match_id;
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
            feat_update.clear();
            for (int i = 0; i < feat_query.size() / db_param_.feat_dimension; ++i)
            {
                std::vector<float> feat(feat_query.begin() + db_param_.feat_dimension * i,
                                        feat_query.begin() + db_param_.feat_dimension * (i + 1));
                //update position first
                if (i == 0 && id == max_id)
                {
                    //create new object first
                    object_db.emplace_back(max_id, example_image, position);
                    max_id++;
                }
                else
                {
                    //update position
                    object_db[id].update_pos(position);
                }

                //update object local feature database
                if (object_db[id].feat_num < db_param_.sim_check_start_threshold)
                {
                    object_db[id].db.add(1, feat.data());
                    object_db[id].feat_num++;
                    feat_update.insert(feat_update.end(), feat.begin(), feat.end());
                }
                else
                {
                    //start similiarity check
                    //checking the similiarity of local database, if the similarity is too high,
                    //we won't add this feature to database to ensure variety
                    std::vector<idx_t> index(1);
                    std::vector<float> distance(1);
                    object_db[id].db.search(1, feat.data(), 1, distance.data(), index.data());

                    //pass the sim check， then we add it to the database
                    if (distance[0] > db_param_.similarity_test_threshold && object_db[id].feat_num <= db_param_.max_feat_num_one_object)
                    {
                        object_db[id].db.add(1, feat.data());
                        object_db[id].feat_num++;
                        feat_update.insert(feat_update.end(), feat.begin(), feat.end());
                    }
                }
            }
        }

        void ReidDatabase::update_feature_db(const std::vector<float> &feat_update, const int id)
        {
            //save the features
            feat_all.insert(feat_all.end(), feat_update.begin(), feat_update.end());
            std::vector<idx_t> ids(feat_update.size() / db_param_.feat_dimension, id);
            id_all.insert(id_all.end(), ids.begin(), ids.end());
            num_feat += feat_update.size() / db_param_.feat_dimension;

            // database mangement strategy
            // small feature database use Brute-force search
            // large feature database use Inverted file to store teh database
            // when the number of features became two times of the number when we build the inverted file, we rebuild the inverted file

            if (is_using_db_small)
            {
                if (num_feat > db_param_.use_inverted_file_db_threshold)
                {
                    //build a new larger database
                    db_large = new faiss::IndexIVFFlat(&db_small, db_param_.feat_dimension, size_t(num_feat / db_param_.nlist_ratio), faiss::METRIC_INNER_PRODUCT);
                    db_large->train(num_feat, feat_all.data()); //TODO this step might perform asychronously
                    db_large->add(num_feat, feat_all.data());
                    is_using_db_small = false;
                    num_feat_when_building_db = num_feat;
                }
                else
                {
                    db_small.add(feat_update.size() / db_param_.feat_dimension, feat_update.data());
                }
            }
            else
            {
                //rebuild a new larger database when the current inverted file become twice the size when it was built to ensure
                //efficiency in searching
                if (num_feat / num_feat_when_building_db > 2)
                {
                    db_large = new faiss::IndexIVFFlat(&db_small, db_param_.feat_dimension, size_t(num_feat / db_param_.nlist_ratio), faiss::METRIC_INNER_PRODUCT);
                    db_large->train(num_feat, feat_all.data()); //TODO this step might perform asychronously
                    db_large->add(num_feat, feat_all.data());
                    num_feat_when_building_db = num_feat;
                }
                else
                {
                    db_large->add(feat_update.size() / db_param_.feat_dimension, feat_update.data());
                }
            }
        }

        void ReidDatabase::report_object_db()
        {
            std::cout << "*****Reid Database Report*****" << std::endl;
            for (const auto ob : object_db)
            {
                std::cout << "id: " << ob.id << " | db num: " << ob.feat_num << std::endl;
            }
            std::cout << "***Reid Database Report Done***" << std::endl;
            std::cout << std::endl;
        }

        void ReidDatabase::convert_index(const std::vector<faiss::Index::idx_t> &index_fake, std::vector<faiss::Index::idx_t> &index)
        {
            index.clear();
            for (auto idf : index_fake)
            {
                index.push_back(id_all[idf]);
            }
        }
    } // namespace reid
} // namespace ptl