//pcl
#include "ptl_tracker/point_cloud_processor.h"
namespace ptl
{
    namespace tracker
    {
        PointCloudProcessor::PointCloudProcessor(pcl::PointCloud<pcl::PointXYZI> pc_orig, PointCloudProcessorParam param)
        {
            _param = param;
            _pc_origin = pc_orig;
        }

        void PointCloudProcessor::resample()
        {
            pcl::VoxelGrid<pcl::PointXYZI> resample_filter;
            resample_filter.setInputCloud(_pc_origin.makeShared());
            resample_filter.setLeafSize(_param.resample_size, _param.resample_size, _param.resample_size);
            resample_filter.filter(pc_resample);
        }

        void PointCloudProcessor::conditonal_filter()
        {
            pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZI>());

            range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT, _param.x_min)));
            range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT, _param.x_max)));
            range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::GT, _param.z_min)));
            range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::LT, _param.z_max)));
            pcl::ConditionalRemoval<pcl::PointXYZI> conditional_filter;
            conditional_filter.setCondition(range_cond);
            conditional_filter.setInputCloud(pc_resample.makeShared()); //设置输入点云
            conditional_filter.setKeepOrganized(false);
            conditional_filter.filter(pc_conditional_filtered); //执行滤波，保存过滤结果在cloud_filtered
        }

        void PointCloudProcessor::statitical_filter()
        {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZI> statistical_filter;
            statistical_filter.setStddevMulThresh(_param.std_dev_thres);
            statistical_filter.setMeanK(_param.mean_k);
            // statistical_filter.setKeepOrganized(true);
            statistical_filter.setInputCloud(pc_conditional_filtered.makeShared());
            statistical_filter.filter(pc_statistical_filtered);
        }

        void PointCloudProcessor::clustering()
        {
            pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
            tree->setInputCloud(pc_statistical_filtered.makeShared());
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclidean_cluster;
            euclidean_cluster.setClusterTolerance(_param.cluster_tolerance);
            euclidean_cluster.setMinClusterSize(_param.cluster_size_min);
            euclidean_cluster.setMaxClusterSize(_param.cluster_size_max);
            euclidean_cluster.setSearchMethod(tree);
            euclidean_cluster.setInputCloud(pc_statistical_filtered.makeShared());
            euclidean_cluster.extract(cluster_indices);
            for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
            {
                pcl::PointCloud<pcl::PointXYZI> pc_clustered_tmp;
                for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
                {
                    pc_clustered_tmp.points.push_back(pc_statistical_filtered.points[*pit]);
                    pc_clustered_tmp.width = pc_clustered_tmp.size();
                    pc_clustered_tmp.height = 1;
                    pc_clustered_tmp.is_dense = true;
                }
                pc_clustered.push_back(pc_clustered_tmp);
            }
        }

        void PointCloudProcessor::cal_centroid()
        {
            for (auto pcc : pc_clustered)
            {
                pcl::PointXYZ centroid;
                pcl::computeCentroid(pcc, centroid);
                centroids.push_back(centroid);
            }
        }

        void PointCloudProcessor::compute(bool use_resample, bool use_conditional_filter, bool use_statistical_filter,
                                          bool use_clustering, bool use_cal_centroid)
        {
            if (use_resample)
                resample();
            if (use_conditional_filter)
                conditonal_filter();
            if (use_statistical_filter)
                statitical_filter();
            if (use_clustering)
                clustering();
            if (use_cal_centroid)
                cal_centroid();
        }
    } // namespace tracker
} // namespace ptl