#pragma once
#include <iostream>
#include <cmath>
//pcl
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>

namespace ptl
{
    namespace tracker
    {
        struct PointCloudProcessorParam
        {
            float resample_size = 0.1;
            float x_min = 0.0;
            float x_max = 10.0;
            float z_min = 0.0;
            float z_max = 5.0;
            float std_dev_thres = 0.1;
            int mean_k = 30;
            float cluster_tolerance = 0.5;
            int cluster_size_min = 30;
            int cluster_size_max = 10000;
        };

        class PointCloudProcessor
        {
        public:
            PointCloudProcessor(const pcl::PointCloud<pcl::PointXYZI> &pc_orig, const PointCloudProcessorParam &param);
            void compute(bool use_resample = true, bool use_conditional_filter = true,
                         bool use_statistical_filter = true, bool use_clustering = true,
                         bool use_cal_centroid = true);
            pcl::PointXYZ get_centroid_with_max_points();
            pcl::PointXYZ get_centroid_closest();

            pcl::PointCloud<pcl::PointXYZI> pc_resample;
            pcl::PointCloud<pcl::PointXYZI> pc_conditional_filtered;
            pcl::PointCloud<pcl::PointXYZI> pc_statistical_filtered;
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_final;
            std::vector<pcl::PointCloud<pcl::PointXYZI>> pc_clustered;
            std::vector<pcl::PointXYZ> centroids;

        private:
            void resample();
            void conditonal_filter();
            void statitical_filter();
            void clustering();
            void cal_centroid();

            PointCloudProcessorParam _param;
            pcl::PointCloud<pcl::PointXYZI> _pc_origin;
        };
    } // namespace tracker
} // namespace ptl