#pragma once

#include "core/rgbd_image.h"
#include "core/intrinsic_matrix.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace dvo
{
namespace visualization
{
class AsyncPointCloudBuilder
{
public:
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
    typedef std::function<void(const PointCloud::Ptr &cloud)> DoneCallback;

    struct BuildJob
    {
    public:
        core::RgbdImage image;
        const Eigen::Affine3d pose;

        BuildJob(const core::RgbdImage &image, const Eigen::Affine3d pose = Eigen::Affine3d::Identity());

        AsyncPointCloudBuilder::PointCloud::Ptr build();

    private:
        AsyncPointCloudBuilder::PointCloud::Ptr cloud_;
    };

    AsyncPointCloudBuilder();
    virtual ~AsyncPointCloudBuilder();

    void build(const dvo::core::RgbdImage &image, const Eigen::Affine3d pose = Eigen::Affine3d::Identity());

    void done(DoneCallback &callback);

private:
    DoneCallback done_;
};

} // namespace visualization
} // namespace dvo