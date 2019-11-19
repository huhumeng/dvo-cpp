#include "visualization/async_point_cloud_builder.h"

#include <tbb/task_scheduler_init.h>
#include <tbb/task.h>

namespace dvo
{
namespace visualization
{
class BuildPointCloudTask : public tbb::task
{
public:
    BuildPointCloudTask(const core::RgbdImage &image, const Eigen::Affine3d &pose, AsyncPointCloudBuilder::DoneCallback &callback) : job_(image, pose),
                                                                                                                                     callback_(callback)
    {
    }

    virtual ~BuildPointCloudTask()
    {
    }

    virtual tbb::task *execute()
    {
        callback_(job_.build());

        return NULL;
    }

private:
    AsyncPointCloudBuilder::BuildJob job_;
    AsyncPointCloudBuilder::DoneCallback &callback_;
};

AsyncPointCloudBuilder::BuildJob::BuildJob(const core::RgbdImage &image, const Eigen::Affine3d pose) : image(image),
                                                                                                       pose(pose)
{
}

AsyncPointCloudBuilder::PointCloud::Ptr AsyncPointCloudBuilder::BuildJob::build()
{
    if (!cloud_)
    {
        image.buildPointCloud();
        Eigen::Matrix<float, 4, -1> pointcloud = pose.cast<float>() * image.pointcloud;

        cloud_.reset(new AsyncPointCloudBuilder::PointCloud());
        cloud_->reserve(image.width * image.height);

        const float *intensity_ptr;
        if (image.hasRgb())
        {
            intensity_ptr = image.rgb.ptr<float>();
        }
        else
        {
            intensity_ptr = image.intensity.ptr<float>();
        }

        int step = image.hasRgb() ? 3 : 1;

        AsyncPointCloudBuilder::PointCloud::PointType p;

        for (int idx = 0; idx < image.width * image.height; ++idx, intensity_ptr += step)
        {
            const Eigen::Matrix<float, 4, 1> &col = pointcloud.col(idx);

            //if(!std::isfinite(col(2)) || image.pointcloud.col(idx)(2) > 2.0) continue;

            p.x = col(0);
            p.y = col(1);
            p.z = col(2);

            if (image.hasRgb())
            {
                p.b = intensity_ptr[0];
                p.g = intensity_ptr[1];
                p.r = intensity_ptr[2];
            }
            else
            {
                p.r = p.g = p.b = *intensity_ptr;
            }

            cloud_->push_back(p);
        }
    }
    return cloud_;
}

AsyncPointCloudBuilder::AsyncPointCloudBuilder()
{
    static tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);
}

AsyncPointCloudBuilder::~AsyncPointCloudBuilder()
{
}

void AsyncPointCloudBuilder::build(const core::RgbdImage &image, const Eigen::Affine3d pose)
{
    BuildPointCloudTask *t = new (tbb::task::allocate_root()) BuildPointCloudTask(image, pose, done_);
    tbb::task::enqueue(*t);
}

void AsyncPointCloudBuilder::done(DoneCallback &callback)
{
    done_ = callback;
}

} // namespace visualization
} // namespace dvo