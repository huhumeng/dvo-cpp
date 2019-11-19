#include "core/rgbd_camera.h"
#include "core/rgbd_image.h"

namespace dvo
{
namespace core
{
RgbdCamera::RgbdCamera(size_t width, size_t height, const IntrinsicMatrix &intrinsics) : width_(width),
                                                                                         height_(height),
                                                                                         intrinsics_(intrinsics)
{
    pointcloud_template_.resize(Eigen::NoChange, width_ * height_);
    int idx = 0;

    for (size_t y = 0; y < height_; ++y)
    {
        for (size_t x = 0; x < width_; ++x, ++idx)
        {
            pointcloud_template_(0, idx) = (x - intrinsics_.ox()) / intrinsics_.fx();
            pointcloud_template_(1, idx) = (y - intrinsics_.oy()) / intrinsics_.fy();
            pointcloud_template_(2, idx) = 1.0;
            pointcloud_template_(3, idx) = 0.0;
        }
    }
}

RgbdImagePtr RgbdCamera::create(const cv::Mat &intensity, const cv::Mat &depth) const
{
    RgbdImagePtr result = std::make_shared<RgbdImage>(*this);

    result->intensity = intensity;
    result->depth = depth;
    result->initialize();

    return result;
}

RgbdImagePtr RgbdCamera::create() const
{
    return std::make_shared<RgbdImage>(*this);
}

void RgbdCamera::buildPointCloud(const cv::Mat &depth, Eigen::Matrix<float, 4, -1> &pointcloud) const
{
    assert(hasSameSize(depth));

    pointcloud.resize(Eigen::NoChange, width_ * height_);

    const float *depth_ptr = depth.ptr<float>();
    int idx = 0;

    for (size_t y = 0; y < height_; ++y)
    {
        for (size_t x = 0; x < width_; ++x, ++depth_ptr, ++idx)
        {
            pointcloud.col(idx) = pointcloud_template_.col(idx) * (*depth_ptr);
            pointcloud(3, idx) = 1.0; // change from line to point
        }
    }
}

RgbdCameraPyramid::RgbdCameraPyramid(const RgbdCamera &base)
{
    // copy construct function
    levels_.push_back(std::make_shared<RgbdCamera>(base));
}

RgbdCameraPyramid::RgbdCameraPyramid(size_t base_width, size_t base_height, const dvo::core::IntrinsicMatrix &base_intrinsics)
{
    levels_.push_back(std::make_shared<RgbdCamera>(base_width, base_height, base_intrinsics));
}

RgbdImagePyramidPtr RgbdCameraPyramid::create(const cv::Mat &base_intensity, const cv::Mat &base_depth)
{
    return std::make_shared<RgbdImagePyramid>(*this, base_intensity, base_depth);
}

void RgbdCameraPyramid::build(size_t levels)
{
    // TODO: when start = 0, program will crash
    size_t start = levels_.size();

    for (size_t idx = start; idx < levels; ++idx)
    {
        RgbdCameraPtr &previous = levels_[idx - 1];

        IntrinsicMatrix intrinsics(previous->intrinsics());
        intrinsics.scale(0.5f);

        levels_.push_back(std::make_shared<RgbdCamera>(previous->width() / 2, previous->height() / 2, intrinsics));
    }
}

const RgbdCamera &RgbdCameraPyramid::level(size_t level)
{
    build(level + 1);

    return *levels_[level];
}

const RgbdCamera &RgbdCameraPyramid::level(size_t level) const
{
    return *levels_[level];
}

} // namespace core
} // namespace dvo