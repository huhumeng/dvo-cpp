#pragma once

#include "core/intrinsic_matrix.h"

#include <opencv2/core/core.hpp>

namespace dvo
{
namespace core
{
class RgbdImage;
typedef std::shared_ptr<RgbdImage> RgbdImagePtr;

class RgbdCamera
{
public:
    RgbdCamera(size_t width, size_t height, const IntrinsicMatrix &intrinsics);
    ~RgbdCamera() = default;

    size_t width() const { return width_; }

    size_t height() const { return height_; }

    const IntrinsicMatrix &intrinsics() const { return intrinsics_; }

    RgbdImagePtr create(const cv::Mat &intensity, const cv::Mat &depth) const;
    RgbdImagePtr create() const;

    void buildPointCloud(const cv::Mat &depth, Eigen::Matrix<float, 4, -1> &pointcloud) const;

private:
    size_t width_, height_;

    bool hasSameSize(const cv::Mat &img) const { return img.cols == width_ && img.rows == height_; }

    IntrinsicMatrix intrinsics_;
    Eigen::Matrix<float, 4, -1> pointcloud_template_;
};

typedef std::shared_ptr<RgbdCamera> RgbdCameraPtr;
typedef std::shared_ptr<const RgbdCamera> RgbdCameraConstPtr;

class RgbdImagePyramid;
typedef std::shared_ptr<RgbdImagePyramid> RgbdImagePyramidPtr;

class RgbdCameraPyramid
{
public:
    RgbdCameraPyramid(const RgbdCamera &base);
    RgbdCameraPyramid(size_t base_width, size_t base_height, const IntrinsicMatrix &base_intrinsics);

    ~RgbdCameraPyramid() = default;

    RgbdImagePyramidPtr create(const cv::Mat &base_intensity, const cv::Mat &base_depth);

    void build(size_t levels);

    const RgbdCamera &level(size_t level);

    const RgbdCamera &level(size_t level) const;

private:
    std::vector<RgbdCameraPtr> levels_;
};

typedef std::shared_ptr<RgbdCameraPyramid> RgbdCameraPyramidPtr;
typedef std::shared_ptr<const RgbdCameraPyramid> RgbdCameraPyramidConstPtr;

} // namespace core
} // namespace dvo