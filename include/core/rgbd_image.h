#pragma once

#include "core/data_types.h"
#include "core/intrinsic_matrix.h"
#include "core/rgbd_camera.h"

#include <opencv2/core/core.hpp>

namespace dvo
{
namespace core
{
class RgbdImage
{
public:
    RgbdImage(const RgbdCamera &camera);
    virtual ~RgbdImage() = default;

    const RgbdCamera &camera() const { return camera_; }

    bool hasIntensity() const { return !intensity.empty(); }
    bool hasDepth() const { return !depth.empty(); }
    bool hasRgb() const { return !rgb.empty(); }

    void initialize();

    void calculateDerivatives();
    bool calculateIntensityDerivatives();
    void calculateDepthDerivatives();

    void buildPointCloud();
    void calculateNormals();

    void buildAccelerationStructure();

    cv::Mat intensity;
    cv::Mat intensity_dx;
    cv::Mat intensity_dy;

    cv::Mat depth;
    cv::Mat depth_dx;
    cv::Mat depth_dy;

    cv::Mat normals, angles;

    cv::Mat rgb;

    Eigen::Matrix<float, 4, -1> pointcloud;

    typedef cv::Vec<float, 8> Vec8f;
    cv::Mat_<Vec8f> acceleration;

    size_t width, height;
    double timestamp;

    // inverse warping
    // transformation is the transformation from reference to this image
    void warpIntensity(const AffineTransform &transformation, const Eigen::Matrix<float, 4, -1> &reference_pointcloud, const IntrinsicMatrix &intrinsics, RgbdImage &result, Eigen::Matrix<float, 4, -1> &transformed_pointcloud);

    // forward warping
    // transformation is the transformation from this image to the reference image
    void warpIntensityForward(const AffineTransform &transformation, const IntrinsicMatrix &intrinsics, RgbdImage &result, cv::Mat_<cv::Vec3d> &cloud);
    
    void warpDepthForward(const AffineTransform &transformation, const IntrinsicMatrix &intrinsics, RgbdImage &result, cv::Mat_<cv::Vec3d> &cloud);
    void warpDepthForwardAdvanced(const AffineTransform &transformation, const IntrinsicMatrix &intrinsics, RgbdImage &result);

    bool inImage(const float &x, const float &y) const { return x >= 0 && x < width && y >= 0 && y < height; }

private:

    bool intensity_requires_calculation_, depth_requires_calculation_, pointcloud_requires_build_;

    const RgbdCamera &camera_;

    enum WarpIntensityOptions
    {
        WithPointCloud,
        WithoutPointCloud,
    };
};

class RgbdImagePyramid
{
public:
    typedef std::shared_ptr<RgbdImagePyramid> Ptr;

    RgbdImagePyramid(RgbdCameraPyramid &camera, const cv::Mat &intensity, const cv::Mat &depth);

    virtual ~RgbdImagePyramid() = default;

    void build(const size_t num_levels);

    RgbdImage &level(size_t idx);

    double timestamp() const;

private:
    RgbdCameraPyramid &camera_;
    std::vector<RgbdImagePtr> levels_;
};


} // namespace core
} // namespace dvo