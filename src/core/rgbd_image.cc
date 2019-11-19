#include "core/rgbd_image.h"
#include "core/interpolation.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace dvo
{
namespace core
{
RgbdImage::RgbdImage(const RgbdCamera &camera) : camera_(camera),
                                                 intensity_requires_calculation_(true),
                                                 depth_requires_calculation_(true),
                                                 pointcloud_requires_build_(true),
                                                 width(0),
                                                 height(0)
{}

void RgbdImage::initialize()
{
    assert(hasIntensity() || hasDepth());

    if (hasIntensity() && hasDepth())
    {
        assert(intensity.size() == depth.size());
    }

    if (hasIntensity())
    {
        // intensity image must to be CV_32FC1
        assert(intensity.type() == cv::DataType<IntensityType>::type && intensity.channels() == 1);
        width = intensity.cols;
        height = intensity.rows;
    }

    if (hasDepth())
    {
        assert(depth.type() == cv::DataType<DepthType>::type && depth.channels() == 1);
        width = depth.cols;
        height = depth.rows;
    }

    intensity_requires_calculation_ = true;
    depth_requires_calculation_ = true;
    pointcloud_requires_build_ = true;
}

void RgbdImage::calculateDerivatives()
{
    calculateIntensityDerivatives();
    calculateDepthDerivatives();
}

bool RgbdImage::calculateIntensityDerivatives()
{
    if (!intensity_requires_calculation_)
        return false;

    assert(hasIntensity());

    calculateDerivativeX<IntensityType>(intensity, intensity_dx);
    calculateDerivativeY<IntensityType>(intensity, intensity_dy);
    
    intensity_requires_calculation_ = false;
    return true;
}

void RgbdImage::calculateDepthDerivatives()
{
    if (!depth_requires_calculation_)
        return;

    assert(hasDepth());

    calculateDerivativeX<DepthType>(depth, depth_dx);
    calculateDerivativeY<DepthType>(depth, depth_dy);

    depth_requires_calculation_ = false;
}

void RgbdImage::buildPointCloud()
{
    if (!pointcloud_requires_build_)
        return;

    assert(hasDepth());

    camera_.buildPointCloud(depth, pointcloud);

    pointcloud_requires_build_ = false;
}

void RgbdImage::calculateNormals()
{
    if (angles.empty())
    {
        normals = cv::Mat::zeros(depth.size(), CV_32FC4);
        angles.create(depth.size(), CV_32FC1);

        float *angle_ptr = angles.ptr<float>();
        cv::Vec4f *normal_ptr = normals.ptr<cv::Vec4f>();

        int x_max = depth.cols - 1;
        int y_max = depth.rows - 1;

        for (int y = 0; y < depth.rows; ++y)
        {
            for (int x = 0; x < depth.cols; ++x, ++angle_ptr, ++normal_ptr)
            {
                int idx1 = y * depth.cols + std::max(x - 1, 0);
                int idx2 = y * depth.cols + std::min(x + 1, x_max);
                int idx3 = std::max(y - 1, 0) * depth.cols + x;
                int idx4 = std::min(y + 1, y_max) * depth.cols + x;

                Eigen::Vector4f::AlignedMapType n(normal_ptr->val);
                n = (pointcloud.col(idx2) - pointcloud.col(idx1)).cross3(pointcloud.col(idx4) - pointcloud.col(idx3));
                n.normalize();

                *angle_ptr = std::abs(n(2));
            }
        }
    }
}

void RgbdImage::buildAccelerationStructure()
{
    if (acceleration.empty())
    {
        calculateDerivatives();
        cv::Mat zeros = cv::Mat::zeros(intensity.size(), intensity.type());

        cv::Mat channels[8] = {intensity, depth, intensity_dx, intensity_dy, depth_dx, depth_dy, zeros, zeros};
        cv::merge(channels, 8, acceleration);
    }
}

void RgbdImage::warpIntensity(const AffineTransform &transformationd, const Eigen::Matrix<float, 4, -1> &reference_pointcloud, const IntrinsicMatrix &intrinsics, RgbdImage &result, Eigen::Matrix<float, 4, -1> &transformed_pointcloud)
{
    Eigen::Affine3f transformation = transformationd.cast<float>();

    cv::Mat warped_image(intensity.size(), intensity.type());
    cv::Mat warped_depth(depth.size(), depth.type());

    float ox = intrinsics.ox();
    float oy = intrinsics.oy();

    float *warped_intensity_ptr = warped_image.ptr<IntensityType>();
    float *warped_depth_ptr = warped_depth.ptr<DepthType>();

    int idx = 0;

    transformed_pointcloud = transformation * reference_pointcloud;

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x, ++idx, ++warped_intensity_ptr, ++warped_depth_ptr)
        {

            const Eigen::Vector4f &p3d = transformed_pointcloud.col(idx);

            if (!std::isfinite(p3d(2)))
            {
                *warped_intensity_ptr = Invalid;
                *warped_depth_ptr = InvalidDepth;
                continue;
            }

            float x_projected = (float)(p3d(0) * intrinsics.fx() / p3d(2) + ox);
            float y_projected = (float)(p3d(1) * intrinsics.fy() / p3d(2) + oy);

            if (inImage(x_projected, y_projected))
            {
                float z = p3d(2);

                *warped_intensity_ptr = Interpolation::bilinearWithDepthBuffer(this->intensity, this->depth, x_projected, y_projected, z);
                *warped_depth_ptr = z;
            }
            else
            {
                *warped_intensity_ptr = Invalid;
                *warped_depth_ptr = InvalidDepth;
            }
        }
    }

    result.intensity = warped_image;
    result.depth = warped_depth;
    result.initialize();
}

void RgbdImage::warpIntensityForward(const AffineTransform &transformationx, const IntrinsicMatrix &intrinsics, RgbdImage &result, cv::Mat_<cv::Vec3d> &cloud)
{
    Eigen::Affine3d transformation = transformationx.cast<double>();

    bool identity = transformation.affine().isIdentity(1e-6);

    cloud = cv::Mat_<cv::Vec3d>(intensity.size(), cv::Vec3d(0, 0, 0));

    cv::Mat warped_image = cv::Mat::zeros(intensity.size(), intensity.type());

    float ox = intrinsics.ox();
    float oy = intrinsics.oy();

    const float *depth_ptr = depth.ptr<float>();
    
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x, ++depth_ptr)
        {
            if (*depth_ptr <= 1e-6f)
                continue;

            float depth = *depth_ptr;
            Eigen::Vector3d p3d((x - ox) * depth / intrinsics.fx(), (y - oy) * depth / intrinsics.fy(), depth);

            if (!identity)
            {
                Eigen::Vector3d p3d_transformed = transformation * p3d;

                float x_projected = (float)(p3d_transformed(0) * intrinsics.fx() / p3d_transformed(2) + ox);
                float y_projected = (float)(p3d_transformed(1) * intrinsics.fy() / p3d_transformed(2) + oy);

                if (inImage(x_projected, y_projected))
                {
                    int xp, yp;
                    xp = (int)std::floor(x_projected);
                    yp = (int)std::floor(y_projected);

                    warped_image.at<IntensityType>(yp, xp) = intensity.at<IntensityType>(y, x);
                }
    
                p3d = p3d_transformed;
            }

            cloud(y, x) = cv::Vec3d(p3d(0), p3d(1), p3d(2));
        }
    }

    if (identity)
    {
        warped_image = intensity;
    }
    
    result.intensity = warped_image;
    result.depth = depth;
    result.initialize();
}


void RgbdImage::warpDepthForward(const AffineTransform &transformationx, const IntrinsicMatrix &intrinsics, RgbdImage &result, cv::Mat_<cv::Vec3d> &cloud)
{
    Eigen::Affine3d transformation = transformationx.cast<double>();

    cloud = cv::Mat_<cv::Vec3d>(depth.size(), cv::Vec3d(0, 0, 0));
    cv::Mat warped_depth = cv::Mat::zeros(depth.size(), depth.type());
    warped_depth.setTo(InvalidDepth);

    float ox = intrinsics.ox();
    float oy = intrinsics.oy();

    const float *depth_ptr = depth.ptr<float>();
   
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x, ++depth_ptr)
        {
            if (!std::isfinite(*depth_ptr))
            {
                continue;
            }

            float depth = *depth_ptr;
            Eigen::Vector3d p3d((x - ox) * depth / intrinsics.fx(), (y - oy) * depth / intrinsics.fy(), depth);
            Eigen::Vector3d p3d_transformed = transformation * p3d;

            float x_projected = (float)(p3d_transformed(0) * intrinsics.fx() / p3d_transformed(2) + ox);
            float y_projected = (float)(p3d_transformed(1) * intrinsics.fy() / p3d_transformed(2) + oy);

            if (inImage(x_projected, y_projected))
            {
                int yi = (int)y_projected, xi = (int)x_projected;

                if (!std::isfinite(warped_depth.at<DepthType>(yi, xi)) || (warped_depth.at<DepthType>(yi, xi) - 0.05) > depth)
                    warped_depth.at<DepthType>(yi, xi) = depth;
            }

            p3d = p3d_transformed;

            cloud(y, x) = cv::Vec3d(p3d(0), p3d(1), p3d(2));
        }
    }

    result.depth = warped_depth;
    result.initialize();
}


void RgbdImage::warpDepthForwardAdvanced(const AffineTransform &transformation, const IntrinsicMatrix &intrinsics, RgbdImage &result)
{
    assert(hasDepth());

    this->buildPointCloud();

    Eigen::Matrix<float, 4, -1> transformed_pointcloud = transformation.cast<float>() * pointcloud;

    cv::Mat warped_depth(depth.size(), depth.type());
    warped_depth.setTo(InvalidDepth);

    float z_factor1 = transformation.rotation()(0, 0) + transformation.rotation()(0, 1) * (intrinsics.fx() / intrinsics.fy());
    float x_factor1 = -transformation.rotation()(2, 0) - transformation.rotation()(2, 1) * (intrinsics.fx() / intrinsics.fy());

    float z_factor2 = transformation.rotation()(1, 1) + transformation.rotation()(1, 0) * (intrinsics.fy() / intrinsics.fx());
    float y_factor2 = -transformation.rotation()(2, 1) - transformation.rotation()(2, 0) * (intrinsics.fy() / intrinsics.fx());

    for (int idx = 0; idx < height * width; ++idx)
    {
        Vector4 p3d = pointcloud.col(idx);
        NumType z = p3d(2);

        if (!std::isfinite(z))
            continue;

        int x_length = (int)std::ceil(z_factor1 + x_factor1 * p3d(0) / z) + 1; // magic +1
        int y_length = (int)std::ceil(z_factor2 + y_factor2 * p3d(1) / z) + 1; // magic +1

        Vector4 p3d_transformed = transformed_pointcloud.col(idx);
        NumType z_transformed = p3d_transformed(2);

        int x_projected = (int)std::floor(p3d_transformed(0) * intrinsics.fx() / z_transformed + intrinsics.ox());
        int y_projected = (int)std::floor(p3d_transformed(1) * intrinsics.fy() / z_transformed + intrinsics.oy());

        // TODO: replace inImage(...) checks, with max(..., 0) on initial value of x_, y_ and  min(..., width/height) for their respective upper bound
        //for (int y_ = y_projected; y_ < y_projected + y_length; y_++)
        //  for (int x_ = x_projected; x_ < x_projected + x_length; x_++)

        int x_begin = std::max(x_projected, 0);
        int y_begin = std::max(y_projected, 0);
        int x_end = std::min(x_projected + x_length, (int)width);
        int y_end = std::min(y_projected + y_length, (int)height);

        for (int y = y_begin; y < y_end; ++y)
        {
            DepthType *v = warped_depth.ptr<DepthType>(y, x_begin);

            for (int x = x_begin; x < x_end; ++x, ++v)
            {
                if (!std::isfinite(*v) || (*v) > z_transformed)
                {
                    (*v) = (DepthType)z_transformed;
                }
            }
        }
    }

    result.depth = warped_depth;
    result.initialize();
}


RgbdImagePyramid::RgbdImagePyramid(RgbdCameraPyramid &camera, const cv::Mat &intensity, const cv::Mat &depth) : camera_(camera)
{
    levels_.push_back(camera_.level(0).create(intensity, depth));
}

void RgbdImagePyramid::build(const size_t num_levels)
{
    if (levels_.size() >= num_levels)
        return;

    // if we already have some levels, we just need to compute the coarser levels
    size_t first = levels_.size();

    for (size_t idx = first; idx < num_levels; ++idx)
    {
        levels_.push_back(camera_.level(idx).create());

        pyrDownMeanSmooth<IntensityType>(levels_[idx - 1]->intensity, levels_[idx]->intensity);
        pyrDownSubsample<DepthType>(levels_[idx - 1]->depth, levels_[idx]->depth);
        levels_[idx]->initialize();
    }
}

RgbdImage &RgbdImagePyramid::level(size_t idx)
{
    assert(idx < levels_.size());

    return *levels_[idx];
}

double RgbdImagePyramid::timestamp() const
{
    return !levels_.empty() ? levels_[0]->timestamp : 0.0;
}
} // namespace core
} // namespace dvo