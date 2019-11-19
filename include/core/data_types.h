#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dvo
{


typedef float float_t;

typedef float_t IntensityType;
static const IntensityType Invalid = std::numeric_limits<IntensityType>::quiet_NaN();

typedef float_t DepthType;
static const DepthType InvalidDepth = std::numeric_limits<DepthType>::quiet_NaN();

// float/double, determines numeric precision
typedef float_t NumType;

typedef Eigen::Matrix<NumType, 6, 6> Matrix6x6;
typedef Eigen::Matrix<NumType, 1, 2> Matrix1x2;
typedef Eigen::Matrix<NumType, 2, 6> Matrix2x6;

typedef Eigen::Matrix<NumType, 6, 1> Vector6;
typedef Eigen::Matrix<NumType, 4, 1> Vector4;

typedef Eigen::Transform<NumType, 3, Eigen::Affine> AffineTransform;

// TODO: rigid transform is subclass of affine3d, change this to sophus::SE3 will be more suitable
typedef Eigen::Affine3d AffineTransformd;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

typedef Eigen::Matrix<float, 8, 1> Vector8f;
typedef Eigen::Matrix<double, 8, 1> Vector8d;

struct alignas(16) PointWithIntensityAndDepth
{
    typedef union alignas(16) {
        float data[4];
        struct
        {
            float x, y, z;
        };
    } Point;

    typedef union alignas(16) {
        float data[8];
        struct
        {
            float i, z, idx, idy, zdx, zdy, time_interpolation;
        };
    } IntensityAndDepth;

    typedef std::vector<PointWithIntensityAndDepth, Eigen::aligned_allocator<PointWithIntensityAndDepth>> VectorType;

    Point point;
    IntensityAndDepth intensity_and_depth;

    Eigen::Vector4f::AlignedMapType getPointVec4f()
    {
        return Eigen::Vector4f::AlignedMapType(point.data);
    }

    Eigen::Vector2f::AlignedMapType getIntensityAndDepthVec2f()
    {
        return Eigen::Vector2f::AlignedMapType(intensity_and_depth.data);
    }

    Eigen::Vector2f::MapType getIntensityDerivativeVec2f()
    {
        return Eigen::Vector2f::MapType(intensity_and_depth.data + 2);
    }

    Eigen::Vector2f::MapType getDepthDerivativeVec2f()
    {
        return Eigen::Vector2f::MapType(intensity_and_depth.data + 4);
    }

    Vector8f::AlignedMapType getIntensityAndDepthWithDerivativesVec8f()
    {
        return Vector8f::AlignedMapType(intensity_and_depth.data);
    }
};

} /* namespace dvo */