#pragma once

#include <Eigen/Core>

namespace dvo
{
namespace core
{

struct IntrinsicMatrix
{
public:
    static IntrinsicMatrix create(float fx, float fy, float ox, float oy);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IntrinsicMatrix() = default;

    IntrinsicMatrix(const IntrinsicMatrix &other);

    float fx() const { return data(0, 0); }
    float fy() const { return data(1, 1); }

    float ox() const { return data(0, 2); }
    float oy() const { return data(1, 2); }

    void scale(float factor);

    Eigen::Matrix3f data;
};


} /* namespace core */
} /* namespace dvo */