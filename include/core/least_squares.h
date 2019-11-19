#pragma once

#include "core/data_types.h"
#include "core/math_sse.h"

#include <opencv2/core/core.hpp>

namespace dvo
{
namespace core
{

class NormalEquationsLeastSquares
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    void initialize();
    void update(const Vector6 &J, const NumType &res, const NumType &weight = 1.0f);
    void update(const Eigen::Matrix<NumType, 2, 6> &J, const Eigen::Matrix<NumType, 2, 1> &res, const Eigen::Matrix<NumType, 2, 2> &weight);
    void finish();
    void solve(Vector6 &x);

    const Matrix6x6 & hessian() const {return A;}
    const Vector6 & error() const {return b;}
    
private:
    OptimizedSelfAdjointMatrix6x6f A_opt;
    Matrix6x6 A;
    Vector6 b;
};
} // namespace core
} // namespace dvo