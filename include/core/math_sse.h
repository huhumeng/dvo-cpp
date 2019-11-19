#pragma once

#include <Eigen/Core>

namespace dvo
{
namespace core
{
/**
 * A 6x6 self adjoint matrix with optimized "rankUpdate(u, scale)" (10x faster than Eigen impl, 1.8x faster than MathSse::addOuterProduct(...)).
 */
class OptimizedSelfAdjointMatrix6x6f
{
public:
    
    void rankUpdate(const Eigen::Matrix<float, 6, 1> &u, const float &alpha);

    void rankUpdate(const Eigen::Matrix<float, 2, 6> &u, const Eigen::Matrix2f &alpha);

    void setZero();

    void toEigen(Eigen::Matrix<float, 6, 6> &m) const;

private:
    alignas(16) float data[24];
};

} // namespace core
} // namespace dvo