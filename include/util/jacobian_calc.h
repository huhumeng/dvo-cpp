#pragma once

#include "core/data_types.h"
namespace dvo
{
namespace util
{
// jacobian computation
inline void computeJacobianOfProjectionAndTransformation(const Vector4 &p, Matrix2x6 &j)
{
    NumType z = 1.0f / p(2);
    NumType z_sqr = 1.0f / (p(2) * p(2));

    j(0, 0) = z;
    j(0, 1) = 0.0f;
    j(0, 2) = -p(0) * z_sqr;
    j(0, 3) = j(0, 2) * p(1);        //j(0, 3) = -p(0) * p(1) * z_sqr;
    j(0, 4) = 1.0f - j(0, 2) * p(0); //j(0, 4) =  (1.0 + p(0) * p(0) * z_sqr);
    j(0, 5) = -p(1) * z;

    j(1, 0) = 0.0f;
    j(1, 1) = z;
    j(1, 2) = -p(1) * z_sqr;
    j(1, 3) = -1.0f + j(1, 2) * p(1); //j(1, 3) = -(1.0 + p(1) * p(1) * z_sqr);
    j(1, 4) = -j(0, 3);               //j(1, 4) =  p(0) * p(1) * z_sqr;
    j(1, 5) = p(0) * z;
}

inline void compute3rdRowOfJacobianOfTransformation(const Vector4 &p, Vector6 &j)
{
    j(0) = 0.0;
    j(1) = 0.0;
    j(2) = 1.0;
    j(3) = p(1);
    j(4) = -p(0);
    j(5) = 0.0;
}

} // namespace util

} // namespace dvo
