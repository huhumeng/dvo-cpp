#include "core/intrinsic_matrix.h"

namespace dvo
{
namespace core
{
IntrinsicMatrix IntrinsicMatrix::create(float fx, float fy, float ox, float oy)
{
    IntrinsicMatrix result;
    result.data.setZero();

    result.data(0, 0) = fx;
    result.data(1, 1) = fy;
    result.data(2, 2) = 1.0f;

    result.data(0, 2) = ox;
    result.data(1, 2) = oy;

    return result;
}

IntrinsicMatrix::IntrinsicMatrix(const IntrinsicMatrix &other) : data(other.data) {}

void IntrinsicMatrix::scale(float factor)
{
    data *= factor;
}

} // namespace core
} // namespace dvo