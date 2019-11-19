#include "core/math_sse.h"

#include <iostream>

#include <immintrin.h>

namespace dvo
{
namespace core
{
void OptimizedSelfAdjointMatrix6x6f::setZero()
{
    memset(data, 0, sizeof(float) * 24);
}

void OptimizedSelfAdjointMatrix6x6f::rankUpdate(const Eigen::Matrix<float, 6, 1> &u, const float &alpha)
{
    __m128 s = _mm_set1_ps(alpha);
    __m128 v1234 = _mm_loadu_ps(u.data());
    __m128 v56xx = _mm_loadu_ps(u.data() + 4);

    __m128 v1212 = _mm_movelh_ps(v1234, v1234);
    __m128 v3434 = _mm_movehl_ps(v1234, v1234);
    __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

    __m128 v1122 = _mm_mul_ps(s, _mm_unpacklo_ps(v1212, v1212));

    _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
    _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
    _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

    __m128 v3344 = _mm_mul_ps(s, _mm_unpacklo_ps(v3434, v3434));

    _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
    _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

    __m128 v5566 = _mm_mul_ps(s, _mm_unpacklo_ps(v5656, v5656));

    _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));
}

void OptimizedSelfAdjointMatrix6x6f::rankUpdate(const Eigen::Matrix<float, 2, 6> &v, const Eigen::Matrix2f &alpha)
{
    /**
   * layout of alpha:
   *
   *   1 2
   *   3 4
   */
    __m128 alpha1324 = _mm_load_ps(alpha.data());           // load first two columns from column major data
    __m128 alpha1313 = _mm_movelh_ps(alpha1324, alpha1324); // first column 2x
    __m128 alpha2424 = _mm_movehl_ps(alpha1324, alpha1324); // second column 2x

    /**
   * layout of v:
   *
   *   1a 2a 3a 4a 5a 6a
   *   1b 2b 3b 4b 5b 6b
   */

    /**
   * layout of u = v * alpha:
   *
   *   1a 2a 3a 4a 5a 6a
   *   1b 2b 3b 4b 5b 6b
   */
    __m128 v1a1b2a2b = _mm_load_ps(v.data() + 0); // load first and second column

    __m128 u1a2a1b2b = _mm_hadd_ps(
        _mm_mul_ps(v1a1b2a2b, alpha1313),
        _mm_mul_ps(v1a1b2a2b, alpha2424));

    __m128 u1a1b1a1b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u2a2b2a2b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(3, 1, 3, 1));

    // upper left 2x2 block of A matrix in row major format
    __m128 b11 = _mm_hadd_ps(
        _mm_mul_ps(u1a1b1a1b, v1a1b2a2b),
        _mm_mul_ps(u2a2b2a2b, v1a1b2a2b));
    _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), b11));

    __m128 v3a3b4a4b = _mm_load_ps(v.data() + 4); // load third and fourth column

    // upper center 2x2 block of A matrix in row major format
    __m128 b12 = _mm_hadd_ps(
        _mm_mul_ps(u1a1b1a1b, v3a3b4a4b),
        _mm_mul_ps(u2a2b2a2b, v3a3b4a4b));
    _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), b12));

    __m128 v5a5b6a6b = _mm_load_ps(v.data() + 8); // load fifth and sixth column

    // upper right 2x2 block of A matrix in row major format
    __m128 b13 = _mm_hadd_ps(
        _mm_mul_ps(u1a1b1a1b, v5a5b6a6b),
        _mm_mul_ps(u2a2b2a2b, v5a5b6a6b));
    _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), b13));

    __m128 u3a4a3b4b = _mm_hadd_ps(
        _mm_mul_ps(v3a3b4a4b, alpha1313),
        _mm_mul_ps(v3a3b4a4b, alpha2424));

    __m128 u3a3b3a3b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u4a4b4a4b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(3, 1, 3, 1));

    // center center 2x2 block of A matrix in row major format
    __m128 b22 = _mm_hadd_ps(
        _mm_mul_ps(u3a3b3a3b, v3a3b4a4b),
        _mm_mul_ps(u4a4b4a4b, v3a3b4a4b));
    _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), b22));

    // center right 2x2 block of A matrix in row major format
    __m128 b23 = _mm_hadd_ps(
        _mm_mul_ps(u3a3b3a3b, v5a5b6a6b),
        _mm_mul_ps(u4a4b4a4b, v5a5b6a6b));
    _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), b23));

    __m128 u5a6a5b6b = _mm_hadd_ps(
        _mm_mul_ps(v5a5b6a6b, alpha1313),
        _mm_mul_ps(v5a5b6a6b, alpha2424));

    __m128 u5a5b5a5b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u6a6b6a6b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(3, 1, 3, 1));

    // bottom right 2x2 block of A matrix in row major format
    __m128 b33 = _mm_hadd_ps(
        _mm_mul_ps(u5a5b5a5b, v5a5b6a6b),
        _mm_mul_ps(u6a6b6a6b, v5a5b6a6b));
    _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), b33));
}

void OptimizedSelfAdjointMatrix6x6f::toEigen(Eigen::Matrix<float, 6, 6> &m) const
{
    Eigen::Matrix<float, 6, 6> tmp;
    size_t idx = 0;

    for (size_t i = 0; i < 6; i += 2)
    {
        for (size_t j = i; j < 6; j += 2)
        {
            tmp(i, j) = data[idx++];
            tmp(i, j + 1) = data[idx++];
            tmp(i + 1, j) = data[idx++];
            tmp(i + 1, j + 1) = data[idx++];
        }
    }

    tmp.selfadjointView<Eigen::Upper>().evalTo(m);
}

} // namespace core
} // namespace dvo