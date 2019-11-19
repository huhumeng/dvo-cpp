#include "core/math_sse.h"
#include "util/tic_toc.h"

#include <iostream>

typedef Eigen::Matrix<float, 6, 6> Hessian;
typedef Eigen::Matrix<float, Eigen::Dynamic, 6> JacobianWorkspace;

constexpr int Dimension = 2;

void do_reduction(const JacobianWorkspace &j, const Eigen::Matrix<float, Dimension, Dimension> &alpha, Hessian &A)
{
    A.setZero();

    for (int idx = 0; idx < j.rows(); idx += Dimension)
    {
        A += j.block<Dimension, 6>(idx, 0).transpose() * alpha * j.block<Dimension, 6>(idx, 0);
    }
}

void do_optimized_reduction(const JacobianWorkspace &j, const Eigen::Matrix<float, Dimension, Dimension> &alpha, Hessian &A)
{
    dvo::core::OptimizedSelfAdjointMatrix6x6f A_opt;
    A_opt.setZero();

    Eigen::Matrix2f my_alpha;
    my_alpha.block<Dimension, Dimension>(0, 0) = alpha;

    for (int idx = 0; idx < j.rows(); idx += Dimension)
    {
        if (Dimension == 1)
            A_opt.rankUpdate(j.block<1, 6>(idx, 0).transpose(), my_alpha(0, 0));
        else
            A_opt.rankUpdate(j.block<2, 6>(idx, 0), my_alpha);
    }

    A_opt.toEigen(A);
}

int main(int argc, char **argv)
{
    Hessian A1, A2;

    Eigen::Matrix<float, Dimension, Dimension> alpha;
    alpha.setRandom();
    alpha(0, 1) = alpha(1, 0);

    JacobianWorkspace J;
    J.resize(640 * 480 * Dimension, Eigen::NoChange);
    J.setRandom();

    dvo::TimerMsecs timer;

    for (int idx = 0; idx < 1e2; idx++)
        do_reduction(J, alpha, A1);

    std::cout << "Unoptimized using " << timer.toc() << " ms." << std::endl;

    timer.tic();
    for (int idx = 0; idx < 1e2; idx++)
        do_optimized_reduction(J, alpha, A2);

    std::cout << "Optimized using " << timer.toc() << " ms." << std::endl;
    
    return 0;
}