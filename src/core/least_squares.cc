#include "core/least_squares.h"

#include <Eigen/Dense>

namespace dvo
{
namespace core
{

constexpr NumType normalizer_inverse = 255. * 255.;
constexpr NumType normalizer = 1. / normalizer_inverse;

void NormalEquationsLeastSquares::initialize()
{
    A.setZero();
    A_opt.setZero();
    b.setZero();
}

void NormalEquationsLeastSquares::update(const Vector6 &J, const NumType &res, const NumType &weight)
{   
    // Hessian += J^t W J
    A_opt.rankUpdate(J, weight);

    // b -= J^T W r
    b -= res * weight * J;
}

void NormalEquationsLeastSquares::update(const Eigen::Matrix<NumType, 2, 6> &J, const Eigen::Matrix<NumType, 2, 1> &res, const Eigen::Matrix<NumType, 2, 2> &weight)
{
    A_opt.rankUpdate(J, weight);
    b -= J.transpose() * weight * res;
}

void NormalEquationsLeastSquares::finish()
{   
    A_opt.toEigen(A);
}

void NormalEquationsLeastSquares::solve(Vector6 &x)
{
    // Using ldlt decompose
    x = A.ldlt().solve(b);

    // or EVD decompose
    // Eigen::SelfAdjointEigenSolver<Matrix6x6> eigensolver(A);
    // Vector6 eigenvalues = eigensolver.eigenvalues();
    // Matrix6x6 eigenvectors = eigensolver.eigenvectors();

    // bool singular = false;

    // for (int i = 0; i < 6; ++i)
    // {
    //     if (eigenvalues(i) < 0.05)
    //     {
    //         singular = true;
    //         throw std::exception();
    //     }
    //     else
    //     {
    //         eigenvalues(i) = 1.0 / eigenvalues(i);
    //     }
    // }

    // x = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
}

} // namespace core
} // namespace dvo
