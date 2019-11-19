#include "core/optimizer.h"

#include <Eigen/Dense>
#include <iostream>

namespace dvo
{
namespace core
{
void IterationStats::InformationEigenValues(Vector6d &eigenvalues) const
{
    Eigen::EigenSolver<Matrix6d> evd(EstimateInformation);
    eigenvalues = evd.eigenvalues().real();

    std::sort(eigenvalues.data(), eigenvalues.data() + 6);
}

double IterationStats::InformationConditionNumber() const
{
    Vector6d ev;
    InformationEigenValues(ev);

    return std::abs(ev(5) / ev(0));
}

bool LevelStats::HasIterationWithIncrement() const
{
    int min = TerminationCriterion == TerminationCriteria::LogLikelihoodDecreased || TerminationCriterion == TerminationCriteria::TooFewConstraints ? 2 : 1;

    return Iterations.size() >= min;
}

IterationStats &LevelStats::LastIterationWithIncrement()
{
    if (!HasIterationWithIncrement())
    {
        std::cerr << "Failed!!!!\n " << *this << std::endl;

        assert(false);
    }

    return TerminationCriterion == TerminationCriteria::LogLikelihoodDecreased ? Iterations[Iterations.size() - 2] : Iterations[Iterations.size() - 1];
}

IterationStats &LevelStats::LastIteration()
{
    return Iterations.back();
}

const IterationStats &LevelStats::LastIteration() const
{
    return Iterations.back();
}

const IterationStats &LevelStats::LastIterationWithIncrement() const
{
    if (!HasIterationWithIncrement())
    {
        std::cerr << "awkward " << *this << std::endl;

        assert(false);
    }
    return TerminationCriterion == TerminationCriteria::LogLikelihoodDecreased ? Iterations[Iterations.size() - 2] : Iterations[Iterations.size() - 1];
}

Result::Result() : LogLikelihood(std::numeric_limits<double>::max())
{
    double nan = std::numeric_limits<double>::quiet_NaN();
    Transformation.linear().setConstant(nan);
    Transformation.translation().setConstant(nan);
    Information.setIdentity();
}

bool Result::isNaN() const
{
    return !std::isfinite(Transformation.matrix().sum()) || !std::isfinite(Information.sum());
}

void Result::setIdentity()
{
    Transformation.setIdentity();
    Information.setIdentity();
    LogLikelihood = 0.0;
}

void Result::clearStatistics()
{
    Statistics.Levels.clear();
}

} // namespace core
} // namespace dvo