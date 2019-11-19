#pragma once

#include <Eigen/Core>

#include "core/param_config.h"

namespace dvo
{
namespace core
{

enum class TerminationCriteria
{
    IterationsExceeded,
    IncrementTooSmall,
    LogLikelihoodDecreased,
    TooFewConstraints,
    NumCriteria
};

struct IterationStats
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    size_t Id, ValidConstraints;

    double TDistributionLogLikelihood;
    Eigen::Vector2d TDistributionMean;
    Eigen::Matrix2d TDistributionPrecision;

    double PriorLogLikelihood;

    Vector6d EstimateIncrement;
    Matrix6d EstimateInformation;

    void InformationEigenValues(Vector6d &eigenvalues) const;

    double InformationConditionNumber() const;
};
typedef std::vector<IterationStats> IterationStatsVector;

struct LevelStats
{
    size_t Id, MaxValidPixels, ValidPixels;
    TerminationCriteria TerminationCriterion;
    IterationStatsVector Iterations;

    bool HasIterationWithIncrement() const;

    IterationStats &LastIterationWithIncrement();
    IterationStats &LastIteration();

    const IterationStats &LastIterationWithIncrement() const;
    const IterationStats &LastIteration() const;
};
typedef std::vector<LevelStats> LevelStatsVector;

struct Stats
{
    LevelStatsVector Levels;
};

struct Result
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    AffineTransformd Transformation;
    Matrix6d Information;
    double LogLikelihood;

    Stats Statistics;

    Result();

    bool isNaN() const;
    void setIdentity();
    void clearStatistics();
};


} // namespace core
} // namespace dvo

template <typename CharT, typename Traits>
std::ostream &operator<<(std::basic_ostream<CharT, Traits> &o, const dvo::core::IterationStats &s)
{
    o << "Iteration: " << s.Id << " ValidConstraints: " << s.ValidConstraints << " DataLogLikelihood: " << s.TDistributionLogLikelihood << " PriorLogLikelihood: " << s.PriorLogLikelihood << std::endl;

    return o;
}


template <typename CharT, typename Traits>
std::ostream &operator<<(std::basic_ostream<CharT, Traits> &o, const dvo::core::LevelStats &s)
{
    std::string termination;

    switch (s.TerminationCriterion)
    {
    case dvo::core::TerminationCriteria::IterationsExceeded:
        termination = "IterationsExceeded";
        break;
    case dvo::core::TerminationCriteria::IncrementTooSmall:
        termination = "IncrementTooSmall";
        break;
    case dvo::core::TerminationCriteria::LogLikelihoodDecreased:
        termination = "LogLikelihoodDecreased";
        break;
    case dvo::core::TerminationCriteria::TooFewConstraints:
        termination = "TooFewConstraints";
        break;
    default:
        break;
    }

    o << "Level: " << s.Id << " Pixel: " << s.ValidPixels << "/" << s.MaxValidPixels << " Termination: " << termination << " Iterations: " << s.Iterations.size() << std::endl;

    for (auto it = s.Iterations.begin(); it != s.Iterations.end(); ++it)
    {
        o << *it;
    }

    return o;
}

template <typename CharT, typename Traits>
std::ostream &operator<<(std::basic_ostream<CharT, Traits> &o, const dvo::core::Stats &s)
{
    o << s.Levels.size() << " levels" << std::endl;

    for (auto it = s.Levels.begin(); it != s.Levels.end(); ++it)
    {
        o << *it;
    }

    return o;
}