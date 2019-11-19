#pragma once

#include "core/weight_calculation.h"

namespace dvo
{
namespace core
{

struct ParamConfig
{
    int FirstLevel, LastLevel;
    int MaxIterationsPerLevel;
    double Precision;
    double Mu; // precision (1/sigma^2) of prior

    bool UseInitialEstimate;
    bool UseWeighting;

    bool UseParallel;

    WeightFunctions::enum_t weight_function_type;
    float weight_function_param;

    ScaleEstimators::enum_t scale_estimator_type;
    float scale_stimator_param;

    float IntensityDerivativeThreshold;
    float DepthDerivativeThreshold;

    ParamConfig();

    size_t getNumLevels() const;

    bool useEstimateSmoothing() const;
};

} // namespace core

} // namespace dvo

template <typename CharT, typename Traits>
std::ostream &operator<<(std::basic_ostream<CharT, Traits> &out, const dvo::core::ParamConfig &config)
{
    out
        << "First Level = " << config.FirstLevel
        << ", Last Level = " << config.LastLevel
        << ", Max Iterations per Level = " << config.MaxIterationsPerLevel
        << ", Precision = " << config.Precision
        << ", Mu = " << config.Mu
        << ", Use Initial Estimate = " << (config.UseInitialEstimate ? "true" : "false")
        << ", Use Weighting = " << (config.UseWeighting ? "true" : "false")
        << ", Scale Estimator = " << dvo::core::ScaleEstimators::str(config.scale_estimator_type)
        << ", Scale Estimator Param = " << config.scale_stimator_param
        << ", Influence Function = " << dvo::core::WeightFunctions::str(config.weight_function_type)
        << ", Influence Function Param = " << config.weight_function_param
        << ", Intensity Derivative Threshold = " << config.IntensityDerivativeThreshold
        << ", Depth Derivative Threshold = " << config.DepthDerivativeThreshold;

    return out;
}