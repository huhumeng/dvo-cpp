#include "core/param_config.h"

namespace dvo
{
namespace core
{

ParamConfig::ParamConfig() : FirstLevel(3),
                             LastLevel(1),
                             MaxIterationsPerLevel(100),
                             Precision(5e-7),
                             UseInitialEstimate(false),
                             UseWeighting(true),
                             Mu(0),
                             weight_function_type(WeightFunctions::TDistribution),
                             weight_function_param(5.f),
                             scale_estimator_type(ScaleEstimators::TDistribution),
                             scale_stimator_param(5.f),
                             IntensityDerivativeThreshold(0.0f),
                             DepthDerivativeThreshold(0.0f)
{
}

size_t ParamConfig::getNumLevels() const
{
    return FirstLevel + 1;
}

bool ParamConfig::useEstimateSmoothing() const
{
    return Mu > 1e-6;
}

} // namespace core
} // namespace dvo