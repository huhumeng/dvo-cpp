#include "core/weight_calculation.h"
#include "util/histogram.h"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace dvo
{
namespace core
{

void TDistributionScaleEstimator::configure(const float &param)
{
    dof = param;
}

float TDistributionScaleEstimator::compute(const cv::Mat &errors) const
{
    float initial_lamda = 1.0f / (initial_sigma * initial_sigma);

    int num = 0;
    float lambda = initial_lamda;

    int iterations = 0;

    do
    {
        ++iterations;
        num = 0;
        initial_lamda = lambda;
        lambda = 0.0f;

        const float *data_ptr = errors.ptr<float>();

        for (size_t idx = 0; idx < errors.size().area(); ++idx, ++data_ptr)
        {
            const float &data = *data_ptr;

            if (std::isfinite(data))
            {
                ++num;
                float data2 = data * data;
                lambda += data2 * ((dof + 1.0f) / (dof + initial_lamda * data2));
            }
        }

        lambda = float(num) / lambda;
    } while (std::abs(lambda - initial_lamda) > 1e-3);

    return std::sqrt(1.0f / lambda);
}

float MADScaleEstimator::compute(const cv::Mat &errors) const
{
    cv::Mat error_hist;
    float median;

    cv::Mat abs_error = cv::abs(errors);

    util::compute1DHistogram(abs_error, error_hist, 0, 255, 1);
    median = util::computeMedianFromHistogram(error_hist, 0, 255);

    return 1.48f * median; // estimate of stddev
}

float NormalDistributionScaleEstimator::compute(const cv::Mat &errors) const
{
    cv::Mat mask = (errors == errors); // mask nans

    cv::Mat mean, stddev;
    cv::meanStdDev(errors, mean, stddev, mask);

    return (float)stddev.at<double>(0, 0);
}

const char *ScaleEstimators::str(enum_t type)
{
    switch (type)
    {
    case ScaleEstimators::Unit:
        return "Unit";
    case ScaleEstimators::TDistribution:
        return "TDistribution";
    case ScaleEstimators::MAD:
        return "MAD";
    case ScaleEstimators::NormalDistribution:
        return "NormalDistribution";
    default:
        break;
    }
    assert(false && "Unknown scale estimator type!");

    return "";
}

ScaleEstimator *ScaleEstimators::get(ScaleEstimators::enum_t type)
{
    static TDistributionScaleEstimator tdistribution;
    static MADScaleEstimator mad;
    static NormalDistributionScaleEstimator normaldistribution;
    static UnitScaleEstimator unit;

    switch (type)
    {
    case ScaleEstimators::Unit:
        return (ScaleEstimator *)&unit;
    case ScaleEstimators::TDistribution:
        return (ScaleEstimator *)&tdistribution;
    case ScaleEstimators::MAD:
        return (ScaleEstimator *)&mad;
    case ScaleEstimators::NormalDistribution:
        return (ScaleEstimator *)&normaldistribution;
    default:
        break;
    }
    assert(false && "Unknown scale estimator type!");

    return 0;
}

inline float TukeyWeightFunction::value(const float &x) const
{
    const float x_square = x * x;

    if (x_square <= b_square)
    {
        const float tmp = 1.0f - x_square / b_square;

        return tmp * tmp;
    }
    else
    {
        return 0.0f;
    }
}

void TukeyWeightFunction::configure(const float &param)
{
    b_square = param * param;
}

inline float TDistributionWeightFunction::value(const float &x) const
{

    return ((dof + 1.0f) / (dof + (x * x)));
}

void TDistributionWeightFunction::configure(const float &param)
{
    dof = param;
}

inline float HuberWeightFunction::value(const float &x) const
{
    const float x_abs = std::abs(x);

    if (x_abs < k)
    {
        return 1.0f;
    }
    else
    {
        return k / x_abs;
    }
}

void HuberWeightFunction::configure(const float &param)
{
    k = param;
}

const char *WeightFunctions::str(enum_t type)
{
    switch (type)
    {
    case Unit:
        return "Unit";
    case TDistribution:
        return "TDistribution";
    case Tukey:
        return "Tukey";
    case Huber:
        return "Huber";
    default:
        assert(false && "Unknown influence function type!");
        break;
    }

    return "";
}

WeightFunction *WeightFunctions::get(WeightFunctions::enum_t type)
{
    static TDistributionWeightFunction tdistribution;
    static TukeyWeightFunction tukey;
    static HuberWeightFunction huber;
    static UnitWeightFunction unit;

    switch (type)
    {
    case Unit:
        return (WeightFunction *)&unit;
    case TDistribution:
        return (WeightFunction *)&tdistribution;
    case Tukey:
        return (WeightFunction *)&tukey;
    case Huber:
        return (WeightFunction *)&huber;
    default:
        assert(false && "Unknown influence function type!");
        break;
    }

    return 0;
}

WeightCalculation::WeightCalculation() : scale_(1.0f)
{
}

void WeightCalculation::calculateScale(const cv::Mat &errors)
{
    // some scale estimators might return 0
    scale_ = std::max(scaleEstimator_->compute(errors), 0.001f);
}

float WeightCalculation::calculateWeight(const float error) const
{
    return weightFunction_->value(error / scale_);
}

void WeightCalculation::calculateWeights(const cv::Mat &errors, cv::Mat &weights)
{
    weights.create(errors.size(), errors.type());

    cv::Mat scaled_errors = errors / scale_;

    const float *err_ptr = scaled_errors.ptr<float>();

    float *weight_ptr = weights.ptr<float>();

    for (size_t idx = 0; idx < errors.size().area(); ++idx, ++err_ptr, ++weight_ptr)
    {
        if (std::isfinite(*err_ptr))
        {
            *weight_ptr = weightFunction_->value(*err_ptr);
        }
        else
        {
            *weight_ptr = 0.0f;
        }
    }
}
} // namespace core
} // namespace dvo