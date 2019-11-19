#pragma once

#include "core/data_types.h"

#include <opencv2/core/core.hpp>

namespace dvo
{
namespace core
{

// interface for scale estimators
class ScaleEstimator
{
public:
    virtual ~ScaleEstimator() = default;
    virtual float compute(const cv::Mat &errors) const = 0;
    virtual void configure(const float &param){};
};

class UnitScaleEstimator : public ScaleEstimator
{
public:
    virtual ~UnitScaleEstimator() = default;
    virtual float compute(const cv::Mat &errors) const override { return 1.0f; };
};

// estimates scale by fitting a t-distribution to the data with the given degrees of freedom
class TDistributionScaleEstimator : public ScaleEstimator
{
public:
    virtual ~TDistributionScaleEstimator() = default;
    
    virtual float compute(const cv::Mat &errors) const override;
    virtual void configure(const float &param) override;

protected:
    float dof = 5.f;
    float initial_sigma = 5.f;
};



// estimates scale by computing the median absolute deviation
class MADScaleEstimator : public ScaleEstimator
{
public:

    virtual ~MADScaleEstimator() = default;
    virtual float compute(const cv::Mat &errors) const override;

};

// estimates scale by computing the standard deviation
class NormalDistributionScaleEstimator : public ScaleEstimator
{
public:
    virtual ~NormalDistributionScaleEstimator() = default;
    virtual float compute(const cv::Mat &errors) const override;
};

struct ScaleEstimators
{
    typedef enum
    {
        Unit,
        NormalDistribution,
        TDistribution,
        MAD
        // don't forget to add to dynamic reconfigure!
    } enum_t;

    static const char *str(enum_t type);

    static ScaleEstimator *get(enum_t type);
};

/**
 * Interface for influence functions. An influence function is the first derivative of a symmetric robust function p(sqrt(x)).
 * The errors are assumed to be normalized to unit variance.
 *
 * See:
 *   "Lucas-Kanade 20 Years On: A Unifying Framework: Part 2"
 */
class WeightFunction
{
public:
    virtual ~WeightFunction() = default;

    virtual float value(const float &x) const = 0;
    virtual void configure(const float &param){};
};

class UnitWeightFunction : public WeightFunction
{
public:

    virtual ~UnitWeightFunction() = default;

    virtual inline float value(const float &x) const { return 1.0f; };
};


class TukeyWeightFunction : public WeightFunction
{
public:

    virtual ~TukeyWeightFunction() = default;
    virtual inline float value(const float &x) const;
    virtual void configure(const float &param);

private:
    float b_square = 4.6851f;
};

class TDistributionWeightFunction : public WeightFunction
{
public:
    virtual ~TDistributionWeightFunction() = default;
    virtual inline float value(const float &x) const;
    virtual void configure(const float &param);

private:
    float dof = 5.0f;
};

class HuberWeightFunction : public WeightFunction
{
public:
    
    virtual ~HuberWeightFunction() = default;
    virtual inline float value(const float &x) const;
    virtual void configure(const float &param);

private:
    float k;
};

struct WeightFunctions
{
    typedef enum
    {
        Unit,
        Tukey,
        TDistribution,
        Huber,
        // don't forget to add to dynamic reconfigure!
    } enum_t;

    static const char *str(enum_t type);

    static WeightFunction *get(enum_t type);
};

class WeightCalculation
{
public:
    WeightCalculation();

    void calculateScale(const cv::Mat &errors);

    float calculateWeight(const float error) const;

    void calculateWeights(const cv::Mat &errors, cv::Mat &weights);

    const ScaleEstimator *scaleEstimator() const { return scaleEstimator_; }
    ScaleEstimator *scaleEstimator() { return scaleEstimator_; }

    WeightCalculation &scaleEstimator(ScaleEstimator * value){scaleEstimator_ = value; return *this;}

    const WeightFunction *weightFunction() const { return weightFunction_; }
    WeightFunction *weightFunction() { return weightFunction_; }

    WeightCalculation &weightFunction(WeightFunction * value){weightFunction_ = value; return *this;}


private:
    ScaleEstimator *scaleEstimator_;

    WeightFunction *weightFunction_;

    float scale_;
};

} // namespace core
} // namespace dvo