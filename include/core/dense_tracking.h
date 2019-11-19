#pragma once

#include "core/weight_calculation.h"
#include "core/point_selection.h"
#include "core/param_config.h"
#include "core/optimizer.h"

namespace dvo
{
namespace core
{
class DenseTracker
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:

    static const ParamConfig &getDefaultConfig();

    DenseTracker(const ParamConfig &cfg = getDefaultConfig());
    DenseTracker(const DenseTracker &other);

    const ParamConfig &configuration() const
    {
        return cfg;
    }

    void configure(const ParamConfig &cfg);

    bool match(RgbdImagePyramid &reference, RgbdImagePyramid &current, AffineTransformd &transformation);
    bool match(PointSelection &reference, RgbdImagePyramid &current, AffineTransformd &transformation);

    bool match(RgbdImagePyramid &reference, RgbdImagePyramid &current, Result &result);
    bool match(PointSelection &reference, RgbdImagePyramid &current, Result &result);

    cv::Mat computeIntensityErrorImage(RgbdImagePyramid &reference, RgbdImagePyramid &current, const AffineTransformd &transformation, size_t level = 0);

    typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> ResidualVectorType;
    typedef std::vector<float> WeightVectorType;

private:
    struct IterationContext
    {
        const ParamConfig &cfg;

        int Level;
        int Iteration;

        double Error, LastError;

        IterationContext(const ParamConfig &cfg);

        // returns true if this is the first iteration
        bool IsFirstIteration() const;

        // returns true if this is the first iteration on the current level
        bool IsFirstIterationOnLevel() const;

        // returns true if this is the first level
        bool IsFirstLevel() const;

        // returns true if this is the last level
        bool IsLastLevel() const;

        bool IterationsExceeded() const;

        // returns LastError - Error
        double ErrorDiff() const;
    };

    ParamConfig cfg;

    IterationContext itctx_;

    WeightCalculation weight_calculation_;
    PointSelection reference_selection_;
    PointSelectionPredicate selection_predicate_;

    PointWithIntensityAndDepth::VectorType points, points_error;

    ResidualVectorType residuals;
    WeightVectorType weights;
};

typedef PointWithIntensityAndDepth::VectorType::iterator PointIterator;
typedef DenseTracker::ResidualVectorType::iterator ResidualIterator;
typedef DenseTracker::WeightVectorType::iterator WeightIterator;
typedef std::vector<uint8_t>::iterator ValidFlagIterator;

struct ComputeResidualsResult
{
    PointIterator first_point_error;
    PointIterator last_point_error;

    ResidualIterator first_residual;
    ResidualIterator last_residual;

    ValidFlagIterator first_valid_flag;
    ValidFlagIterator last_valid_flag;
};

void computeResiduals(const PointIterator &first_point, const PointIterator &last_point, const RgbdImage &current, const IntrinsicMatrix &intrinsics, const Eigen::Affine3f transform, const Vector8f &reference_weight, const Vector8f &current_weight, ComputeResidualsResult &result);

void computeResidualsSse(const PointIterator &first_point, const PointIterator &last_point, const RgbdImage &current, const IntrinsicMatrix &intrinsics, const Eigen::Affine3f transform, const Vector8f &reference_weight, const Vector8f &current_weight, ComputeResidualsResult &result);
void computeResidualsAndValidFlagsSse(const PointIterator &first_point, const PointIterator &last_point, const RgbdImage &current, const IntrinsicMatrix &intrinsics, const Eigen::Affine3f transform, const Vector8f &reference_weight, const Vector8f &current_weight, ComputeResidualsResult &result);

float computeCompleteDataLogLikelihood(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Vector2f &mean, const Eigen::Matrix2f &precision);

float computeWeightedError(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Matrix2f &precision);
float computeWeightedErrorSse(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Matrix2f &precision);

//Eigen::Vector2f computeMean(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight);

Eigen::Matrix2f computeScale(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Vector2f &mean);
Eigen::Matrix2f computeScaleSse(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Vector2f &mean);

void computeWeights(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Vector2f &mean, const Eigen::Matrix2f &precision);
void computeWeightsSse(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, const Eigen::Vector2f &mean, const Eigen::Matrix2f &precision);

void computeMeanScaleAndWeights(const ResidualIterator &first_residual, const ResidualIterator &last_residual, const WeightIterator &first_weight, Eigen::Vector2f &mean, Eigen::Matrix2f &precision);

} // namespace core
} // namespace dvo
