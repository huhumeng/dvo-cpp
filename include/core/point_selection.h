#pragma once

#include "core/rgbd_image.h"

namespace dvo
{
namespace core
{
class PointSelectionPredicate
{
public:
    PointSelectionPredicate() : intensity_threshold(0.0f),
                                depth_threshold(0.0f)
    {
    }

    bool isPointOk(const size_t &x, const size_t &y, const float &z, const float &idx, const float &idy, const float &zdx, const float &zdy) const
    {
        return z == z && zdx == zdx && zdy == zdy && (std::abs(idx) > intensity_threshold || std::abs(idy) > intensity_threshold || std::abs(zdx) > depth_threshold || std::abs(zdy) > depth_threshold);
    }

    float intensity_threshold;
    float depth_threshold;
};

class PointSelection
{
public:
    typedef PointWithIntensityAndDepth::VectorType PointVector;
    typedef PointVector::iterator PointIterator;

    PointSelection(const PointSelectionPredicate &predicate);
    PointSelection(RgbdImagePyramid &pyramid, const PointSelectionPredicate &predicate);

    ~PointSelection() = default;

    RgbdImagePyramid &getRgbdImagePyramid();

    void setRgbdImagePyramid(RgbdImagePyramid &pyramid);

    size_t getMaximumNumberOfPoints(const size_t &level);

    void select(const size_t &level, PointIterator &first_point, PointIterator &last_point);

    bool getDebugIndex(const size_t &level, cv::Mat &dbg_idx);

    void debug(bool v)
    {
        debug_ = v;
    }

    bool debug() const
    {
        return debug_;
    }

private:
    
    struct Storage
    {
    public:
        PointVector points;
        PointIterator points_end;
        bool is_cached;

        cv::Mat debug_idx;

        Storage();
        void allocate(size_t max_points);
    };

    PointIterator selectPointsFromImage(const RgbdImage &img, const PointIterator &first_point, const PointIterator &last_point, cv::Mat &debug_idx);

    RgbdImagePyramid *pyramid_;
    std::vector<Storage> storage_;
    const PointSelectionPredicate &predicate_;

    bool debug_;
};

} // namespace core
} // namespace dvo