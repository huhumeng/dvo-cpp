#pragma once

#include <opencv2/core/core.hpp>

namespace dvo
{
namespace util
{

/**
 * Computes the one dimensional histogram of the given data. The values have to be in the range [min max].
 *
 * See: cv::calcHist(...)
 */
void compute1DHistogram(const cv::Mat& data, cv::Mat& histogram, float min, float max, float binWidth);

float computeMedianFromHistogram(const cv::Mat& histogram, float min, float max);

float computeEntropyFromHistogram(const cv::Mat& histogram);

} /* namespace util */
} /* namespace dvo */