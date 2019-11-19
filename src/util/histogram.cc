#include "util/histogram.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace dvo
{
namespace util
{

/**
 * Computes the number of bins for a histogram with values in the range of [min max] and binWidth values in each bin.
 */
int getNumberOfBins(float min, float max, float binWidth)
{
    return (int)((max - min + 1) / binWidth);
}

int countElementsInHistogram(const cv::Mat &histogram)
{
    int num = 0;
    const float *histogram_ptr = histogram.ptr<float>();

    for (size_t idx = 0; idx < histogram.size().area(); ++idx, ++histogram_ptr)
    {
        num += (int)*histogram_ptr;
    }

    return num;
}

void compute1DHistogram(const cv::Mat &data, cv::Mat &histogram, float min, float max, float binWidth)
{
    cv::Mat mask;

    std::vector<int> channels = {0};
    std::vector<int> nbins = {getNumberOfBins(min, max, binWidth)};

    std::vector<float> range = {min, max};

    // seems to ignore nan values
    cv::calcHist(data, channels, cv::Mat(), histogram, nbins, range);
}

float computeMedianFromHistogram(const cv::Mat &histogram, float min, float max)
{
    float total_half = countElementsInHistogram(histogram) / 2.0f;
    const float *histogram_ptr = histogram.ptr<float>();

    float median = 0.0f;
    float acc = 0.0f;

    for (size_t idx = 0; idx < histogram.size().area(); ++idx, ++histogram_ptr)
    {
        acc += *histogram_ptr;

        if (acc > total_half)
        {
            median = idx;
            break;
        }
    }

    return median + min;
}

float computeEntropyFromHistogram(const cv::Mat &histogram)
{
    float sum = countElementsInHistogram(histogram);
    const float *histogram_ptr = histogram.ptr<float>();
    float entropy = 0.0;

    for (size_t idx = 0; idx < histogram.size().area(); ++idx, ++histogram_ptr)
    {
        if (*histogram_ptr < 1.0f)
            continue;

        float p = *histogram_ptr / sum;

        entropy -= p * std::log(p);
    }

    return entropy / std::log(2.0);
}



} /* namespace util */
} /* namespace dvo */
