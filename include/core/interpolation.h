#pragma once

#include "core/data_types.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace dvo
{
namespace core
{

struct Interpolation
{
    static IntensityType none(const cv::Mat &img, float x, float y);
    static IntensityType bilinear(const cv::Mat &img, float x, float y);
    static IntensityType bilinearWithDepthBuffer(const cv::Mat &intensity, const cv::Mat &depth, float x, float y, float z);
};

template <typename T>
void pyrDownMeanSmooth(const cv::Mat &in, cv::Mat &out)
{
    out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

    for (int y = 0; y < out.rows; ++y)
    {
        for (int x = 0; x < out.cols; ++x)
        {
            int x0 = x * 2;
            int x1 = x0 + 1;
            int y0 = y * 2;
            int y1 = y0 + 1;

            out.at<T>(y, x) = (T)((in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f);
        }
    }
}

template <typename T>
void pyrDownMeanSmoothIgnoreInvalid(const cv::Mat &in, cv::Mat &out)
{
    out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

    for (int y = 0; y < out.rows; ++y)
    {
        for (int x = 0; x < out.cols; ++x)
        {
            int x0 = x * 2;
            int x1 = x0 + 1;
            int y0 = y * 2;
            int y1 = y0 + 1;

            T total = 0;
            int cnt = 0;

            auto ignore_invalid = [&cnt, &total, &in](int x, int y) {
                if (std::isfinite(in.at<T>(y, x)))
                {
                    total += in.at<T>(y, x);
                    cnt++;
                }
            };
            
            ignore_invalid(x0, y0);
            ignore_invalid(x0, y1);
            ignore_invalid(x1, y0);
            ignore_invalid(x1, y1);
        
            if (cnt > 0)
            {
                out.at<T>(y, x) = total / (T)cnt;
            }
            else
            {
                out.at<T>(y, x) = InvalidDepth;
            }
        }
    }
}

template <typename T>
void pyrDownMedianSmooth(const cv::Mat &in, cv::Mat &out)
{
    out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

    cv::Mat in_smoothed;
    cv::medianBlur(in, in_smoothed, 3);

    for (int y = 0; y < out.rows; ++y)
    {
        for (int x = 0; x < out.cols; ++x)
        {
            out.at<T>(y, x) = in_smoothed.at<T>(y * 2, x * 2);
        }
    }
}

template <typename T>
void pyrDownSubsample(const cv::Mat &in, cv::Mat &out)
{
    out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

    for (int y = 0; y < out.rows; ++y)
    {
        for (int x = 0; x < out.cols; ++x)
        {
            out.at<T>(y, x) = in.at<T>(y * 2, x * 2);
        }
    }
}

template <typename T>
void calculateDerivativeX(const cv::Mat &img, cv::Mat &result)
{
    result.create(img.size(), img.type());
    result = 0;

    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 1; x < img.cols - 1; ++x)
        {
            result.at<T>(y, x) = (T)(img.at<T>(y, x + 1) - img.at<T>(y, x - 1)) * 0.5f;
        }
    }
}

template <typename T>
void calculateDerivativeY(const cv::Mat &img, cv::Mat &result)
{
    result.create(img.size(), img.type());
    result = 0;

    for (int y = 1; y < img.rows - 1; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            result.at<T>(y, x) = (T)(img.at<T>(y + 1, x) - img.at<T>(y - 1, x)) * 0.5f;
        }
    }
}



} /* namespace core */
} /* namespace dvo */