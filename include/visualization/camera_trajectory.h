#pragma once

#include <functional>

namespace dvo
{
namespace visualization
{
struct Color
{
public:
    static const Color &red()
    {
        static Color red(1.0, 0.2, 0.2);
        return red;
    }
    static const Color &green()
    {
        static Color green(0.2, 1.0, 0.2);
        return green;
    }
    static const Color &blue()
    {
        static Color blue(0.2, 0.2, 1.0);
        return blue;
    }

    Color() : r(0), g(0), b(0)
    {
    }
    Color(double r, double g, double b) : r(r), g(g), b(b)
    {
    }

    double r, g, b;
};

class CameraVisualizer
{
public:

    // std::function<>
};


} // namespace visualization
} // namespace dvo