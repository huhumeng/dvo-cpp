#pragma once

#include <chrono>

namespace dvo
{

namespace util
{

using std::chrono::hours;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::minutes;
using std::chrono::nanoseconds;
using std::chrono::seconds;

namespace details
{
template <typename _Clock, typename _Dur>
using time_point = std::chrono::time_point<_Clock, _Dur>;
using default_clock_t = std::chrono::high_resolution_clock;
} // namespace details

/**
 * @brief Timer. A tic-toc timer.
 *
 * Mesure the elapsed time between construction - or tic() -
 * and toc(). The elapsed time is expressed in unit.
 *
 * @param unit. The time unit.
 * @see unit
 */
template <typename unit>
class Timer
{
public:
    /**
     * @brief Timer. Launch the timer.
     */
    Timer() : start_(now()) {}

    /**
     * @brief ~Timer. Default desctructor.
     */
    ~Timer() = default;

    /**
     * @brief tic. Reset the timer.
     */
    void tic() { start_ = now(); }

    /**
     * @brief toc. Return this elapsed time since construction or last tic().
     * @return double. The elapsed time.
     * @see tic()
     */
    template <typename T = int64_t>
    T toc()
    {
        return static_cast<T>(cast_d(now() - start_).count());
    }

protected:
    details::time_point<details::default_clock_t, unit> start_;

    template <typename... Args>
    auto cast_d(Args &&... args) -> decltype(
        std::chrono::duration_cast<unit>(std::forward<Args>(args)...))
    {
        return std::chrono::duration_cast<unit>(std::forward<Args>(args)...);
    }

    auto now() -> decltype(start_)
    {
        return std::chrono::time_point_cast<unit>(details::default_clock_t::now());
    }
};

} // namespace util

using TimerSecs = util::Timer<util::seconds>;
using TimerMsecs = util::Timer<util::milliseconds>;
using TimerUsecs = util::Timer<util::microseconds>;
using TimerNsecs = util::Timer<util::nanoseconds>;
} // namespace dvo
