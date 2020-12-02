#include <chrono>
class timer
{

public:
    timer()
    {
        t_start = std::chrono::steady_clock::now();
    }
    std::chrono::steady_clock::time_point t_start, t_end;
    void tic()
    {
        t_start = std::chrono::steady_clock::now();
    }

    double toc()
    {
        t_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    }
};