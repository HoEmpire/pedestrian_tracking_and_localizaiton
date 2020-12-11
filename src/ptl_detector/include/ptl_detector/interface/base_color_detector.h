#pragma once

namespace ptl
{
    namespace detector
    {

        struct ColorDetectionResult
        {
            int color = -1;
            float prob = 0.0;
            ColorDetectionResult(int color = -1, float prob = 0.0)
                : color(color), prob(prob) {}
        };

    } // namespace detector
} // namespace ptl