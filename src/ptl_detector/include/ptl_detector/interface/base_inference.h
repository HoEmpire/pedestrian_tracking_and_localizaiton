#pragma once

namespace ptl
{
    namespace detector
    {

        class BaseInference
        {
        public:
            BaseInference() = default;
            virtual ~BaseInference() = default;

            virtual bool Init() = 0;
            virtual void Infer() = 0;
        };

    } // namespace detector
} // namespace ptl
