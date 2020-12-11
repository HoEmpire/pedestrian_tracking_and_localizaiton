#pragma once

#include "log.h"
#include "macros.h"
#include "ros/ros.h"

namespace ptl
{
    namespace detector
    {

#ifndef GPARAM
#define GPARAM(x, y)                               \
    do                                             \
    {                                              \
        if (!nh_->getParam(x, y))                  \
        {                                          \
            ROS_FATAL("get pararm " #x " error!"); \
        }                                          \
    } while (0)
#endif

#ifndef DECLARE_ROS_INITIALIZATION
#define DECLARE_ROS_INITIALIZATION(classname)                   \
public:                                                         \
    explicit inline classname(ros::NodeHandle *nh) : nh_(nh){}; \
                                                                \
protected:                                                      \
    ros::NodeHandle *nh_;
#endif

        class RosNodeHandler
        {
        public:
            ros::NodeHandle *GetNh()
            {
                if (nh_ == nullptr)
                {
                    nh_ = new ros::NodeHandle("~");
                }
                //    AERROR_IF(nh_ == nullptr) << "Invalid node handler for ROS!";
                //    assert(nh_ != nullptr);
                return nh_;
            }

            void SetNh(ros::NodeHandle *nh) { nh_ = nh; }

        private:
            ros::NodeHandle *nh_{};

            DECLARE_SINGLETON(RosNodeHandler)
        };

    } // namespace detector
} // namespace ptl