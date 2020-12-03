cmake_minimum_required(VERSION 3.5)

project(ptl_detector)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")

find_package(catkin 
    REQUIRED 
    COMPONENTS
    std_msgs
    sensor_msgs
    roscpp
    rospy
    usfs_inference
    usfs_common
    # cv_bridge
    # image_transport
    )

find_package(OpenCV 4.4.0 REQUIRED)


# ROS
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp
    DEPENDS)


# common include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

include_directories( "/usr/local/include/opentracker" )
find_library(OpenTracker_LIBS libopentracker.so /usr/local/lib)

add_executable(detector src/detector.cpp) 
target_link_libraries(detector ${catkin_LIBRARIES} ${OpenCV_LIBS})

