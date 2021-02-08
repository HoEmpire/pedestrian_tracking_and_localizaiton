cmake_minimum_required(VERSION 3.5)

project(ptl_detector)

# set compilation
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-Wall -Wextra -Wno-unused-parameter -O2)

# add useful marcos
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/util.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/modules)

# find dependencies
find_package(OpenCV 4 REQUIRED)
find_package(Glog REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/yolo-tensorrt)

# ROS
find_package(catkin 
    REQUIRED 
    COMPONENTS
    std_msgs
    sensor_msgs
    roscpp
    rospy
    ptl_msgs
    cv_bridge
    image_transport
    )

# ROS
catkin_package(
    INCLUDE_DIRS include ${CMAKE_CURRENT_SOURCE_DIR}/external/yolo-tensorrt/modules
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp
    DEPENDS)


# common include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/yolo-tensorrt/modules
    ${GLOG_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# find all source files
file(GLOB_RECURSE
    src_files
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*/*.cc
    )

# common library
add_library(
    ${PROJECT_NAME}
    SHARED
    ${src_files}
)
target_link_libraries(
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${OpenCV_LIBS}
    yolo_trt
)

# apps
# add_simple_apps()



