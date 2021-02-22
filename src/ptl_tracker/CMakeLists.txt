cmake_minimum_required(VERSION 3.5)

project(ptl_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")

find_package(catkin 
    REQUIRED 
    COMPONENTS
    roscpp
    ptl_msgs
    cv_bridge
    tf2_ros
    tf2_sensor_msgs
    tf2_geometry_msgs
    pcl_ros
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp ptl_msgs tf2_sensor_msgs
    DEPENDS)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED) 

include_directories( 
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)
add_library(ptl_tracker 
            src/local_object.cpp 
            src/tracker.cpp 
            src/kalman_filter.cpp 
            src/kalman_filter_3d.cpp 
            src/optical_flow.cpp 
            src/point_cloud_processor.cpp
        )
target_link_libraries(ptl_tracker ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PCL_LIBRARIES}) 
add_dependencies(ptl_tracker ptl_msgs_generate_messages_cpp)
