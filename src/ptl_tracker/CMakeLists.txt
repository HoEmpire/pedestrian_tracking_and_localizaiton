cmake_minimum_required(VERSION 3.5)

project(ptl_tracker)

set(CMAKE_BUILD_TYPE "Debug")
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

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED) 

include_directories( 
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    "/usr/local/include/opentracker" 
)
find_library(OpenTracker_LIBS libopentracker.so /usr/local/lib)
add_library(tracker_lib src/local_object.cpp src/tracker.cpp src/kalman_filter.cpp src/kalman_filter_3d.cpp)
target_link_libraries(tracker_lib ${OpenTracker_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES}) 

add_library(pcp_lib src/point_cloud_processor.cpp)

add_executable(tracker src/main.cpp) 
target_link_libraries(tracker tracker_lib pcp_lib)
add_dependencies(tracker ptl_msgs_generate_messages_cpp)

add_executable(tf_publisher src/tf_publisher.cpp) 
target_link_libraries(tf_publisher ${catkin_LIBRARIES})
