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
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp ptl_msgs
    DEPENDS)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories( 
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    "/usr/local/include/opentracker" 
)
find_library(OpenTracker_LIBS libopentracker.so /usr/local/lib)
add_library(tracker_lib src/local_object.cpp src/tracker.cpp)
target_link_libraries(tracker_lib ${OpenTracker_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES}) 

add_executable(tracker src/main.cpp) 
target_link_libraries(tracker tracker_lib)
add_dependencies(tracker ptl_msgs_generate_messages_cpp)
