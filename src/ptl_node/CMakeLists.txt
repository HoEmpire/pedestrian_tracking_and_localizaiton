cmake_minimum_required(VERSION 3.5)

project(ptl_node)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -pthread")

find_package(catkin 
    REQUIRED 
    COMPONENTS
    roscpp
    ptl_tracker
    ptl_reid_cpp
    pcl_ros
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp ptl_reid_cpp ptl_tracker
    DEPENDS)

include_directories( 
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_library(ptl_node src/node.cpp)
target_link_libraries(ptl_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(node app/main.cpp) 
target_link_libraries(node ptl_node)
