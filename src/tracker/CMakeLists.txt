cmake_minimum_required(VERSION 3.5)

project(tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")

find_package(catkin 
    REQUIRED 
    COMPONENTS
    roscpp
)

catkin_package(
    CATKIN_DEPENDS roscpp
    DEPENDS)

find_package(OpenCV 3 REQUIRED)
include_directories( "/usr/local/include/opentracker" )
find_library(OpenTracker_LIBS libopentracker.so /usr/local/lib)

add_executable(tracker src/tracker.cpp) 
target_link_libraries(tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${OpenTracker_LIBS}) 
