cmake_minimum_required(VERSION 3.5)

project(ptl_reid)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")

find_package(catkin 
    REQUIRED 
    COMPONENTS
)

catkin_package(CATKIN_DEPENDS)

