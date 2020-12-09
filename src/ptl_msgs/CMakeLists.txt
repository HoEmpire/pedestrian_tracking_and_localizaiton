cmake_minimum_required(VERSION 3.5)

project(ptl_msgs)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")

find_package(catkin 
    REQUIRED 
    COMPONENTS
    std_msgs
    sensor_msgs
    message_generation
    )

add_message_files(
    FILES
    ImageBlock.msg
    DeadTracker.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

