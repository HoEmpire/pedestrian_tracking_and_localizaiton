cmake_minimum_required(VERSION 3.5)

project(ptl_reid_cpp)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")

find_package(catkin 
    REQUIRED 
    COMPONENTS
    roscpp
    cv_bridge
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp
    DEPENDS)

find_package(OpenCV 4 REQUIRED)
find_library(Faiss_LIBS libfaiss.a /usr/local/lib)

include_directories( 
    include
    ${catkin_INCLUDE_DIRS}
)
add_library(reid_lib src/reid_database.cpp)
target_link_libraries(reid_lib ${OpenCV_LIBS} ${catkin_LIBRARIES}) 

# add_library(pcp_lib src/point_cloud_processor.cpp)

# add_executable(tracker src/main.cpp) 
# target_link_libraries(tracker tracker_lib pcp_lib)
# add_dependencies(tracker ptl_msgs_generate_messages_cpp)