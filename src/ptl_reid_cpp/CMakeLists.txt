cmake_minimum_required(VERSION 3.5)

project(ptl_reid_cpp)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -pthread")

# add useful marcos
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/util.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/modules)

find_package(catkin 
    REQUIRED 
    COMPONENTS
    roscpp
    ptl_detector
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp ptl_detector
    DEPENDS)

find_package(OpenCV 4 REQUIRED)
find_package(Threads)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_library(Faiss_LIBS libfaiss.a /usr/local/lib)
find_package(OpenMP REQUIRED)
find_package(LAPACK REQUIRED)

include_directories( 
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS} 
    ${TensorRT_INCLUDE_DIRS}
)
add_library(ptl_reid_cpp src/reid_database.cpp src/reid_inference.cpp src/reid.cpp)
target_link_libraries(
    ptl_reid_cpp 
    ${OpenCV_LIBS} 
    ${catkin_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TensorRT_LIBRARIES}
    ${Faiss_LIBS} 
    OpenMP::OpenMP_CXX 
    ${LAPACK_LIBRARIES}
    ) 

add_executable(reid app/main.cpp) 
target_link_libraries(reid ptl_reid_cpp)