cmake_minimum_required(VERSION 3.1)
project(yolo_trt)

file(GLOB_RECURSE sources modules/*.hpp modules/*.cpp modules/*.h modules/*.cu)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#cuda
find_package(CUDA REQUIRED)

message("-- CUDA version:${CUDA_VERSION}")

if (CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected")
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
    )
endif ()

find_package(OpenCV REQUIRED)

#generate detector lib
cuda_add_library(${PROJECT_NAME} SHARED ${sources})
target_include_directories(${PROJECT_NAME} PRIVATE modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin nvcaffe_parser "stdc++fs" ${OpenCV_LIBS})