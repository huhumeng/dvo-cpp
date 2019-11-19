cmake_minimum_required(VERSION 3.2)
project(dvo-cpp)

set(CMAKE_C_COMPILER "ccache")
SET(CMAKE_CXX_COMPILER_ARG1 "clang")

set(CMAKE_CXX_COMPILER "ccache")
SET(CMAKE_CXX_COMPILER_ARG1 "clang++")

set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(OpenCV 3.2 REQUIRED)

find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})


include_directories(
    include
    /usr/local/include/eigen3
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_core
    src/core/interpolation.cc
    src/core/intrinsic_matrix.cc
    src/core/math_sse.cc
    src/core/least_squares.cc
    src/core/rgbd_camera.cc
    src/core/rgbd_image.cc
    src/core/point_selection.cc
    src/core/weight_calculation.cc
    src/core/param_config.cc
    src/core/dense_tracking.cc
    src/core/optimizer.cc

    src/util/histogram.cc

    src/visualization/async_point_cloud_builder.cc
)
target_link_libraries(${PROJECT_NAME}_core
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

add_executable(ex_sse example/ex_sse.cc)
target_link_libraries(ex_sse ${PROJECT_NAME}_core)

add_executable(ex_rgbd_image example/ex_rgbd_image.cc)
target_link_libraries(ex_rgbd_image ${PROJECT_NAME}_core)
