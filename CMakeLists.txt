cmake_minimum_required(VERSION 3.0.2)
project(depth_image_extractor)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
add_compile_options(-std=c++17 -O3 -DPROFILE -DBROWNCONRADY -DDEBUG_KALMAN)

## Fastest compile options
#add_compile_options(-std=c++17 -O3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(/home/antoine/Downloads/TensorRT-8.4.1.5/include)
link_directories(/home/antoine/Downloads/TensorRT-8.4.1.5/lib)
include_directories($(EIGEN3_INCLUDE_DIR))


catkin_package(
  DEPENDS  OpenCV
  INCLUDE_DIRS include
#  LIBRARIES depth_image_extractor
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(ObjectDetection src/ObjectDetection.cpp)
add_library(PoseEstimator src/PoseEstimator.cpp)
add_library(KalmanFilter src/KalmanFilter.cpp)
add_library(Hungarian src/Hungarian.cpp)

add_executable(depth_extractor src/depth_extractor.cpp)
add_executable(tensorRT_detection src/tensorRT_detection.cpp)
add_executable(tensorRT_detection_v2 src/tensorRT_detection_v2.cpp)
add_executable(tensorRT_detection_v3 src/tensorRT_detection_v3.cpp)


target_link_libraries(depth_extractor
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)
target_link_libraries(tensorRT_detection
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    cudart
)
target_link_libraries(tensorRT_detection_v2
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    cudart
)

target_link_libraries(tensorRT_detection_v3
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    Hungarian
    ObjectDetection
    PoseEstimator
    KalmanFilter
    nvinfer
    cudart
)