cmake_minimum_required(VERSION 3.0.2)
project(detect_and_track)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
add_compile_options(-std=c++17 -O3 -DPUBLISH_DETECTION_IMAGE -DPUBLISH_DETECTION_WITH_POSITION -DDEBUG_TRACKER -DPROFILE)

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
  geometry_msgs
  std_msgs
  message_generation
  tf2_ros
  tf2_geometry_msgs
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

add_message_files(
   FILES
   BoundingBox2D.msg
   BoundingBoxes2D.msg
   PositionBoundingBox2D.msg
   PositionBoundingBox2DArray.msg
   PositionID.msg
   PositionIDArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  DEPENDS  OpenCV
  INCLUDE_DIRS include
#  LIBRARIES detect_and_track
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(ObjectDetection src/ObjectDetection.cpp)
add_library(Tracker src/Tracker.cpp)
add_library(PoseEstimator src/PoseEstimator.cpp)
add_library(KalmanFilter src/KalmanFilter.cpp)
add_library(Hungarian src/Hungarian.cpp)
add_library(DetectionUtils src/DetectionUtils.cpp)
add_library(Utils src/utils.cpp)
add_library(ROSWrappers src/ROSWrappers.cpp)

#add_executable(depth_extractor src/depth_extractor.cpp)
add_executable(detect_node src/detect_node.cpp)
add_executable(detect_and_locate_node src/detect_and_locate_node.cpp)
add_executable(detect_and_track2D_node src/detect_and_track2D_node.cpp)
add_executable(detect_track2D_and_locate_node src/detect_track2D_and_locate_node.cpp)
add_executable(track2D_node src/track2D_node.cpp)


#target_link_libraries(depth_extractor
#    ${catkin_LIBRARIES}
#    ${OpenCV_LIBS}
#)

target_link_libraries(detect_node
    ROSWrappers
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    DetectionUtils    
    Tracker
    Hungarian
    PoseEstimator
    KalmanFilter
    ObjectDetection
    Utils
    nvinfer
    cudart
)

target_link_libraries(detect_and_locate_node
    ROSWrappers
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    DetectionUtils    
    Tracker
    Hungarian
    PoseEstimator
    KalmanFilter
    ObjectDetection
    Utils
    nvinfer
    cudart
)

target_link_libraries(track2D_node
    ROSWrappers
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    DetectionUtils    
    Tracker
    Hungarian
    PoseEstimator
    KalmanFilter
    ObjectDetection
    Utils
    nvinfer
    cudart
)

target_link_libraries(detect_and_track2D_node
    ROSWrappers
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    DetectionUtils    
    Tracker
    Hungarian
    PoseEstimator
    KalmanFilter
    ObjectDetection
    Utils
    nvinfer
    cudart
)

target_link_libraries(detect_track2D_and_locate_node
    ROSWrappers
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    DetectionUtils    
    Tracker
    Hungarian
    PoseEstimator
    KalmanFilter
    ObjectDetection
    Utils
    nvinfer
    cudart
)