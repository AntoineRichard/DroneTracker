cmake_minimum_required(VERSION 3.5)
project(detect_and_track)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

## Compile as C++17
add_compile_options(-std=c++17 -O3 -DPUBLISH_DETECTION_IMAGE -DPUBLISH_DETECTION_WITH_POSITION -DDEBUG_POSE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

message("-- CUDA version: ${CUDA_VERSION}")

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(/home/antoine/Downloads/TensorRT-8.4.1.5/include)
link_directories(/home/antoine/Downloads/TensorRT-8.4.1.5/lib)
include_directories($(EIGEN3_INCLUDE_DIR))

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

########################
## Message Generation ##
########################

set(msg_files
  "msg/BoundingBox2D.msg"
  "msg/BoundingBoxes2D.msg"
  "msg/PositionBoundingBox2D.msg"
  "msg/PositionBoundingBox2DArray.msg"
  "msg/PositionID.msg"
  "msg/PositionIDArray.msg"
)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES
  std_msgs
  geometry_msgs
  builtin_interfaces
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)


###########
## Build ##
###########

add_library(ObjectDetection src/ObjectDetection.cpp)
add_library(Tracker src/Tracker.cpp)
add_library(PoseEstimator src/PoseEstimator.cpp)
add_library(KalmanFilter src/KalmanFilter.cpp)
add_library(Hungarian src/Hungarian.cpp)
add_library(DetectionUtils src/DetectionUtils.cpp)
add_library(Utils src/utils.cpp)
add_library(ROS2Wrappers src/ROS2Wrappers.cpp)

add_executable(detect_node src/detect_node.cpp)
add_executable(detect_and_locate_node src/detect_and_locate_node.cpp)
add_executable(detect_and_track2D_node src/detect_and_track2D_node.cpp)
add_executable(detect_track2D_and_locate_node src/detect_track2D_and_locate_node.cpp)
add_executable(track2D_node src/track2D_node.cpp)

target_link_libraries(detect_node
    ROS2Wrappers
    ${cv_bridge_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
    ${rosidl_default_generators_LIBRARIES}
    ${image_transport_LIBRARIES}
    ${tf2_LIBRARIES}
    ${tf2_geometry_msgs_LIBRARIES}
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
    ROS2Wrappers
    ${cv_bridge_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
    ${rosidl_default_generators_LIBRARIES}
    ${image_transport_LIBRARIES}
    ${tf2_LIBRARIES}
    ${tf2_geometry_msgs_LIBRARIES}
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
    ROS2Wrappers
    ${cv_bridge_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
    ${rosidl_default_generators_LIBRARIES}
    ${image_transport_LIBRARIES}
    ${tf2_LIBRARIES}
    ${tf2_geometry_msgs_LIBRARIES}
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
    ROS2Wrappers
    ${cv_bridge_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
    ${rosidl_default_generators_LIBRARIES}
    ${image_transport_LIBRARIES}
    ${tf2_LIBRARIES}
    ${tf2_geometry_msgs_LIBRARIES}
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
    ROS2Wrappers
    ${cv_bridge_LIBRARIES}
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
    ${rosidl_default_generators_LIBRARIES}
    ${image_transport_LIBRARIES}
    ${tf2_LIBRARIES}
    ${tf2_geometry_msgs_LIBRARIES}
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

rosidl_target_interfaces(ROS2Wrappers
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_target_interfaces(detect_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_target_interfaces(detect_and_locate_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_target_interfaces(track2D_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_target_interfaces(detect_and_track2D_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_target_interfaces(detect_track2D_and_locate_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

#############
## Install ##
#############

install(TARGETS
  ROS2Wrappers
  DetectionUtils    
  Tracker
  Hungarian
  PoseEstimator
  KalmanFilter
  ObjectDetection
  DESTINATION lib
)

install(TARGETS
  detect_node
  detect_and_locate_node
  track2D_node
  detect_and_track2D_node
  detect_track2D_and_locate_node
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()