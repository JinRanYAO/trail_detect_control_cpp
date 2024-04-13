cmake_minimum_required(VERSION 3.0.2)
project(trail_detect)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
SET(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  nav_msgs
  geometry_msgs
  cyber_msgs
  cv_bridge
  tf
  tf_conversions
  tf2_ros
  image_transport
  yolov8_pose
)

find_package(OpenCV REQUIRED)

find_package(Eigen3)

find_package(Ceres REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES trail_detect
 CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs visualization_msgs nav_msgs geometry_msgs cyber_msgs cv_bridge tf tf_conversions tf2_ros image_transport yolov8_pose
)

set(TensorRT_ROOT /home/tongyao/TensorRT-8.4.2.4)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)

set(SAMPLES_COMMON_DIR ${TensorRT_ROOT}/samples/common)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR} 
  ${SAMPLES_COMMON_DIR}
  ${yolov8_pose_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib src/ekf.cpp src/tracker.cpp src/rs_planning.cpp src/planning.cpp src/optimizer.cpp)

add_executable(${PROJECT_NAME}_node src/main.cpp)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${yolov8_pose_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${CERES_LIBRARIES}
  yolov8_pose
)