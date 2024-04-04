cmake_minimum_required(VERSION 3.10)
project(yolov8_pose LANGUAGES C CXX CUDA)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  message_generation
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge image_transport message_runtime
 DEPENDS CUDA OpenCV
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

set(TensorRT_ROOT /home/tongyao/TensorRT-8.4.2.4)

find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVCAFFE_PARSER NAMES nvcaffe_parser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)

set(SAMPLES_COMMON_DIR ${TensorRT_ROOT}/samples/common)

include_directories(
  include
  src 
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS} 
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR} 
  ${SAMPLES_COMMON_DIR}
)

link_directories(
  ${TensorRT_ROOT}/lib
  ${TensorRT_ROOT}/lib64
)

file(GLOB CPPS 
  src/*.cpp
  src/*.cu
  include/*.cu
  include/*.cpp
  ${TensorRT_ROOT}/samples/common/logger.cpp 
  ${TensorRT_ROOT}/samples/common/sampleOptions.cpp
)

add_library(yolov8_pose ${CPPS})
target_link_libraries(yolov8_pose
  ${ALL_LIBS}
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TRT_NVINFER} 
  ${TRT_NVINFER_PLUGIN} 
  ${TRT_NVONNX_PARSER} 
  ${TRT_NVCAFFE_PARSER}
)

install(TARGETS yolov8_pose
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/yolov8_pose/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN ".*h"
)

set_property(TARGET yolov8_pose PROPERTY CUDA_ARCHITECTURES 50 61 72 75)
target_compile_options(yolov8_pose PUBLIC 
   $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo; --use_fast_math --disable-warnings>)