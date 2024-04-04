cmake_minimum_required(VERSION 2.8.3)
project(cyber_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  LocalizationEstimate.msg
  LocalTrajPoint.msg
  LocalTrajList.msg
  Object.msg
  ObjectArray.msg
  VehicleSpeedFeedback.msg
  SpeedFeedbackAGV.msg
  VehicleSteerFeedback.msg
  speedcmd.msg
  steercmd.msg
  body_status_feedback.msg
  MaterialBoxPose.msg
  GPGGA_MSG.msg
  Heading.msg
  JY901.msg
  ButtonState.msg
  CanFrame.msg
  arm_action.msg

)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

## The catkin_package macro generates cmake config files for your package
catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs roscpp sensor_msgs std_msgs
)
