# Introduction
The aim of this project is to control a tractor to achieve automatic docking. In this repo, a .trt model (keypoint detection, based on yolov8-pose) is already obtained, which can be used to output the pixel coordinates of the hitch in the image. The code in this repo includes keypoint detection and tracking, the recovery of the spatial positions of keypoints, and the control of the tractor. The entire system runs at a frequency of 12.5Hz on the NVIDIA Jetson Nano.
# Environment
ROS, Eigen, Ceres, OpenCV, CUDA
# Files
- cyber_msgs: .msg files required by ROS
- yolov8_pose: To achieve real-time inference process on CUDA, based on https://github.com/FeiYull/TensorRT-Alpha, and optimize kernel_function.cu to accelerate.
- trail_detect:
  - config: parameters, including topic, camera params, vehicle params, and detection and tracking params
  - ekf.cpp: Utilize angular velocity, speed and steer feedback to compute the tractor position and pose based on EKF.
  - optimizer.cpp: To restore the spatial position of the hitch keypoints based on prior geometric constraints.
  - tracker.cpp: To add a tracking module based on the Kalman Filter to solve the inter-frame instability of keypoints detection.
  - main.cpp: Framework, communication among each module, and handle the whole procession in this system.
- trail_control:
  - config: parameters, including topic, control params
  - control.cpp: Compute the control signal using kinematic model and visual servo respectively, and output the hybrid control signal.
# Video
<p align="center">
    <img src="example1-vehicle.gif" width="500" alt="Example1-vehicle">
    <img src="example1-control.gif" width="500" alt="Example1-control">
</p>

<p align="center">
    <img src="example2-vehicle.gif" width="500" alt="Example2-vehicle">
    <img src="example2-control.gif" width="500" alt="Example2-control">
</p>

Note: The human driver only acts as a safety officer and does not participate in vehicle control.
