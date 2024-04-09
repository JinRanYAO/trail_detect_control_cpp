#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <cyber_msgs/VehicleSpeedFeedback.h>
#include <cyber_msgs/VehicleSteerFeedback.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "app_yolov8_pose.h"
#include "ekf.h"
#include "tracker.h"
#include "util.h"
#include "optimizer.cpp"
#include "planning.cpp"
// #include "rs_planning.cpp"

utils::InitParameter param;
KeypointDetector* detector_ptr = nullptr;

EKF* ekf_pose_ptr = nullptr;

std::vector<KalmanFilter> trackers(3);
std::vector<Eigen::Vector4d> means(3);
std::vector<Eigen::Vector4d> prev_means(3);
std::vector<Eigen::Matrix4d> covariances(3);
std::vector<bool> trackers_init = {false, false, false};
cv::Mat prev_image_gray;
bool prev_flag = false;
cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);
cv::Size winSize(15, 15);
int maxLevel = 2;

Eigen::Matrix3d intrinsic_matrix;
Eigen::Vector4d distortion_coeffs;
Eigen::Vector3d rot_vec, trans_vec;
cv::Mat intrinsic_matrix_cv(3, 3, CV_64F), distortion_coeffs_cv(4, 1, CV_64F), rot_vec_cv(3, 1, CV_64F), trans_vec_cv(3, 1, CV_64F);
double r_wheel, hitch_height, front_hitch_length, trailer_length, trailer_width, rear_hitch_length, rear_height, rear_hitch_height;

tf::Transform cam_transform;
cv::Mat rvec_m;
Eigen::Matrix3d rvec_m_eigen;
tf::Matrix3x3 tf3d;
Eigen::Vector3d cam_t;
tf::Quaternion tfqt;

double std_position, std_velocity, std_mea_detect, std_mea_LK, std_mea_3D;

double curr_yaw = 0.0;
double velocity = 0.0;

cv::Mat img_sub;
cv::Mat img_sub_resize;
int size_w, size_h;
double scale_w, scale_h;

bool two_image = false;
bool first_leastsq = false;
bool point3d_init = false;
double lq_cost_min = 100000.0;
cv::Mat last_keypoints;
cv::Mat now_keypoints;
tf::StampedTransform last_pos;
tf::StampedTransform now_pos;
Eigen::Matrix4d T_cw_now;
double keypoints_3D_guess[6] = {0, -5, 0, -5, 0, -5};
Eigen::Matrix3d keypoints_3D_sq;
Eigen::Matrix3d keypoints_3D_save;
Eigen::Matrix3d keypoints_3D;

ros::Publisher vis_pub, points_pub, path_pub;

tf::TransformBroadcaster* pose_broadcaster_ptr = nullptr;
geometry_msgs::TransformStamped pose_stamped;
geometry_msgs::TransformStamped camera_stamped;
tf::TransformListener* tf_listener_ptr = nullptr;

std::vector<State> paths;

void imageCallback(const sensor_msgs::CompressedImageConstPtr& img_msg);
void imuCallback(const geometry_msgs::Vector3StampedConstPtr& imu_msg);
void speedCallback(const cyber_msgs::VehicleSpeedFeedbackConstPtr& speed_msg);
void steerCallback(const cyber_msgs::VehicleSteerFeedbackConstPtr& steer_msg);

void filterCallback(const ros::TimerEvent&);

cv::Mat track(const cv::Mat& image, const cv::Mat& box, const std::vector<cv::Mat>& points);
void triangulate(const cv::Mat& last_keypoints, const cv::Mat& now_keypoints, const tf::StampedTransform& last_pos, const tf::StampedTransform& now_pos);

void pose_publish();
cv::Mat path_publish(const cv::Mat& image);
void points_publish();

int main(int argc, char** argv){
    ros::init(argc, argv, "trail_detect_control_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string image_topic, imu_topic, speed_feedback_topic, steer_feedback_topic;

    std::string vis_topic, points_topic, path_topic;

    std::string trt_file;
    double src_w, src_h;
    float conf_thresh, iou_thresh;

    std::vector<double> intrinsic_params, distortion_params, rot_params, trans_params;

    pnh.param<std::string>("topics/image_topic", image_topic, "/axis/image_raw/compressed");
    pnh.param<std::string>("topics/imu_topic", imu_topic, "/imu/angular_velocity");
    pnh.param<std::string>("topics/speed_feedback_topic", speed_feedback_topic, "/e100/speed_feedback");
    pnh.param<std::string>("topics/steer_feedback_topic", steer_feedback_topic, "/e100/steer_feedback");

    pnh.param<std::string>("topic/vis_topic", vis_topic, "/vis_result");
    pnh.param<std::string>("topic/points_topic", points_topic, "/points3d");
    pnh.param<std::string>("topic/path_topic", path_topic, "/rs_path");

    pnh.param<std::string>("yolov8_pose/trt_file", trt_file, "/home/tongyao/tensorrt-alpha/data/yolov8-pose/best-384.trt");
    pnh.param<int>("yolov8_pose/size_w", size_w, 640);
    pnh.param<int>("yolov8_pose/size_h", size_h, 384);
    pnh.param<double>("yolov8_pose/src_w", src_w, 1920);
    pnh.param<double>("yolov8_pose/src_h", src_h, 1080);
    pnh.param<float>("yolov8_pose/conf_thresh", conf_thresh, 0.5);
    pnh.param<float>("yolov8_pose/iou_thresh", iou_thresh, 0.7);

    pnh.getParam("fisheye_params/intrinsic_matrix", intrinsic_params);
    pnh.getParam("fisheye_params/distortion_coeffs", distortion_params);
    pnh.getParam("fisheye_params/rot_vec", rot_params);
    pnh.getParam("fisheye_params/trans_vec", trans_params);

    pnh.getParam("vehicle_params/r_wheel", r_wheel);
    pnh.getParam("vehicle_params/hitch_height", hitch_height);
    pnh.getParam("vehicle_params/front_hitch_length", front_hitch_length);
    pnh.getParam("vehicle_params/trailer_length", trailer_length);
    pnh.getParam("vehicle_params/trailer_width", trailer_width);
    pnh.getParam("vehicle_params/rear_hitch_length", rear_hitch_length);
    pnh.getParam("vehicle_params/rear_height", rear_height);
    pnh.getParam("vehicle_params/rear_hitch_height", rear_hitch_height);

    pnh.getParam("kf_tracker/std_weight_position", std_position);
    pnh.getParam("kf_tracker/std_weight_velocity", std_velocity);
    pnh.getParam("kf_tracker/std_weight_mea_detect", std_mea_detect);
    pnh.getParam("kf_tracker/std_weight_mea_LK", std_mea_LK);
    pnh.getParam("kf_tracker/std_weight_mea_3D", std_mea_3D);

    intrinsic_matrix << intrinsic_params[0], intrinsic_params[1], intrinsic_params[2],
                        intrinsic_params[3], intrinsic_params[4], intrinsic_params[5],
                        intrinsic_params[6], intrinsic_params[7], intrinsic_params[8];
    distortion_coeffs << distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3];
    rot_vec << rot_params[0], rot_params[1], rot_params[2];
    trans_vec << trans_params[0], trans_params[1], trans_params[2];

    for (int i = 0; i < intrinsic_matrix.rows(); ++i) {
        for (int j = 0; j < intrinsic_matrix.cols(); ++j) {
            intrinsic_matrix_cv.at<double>(i, j) = intrinsic_matrix(i, j);
        }
    }
    for (int i = 0; i < distortion_coeffs.rows(); ++i) {
        for (int j = 0; j < distortion_coeffs.cols(); ++j) {
            distortion_coeffs_cv.at<double>(i, j) = distortion_coeffs(i, j);
        }
    }
    for (int i = 0; i < rot_vec.rows(); ++i) {
        for (int j = 0; j < rot_vec.cols(); ++j) {
            rot_vec_cv.at<double>(i, j) = rot_vec(i, j);
        }
    }
    for (int i = 0; i < trans_vec.rows(); ++i) {
        for (int j = 0; j < trans_vec.cols(); ++j) {
            trans_vec_cv.at<double>(i, j) = trans_vec(i, j);
        }
    }

    static tf2_ros::StaticTransformBroadcaster world_broadcaster;
    geometry_msgs::TransformStamped world_transform;
    world_transform.header.stamp = ros::Time::now();
    world_transform.header.frame_id = "world";
    world_transform.child_frame_id = "base_link";
    world_transform.transform.translation.x = 0.0;
    world_transform.transform.translation.y = 0.0;
    world_transform.transform.translation.z = 0.0;
    world_transform.transform.rotation.x = 0.0;
    world_transform.transform.rotation.y = 0.0;
    world_transform.transform.rotation.z = 0.0;
    world_transform.transform.rotation.w = 1.0;
    world_broadcaster.sendTransform(world_transform);

    cv::Rodrigues(rot_vec_cv, rvec_m);
    cv::cv2eigen(rvec_m, rvec_m_eigen);
    tf::matrixEigenToTF(rvec_m_eigen.transpose(), tf3d);
    cam_t = -rvec_m_eigen.transpose() * trans_vec;
    cam_transform.setOrigin(tf::Vector3(cam_t(0), cam_t(1), cam_t(2)));
    tf3d.getRotation(tfqt);
    cam_transform.setRotation(tfqt);

    vis_pub = nh.advertise<sensor_msgs::Image>(vis_topic, 10);
    points_pub = nh.advertise<visualization_msgs::MarkerArray>(points_topic, 10);
    path_pub = nh.advertise<nav_msgs::Path>(path_topic, 1);

    tf::TransformBroadcaster pose_broadcaster;
    pose_broadcaster_ptr = &pose_broadcaster;
    tf::TransformListener tf_listener;
    tf_listener_ptr = &tf_listener;

    EKF ekf_pose;
    ekf_pose_ptr = &ekf_pose;

    setYolov8Params(param, size_w, size_h, conf_thresh, iou_thresh);
    scale_w = src_w / size_w;
    scale_h = src_h / size_h;
    KeypointDetector detector(param, trt_file);
    detector_ptr = &detector;
    cv::Mat test = cv::Mat::ones(size_w, size_h, CV_8UC3);
    cv::Mat result_img;
    cv::Mat box;
    std::vector<cv::Mat> points;
    std::tie(result_img, box, points) = detector_ptr->inference(test, param);

    ros::Subscriber image_sub = nh.subscribe(image_topic, 10, imageCallback);
    ros::Subscriber imu_sub = nh.subscribe(imu_topic, 80, imuCallback);
    ros::Subscriber speed_sub = nh.subscribe(speed_feedback_topic, 20, speedCallback);
    ros::Subscriber steer_sub = nh.subscribe(steer_feedback_topic, 40, steerCallback);

    ros::Timer filter = nh.createTimer(ros::Duration(0.01), filterCallback);

    // ros::spin();
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}

void imageCallback(const sensor_msgs::CompressedImageConstPtr& img_msg){
    img_sub = cv::imdecode(cv::Mat(img_msg->data), 1);
    cv::resize(img_sub, img_sub_resize, cv::Size(size_w, size_h));
    cv::Mat result_img;
    cv::Mat box;
    std::vector<cv::Mat> points;
    std::tie(result_img, box, points) = detector_ptr->inference(img_sub_resize, param);
    result_img = track(img_sub, box, points);

    tf_listener_ptr->lookupTransform("world", "camera_link", ros::Time(0), now_pos);
    std::vector<cv::Point2d> keypoints_cv;
    for (int i = 0; i < 3; i++)
    {
        if (trackers_init[i] == true)
        {
            keypoints_cv.emplace_back(cv::Point2d(means[i](0), means[i](1)));
        }
        
    }
    // for (const auto &point: points)
    // {
    //     std::cout << "point: " << point << std::endl;
    //     keypoints_cv.emplace_back(cv::Point2d(point.at<double>(0, 1), point.at<double>(0, 2)));
    // }
    cv::Mat keypoints(keypoints_cv);
    std::vector<cv::Point2d> undistort_points_cv;
    cv::fisheye::undistortPoints(keypoints, undistort_points_cv, intrinsic_matrix_cv, distortion_coeffs_cv, cv::noArray(), intrinsic_matrix_cv);

    cv::Mat undistort_points(static_cast<int>(undistort_points_cv.size()), 2, CV_64F);

    for (int i = 0; i < undistort_points.rows; ++i)
    {
        undistort_points.at<double>(i, 0) = undistort_points_cv[i].x;
        undistort_points.at<double>(i, 1) = undistort_points_cv[i].y;
    }

    if (two_image == false)
    {
        last_keypoints = undistort_points;
        last_pos = now_pos;
        two_image = true;
        return;
    }

    now_keypoints = undistort_points;

    triangulate(last_keypoints, now_keypoints, last_pos, now_pos);

    last_keypoints = now_keypoints;
    last_pos = now_pos;

    if (point3d_init == true)
    {
        double sx = ekf_pose_ptr->X(0) - front_hitch_length * std::cos(ekf_pose_ptr->X(2));
        double sy = ekf_pose_ptr->X(1) - front_hitch_length * std::sin(ekf_pose_ptr->X(2));
        double syaw = ekf_pose_ptr->X(2);
        double gx = keypoints_3D(2, 0);
        double gy = keypoints_3D(2, 1);
        double k_r_hitch = - (keypoints_3D(0, 0) - keypoints_3D(1, 0)) / (keypoints_3D(0, 1) - keypoints_3D(1, 1));
        double gyaw = (k_r_hitch > 0) ? std::atan(k_r_hitch) : (std::atan(k_r_hitch) + M_PI);
        State start {sx, sy, syaw};
        State goal {gx, gy, gyaw};
        // std::tie(pathxs, pathys, pathyaws) = reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, 1.0);
        paths = planPath(start, goal, 1.0);
    }

    cv::Mat pub_img = path_publish(result_img);
    sensor_msgs::ImagePtr pub_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", pub_img).toImageMsg();
    vis_pub.publish(pub_img_msg);
}

void imuCallback(const geometry_msgs::Vector3StampedConstPtr& imu_msg){
    curr_yaw = imu_msg->vector.z;
    Eigen::Vector2d Z_vel;
    Z_vel << velocity, curr_yaw;
    Eigen::Matrix2d R_vel;
    R_vel << std::pow(ekf_pose_ptr->speed_err, 2), 0.0, 0.0, std::pow(ekf_pose_ptr->imu_err, 2);
    double t_vel = ros::Time::now().toSec();
    ekf_pose_ptr->velStateUpdate(Z_vel, R_vel, t_vel);
    pose_publish();
}

void speedCallback(const cyber_msgs::VehicleSpeedFeedbackConstPtr& speed_msg){
    velocity = - speed_msg->speed_cmps / 100.0;
    Eigen::Vector2d Z_vel;
    Z_vel << velocity, curr_yaw;
    Eigen::Matrix2d R_vel;
    R_vel << std::pow(ekf_pose_ptr->speed_err, 2), 0.0, 0.0, std::pow(ekf_pose_ptr->imu_err, 2);
    double t_vel = ros::Time::now().toSec();
    ekf_pose_ptr->velStateUpdate(Z_vel, R_vel, t_vel);
    pose_publish();
}

void steerCallback(const cyber_msgs::VehicleSteerFeedbackConstPtr& steer_msg){
    // ekf_pose_ptr->steer_angle = - (steer_msg->steer_0p1d - 40) / 10 * 0.0625 * M_PI / 180.0;
    ekf_pose_ptr->steer_angle = - (steer_msg->steer_0p1d - 40) * 0.00003472 * M_PI;
}

void filterCallback(const ros::TimerEvent&){
    double t_output = ros::Time::now().toSec();
    ekf_pose_ptr->readX(t_output);
    pose_publish();
}

<<<<<<< HEAD
cv::Mat track(const cv::Mat image, const cv::Mat box, const std::vector<cv::Mat> points){
    cv::Mat image_copy = image.clone(); 
    double w = scale_w * (box.at<double>(0, 2) - box.at<double>(0, 0));
    double h = scale_h * (box.at<double>(0, 3) - box.at<double>(0, 1));
=======
cv::Mat track(const cv::Mat& image, const cv::Mat& box, const std::vector<cv::Mat>& points){
    // cv::Mat image_copy = image.clone(); 
    double w = box.at<double>(0, 2) - box.at<double>(0, 0);
    double h = box.at<double>(0, 3) - box.at<double>(0, 1);
>>>>>>> 6949a747dbd96167df806a607c6fcd9e717816e5
    for (const auto& point : points)
    {
        int kpi = (int)point.at<double>(0, 0);
        double x = point.at<double>(0, 1) * scale_w;
        double y = point.at<double>(0, 2) * scale_h;
        double conf = point.at<double>(0, 3);

        Eigen::Vector2d mea_pos(x, y);

        if (trackers_init[kpi] == false){
            KalmanFilter tracker(std_position, std_velocity, std_mea_detect, std_mea_LK, std_mea_3D);
            Eigen::Vector4d mean;
            Eigen::Matrix4d covariance;
            std::tie(mean, covariance) = tracker.initiate(mea_pos, w, h);
            trackers[kpi] = tracker;
            means[kpi] = mean;
            covariances[kpi] = covariance;
            trackers_init[kpi] = true;
        } else
        {
            cv::Point point(cv::saturate_cast<int>(x), cv::saturate_cast<int>(y));
            std::tie(means[kpi], covariances[kpi]) = trackers[kpi].predict(means[kpi], covariances[kpi], w, h);
            std::tie(means[kpi], covariances[kpi]) = trackers[kpi].update(means[kpi], covariances[kpi], mea_pos, w, h, "detect");
            cv::circle(image_copy, point, 5, cv::Scalar(255, 255, 255), -1);
        }
    }

    cv::Mat curr_image_gray;
    // cv::cvtColor(image_copy, curr_image_gray, cv::COLOR_BGR2GRAY);

    // if (prev_flag == true)
    // {
    //     std::vector<cv::Point2f> cv_points(prev_means.size());
    //     std::transform(prev_means.begin(), prev_means.end(), cv_points.begin(),[](const Eigen::Vector4d& ev) { return cv::Point2f((float)ev(0), (float)ev(1)); });
    //     std::vector<cv::Point2f> curr_keypoints_LK;
    //     std::vector<uchar> status;
    //     std::vector<float> err;
    //     cv::calcOpticalFlowPyrLK(prev_image_gray, curr_image_gray, cv_points, curr_keypoints_LK, status, err, winSize, maxLevel, criteria);

    //     std::vector<cv::Point2f> good_points;
    //     for (size_t i = 0; i < status.size(); i++) {
    //         if (status[i]) {
    //             good_points.emplace_back(curr_keypoints_LK[i]);
    //         }
    //     }
    //     for (size_t i = 0; i < std::min((int)good_points.size(), 3); i++)
    //     {
    //         cv::circle(image_copy, good_points[i], 5, cv::Scalar(0, 0, 0), -1);
    //         Eigen::Vector2d mea_LK((double)good_points[i].x, (double)good_points[i].y);
    //         std::tie(means[i], covariances[i]) = trackers[i].update(means[i], covariances[i], mea_LK, w, h, "LK");
    //     }
    // }

    // if (point3d_init == true)
    // {
    //     Eigen::MatrixXd keypoints_3D_homogeneous(4, keypoints_3D.rows());
    //     keypoints_3D_homogeneous << keypoints_3D.transpose(), Eigen::RowVectorXd::Ones(keypoints_3D.rows());
    //     Eigen::MatrixXd keypoints_3D_transformed = T_cw_now * keypoints_3D_homogeneous;
    //     std::vector<cv::Point3d> keypoints_3D_cam;
    //     keypoints_3D_cam.reserve(keypoints_3D_transformed.cols());
    //     for (int i = 0; i < keypoints_3D_transformed.cols(); ++i)
    //     {
    //         keypoints_3D_cam.emplace_back(cv::Point3d(keypoints_3D_transformed(0, i), keypoints_3D_transformed(1, i), keypoints_3D_transformed(2, i)));
    //     }
    //     std::vector<cv::Point2d> keypoints_2D_cam;
    //     cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    //     cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    //     cv::fisheye::projectPoints(keypoints_3D_cam, keypoints_2D_cam, rvec, tvec, intrinsic_matrix_cv, distortion_coeffs_cv);

    //     for(int i = 0; i < std::min((int)keypoints_2D_cam.size(), 3); ++i) {
    //         if (trackers_init[i] == true) {
    //             cv::circle(image_copy, keypoints_2D_cam[i], 5, cv::Scalar(0, 255, 0), -1);
    //             Eigen::Vector2d mea_3D(keypoints_2D_cam[i].x, keypoints_2D_cam[i].y);
    //             std::tie(means[i], covariances[i]) = trackers[i].update(means[i], covariances[i], mea_3D, w, h, "3D");
    //         }
    //     }
    // }

    // for (auto & mean : means)
    // {
    //     cv::Point point(cv::saturate_cast<int>(mean(0)), cv::saturate_cast<int>(mean(1)));
    //     cv::circle(image_copy, point, 5, cv::Scalar(0, 255, 0), -1);
    // }
    
    prev_means = means;
    prev_image_gray = curr_image_gray;
    prev_flag = true;
    
    // return image;
    return image_copy;
}

void triangulate(const cv::Mat& last_keypoints, const cv::Mat& now_keypoints, const tf::StampedTransform& last_pos, const tf::StampedTransform& now_pos){
    Eigen::Matrix4d T_cw_last = geo_to_eigen(last_pos).inverse();
    T_cw_now = geo_to_eigen(now_pos).inverse();

    Eigen::Matrix<double, 3, 4> projMatLast = T_cw_last.block<3, 4>(0, 0);
    Eigen::Matrix<double, 3, 4> projMatNow = T_cw_now.block<3, 4>(0, 0);
    // std::cout << "last_keypoints: " << last_keypoints << std::endl;
    // std::cout << "now_keypoints: " << now_keypoints << std::endl;
    // cv::Mat n_last_keypoints = pixel2cam(last_keypoints);
    // cv::Mat n_now_keypoints = pixel2cam(now_keypoints);

    if (first_leastsq == true)
    {
        Eigen::Matrix<double, 2, 3> guess_eigen = keypoints_3D.block<3, 2>(0, 0).transpose();
        std::copy(guess_eigen.data(), guess_eigen.data()+6, keypoints_3D_guess);
    }   

    double lq_cost;
    Eigen::Matrix<double, Eigen::Dynamic, 2> last_keypoints_eigen;
    last_keypoints_eigen.resize(last_keypoints.rows, 2);
    cv::cv2eigen(last_keypoints, last_keypoints_eigen);
    Eigen::Matrix<double, Eigen::Dynamic, 2> now_keypoints_eigen;
    now_keypoints_eigen.resize(now_keypoints.rows, 2);
    cv::cv2eigen(now_keypoints, now_keypoints_eigen);

    double t1 = ros::Time::now().toSec();
    if (last_keypoints.rows==3 && now_keypoints.rows==3)
    {
        std::tie(keypoints_3D_sq, lq_cost) = compute_keypoint3d(last_keypoints_eigen, now_keypoints_eigen, T_cw_last, T_cw_now, intrinsic_matrix, keypoints_3D_guess);
    } else
    {
        std::tie(keypoints_3D_sq, lq_cost) = compute_keypoint3d(last_keypoints_eigen.block<2, 2>(0, 0), now_keypoints_eigen.block<2, 2>(0, 0), T_cw_last, T_cw_now, intrinsic_matrix, keypoints_3D_guess);
    }
    std::cout << "optimize time: " << ros::Time::now().toSec()-t1 << std::endl;
    std::cout << "keypoints_3d: " << keypoints_3D_sq << std::endl;
    
    first_leastsq = true;
    if (lq_cost < lq_cost_min)
    {
        keypoints_3D = keypoints_3D_sq;
    }
    else
    {
        keypoints_3D = keypoints_3D_save;
    }
    keypoints_3D_save = keypoints_3D;

    point3d_init = true;
    points_publish();
}

void pose_publish(){
    
    tf::Transform base_transform;
    base_transform.setOrigin(tf::Vector3(ekf_pose_ptr->X(0), ekf_pose_ptr->X(1), r_wheel));
    tf::Quaternion base_q;
    base_q.setRPY(0, 0, ekf_pose_ptr->X(2));
    base_transform.setRotation(base_q);
    pose_broadcaster_ptr->sendTransform(tf::StampedTransform(base_transform, ros::Time::now(), "world", "base_link"));
    pose_broadcaster_ptr->sendTransform(tf::StampedTransform(cam_transform, ros::Time::now(), "base_link", "camera_link"));
}

cv::Mat path_publish(const cv::Mat& image){
    cv::Mat image_copy = image.clone();
    nav_msgs::Path path;
    path.header.frame_id = "world";

    for (size_t i = 0; i < paths.size(); i++)
    {
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "world";

        pose.pose.position.x = paths[i].x;
        pose.pose.position.y = paths[i].y;
        pose.pose.position.z = 0.25;

        geometry_msgs::Quaternion q = tf::createQuaternionMsgFromRollPitchYaw(0, 0, paths[i].yaw);
        pose.pose.orientation.w = q.w;
        pose.pose.orientation.x = q.x;
        pose.pose.orientation.y = q.y;
        pose.pose.orientation.z = q.z;

        path.poses.emplace_back(pose);

        geometry_msgs::PointStamped world_point, base_link_point;
        world_point.header.frame_id = "world";
        world_point.point.x = paths[i].x;
        world_point.point.y = paths[i].y;
        world_point.point.z = 0.25; 

        tf_listener_ptr->transformPoint("base_link", world_point, base_link_point);
        std::vector<cv::Point3d> points_to_project({ cv::Point3d(base_link_point.point.x, base_link_point.point.y, base_link_point.point.z) });
        std::vector<cv::Point2d> projected_points;
        cv::fisheye::projectPoints(points_to_project, projected_points, rot_vec_cv, trans_vec_cv, intrinsic_matrix_cv, distortion_coeffs_cv);
        cv::circle(image_copy, projected_points[0], 5, cv::Scalar(0, 0, 255), -1);
    }
    
    path_pub.publish(path);

    // double position_x = 0.0;
    // double position_y = 0.0;
    // double yaw = 0.0;

    // for (int i = 0; i < 10; ++i) {
    //     position_x += velocity * std::cos(yaw) * 0.1;
    //     position_y += velocity * std::sin(yaw) * 0.1;
    //     yaw += curr_yaw * 0.1;

    //     double position_hole_x = position_x - 0.69 * std::cos(yaw);
    //     double position_hole_y = position_y - 0.69 * std::sin(yaw);
    //     double position_hole_z = 0.032;

    //     std::vector<cv::Point3d> objectPoints({ cv::Point3d(position_hole_x, position_hole_y, position_hole_z) });
    //     std::vector<cv::Point2d> imagePoints;
    //     cv::fisheye::projectPoints(objectPoints, imagePoints, rot_vec_cv, trans_vec_cv, intrinsic_matrix_cv, distortion_coeffs_cv);
    //     cv::circle(image_copy, imagePoints[0], 5, cv::Scalar(0, 255, 0), -1);
    // }
    return image_copy;
}

void points_publish(){
    visualization_msgs::MarkerArray points_vis;
    points_vis.markers.clear();

    visualization_msgs::Marker right;
    right.header.frame_id = "world";
    right.type = visualization_msgs::Marker::SPHERE;
    right.action = visualization_msgs::Marker::ADD;
    right.id = 0;
    right.pose.position.x = keypoints_3D(0, 0);
    right.pose.position.y = keypoints_3D(0, 1);
    right.pose.position.z = keypoints_3D(0, 2);
    right.pose.orientation.w = 1.0;
    right.scale.x = 0.05;
    right.scale.y = 0.05;
    right.scale.z = 0.05;
    right.color.a = 1.0;
    right.color.r = 1.0;
    right.color.g = 0.0;
    right.color.b = 0.0;

    visualization_msgs::Marker left;
    left.header.frame_id = "world";
    left.type = visualization_msgs::Marker::SPHERE;
    left.action = visualization_msgs::Marker::ADD;
    left.id = 1;
    left.pose.position.x = keypoints_3D(1, 0);
    left.pose.position.y = keypoints_3D(1, 1);
    left.pose.position.z = keypoints_3D(1, 2);
    left.pose.orientation.w = 1.0;
    left.scale.x = 0.05;
    left.scale.y = 0.05;
    left.scale.z = 0.05;
    left.color.a = 1.0;
    left.color.r = 0.0;
    left.color.g = 1.0;
    left.color.b = 0.0;

    visualization_msgs::Marker hitch;
    hitch.header.frame_id = "world";
    hitch.type = visualization_msgs::Marker::SPHERE;
    hitch.action = visualization_msgs::Marker::ADD;
    hitch.id = 2;
    hitch.pose.position.x = keypoints_3D(2, 0);
    hitch.pose.position.y = keypoints_3D(2, 1);
    hitch.pose.position.z = keypoints_3D(2, 2);
    hitch.pose.orientation.w = 1.0;
    hitch.scale.x = 0.05;
    hitch.scale.y = 0.05;
    hitch.scale.z = 0.05;
    hitch.color.a = 1.0;
    hitch.color.r = 0.0;
    hitch.color.g = 0.0;
    hitch.color.b = 1.0;

    points_vis.markers.emplace_back(right);
    points_vis.markers.emplace_back(left);
    points_vis.markers.emplace_back(hitch);

    points_pub.publish(points_vis);
}
