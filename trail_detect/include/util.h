#pragma once
#include <iostream>
#include <tf/tf.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>

Eigen::Matrix4d geo_to_eigen(const tf::StampedTransform & pos){
    Eigen::Vector3d trans(pos.getOrigin().x(), pos.getOrigin().y(), pos.getOrigin().z());
    Eigen::Quaterniond quat(pos.getRotation().w(), pos.getRotation().x(), pos.getRotation().y(), pos.getRotation().z());
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R = quat.toRotationMatrix();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = trans;
    return T;
}

Eigen::Matrix4d data_to_transform(const Eigen::Matrix3d& r_matrix, const Eigen::Vector3d& t_position) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat.block<3, 3>(0, 0) = r_matrix;
    mat.block<3, 1>(0, 3) = t_position;
    return mat;
}

cv::Mat pixel2cam(const cv::Mat& keypoints) {
    cv::Mat n_keypoints = cv::Mat::zeros(keypoints.size(), keypoints.type());
    for (int i = 0; i < keypoints.rows; ++i) {
        n_keypoints.at<double>(i, 0) = (keypoints.at<double>(i, 0) - 944.62) / 561.70;
        n_keypoints.at<double>(i, 1) = (keypoints.at<double>(i, 1) - 549.53) / 562.83;
    }
    return n_keypoints;
}

// Eigen::Matrix<double, 3, Eigen::Dynamic> triangulatePoints(const Eigen::Matrix<double, 3, 4>& projMat1, const Eigen::Matrix<double, 3, 4>& projMat2, const cv::Mat& pts1_cv, const cv::Mat& pts2_cv) {

//     Eigen::Matrix<double, Eigen::Dynamic, 2> pts1, pts2;
//     pts1.resize(pts1_cv.rows, Eigen::NoChange);
//     pts2.resize(pts2_cv.rows, Eigen::NoChange);
//     std::cout << "cv1 trian: " << pts1_cv << std::endl;
//     std::cout << "cv2 trian: " << pts2_cv << std::endl;
//     cv::cv2eigen(pts1_cv, pts1);
//     cv::cv2eigen(pts2_cv, pts2);
//     std::cout << "eigen trian: " << pts1 << std::endl;
//     std::cout << "eigen trian: " << pts2 << std::endl;

//     int num_points = pts1.rows();
//     Eigen::Matrix<double, 3, Eigen::Dynamic> points_3d(3, num_points);

//     for (int i = 0; i < num_points; ++i) {
//         Eigen::Matrix4d A;
//         A.row(0) = pts1(i, 0) * projMat1.row(2) - projMat1.row(0);
//         A.row(1) = pts1(i, 1) * projMat1.row(2) - projMat1.row(1);
//         A.row(2) = pts2(i, 0) * projMat2.row(2) - projMat2.row(0);
//         A.row(3) = pts2(i, 1) * projMat2.row(2) - projMat2.row(1);
//         Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
//         Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
//         point_3d_homo /= point_3d_homo(3);
//         points_3d.col(i) = point_3d_homo.head(3);
//     }

//     return points_3d;
// }