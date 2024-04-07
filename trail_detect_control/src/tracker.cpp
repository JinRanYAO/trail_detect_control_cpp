#include "tracker.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>

KalmanFilter::KalmanFilter(const double& std_position, const double& std_velocity, const double& std_mea_detect, const double& std_mea_LK, const double& std_mea_3D){
    ndim = 2;
    dt = 1.0;
    motion_mat = Eigen::Matrix4d::Identity();
    for (int i = 0; i < ndim; ++i)
    {
        motion_mat(i, ndim + 1) = dt;
    }
    update_mat = Eigen::Matrix<double, 2, 4>::Identity();
    std_weight_position = std_position;
    std_weight_velocity = std_velocity;
    std_weight_mea_detect = std_mea_detect;
    std_weight_mea_LK = std_mea_LK;
    std_weight_mea_3D = std_mea_3D;
}

std::tuple<Eigen::Vector4d, Eigen::Matrix4d> KalmanFilter::initiate(const Eigen::Vector2d& measurement, const double& w, const double& h){
    Eigen::Vector2d mean_pos = measurement;
    Eigen::Vector2d mean_vel = Eigen::Vector2d::Zero();
    Eigen::Vector4d mean = Eigen::Vector4d(2 * ndim);
    mean << mean_pos, mean_vel;
    Eigen::Vector4d std_vec;
    std_vec << 2 * std_weight_position * w, 2 * std_weight_position * h, 10 * std_weight_velocity * w, 10 * std_weight_velocity * h;
    Eigen::Matrix4d covariance = std_vec.array().square().matrix().asDiagonal();
    return std::make_tuple(mean, covariance);
}

std::tuple<Eigen::Vector4d, Eigen::Matrix4d> KalmanFilter::predict(const Eigen::Vector4d& mean, const Eigen::Matrix4d& covariance, const double& w, const double& h){
    Eigen::Vector2d std_pos(std_weight_position * w, std_weight_position * h);
    Eigen::Vector2d std_vel(std_weight_velocity * w, std_weight_velocity * h);
    Eigen::Vector4d std_vec;
    std_vec << std_pos, std_vel;
    Eigen::Matrix4d motion_cov = std_vec.array().square().matrix().asDiagonal();
    Eigen::Vector4d pre_mean = motion_mat * mean;
    Eigen::Matrix4d pre_covariance = motion_mat * covariance * motion_mat.transpose() + motion_cov;
    return std::make_tuple(pre_mean, pre_covariance);
}

std::tuple<Eigen::Vector2d, Eigen::Matrix2d> KalmanFilter::project(const Eigen::Vector4d& mean, const Eigen::Matrix4d& covariance, const double& w, const double& h, const std::string& mea_type) {
    double std_weight_mea;
    if (mea_type == "detect") {
        std_weight_mea = std_weight_mea_detect;
    } else if (mea_type == "LK") {
        std_weight_mea = std_weight_mea_LK;
    } else {
        std_weight_mea = std_weight_mea_3D;
    }
    Eigen::Vector2d std(std_weight_mea * w, std_weight_mea * h);
    Eigen::Matrix2d innovation_cov = std.array().square().matrix().asDiagonal();
    Eigen::Vector2d pro_mean = update_mat * mean;
    Eigen::Matrix2d pro_covariance = update_mat * covariance * update_mat.transpose() + innovation_cov;
    return std::make_tuple(pro_mean, pro_covariance);
}

std::tuple<Eigen::Vector4d, Eigen::Matrix4d> KalmanFilter::update(const Eigen::Vector4d& mean, const Eigen::Matrix4d& covariance, const Eigen::Vector2d& measurement, const double& w, const double& h, const std::string& mea_type) {
    Eigen::Vector2d pro_mean;
    Eigen::Matrix2d pro_cov;
    std::tie(pro_mean, pro_cov) = project(mean, covariance, w, h, mea_type);
    Eigen::LLT<Eigen::MatrixXd> lltOfProjectedCov(pro_cov);
    Eigen::Matrix<double, 4, 2> kalman_gain = covariance * update_mat.transpose() * lltOfProjectedCov.solve(Eigen::Matrix2d::Identity());
    Eigen::Vector2d innovation = measurement - pro_mean;
    Eigen::Vector4d new_mean = mean + kalman_gain * innovation;
    Eigen::Matrix4d new_covariance = covariance - kalman_gain * pro_cov * kalman_gain.transpose();
    // if (innovation.cwiseAbs().maxCoeff() <= 0.15 * w)
    // {
    //     return std::make_tuple(new_mean, new_covariance);
    // }
    // else
    // {
    //     return std::make_tuple(mean, covariance);
    // }
    return std::make_tuple(new_mean, new_covariance);
}