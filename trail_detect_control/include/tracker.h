#ifndef TRACKER_H
#define TRACKER_H
#include <tuple>
#include <eigen3/Eigen/Dense>

class KalmanFilter{
public:
    KalmanFilter(){}
    KalmanFilter(const double& std_position, const double& std_velocity, const double& std_mea_detect, const double& std_mea_LK, const double& std_mea_3D);
    ~KalmanFilter(){}

    std::tuple<Eigen::Vector4d, Eigen::Matrix4d> initiate(const Eigen::Vector2d measurement, const double w, const double h);
    std::tuple<Eigen::Vector4d, Eigen::Matrix4d> predict(const Eigen::Vector4d mean, const Eigen::Matrix4d covariance, const double w, const double h);
    std::tuple<Eigen::Vector2d, Eigen::Matrix2d> project(const Eigen::Vector4d mean, const Eigen::Matrix4d covariance, const double w, const double h, const std::string mea_type);
    std::tuple<Eigen::Vector4d, Eigen::Matrix4d> update(const Eigen::Vector4d mean, const Eigen::Matrix4d covariance, const Eigen::Vector2d measurement, const double w, const double h, const std::string mea_type);

private:
    int ndim;
    double dt;
    Eigen::Matrix4d motion_mat;
    Eigen::Matrix<double, 2, 4> update_mat;
    double std_weight_position;
    double std_weight_velocity;
    double std_weight_mea_detect;
    double std_weight_mea_LK;
    double std_weight_mea_3D;
};

#endif