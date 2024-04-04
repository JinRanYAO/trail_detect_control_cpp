#ifndef EKF_H
#define EKF_H
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include "geometry_msgs/Pose2D.h"

class EKF {
public:
    
    EKF(){
        X = Eigen::VectorXd::Zero(5);
        X(2) = M_PI / 2;
        P = Eigen::MatrixXd::Zero(5,5);
        speed_err = 0.001;
        imu_err = 0.004;
        steer_angle = 0.0;
        time_now = ros::Time::now().toSec();
    }
    ~EKF(){}

    void measureUpdate(Eigen::Vector3d &Z_mea, const Eigen::Matrix3d &R_mea, const double time);

    void velStateUpdate(const Eigen::Vector2d &Z_vel, const Eigen::Matrix2d &R_vel, const double time);

    void readX(double time);

    Eigen::VectorXd X;
    Eigen::MatrixXd P;
    double steer_angle;
    double speed_err;
    double imu_err;

private:
    double time_now;
    void statePrediction(double dt);
};

#endif