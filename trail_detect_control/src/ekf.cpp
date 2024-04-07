#include "ekf.h"

void EKF::measureUpdate(Eigen::Vector3d &Z_mea, const Eigen::Matrix3d &R_mea, const double& time)
{
    double dt = time - time_now;
    time_now = time;
    statePrediction(dt);

    Eigen::MatrixXd H(3,5);
    Eigen::MatrixXd I(5,5);
    Eigen::MatrixXd S(3,3);
    Eigen::MatrixXd K(5,3);
    Eigen::Vector3d Y;

    H << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0;
    I = Eigen::MatrixXd::Identity(5,5);

    Y = Z_mea - H * X;
    S = H * P * H.transpose() + R_mea;
    K = P * H.transpose() * S.inverse();
    X = X + K * Y;
    P = (I - K * H) * P;

}

void EKF::velStateUpdate(const Eigen::Vector2d &Z_vel, const Eigen::Matrix2d &R_vel, const double& time)
{
    double dt = time - time_now;
    time_now = time;
    statePrediction(dt);

    Eigen::MatrixXd H(2,5);
    Eigen::MatrixXd I(5,5);
    Eigen::MatrixXd S(2,2);
    Eigen::MatrixXd K(5,2);
    Eigen::Vector2d Y;

    H << 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0;
    I = Eigen::MatrixXd::Identity(5,5);

    Y = Z_vel - H * X;
    S = H * P * H.transpose() + R_vel;
    K = P * H.transpose() * S.inverse();
    X = X + K * Y;
    P = (I - K * H) * P;
}

void EKF::readX(double& time)
{
    double dt = time - time_now;
    time_now = time;
    statePrediction(dt);
}

void EKF::statePrediction(double& dt)
{
    double x = X(0);
    double y = X(1);
    double yaw = X(2);
    double v = X(3);
    double w = X(4);

    Eigen::MatrixXd F(5,5);
    Eigen::MatrixXd Q(5,5);
    Eigen::MatrixXd G(5,2);
    Eigen::MatrixXd E(2,2);

    if (w < 1e-5)
    {
        X(0) = x + v * cos(yaw)*dt;
        X(1) = y + v * sin(yaw)*dt;
    }
    else
    {
        X(0) = x + v/w*(sin(yaw+w*dt)-sin(yaw));
        X(1) = y + v/w*(cos(yaw)-cos(yaw+w*dt));
    }

    X(2) = yaw + w*dt;
    X(3) = v;
    X(4) = w;

    if (w < 1e-5)
    {
        F << 1, 0, -v*sin(yaw)*dt,  cos(yaw)*dt, 0,
                0, 1, v*cos(yaw)*dt, cos(yaw)*dt, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;
    }
    else
    {
        F << 1, 0, (v*(cos(yaw + dt*w) - cos(yaw)))/w,  (sin(yaw + dt*w) - sin(yaw))/w, (dt*v*cos(yaw + dt*w))/w - (v*(sin(yaw + dt*w) - sin(yaw)))/pow(w,2),
                0, 1, (v*(sin(yaw + dt*w) - sin(yaw)))/w, -(cos(yaw + dt*w) - cos(yaw))/w, (v*(cos(yaw + dt*w) - cos(yaw)))/pow(w,2) + (dt*v*sin(yaw + dt*w))/w,
                0, 0,                                  1,                               0,                                                                   dt,
                0, 0,                                  0,                               1,                                                                    0,
                0, 0,                                  0,                               0,                                                                    1;
    }

    G << cos(yaw)*pow(dt,2)/2, 0,
            sin(yaw)*pow(dt,2)/2, 0,
            0, pow(dt,2)/2,
            dt, 0,
            0, dt;
    E << pow(5,2), 0,
            0, pow(0.1,2);
    
    Q = G * E * G.transpose();
    P = F * P * F.transpose() + Q;
}