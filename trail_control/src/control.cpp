#include <cmath>
#include <ros/ros.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

double steer_degree;
double wheel_base_2;
int control_mode;

double high_speed, mid_speed, low_speed;
double kp_dist_steer, kp_angle_steer;
double dist_far, dist_near, dist_thres;
double pixel_thres, kp_pixel_steer;
double weight_dist;
double degree_to_steer_ratio;
double wheel_base;
double lq_cost_thres;

double d_speed, d_brake;
double d_steer_pos, d_steer_pixel, d_steer;

std::vector<double> pose_error;
std::vector<double> pixel_error;

ros::Publisher speed_pub, steer_pub;

std_msgs::Int32MultiArray speed_cmd_msg;
std_msgs::Float64 steer_cmd_msg;

void posCallback(const std_msgs::Float64MultiArrayConstPtr &msg);
void pixelCallback(const std_msgs::Float64MultiArrayConstPtr &msg);

int main(int argc, char** argv){

    ros::init(argc, argv, "trail_control_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string r_hitch_topic, f_hitch_topic, speed_feedback_topic;
    std::string speed_topic, steer_topic;

    pnh.param<std::string>("topic/rear_hitch_pos_topic", r_hitch_topic, "/hitch_pos");
    pnh.param<std::string>("topic/front_hitch_pixel_topic", f_hitch_topic, "/hitch_pixel");
    pnh.param<std::string>("topics/speed_feedback_topic", speed_feedback_topic, "/e100/speed_feedback");

    pnh.param<std::string>("topic/speed_topic", speed_topic, "/speed_cmd");
    pnh.param<std::string>("topic/steer_topic", steer_topic, "/steer_cmd");

    pnh.param<int>("control/mode", control_mode, 1);
    pnh.param<double>("control/high_speed", high_speed, 20);
    pnh.param<double>("control/mid_speed", mid_speed, 10);
    pnh.param<double>("control/low_speed", low_speed, 5);
    pnh.param<double>("control/kp_dist_steer", kp_dist_steer, 2.0);
    pnh.param<double>("control/kp_angle_steer", kp_angle_steer, 1.0);
    pnh.param<double>("control/dist_far", dist_far, 5.0);
    pnh.param<double>("control/dist_near", dist_near, 2.5);
    pnh.param<double>("control/kp_pixel_steer", kp_pixel_steer, 0.1);
    pnh.param<double>("control/pixel_thres", pixel_thres, 5.0);
    pnh.param<double>("control/dist_thres", dist_thres, 0.03);
    pnh.param<double>("control/weight_dist", weight_dist, 0.5);
    pnh.param<double>("control/degree_to_steer_ratio", degree_to_steer_ratio, 160.0);
    pnh.param<double>("control/wheel_base", wheel_base, 1.02);
    pnh.param<double>("control/lq_cost_thres", lq_cost_thres, 20.0);

    ros::Subscriber r_hitch_sub = nh.subscribe(r_hitch_topic, 1, posCallback);
    ros::Subscriber f_hitch_sub = nh.subscribe(f_hitch_topic, 1, pixelCallback);

    speed_pub = nh.advertise<std_msgs::Int32MultiArray>(speed_topic, 1);
    steer_pub = nh.advertise<std_msgs::Float64>(steer_topic, 1);

    steer_degree = 180.0 / M_PI * degree_to_steer_ratio;
    wheel_base_2 = 2.0 * wheel_base;

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    return 0;
}

void posCallback(const std_msgs::Float64MultiArrayConstPtr &msg){

    double x_error = msg->data[0];
    double y_error = msg->data[1];
    double yaw_now = msg->data[2];
//    double ori_error = msg->data[3];
    double lq_cost = msg->data[4];

    double dist_error = std::hypot(x_error, y_error);
    double angle_error = std::atan2(y_error, x_error) - yaw_now;

    d_steer_pos = std::atan2(wheel_base_2 * std::sin(angle_error) / dist_error, 1.0) * steer_degree;

    // d_steer_pos = (kp_dist_steer * pose_error[1] + kp_angle_steer * pose_error[2]) * 180.0 / M_PI * degree_to_steer_ratio;

    if ((dist_error > dist_far) && (lq_cost <= lq_cost_thres))
    {
        d_speed = - high_speed;
        d_steer = d_steer_pos;
        d_brake = 0.0;
    }
    else if ((dist_error > dist_near) && (lq_cost <= lq_cost_thres))
    {
        d_speed = - mid_speed;
        d_steer = d_steer_pos * weight_dist + d_steer_pixel * (1 - weight_dist);
        d_brake = 5.0;
    }
    else
    {
        d_speed = - low_speed;
        d_steer = d_steer_pixel;
        d_brake = 8.0;
    }

    if ((std::abs(pixel_error[0]) <= pixel_thres && std::abs(pixel_error[1]) <= pixel_thres) || (dist_error <= dist_thres))
    {
        d_speed = 0.0;
        d_brake = 10.0;
    }
    else
    {
        // d_brake = (d_speed - curr_speed) / 10.0 * (-5.739404974) - 5.8701266;
        // d_brake = 0.0;
        if (d_steer > 6000.0)
        {
            d_steer = 6000.0;
        } else if (d_steer < -6000.0)
        {
            d_steer = -6000.0;
        }        
    }
    
    speed_cmd_msg.data = {static_cast<int>(d_speed), static_cast<int>(d_brake)};
//    speed_cmd_msg.data.emplace_back((int)d_speed);
//    speed_cmd_msg.data.emplace_back((int)d_brake);
    steer_cmd_msg.data = d_steer;

    speed_pub.publish(speed_cmd_msg);
    steer_pub.publish(steer_cmd_msg);
    
}

void pixelCallback(const std_msgs::Float64MultiArrayConstPtr &msg){
    pixel_error = msg->data;
    d_steer_pixel = pixel_error[0] * kp_pixel_steer * degree_to_steer_ratio;
}