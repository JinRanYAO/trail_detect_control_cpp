#include <iostream>
#include <cmath>
#include <vector>

struct State {
    double x, y, yaw;
};

inline double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle <= -M_PI) angle += 2 * M_PI;
    return angle;
}

std::vector<State> planPath(const State& start, const State& goal, const double r) {
    std::vector<State> path;

    // 计算两点之间的直线距离
    double dx = goal.x - start.x;
    double dy = goal.y - start.y;
    double direct_distance = std::hypot(dx, dy);
    
    // 计算从当前朝向到目标点朝向需转动的角度
    double start_to_goal_yaw = std::atan2(dy, dx);
    double turn_angle = normalizeAngle(start_to_goal_yaw - start.yaw);
    
    // 简化的假设：转向角度等分为10个点来模拟圆弧
    const int num_arc_points = 10;
    for (int i = 0; i < num_arc_points; ++i) {
        double fraction = static_cast<double>(i) / num_arc_points;
        State pose;
        pose.x = start.x;
        pose.y = start.y;
        pose.yaw = normalizeAngle(start.yaw + turn_angle * fraction);
        path.emplace_back(pose);
    }

    // 连接直线路径点
    const int num_line_points = 10;
    for (int i = 1; i <= num_line_points; ++i) {
        double fraction = static_cast<double>(i) / num_line_points;
        State pose;
        pose.x = start.x + fraction * direct_distance * std::cos(start_to_goal_yaw);
        pose.y = start.y + fraction * direct_distance * std::sin(start_to_goal_yaw);
        pose.yaw = start_to_goal_yaw; // 直行时保持转向后的朝向不变
        path.emplace_back(pose);
    }

    // 最后，将目标点作为路径的最后一个点
    path.emplace_back(goal);

    return path;
}