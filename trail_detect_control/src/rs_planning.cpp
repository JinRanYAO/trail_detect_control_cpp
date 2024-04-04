#include <cmath>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>

struct Path {
    std::vector<double> lengths;
    std::vector<char> ctypes;
    double L = 0.0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> yaw;
    std::vector<int> directions;
};

double mod2pi(double x) {
    double v = fmod(x, 2.0 * M_PI);
    if (v < -M_PI) {
        v += 2.0 * M_PI;
    } else if (v > M_PI) {
        v -= 2.0 * M_PI;
    }
    return v;
}

double pi_2_pi(double x) {

    x = fmod(x + M_PI, 2 * M_PI) - M_PI;
    return x;
}

std::tuple<double, double> polar(double x, double y) {
    double r = std::hypot(x, y);
    double theta = std::atan2(y, x);
    return std::make_tuple(r, theta);
}

bool is_same_path(const Path& path1, const Path& path2, double step_size) {
    return (path1.ctypes == path2.ctypes) && (std::abs(std::accumulate(path2.lengths.begin(), path2.lengths.end(), 0.0, [](double acc, double len) { return acc + std::abs(len); }) - path1.L) <= step_size);
}

std::tuple<bool, double, double, double> straight_left_straight(double x, double y, double phi) {
    phi = mod2pi(phi);
    if (0.01 * M_PI < phi && phi < 0.99 * M_PI && y != 0) {
        double xd = -y / std::tan(phi) + x;
        double t = xd - std::tan(phi / 2.0);
        double u = phi;
        double v = std::copysign(1.0, y) * std::hypot(x - xd, y) - std::tan(phi / 2.0);
        return std::make_tuple(true, t, u, v);
    }

    return std::make_tuple(false, 0.0, 0.0, 0.0);
}

std::tuple<bool, double, double, double> left_straight_left(double x, double y, double phi) {
    double u, t;
    std::tie(u, t) = polar(x - std::sin(phi), y - 1.0 + std::cos(phi));
    if (t >= 0.0) {
        double v = mod2pi(phi - t);
        if (v >= 0.0) {
            return std::make_tuple(true, t, u, v);
        }
    }
    return std::make_tuple(false, 0.0, 0.0, 0.0);
}

std::tuple<bool, double, double, double> left_right_left(double x, double y, double phi) {
    double u1, t1;
    std::tie(u1, t1) = polar(x - std::sin(phi), y - 1.0 + std::cos(phi));
    if (u1 <= 4.0) {
        double u = -2.0 * std::asin(0.25 * u1);
        double t = mod2pi(t1 + 0.5 * u + M_PI);
        double v = mod2pi(phi - t + u);

        if (t >= 0.0 && u <= 0.0) {
            return std::make_tuple(true, t, u, v);
        }
    }
    return std::make_tuple(false, 0.0, 0.0, 0.0);
}

std::tuple<bool, double, double, double> left_straight_right(double x, double y, double phi) {
    double u1, t1;
    std::tie(u1, t1) = polar(x + std::sin(phi), y - 1.0 - std::cos(phi));
    u1 = u1 * u1;
    if (u1 >= 4.0) {
        double u = std::sqrt(u1 - 4.0);
        double theta = std::atan2(2.0, u);
        double t = mod2pi(t1 + theta);
        double v = mod2pi(t - phi);

        if (t >= 0.0 && v >= 0.0) {
            return std::make_tuple(true, t, u, v);
        }
    }
    return std::make_tuple(false, 0.0, 0.0, 0.0);
}

std::vector<Path> set_path(std::vector<Path>& paths, const std::vector<double>& lengths, const std::vector<char>& ctypes, double step_size) {
    Path path;
    path.ctypes = ctypes;
    path.lengths = lengths;
    path.L = std::accumulate(path.lengths.begin(), path.lengths.end(), 0.0, [](double acc, double len) { return acc + std::abs(len); });

    if (std::find_if(paths.begin(), paths.end(), [&](const Path& p){ return is_same_path(p, path, step_size); }) != paths.end()) {
        return paths;
    }

    if (path.L <= step_size) {
        return paths;
    }

    paths.push_back(path);
    return paths;
}

std::vector<Path> straight_curve_straight(double x, double y, double phi, std::vector<Path>& paths, double step_size) {
    bool flag;
    double t, u, v;
    
    std::tie(flag, t, u, v) = straight_left_straight(x, y, phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'S', 'L', 'S'}, step_size);
    }

    std::tie(flag, t, u, v) = straight_left_straight(x, -y, -phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'S', 'R', 'S'}, step_size);
    }

    return paths;
}

std::vector<Path> curve_straight_curve(double x, double y, double phi, std::vector<Path>& paths, double step_size) {
    bool flag;
    double t, u, v;
    
    std::tie(flag, t, u, v) = left_straight_left(x, y, phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'L', 'S', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_left(-x, y, -phi);
    if (flag) {
        paths = set_path(paths, {-t, -u, -v}, {'L', 'S', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_left(x, -y, -phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'R', 'S', 'R'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_left(-x, -y, phi);
    if (flag) {
        paths = set_path(paths, {-t, -u, -v}, {'R', 'S', 'R'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_right(x, y, phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'L', 'S', 'R'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_right(-x, y, -phi);
    if (flag) {
        paths = set_path(paths, {-t, -u, -v}, {'L', 'S', 'R'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_right(x, -y, -phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'R', 'S', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_straight_right(-x, -y, phi);
    if (flag) {
        paths = set_path(paths, {-t, -u, -v}, {'R', 'S', 'L'}, step_size);
    }

    return paths;
}

std::vector<Path> curve_curve_curve(double x, double y, double phi, std::vector<Path>& paths, double step_size) {
    bool flag;
    double t, u, v;
    
    std::tie(flag, t, u, v) = left_right_left(x, y, phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'L', 'R', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_right_left(-x, y, -phi);
    if (flag) {
        paths = set_path(paths, {-t, -u, -v}, {'L', 'R', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_right_left(x, -y, -phi);
    if (flag) {
        paths = set_path(paths, {t, u, v}, {'R', 'L', 'R'}, step_size);
    }

    std::tie(flag, t, u, v) = left_right_left(-x, -y, phi);
    if (flag) {
        paths = set_path(paths, {-t, -u, -v}, {'R', 'L', 'R'}, step_size);
    }

    double xb = x * std::cos(phi) + y * std::sin(phi);
    double yb = x * std::sin(phi) - y * std::cos(phi);

    std::tie(flag, t, u, v) = left_right_left(xb, yb, phi);
    if (flag) {
        paths = set_path(paths, {v, u, t}, {'L', 'R', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_right_left(-xb, yb, -phi);
    if (flag) {
        paths = set_path(paths, {-v, -u, -t}, {'L', 'R', 'L'}, step_size);
    }

    std::tie(flag, t, u, v) = left_right_left(xb, -yb, -phi);
    if (flag) {
        paths = set_path(paths, {v, u, t}, {'R', 'L', 'R'}, step_size);
    }

    std::tie(flag, t, u, v) = left_right_left(-xb, -yb, phi);
    if (flag) {
        paths = set_path(paths, {-v, -u, -t}, {'R', 'L', 'R'}, step_size);
    }

    return paths;
}

std::vector<Path> generate_path(const std::vector<double>& q0, const std::vector<double>& q1, double max_curvature, double step_size) {
    double dx = q1[0] - q0[0];
    double dy = q1[1] - q0[1];
    double dth = q1[2] - q0[2];
    double c = std::cos(q0[2]);
    double s = std::sin(q0[2]);
    double x = (c * dx + s * dy) * max_curvature;
    double y = (-s * dx + c * dy) * max_curvature;

    std::vector<Path> paths;
    paths = straight_curve_straight(x, y, dth, paths, step_size);
    paths = curve_straight_curve(x, y, dth, paths, step_size);
    paths = curve_curve_curve(x, y, dth, paths, step_size);

    return paths;
}

std::vector<std::vector<double>> calc_interpolate_dists_list(const std::vector<double>& lengths, double step_size) {

    std::vector<std::vector<double>> interpolate_dists_list;

    for (const auto& length : lengths) {
        double d_dist = (length >= 0.0) ? step_size : -step_size;

        std::vector<double> interp_dists;
        for (double dist = 0.0; dist < std::abs(length); dist += std::abs(d_dist)) {
            interp_dists.push_back(dist);
        }

        interp_dists.push_back(length);
        interpolate_dists_list.push_back(interp_dists);
    }

    return interpolate_dists_list;
}

std::tuple<double, double, double, int> interpolate(double dist, double length, char mode, double max_curvature, double origin_x, double origin_y, double origin_yaw) {

    double x, y, yaw;

    if (mode == 'S') {
        x = origin_x + dist / max_curvature * std::cos(origin_yaw);
        y = origin_y + dist / max_curvature * std::sin(origin_yaw);
        yaw = origin_yaw;
    } else {
        double ldx = std::sin(dist) / max_curvature;
        double ldy = 0.0;

        if (mode == 'L') {
            ldy = (1.0 - std::cos(dist)) / max_curvature;
            yaw = origin_yaw + dist;
        } else if (mode == 'R') {
            ldy = (1.0 - std::cos(dist)) / -max_curvature;
            yaw = origin_yaw - dist;
        }
        
        double gdx = std::cos(-origin_yaw) * ldx + std::sin(-origin_yaw) * ldy;
        double gdy = -std::sin(-origin_yaw) * ldx + std::cos(-origin_yaw) * ldy;
        x = origin_x + gdx;
        y = origin_y + gdy;
    }

    int direction = (length > 0.0) ? 1 : -1;

    return {x, y, yaw, direction};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<int>> generate_local_course(const std::vector<double>& lengths, const std::vector<char>& modes, double max_curvature, double step_size) {
    auto interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size);

    double origin_x = 0.0, origin_y = 0.0, origin_yaw = 0.0;

    std::vector<double> xs, ys, yaws;
    std::vector<int> directions;

    for (int i = 0; i < lengths.size(); ++i) {
        for (double dist : interpolate_dists_list[i]) {
            double x, y, yaw;
            int direction;
            std::tie(x, y, yaw, direction) = interpolate(dist, lengths[i], modes[i], max_curvature, origin_x, origin_y, origin_yaw);
            xs.push_back(x);
            ys.push_back(y);
            yaws.push_back(yaw);
            directions.push_back(direction);
        }

        origin_x = xs.back();
        origin_y = ys.back();
        origin_yaw = yaws.back();
    }

    return {xs, ys, yaws, directions};
}

std::vector<Path> calc_paths(double sx, double sy, double syaw, double gx, double gy, double gyaw, double maxc, double step_size) {
    std::vector<double> q0 = {sx, sy, syaw};
    std::vector<double> q1 = {gx, gy, gyaw};

    std::vector<Path> paths = generate_path(q0, q1, maxc, step_size);
    for (Path& path : paths) {
        std::vector<double> xs, ys, yaws;
        std::vector<int> directions;
        std::tie(xs, ys, yaws, directions) = generate_local_course(path.lengths, path.ctypes, maxc, step_size * maxc);

        for (int i = 0; i < xs.size(); ++i) {
            double x = xs[i], y = ys[i], yaw = yaws[i];
            path.x.push_back(std::cos(-q0[2]) * x + std::sin(-q0[2]) * y + q0[0]);
            path.y.push_back(-std::sin(-q0[2]) * x + std::cos(-q0[2]) * y + q0[1]);
            path.yaw.push_back(pi_2_pi(yaw + q0[2]));
        }

        path.directions = directions;
        std::transform(path.lengths.begin(), path.lengths.end(), path.lengths.begin(), [maxc](double length) { return length / maxc; });
        path.L /= maxc;
    }

    return paths;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> reeds_shepp_path_planning(double sx, double sy, double syaw, double gx, double gy, double gyaw, double maxc, double step_size = 0.2) {
    auto paths = calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size);

    if (paths.empty()) {
        return {};
    }

    auto best_path_it = std::min_element(paths.begin(), paths.end(), [](const Path& a, const Path& b) { return std::abs(a.L) < std::abs(b.L); });
    Path b_path = *best_path_it;

    return {b_path.x, b_path.y, b_path.yaw};
}