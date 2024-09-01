#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>
#include <vector>
#include <tuple>
#include <cmath>

Eigen::Matrix3d guess3rdPoint(const Eigen::Matrix<double, 3, 2> & points){
    double mid_x = (points(0, 0) + points(0, 1)) / 2.0;
    double mid_y = (points(1, 0) + points(1, 1)) / 2.0;
    double k = - (points(0, 0) - points(0, 1)) / (points(1, 0) - points(1, 1));
    double delta_x = 0.5 / std::sqrt(1 + k*k);

    Eigen::Vector3d point_3rd;
    if (k >= 0)
    {
        point_3rd << (mid_x + delta_x), (mid_y + k * delta_x), 0.25;
    }
    else
    {
        point_3rd << (mid_x - delta_x), (mid_y - k * delta_x), 0.25;
    }
    
    Eigen::Matrix3d three_points;
    three_points << points, point_3rd;
    
    return three_points;
}

template <typename P3>
Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> fromHomog(const Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> & X)
{
    Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> nonHomog(2, X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        nonHomog(0, i) = X(0, i) / X(2, i);
        nonHomog(1, i) = X(1, i) / X(2, i);
    }
    
    return nonHomog;
}

template <typename P3>
Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> reprojection(const Eigen::Matrix<P3, 4, 4> &Trans, const Eigen::Matrix<P3, 4, 3> &P, const Eigen::Matrix<P3, 3, 3> &K){

    Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> points_proj = K * Trans.topRows(3) * P;
    Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> nhom_points_proj = fromHomog<P3>(points_proj);

    return nhom_points_proj.transpose();
}

struct Residuals3
{
    Residuals3(const Eigen::Matrix<double, 3, 2> &p1, const Eigen::Matrix<double, 3, 2> &p2, const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2,
              const Eigen::Vector3d s, const Eigen::Matrix3d &K) : p1_(p1), p2_(p2), T1_(T1), T2_(T2), s_(s), K_(K) {}

    template <typename P3>
    bool operator()(const P3* const params, P3* residuals) const {
        Eigen::Matrix<P3, 4, 3> P;
        P << Eigen::Matrix<P3, 4, 1>(params[0], params[1], P3(s_(0)), P3(1.0)),
             Eigen::Matrix<P3, 4, 1>(params[2], params[3], P3(s_(1)), P3(1.0)),
             Eigen::Matrix<P3, 4, 1>(params[4], params[5], P3(s_(2)), P3(1.0));
        
        Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> pr1 = reprojection<P3>(T1_.cast<P3>(), P, K_.cast<P3>());
        Eigen::Matrix<P3, Eigen::Dynamic, Eigen::Dynamic> pr2 = reprojection<P3>(T2_.cast<P3>(), P, K_.cast<P3>());

        residuals[0] = P3(p1_(0, 0)) - pr1(0, 0);
        residuals[1] = P3(p1_(0, 1)) - pr1(0, 1);
        residuals[2] = P3(p2_(0, 0)) - pr2(0, 0);
        residuals[3] = P3(p2_(0, 1)) - pr2(0, 1);
        residuals[4] = P3(p1_(1, 0)) - pr1(1, 0);
        residuals[5] = P3(p1_(1, 1)) - pr1(1, 1);
        residuals[6] = P3(p2_(1, 0)) - pr2(1, 0);
        residuals[7] = P3(p2_(1, 1)) - pr2(1, 1);
        residuals[8] = P3(p1_(2, 0)) - pr1(2, 0);
        residuals[9] = P3(p1_(2, 1)) - pr1(2, 1);
        residuals[10] = P3(p2_(2, 0)) - pr2(2, 0);
        residuals[11] = P3(p2_(2, 1)) - pr2(2, 1);
        // residuals[12] = P3(0.9) - sqrt(ceres::pow((params[0] - params[2]), 2) + ceres::pow((params[1] - params[3]), 2));
        // residuals[13] = P3(0.675) - sqrt(ceres::pow((params[0] - params[4]), 2) + ceres::pow((params[1] - params[5]), 2) + 0.0025);
        // residuals[14] = P3(0.675) - sqrt(ceres::pow((params[2] - params[4]), 2) + ceres::pow((params[3] - params[5]), 2) + 0.0025);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix<double, 3, 2> &p1, const Eigen::Matrix<double, 3, 2> &p2, const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2,
                                        const Eigen::Vector3d s, const Eigen::Matrix3d &K)
    {
        return (new ceres::AutoDiffCostFunction<Residuals3, 12, 6>(
            new Residuals3(p1, p2, T1, T2, s, K)));
    }

private:
    const Eigen::Matrix<double, 3, 2> p1_;
    const Eigen::Matrix<double, 3, 2> p2_;
    const Eigen::Matrix4d T1_;
    const Eigen::Matrix4d T2_;
    const Eigen::Vector3d s_;
    const Eigen::Matrix3d K_;
};

template <typename P2>
Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> fromHomog2(const Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> & X)
{
    Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> nonHomog(2, X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        nonHomog(0, i) = X(0, i) / X(2, i);
        nonHomog(1, i) = X(1, i) / X(2, i);
    }
    
    return nonHomog;
}

template <typename P2>
Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> reprojection2(const Eigen::Matrix<P2, 4, 4> &Trans, const Eigen::Matrix<P2, 4, 2> &P, const Eigen::Matrix<P2, 3, 3> &K){

    Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> points_proj = K * Trans.topRows(3) * P;
    Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> nhom_points_proj = fromHomog2<P2>(points_proj);

    return nhom_points_proj.transpose();
}

struct Residuals2
{
    Residuals2(const Eigen::Matrix<double, 2, 2> &p1, const Eigen::Matrix<double, 2, 2> &p2, const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2,
              const Eigen::Vector2d s, const Eigen::Matrix3d &K) : p1_(p1), p2_(p2), T1_(T1), T2_(T2), s_(s), K_(K) {}

    template <typename P2>
    bool operator()(const P2* const params, P2* residuals) const {
        Eigen::Matrix<P2, 4, 2> P;
        P << Eigen::Matrix<P2, 4, 1>(params[0], params[1], P2(s_(0)), P2(1.0)),
             Eigen::Matrix<P2, 4, 1>(params[2], params[3], P2(s_(1)), P2(1.0));
        
        Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> pr1 = reprojection2<P2>(T1_.cast<P2>(), P, K_.cast<P2>());
        Eigen::Matrix<P2, Eigen::Dynamic, Eigen::Dynamic> pr2 = reprojection2<P2>(T2_.cast<P2>(), P, K_.cast<P2>());

        residuals[0] = P2(p1_(0, 0)) - pr1(0, 0);
        residuals[1] = P2(p1_(0, 1)) - pr1(0, 1);
        residuals[2] = P2(p2_(0, 0)) - pr2(0, 0);
        residuals[3] = P2(p2_(0, 1)) - pr2(0, 1);
        residuals[4] = P2(p1_(1, 0)) - pr1(1, 0);
        residuals[5] = P2(p1_(1, 1)) - pr1(1, 1);
        residuals[6] = P2(p2_(1, 0)) - pr2(1, 0);
        residuals[7] = P2(p2_(1, 1)) - pr2(1, 1);
        // residuals[8] = P2(0.9) - sqrt(ceres::pow((params[0] - params[2]), 2) + ceres::pow((params[1] - params[3]), 2));

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix<double, 2, 2> &p1, const Eigen::Matrix<double, 2, 2> &p2, const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2,
                                        const Eigen::Vector2d s, const Eigen::Matrix3d &K)
    {
        return (new ceres::AutoDiffCostFunction<Residuals2, 8, 4>(
            new Residuals2(p1, p2, T1, T2, s, K)));
    }

private:
    const Eigen::Matrix<double, 2, 2> p1_;
    const Eigen::Matrix<double, 2, 2> p2_;
    const Eigen::Matrix4d T1_;
    const Eigen::Matrix4d T2_;
    const Eigen::Vector2d s_;
    const Eigen::Matrix3d K_;
};

std::tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>, double> compute_keypoint3d(const Eigen::Matrix<double, Eigen::Dynamic, 2> &p1, const Eigen::Matrix<double, Eigen::Dynamic, 2> &p2,
                                                            const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2, const Eigen::Matrix3d &K, double* guess)
{
    Eigen::Matrix<double, Eigen::Dynamic, 3> P_optimized;
    int n = std::min(p1.rows(), p2.rows());
    double cost;

    if (n == 3)
    {
        Eigen::Vector3d s(3);
        s << 0.3, 0.3, 0.25;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.max_num_iterations = 200;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        // options.minimizer_progress_to_stdout = true;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
        ceres::Solver::Summary summary;

        ceres::Problem problem;
        ceres::CostFunction* cost_function = Residuals3::Create(p1, p2, T1, T2, s, K);
        problem.AddParameterBlock(guess, 6);
        problem.SetParameterLowerBound(guess, 0, -10);
        problem.SetParameterUpperBound(guess, 0, 10);
        problem.SetParameterLowerBound(guess, 1, -20);
        problem.SetParameterUpperBound(guess, 1, -2);
        problem.SetParameterLowerBound(guess, 2, -10);
        problem.SetParameterUpperBound(guess, 2, 10);
        problem.SetParameterLowerBound(guess, 3, -20);
        problem.SetParameterUpperBound(guess, 3, -2);
        problem.SetParameterLowerBound(guess, 4, -10);
        problem.SetParameterUpperBound(guess, 4, 10);
        problem.SetParameterLowerBound(guess, 5, -20);
        problem.SetParameterUpperBound(guess, 5, -2);
        problem.AddResidualBlock(cost_function, nullptr, guess);

        ceres::Solve(options, &problem, &summary);

        P_optimized.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            P_optimized(i, 0) = guess[2*i];
            P_optimized(i, 1) = guess[2*i+1];
            P_optimized(i, 2) = s(i);
        }
        cost = summary.final_cost;
    }
    else
    {
        Eigen::Vector2d s(2);
        s << 0.3, 0.3;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.max_num_iterations = 200;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        // options.minimizer_progress_to_stdout = true;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
        ceres::Solver::Summary summary;

        ceres::Problem problem;
        ceres::CostFunction* cost_function = Residuals2::Create(p1.topRows(n), p2.topRows(n), T1, T2, s, K);
        problem.AddParameterBlock(guess, 4);
        problem.SetParameterLowerBound(guess, 0, -10);
        problem.SetParameterUpperBound(guess, 0, 10);
        problem.SetParameterLowerBound(guess, 1, -20);
        problem.SetParameterUpperBound(guess, 1, -2);
        problem.SetParameterLowerBound(guess, 2, -10);
        problem.SetParameterUpperBound(guess, 2, 10);
        problem.SetParameterLowerBound(guess, 3, -20);
        problem.SetParameterUpperBound(guess, 3, -2);
        problem.AddResidualBlock(cost_function, nullptr, guess);

        ceres::Solve(options, &problem, &summary);

        P_optimized.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            P_optimized(i, 0) = guess[2*i];
            P_optimized(i, 1) = guess[2*i+1];
            P_optimized(i, 2) = s(i);
        }

        P_optimized = guess3rdPoint(P_optimized.transpose()).transpose();
        cost = summary.final_cost;
    }
    
    return std::make_tuple(P_optimized, cost);
}