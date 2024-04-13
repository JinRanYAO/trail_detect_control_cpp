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

template <typename T>
Eigen::Matrix<T, 2, 1> fromHomog(const Eigen::Matrix<T, 3, 1> & X)
{
    Eigen::Matrix<T, 2, 1> nonHomog(X(0)/X(2), X(1)/X(2));
    
    return nonHomog;
}

template <typename T>
Eigen::Matrix<T, 2, 1> reprojection(const Eigen::Matrix<T, 4, 4> &Trans, const Eigen::Matrix<T, 4, 1> &P, const Eigen::Matrix<T, 3, 3> &K){

    Eigen::Matrix<T, 3, 1> points_proj = K * Trans.topRows(3) * P;
    Eigen::Matrix<T, 2, 1> nhom_points_proj = fromHomog<T>(points_proj);

    return nhom_points_proj;
}

struct Residual
{
    Residual(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2,
              const double s, const Eigen::Matrix3d &K) : p1_(p1), p2_(p2), T1_(T1), T2_(T2), s_(s), K_(K) {}

    template <typename T>
    bool operator()(const T* const ps, T* residuals) const {
        Eigen::Matrix<T, 4, 1> P(ps[0], ps[1], T(s_), T(1.0));
        
        Eigen::Matrix<T, 2, 1> pr1 = reprojection<T>(T1_.cast<T>(), P, K_.cast<T>());
        Eigen::Matrix<T, 2, 1> pr2 = reprojection<T>(T2_.cast<T>(), P, K_.cast<T>());

        residuals[0] = T(p1_(0)) - pr1(0);
        residuals[1] = T(p1_(1)) - pr1(1);
        residuals[2] = T(p2_(0)) - pr2(0);
        residuals[3] = T(p2_(1)) - pr2(1);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2,
                                        const double s, const Eigen::Matrix3d &K)
    {
        return (new ceres::AutoDiffCostFunction<Residual, 4, 2>(
            new Residual(p1, p2, T1, T2, s, K)));
    }

private:
    const Eigen::Vector2d p1_;
    const Eigen::Vector2d p2_;
    const Eigen::Matrix4d T1_;
    const Eigen::Matrix4d T2_;
    const double s_;
    const Eigen::Matrix3d K_;
};

std::tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>, double> compute_keypoint3d(const Eigen::Matrix<double, Eigen::Dynamic, 2> &p1, const Eigen::Matrix<double, Eigen::Dynamic, 2> &p2,
                                                            const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2, const Eigen::Matrix3d &K, const Eigen::Matrix<double, 3, 2> &x0_guess)
{
    Eigen::Matrix<double, Eigen::Dynamic, 3> P_optimized;
    int n = std::min(p1.rows(), p2.rows());
    double cost;
    double params[6] = {0, -5, 0, -5, 0, -5};

    if (n == 3)
    {
        Eigen::Vector3d s(0.3, 0.3, 0.25);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        // options.minimizer_progress_to_stdout = true;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
        ceres::Solver::Summary summary;

        ceres::Problem problem;
        for (int i = 0; i < n; i++)
        {
            Eigen::Vector2d p1_v(p1(i, 0), p1(i, 1));
            Eigen::Vector2d p2_v(p2(i, 0), p2(i, 1));
            ceres::CostFunction* cost_function = Residual::Create(p1_v, p2_v, T1, T2, s(i), K);
            double* param_ptr = params + 2 * i;
            problem.AddParameterBlock(param_ptr, 2);
            problem.SetParameterLowerBound(param_ptr, 0, -8);
            problem.SetParameterUpperBound(param_ptr, 0, 8);
            problem.SetParameterLowerBound(param_ptr, 1, -15);
            problem.SetParameterUpperBound(param_ptr, 1, 0);
            problem.AddResidualBlock(cost_function, nullptr, param_ptr);
        }

        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << std::endl;

        P_optimized.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            P_optimized(i, 0) = params[2*i];
            P_optimized(i, 1) = params[2*i+1];
            P_optimized(i, 2) = s(i);
        }
        cost = summary.final_cost;
    }
    else
    {
        Eigen::Vector2d s(0.3, 0.3);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        // options.minimizer_progress_to_stdout = true;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
        ceres::Solver::Summary summary;

        ceres::Problem problem;
        for (int i = 0; i < n; i++)
        {
            Eigen::Vector2d p1_v(p1(i, 0), p1(i, 1));
            Eigen::Vector2d p2_v(p2(i, 0), p2(i, 1));
            ceres::CostFunction* cost_function = Residual::Create(p1_v, p2_v, T1, T2, s(i), K);
            double* param_ptr = params + 2 * i;
            problem.AddParameterBlock(param_ptr, 2);
            problem.SetParameterLowerBound(param_ptr, 0, -8);
            problem.SetParameterUpperBound(param_ptr, 0, 8);
            problem.SetParameterLowerBound(param_ptr, 1, -15);
            problem.SetParameterUpperBound(param_ptr, 1, 0);
            problem.AddResidualBlock(cost_function, nullptr, param_ptr);
        }

        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << std::endl;

        P_optimized.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            P_optimized(i, 0) = params[2*i];
            P_optimized(i, 1) = params[2*i+1];
            P_optimized(i, 2) = s(i);
        }
        P_optimized = guess3rdPoint(P_optimized.transpose()).transpose();
        cost = summary.final_cost;
    }
    
    return std::make_tuple(P_optimized, cost);
}