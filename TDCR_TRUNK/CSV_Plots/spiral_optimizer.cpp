#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace py = pybind11;

class Spiral {
public:
    double x0, y0, a, b, theta_off;
    bool valid;
    
    Spiral() : x0(0), y0(0), a(1), b(0), theta_off(0), valid(false) {}
    Spiral(double x0, double y0, double a, double b, double theta_off) 
        : x0(x0), y0(y0), a(a), b(b), theta_off(theta_off), valid(true) {}
};

double safe_clip(double value, double min_val, double max_val) {
    return std::max(min_val, std::min(max_val, value));
}

double safe_exp(double x) {
    return std::exp(safe_clip(x, -20.0, 20.0));
}

std::vector<double> unwrap_angles(const std::vector<double>& theta) {
    std::vector<double> result = theta;
    for (size_t i = 1; i < result.size(); ++i) {
        while (result[i] - result[i-1] > M_PI) {
            result[i] -= 2 * M_PI;
        }
        while (result[i] - result[i-1] < -M_PI) {
            result[i] += 2 * M_PI;
        }
    }
    return result;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) {
        return 0.5 * (v[n/2 - 1] + v[n/2]);
    } else {
        return v[n/2];
    }
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double idx = p/100.0 * (n - 1);
    size_t lower = static_cast<size_t>(std::floor(idx));
    size_t upper = static_cast<size_t>(std::ceil(idx));
    
    if (lower == upper || upper >= n) {
        return v[std::min(lower, n-1)];
    }
    
    double weight = idx - lower;
    return v[lower] * (1 - weight) + v[upper] * weight;
}

std::vector<bool> detect_outliers(const std::vector<double>& x, const std::vector<double>& y, double threshold_pct = 85) {
    double center_x = median(x);
    double center_y = median(y);
    
    std::vector<double> distances;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - center_x;
        double dy = y[i] - center_y;
        distances.push_back(std::sqrt(dx*dx + dy*dy));
    }
    
    double threshold = percentile(distances, threshold_pct);
    
    std::vector<bool> mask(x.size());
    for (size_t i = 0; i < distances.size(); ++i) {
        mask[i] = distances[i] < threshold;
    }
    
    return mask;
}

std::pair<double, double> weighted_polyfit(const std::vector<double>& theta, 
                                         const std::vector<double>& log_r, 
                                         const std::vector<double>& weights) {
    if (theta.size() < 2) return {0.0, median(log_r)};
    
    double sum_w = 0, sum_wx = 0, sum_wy = 0, sum_wxx = 0, sum_wxy = 0;
    
    for (size_t i = 0; i < theta.size(); ++i) {
        sum_w += weights[i];
        sum_wx += weights[i] * theta[i];
        sum_wy += weights[i] * log_r[i];
        sum_wxx += weights[i] * theta[i] * theta[i];
        sum_wxy += weights[i] * theta[i] * log_r[i];
    }
    
    double denom = sum_w * sum_wxx - sum_wx * sum_wx;
    if (std::abs(denom) < 1e-12) return {0.0, median(log_r)};
    
    double slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
    double intercept = (sum_wy * sum_wxx - sum_wx * sum_wxy) / denom;
    
    return {slope, intercept};
}

double spiral_cost(const std::vector<double>& params, const std::vector<double>& x, const std::vector<double>& y) {
    double x0 = params[0], y0 = params[1], loga = params[2], b = params[3], theta_off = params[4];
    
    std::vector<double> theta_data, r_data;
    for (size_t i = 0; i < x.size(); ++i) {
        theta_data.push_back(std::atan2(y[i] - y0, x[i] - x0));
        r_data.push_back(std::sqrt((x[i] - x0)*(x[i] - x0) + (y[i] - y0)*(y[i] - y0)));
    }
    
    theta_data = unwrap_angles(theta_data);
    
    // Check if all radii are too small
    bool all_small = true;
    for (double r : r_data) {
        if (r >= 1e-8) {
            all_small = false;
            break;
        }
    }
    if (all_small) return 1e10;
    
    double total_loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double model_log_r = loga + b * (theta_data[i] + theta_off);
        double log_r_data = std::log(std::max(r_data[i], 1e-8));
        double residual = log_r_data - model_log_r;
        total_loss += residual * residual;
    }
    
    double reg = 0.001 * (b*b + theta_off*theta_off);
    return total_loss + reg;
}

std::vector<double> gradient_descent(const std::vector<double>& x, const std::vector<double>& y, 
                                   std::vector<double> params0, 
                                   const std::vector<std::pair<double, double>>& bounds,
                                   int maxiter = 1000) {
    std::vector<double> params = params0;
    std::vector<double> best_params = params;
    double best_cost = spiral_cost(params, x, y);
    
    std::vector<double> learning_rates = {0.3, 0.3, 0.03, 0.03, 0.03};

    std::vector<double> momentum(5, 0.0);
    double momentum_decay = 0.9;
    
    for (int iter = 0; iter < maxiter; ++iter) {
        // Numerical gradient
        std::vector<double> grad(5);
        double eps = 1e-6;
        
        for (int i = 0; i < 5; ++i) {
            std::vector<double> params_plus = params, params_minus = params;
            params_plus[i] += eps;
            params_minus[i] -= eps;
            
            double cost_plus = spiral_cost(params_plus, x, y);
            double cost_minus = spiral_cost(params_minus, x, y);
            
            grad[i] = (cost_plus - cost_minus) / (2 * eps);
        }
        
        // Update with momentum
        for (int i = 0; i < 5; ++i) {
            momentum[i] = momentum_decay * momentum[i] - learning_rates[i] * grad[i];
            params[i] += momentum[i];
            
            // Apply bounds
            params[i] = std::max(bounds[i].first, std::min(bounds[i].second, params[i]));
        }
        
        double cost = spiral_cost(params, x, y);
        if (cost < best_cost) {
            best_cost = cost;
            best_params = params;
        }
        
        // Early stopping
        if (cost < 1.0 or (iter > 50 && cost > best_cost * 2)) break;
        
        // Adaptive learning rate
        if (iter % 50 == 0 && iter > 0) {
            for (double& lr : learning_rates) lr *= 0.95;
        }
    }
    
    return best_params;
}

Spiral fit_spiral_fast(py::array_t<double> x_arr, py::array_t<double> y_arr) {
    auto x_buf = x_arr.request();
    auto y_buf = y_arr.request();
    
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    
    std::vector<double> x(x_ptr, x_ptr + x_buf.size);
    std::vector<double> y(y_ptr, y_ptr + y_buf.size);
    
    // Remove NaN values
    std::vector<double> x_clean, y_clean;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            x_clean.push_back(x[i]);
            y_clean.push_back(y[i]);
        }
    }
    
    if (x_clean.size() < 6) return Spiral();
    
    // Outlier detection
    auto keep_mask = detect_outliers(x_clean, y_clean, 85);
    std::vector<double> x_final, y_final;
    for (size_t i = 0; i < x_clean.size(); ++i) {
        if (keep_mask[i]) {
            x_final.push_back(x_clean[i]);
            y_final.push_back(y_clean[i]);
        }
    }
    
    if (x_final.size() < 6) return Spiral();
    
    // Initial parameter estimation
    double x0_init = median(x_final);
    double y0_init = median(y_final);
    
    std::vector<double> theta, r;
    for (size_t i = 0; i < x_final.size(); ++i) {
        theta.push_back(std::atan2(y_final[i] - y0_init, x_final[i] - x0_init));
        r.push_back(std::sqrt((x_final[i] - x0_init)*(x_final[i] - x0_init) + 
                             (y_final[i] - y0_init)*(y_final[i] - y0_init)));
    }
    
    theta = unwrap_angles(theta);
    
    // Filter by radius
    double min_r = percentile(r, 5);
    std::vector<double> theta_fit, r_fit;
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] > min_r) {
            theta_fit.push_back(theta[i]);
            r_fit.push_back(r[i]);
        }
    }
    
    if (theta_fit.size() < 5) return Spiral();
    
    // Weighted fitting
    double max_r = *std::max_element(r_fit.begin(), r_fit.end());
    std::vector<double> weights, log_r_fit;
    for (size_t i = 0; i < r_fit.size(); ++i) {
        weights.push_back(std::sqrt(r_fit[i]) / max_r);
        log_r_fit.push_back(std::log(std::max(r_fit[i], 1e-8)));
    }
    
    auto [b0, loga0] = weighted_polyfit(theta_fit, log_r_fit, weights);
    
    // Set up optimization
    double x_range = *std::max_element(x_final.begin(), x_final.end()) - 
                    *std::min_element(x_final.begin(), x_final.end());
    double y_range = *std::max_element(y_final.begin(), y_final.end()) - 
                    *std::min_element(y_final.begin(), y_final.end());
    
    std::vector<std::pair<double, double>> bounds = {
        {*std::min_element(x_final.begin(), x_final.end()) - 0.05*x_range, 
         *std::max_element(x_final.begin(), x_final.end()) + 0.05*x_range},
        {*std::min_element(y_final.begin(), y_final.end()) - 0.05*y_range, 
         *std::max_element(y_final.begin(), y_final.end()) + 0.05*y_range},
        {-10.0, 10.0}, {-2.5, 2.5}, {-2*M_PI, 2*M_PI}
    };
    
    std::vector<double> params0 = {x0_init, y0_init, loga0, b0, 0.0};
    
    // Optimize
    std::vector<double> best_params = gradient_descent(x_final, y_final, params0, bounds, 1000);
    
    double x0_fit = best_params[0];
    double y0_fit = best_params[1]; 
    double loga_fit = best_params[2];
    double b_fit = best_params[3];
    double theta_off_fit = best_params[4];
    
    double a_fit = std::exp(safe_clip(loga_fit, -20.0, 20.0));
    
    return Spiral(x0_fit, y0_fit, a_fit, b_fit, theta_off_fit);
}

py::array_t<double> calculate_spiral_errors(py::array_t<double> x_arr, py::array_t<double> y_arr, 
                                           double x0, double y0, double a, double b, double theta_off) {
    auto x_buf = x_arr.request();
    auto y_buf = y_arr.request();
    
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    
    std::vector<double> x(x_ptr, x_ptr + x_buf.size);
    std::vector<double> y(y_ptr, y_ptr + y_buf.size);
    
    std::vector<double> theta_data, r_data;
    for (size_t i = 0; i < x.size(); ++i) {
        theta_data.push_back(std::atan2(y[i] - y0, x[i] - x0));
        r_data.push_back(std::sqrt((x[i] - x0)*(x[i] - x0) + (y[i] - y0)*(y[i] - y0)));
    }
    
    theta_data = unwrap_angles(theta_data);
    
    std::vector<double> distances(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double r_model = a * safe_exp(b * (theta_data[i] + theta_off));
        distances[i] = std::abs(r_data[i] - r_model);
    }
    
    return py::cast(distances);
}

PYBIND11_MODULE(spiral_optimizer, m) {
    py::class_<Spiral>(m, "Spiral")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>())
        .def_readwrite("x0", &Spiral::x0)
        .def_readwrite("y0", &Spiral::y0)
        .def_readwrite("a", &Spiral::a)
        .def_readwrite("b", &Spiral::b)
        .def_readwrite("theta_off", &Spiral::theta_off)
        .def_readwrite("valid", &Spiral::valid);
    
    m.def("fit_spiral_fast", &fit_spiral_fast, "Fast spiral fitting");
    m.def("calculate_spiral_errors", &calculate_spiral_errors, "Calculate spiral fit errors");
}
