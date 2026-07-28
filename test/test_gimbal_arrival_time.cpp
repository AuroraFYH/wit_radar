#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "gimbal_arrival_time.hpp"

namespace {

bool close_to(double value, double expected, double tolerance) {
    return std::abs(value - expected) <= tolerance;
}

}  // namespace

int main() {
    try {
        wit_radar::communication::GimbalArrivalTimeParameters parameters;
        parameters.enabled = true;
        parameters.yaw = {100.0, 200.0};
        parameters.pitch = {80.0, 160.0};
        parameters.settle_margin_ms = 20.0;
        wit_radar::communication::GimbalArrivalTimeEstimator estimator(parameters);

        const auto triangular = estimator.estimate({0.0F, 0.0F}, {10.0F, 0.0F});
        if (!triangular.has_value() || !close_to(triangular->yaw_motion_seconds, 2.0 * std::sqrt(10.0 / 200.0),
                                                 1e-9) ||
            !close_to(triangular->total_seconds, triangular->yaw_motion_seconds + 0.02, 1e-9)) {
            throw std::runtime_error("Triangular yaw motion estimate is invalid.");
        }

        const auto trapezoidal = estimator.estimate({0.0F, 0.0F}, {100.0F, 40.0F});
        if (!trapezoidal.has_value() || !close_to(trapezoidal->yaw_motion_seconds, 1.5, 1e-9) ||
            !close_to(trapezoidal->pitch_motion_seconds, 1.0, 1e-9) ||
            !close_to(trapezoidal->total_seconds, 1.52, 1e-9)) {
            throw std::runtime_error("Trapezoidal two-axis motion estimate is invalid.");
        }

        const auto wrapped_yaw = estimator.estimate({179.0F, 0.0F}, {-179.0F, 0.0F});
        if (!wrapped_yaw.has_value() || !close_to(wrapped_yaw->yaw_motion_seconds, 0.2, 1e-9)) {
            throw std::runtime_error("Yaw wrap-around motion estimate is invalid.");
        }
        std::cout << "Gimbal arrival time test: total=" << trapezoidal->total_seconds << " s\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Gimbal arrival time test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
