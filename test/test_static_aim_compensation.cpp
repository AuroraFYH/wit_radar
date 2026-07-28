#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "static_aim_compensation.hpp"

namespace {

bool close_to(double first, double second, double tolerance = 1e-9) {
    return std::abs(first - second) <= tolerance;
}

}  // namespace

int main() {
    try {
        const wit_radar::StaticAimCompensationParameters parameters{
            .enabled = true,
            .range_min_m = 18.9,
            .range_max_m = 25.6,
            .yaw_constant_deg = 0.607,
            .yaw_inverse_range_deg_m = -10.649,
            .pitch_constant_deg = 0.259,
        };
        const wit_radar::StaticAimCompensationResult result =
            wit_radar::evaluate_static_aim_compensation(parameters, 20.0);
        if (!result.applied || !close_to(result.yaw_correction_deg, 0.07455) ||
            !close_to(result.pitch_correction_deg, 0.259)) {
            throw std::runtime_error("Static compensation produced an unexpected correction.");
        }
        if (wit_radar::evaluate_static_aim_compensation(parameters, 18.8).applied ||
            wit_radar::evaluate_static_aim_compensation(parameters, 25.7).applied) {
            throw std::runtime_error("Static compensation applied outside its calibrated range.");
        }
        std::cout << "Static aim compensation test passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Static aim compensation test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
