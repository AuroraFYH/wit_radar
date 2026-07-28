#include "static_aim_compensation.hpp"

#include <cmath>
#include <stdexcept>

namespace wit_radar {

StaticAimCompensationResult evaluate_static_aim_compensation(
    const StaticAimCompensationParameters& parameters, double range_m) {
    if (!parameters.enabled) {
        return {};
    }
    if (!std::isfinite(parameters.range_min_m) || !std::isfinite(parameters.range_max_m) ||
        parameters.range_min_m <= 0.0 || parameters.range_max_m < parameters.range_min_m ||
        !std::isfinite(parameters.yaw_constant_deg) ||
        !std::isfinite(parameters.yaw_inverse_range_deg_m) ||
        !std::isfinite(parameters.pitch_constant_deg)) {
        throw std::invalid_argument("Static aim compensation parameters are invalid.");
    }
    if (!std::isfinite(range_m) || range_m < parameters.range_min_m ||
        range_m > parameters.range_max_m) {
        return {};
    }

    const double yaw_correction_deg =
        parameters.yaw_constant_deg + parameters.yaw_inverse_range_deg_m / range_m;
    if (!std::isfinite(yaw_correction_deg)) {
        throw std::invalid_argument("Static aim compensation result is invalid.");
    }
    return {true, yaw_correction_deg, parameters.pitch_constant_deg};
}

}  // namespace wit_radar
