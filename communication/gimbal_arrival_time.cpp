#include "gimbal_arrival_time.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wit_radar::communication {
namespace {

bool valid_limits(const GimbalAxisMotionLimits& limits) {
    return std::isfinite(limits.max_speed_deg_s) && std::isfinite(limits.max_acceleration_deg_s2) &&
           limits.max_speed_deg_s > 0.0 && limits.max_acceleration_deg_s2 > 0.0;
}

}  // namespace

GimbalArrivalTimeEstimator::GimbalArrivalTimeEstimator(GimbalArrivalTimeParameters parameters)
    : parameters_(parameters) {
    if (!std::isfinite(parameters_.settle_margin_ms) || parameters_.settle_margin_ms < 0.0 ||
        parameters_.prediction_iterations <= 0 || !std::isfinite(parameters_.convergence_ms) ||
        parameters_.convergence_ms < 0.0 ||
        (parameters_.enabled && (!valid_limits(parameters_.yaw) || !valid_limits(parameters_.pitch)))) {
        throw std::invalid_argument("Gimbal arrival-time parameters are invalid.");
    }
}

std::optional<double> GimbalArrivalTimeEstimator::axis_motion_seconds(
    double distance_degrees, const GimbalAxisMotionLimits& limits) {
    if (!std::isfinite(distance_degrees) || distance_degrees < 0.0 || !valid_limits(limits)) {
        return std::nullopt;
    }
    if (distance_degrees <= 1e-9) {
        return 0.0;
    }
    const double transition_distance =
        limits.max_speed_deg_s * limits.max_speed_deg_s / limits.max_acceleration_deg_s2;
    if (distance_degrees <= transition_distance) {
        return 2.0 * std::sqrt(distance_degrees / limits.max_acceleration_deg_s2);
    }
    return distance_degrees / limits.max_speed_deg_s +
           limits.max_speed_deg_s / limits.max_acceleration_deg_s2;
}

std::optional<GimbalArrivalTimeEstimate> GimbalArrivalTimeEstimator::estimate(
    const GimbalAngles& current, const GimbalAngles& target) const {
    if (!std::isfinite(current.yaw) || !std::isfinite(current.pitch) || !std::isfinite(target.yaw) ||
        !std::isfinite(target.pitch)) {
        return std::nullopt;
    }
    if (!parameters_.enabled) {
        return GimbalArrivalTimeEstimate{};
    }
    const double yaw_distance_degrees = std::abs(std::remainder(target.yaw - current.yaw, 360.0F));
    const double pitch_distance_degrees = std::abs(target.pitch - current.pitch);
    const std::optional<double> yaw_seconds = axis_motion_seconds(yaw_distance_degrees, parameters_.yaw);
    const std::optional<double> pitch_seconds = axis_motion_seconds(pitch_distance_degrees, parameters_.pitch);
    if (!yaw_seconds.has_value() || !pitch_seconds.has_value()) {
        return std::nullopt;
    }
    GimbalArrivalTimeEstimate estimate;
    estimate.yaw_motion_seconds = yaw_seconds.value();
    estimate.pitch_motion_seconds = pitch_seconds.value();
    estimate.total_seconds = std::max(estimate.yaw_motion_seconds, estimate.pitch_motion_seconds) +
                             parameters_.settle_margin_ms / 1000.0;
    return estimate;
}

const GimbalArrivalTimeParameters& GimbalArrivalTimeEstimator::parameters() const noexcept {
    return parameters_;
}

}  // namespace wit_radar::communication
