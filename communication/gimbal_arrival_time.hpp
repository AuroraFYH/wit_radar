#pragma once

#include <optional>

#include "robot_protocol.hpp"

namespace wit_radar::communication {

struct GimbalAxisMotionLimits {
    double max_speed_deg_s = 0.0;
    double max_acceleration_deg_s2 = 0.0;
};

struct GimbalArrivalTimeParameters {
    bool enabled = false;
    GimbalAxisMotionLimits yaw;
    GimbalAxisMotionLimits pitch;
    double settle_margin_ms = 0.0;
    int prediction_iterations = 3;
    double convergence_ms = 1.0;
};

struct GimbalArrivalTimeEstimate {
    double yaw_motion_seconds = 0.0;
    double pitch_motion_seconds = 0.0;
    double total_seconds = 0.0;
};

class GimbalArrivalTimeEstimator {
public:
    explicit GimbalArrivalTimeEstimator(GimbalArrivalTimeParameters parameters = {});

    std::optional<GimbalArrivalTimeEstimate> estimate(const GimbalAngles& current,
                                                       const GimbalAngles& target) const;

    const GimbalArrivalTimeParameters& parameters() const noexcept;

private:
    static std::optional<double> axis_motion_seconds(double distance_degrees,
                                                     const GimbalAxisMotionLimits& limits);

    GimbalArrivalTimeParameters parameters_;
};

}  // namespace wit_radar::communication
