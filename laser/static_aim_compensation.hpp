#pragma once

namespace wit_radar {

struct StaticAimCompensationParameters {
    bool enabled = false;
    double range_min_m = 0.0;
    double range_max_m = 0.0;
    double yaw_constant_deg = 0.0;
    double yaw_inverse_range_deg_m = 0.0;
    double pitch_constant_deg = 0.0;
};

struct StaticAimCompensationResult {
    bool applied = false;
    double yaw_correction_deg = 0.0;
    double pitch_correction_deg = 0.0;
};

StaticAimCompensationResult evaluate_static_aim_compensation(
    const StaticAimCompensationParameters& parameters, double range_m);

}  // namespace wit_radar
