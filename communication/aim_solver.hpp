#pragma once

#include <optional>

#include <opencv2/core/types.hpp>

namespace wit_radar::communication {

struct AimSolution {
    float target_yaw = 0.0F;
    float target_pitch = 0.0F;
};

class WorldTargetSolver {
public:
    static std::optional<AimSolution> solve(const cv::Point3d& target_in_world,
                                            const cv::Point3d& gimbal_origin_in_world);
};

class AbsoluteLaserAimSolver {
public:
    // Gimbal coordinates are X forward, Y left, Z up. Positive yaw is counter-clockwise viewed
    // from above; positive pitch raises the muzzle. The gimbal applies yaw first, then local pitch.
    static cv::Matx33d rotation_gimbal_to_world(float yaw_degrees, float pitch_degrees);

    // target_in_world is fixed for a stationary target. beam_point_in_gimbal and
    // beam_direction_in_gimbal describe the red laser beam as mounted on the gimbal.
    static std::optional<AimSolution> solve(const cv::Point3d& target_in_world,
                                            const cv::Point3d& beam_point_in_gimbal,
                                            const cv::Point3d& beam_direction_in_gimbal);
};

}  // namespace wit_radar::communication
