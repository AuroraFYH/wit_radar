#include "aim_solver.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace wit_radar::communication {
namespace {

constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;
constexpr double degrees_to_radians = 3.14159265358979323846 / 180.0;

cv::Point3d apply_rotation(const cv::Matx33d& rotation, const cv::Point3d& point) {
    const cv::Vec3d transformed = rotation * cv::Vec3d(point.x, point.y, point.z);
    return {transformed[0], transformed[1], transformed[2]};
}

double dot(const cv::Point3d& first, const cv::Point3d& second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

float normalize_degrees(float angle) {
    return std::remainder(angle, 360.0F);
}

double beam_line_error_squared(const cv::Point3d& target_in_world,
                                const cv::Point3d& beam_point_in_gimbal,
                                const cv::Point3d& beam_direction_in_gimbal,
                                float yaw_degrees, float pitch_degrees) {
    const cv::Matx33d rotation =
        AbsoluteLaserAimSolver::rotation_gimbal_to_world(yaw_degrees, pitch_degrees);
    const cv::Point3d point_in_world = apply_rotation(rotation, beam_point_in_gimbal);
    const cv::Point3d direction_in_world = apply_rotation(rotation, beam_direction_in_gimbal);
    const cv::Point3d target_delta = target_in_world - point_in_world;
    const double distance_along_beam = dot(target_delta, direction_in_world);
    if (distance_along_beam <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const cv::Point3d perpendicular = target_delta - direction_in_world * distance_along_beam;
    return dot(perpendicular, perpendicular);
}

}  // namespace

std::optional<AimSolution> WorldTargetSolver::solve(const cv::Point3d& target_in_world,
                                                     const cv::Point3d& gimbal_origin_in_world) {
    const cv::Point3d direction = target_in_world - gimbal_origin_in_world;
    const double horizontal_distance = std::hypot(direction.x, direction.y);
    if (horizontal_distance < 1e-6) {
        return std::nullopt;
    }
    AimSolution solution;
    solution.target_yaw = static_cast<float>(std::atan2(direction.y, direction.x) * radians_to_degrees);
    solution.target_pitch =
        static_cast<float>(std::atan2(direction.z, horizontal_distance) * radians_to_degrees);
    return solution;
}

cv::Matx33d AbsoluteLaserAimSolver::rotation_gimbal_to_world(float yaw_degrees, float pitch_degrees) {
    const double yaw = static_cast<double>(yaw_degrees) * degrees_to_radians;
    const double pitch = static_cast<double>(pitch_degrees) * degrees_to_radians;
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double cos_pitch = std::cos(pitch);
    const double sin_pitch = std::sin(pitch);
    const cv::Matx33d yaw_rotation{cos_yaw, -sin_yaw, 0.0, sin_yaw, cos_yaw, 0.0,
                                   0.0,     0.0,      1.0};
    const cv::Matx33d pitch_rotation{cos_pitch, 0.0, -sin_pitch, 0.0, 1.0, 0.0,
                                     sin_pitch, 0.0, cos_pitch};
    return yaw_rotation * pitch_rotation;
}

std::optional<AimSolution> AbsoluteLaserAimSolver::solve(
    const cv::Point3d& target_in_world, const cv::Point3d& beam_point_in_gimbal,
    const cv::Point3d& beam_direction_in_gimbal) {
    const double direction_length = cv::norm(beam_direction_in_gimbal);
    if (!std::isfinite(direction_length) || direction_length < 1e-9 ||
        !std::isfinite(target_in_world.x) || !std::isfinite(target_in_world.y) ||
        !std::isfinite(target_in_world.z)) {
        return std::nullopt;
    }
    const cv::Point3d normalized_direction = beam_direction_in_gimbal * (1.0 / direction_length);
    const std::optional<AimSolution> target_direction = WorldTargetSolver::solve(target_in_world, {});
    const std::optional<AimSolution> beam_direction = WorldTargetSolver::solve(normalized_direction, {});
    if (!target_direction.has_value() || !beam_direction.has_value()) {
        return std::nullopt;
    }

    float best_yaw = normalize_degrees(target_direction->target_yaw - beam_direction->target_yaw);
    float best_pitch = target_direction->target_pitch - beam_direction->target_pitch;
    double best_error = beam_line_error_squared(target_in_world, beam_point_in_gimbal, normalized_direction,
                                                 best_yaw, best_pitch);
    constexpr std::array<float, 6> search_steps{5.0F, 1.0F, 0.25F, 0.05F, 0.01F, 0.002F};
    for (const float step : search_steps) {
        bool improved = true;
        while (improved) {
            improved = false;
            for (const int yaw_direction : {-1, 0, 1}) {
                for (const int pitch_direction : {-1, 0, 1}) {
                    if (yaw_direction == 0 && pitch_direction == 0) {
                        continue;
                    }
                    const float candidate_yaw =
                        normalize_degrees(best_yaw + static_cast<float>(yaw_direction) * step);
                    const float candidate_pitch = best_pitch + static_cast<float>(pitch_direction) * step;
                    const double candidate_error = beam_line_error_squared(
                        target_in_world, beam_point_in_gimbal, normalized_direction, candidate_yaw,
                        candidate_pitch);
                    if (candidate_error + 1e-16 < best_error) {
                        best_yaw = candidate_yaw;
                        best_pitch = candidate_pitch;
                        best_error = candidate_error;
                        improved = true;
                    }
                }
            }
        }
    }
    if (!std::isfinite(best_error)) {
        return std::nullopt;
    }
    return AimSolution{best_yaw, best_pitch};
}

}  // namespace wit_radar::communication
