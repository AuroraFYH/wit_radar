#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "aim_solver.hpp"

namespace {

cv::Point3d rotate(const cv::Matx33d& rotation, const cv::Point3d& point) {
    const cv::Vec3d transformed = rotation * cv::Vec3d(point.x, point.y, point.z);
    return {transformed[0], transformed[1], transformed[2]};
}

float angular_difference(float first, float second) {
    return std::abs(std::remainder(first - second, 360.0F));
}

}  // namespace

int main() {
    try {
        constexpr float expected_yaw = 17.0F;
        constexpr float expected_pitch = -8.0F;
        const cv::Point3d beam_point{0.025, -0.010, 0.003};
        const cv::Point3d beam_direction{0.045, 0.020, 0.999};
        const cv::Matx33d expected_rotation =
            wit_radar::communication::AbsoluteLaserAimSolver::rotation_gimbal_to_world(expected_yaw,
                                                                                          expected_pitch);
        const cv::Point3d target_in_world = rotate(expected_rotation, beam_point) +
                                            rotate(expected_rotation, beam_direction) * 4.0;

        const auto solution = wit_radar::communication::AbsoluteLaserAimSolver::solve(
            target_in_world, beam_point, beam_direction);
        if (!solution.has_value() || angular_difference(solution->target_yaw, expected_yaw) > 0.02F ||
            std::abs(solution->target_pitch - expected_pitch) > 0.02F) {
            throw std::runtime_error("Absolute laser aim did not recover the expected orientation.");
        }
        for (const wit_radar::communication::AimSolution feedback : {
                 wit_radar::communication::AimSolution{-12.0F, 4.0F},
                 wit_radar::communication::AimSolution{31.0F, -15.0F}}) {
            const cv::Matx33d current_rotation =
                wit_radar::communication::AbsoluteLaserAimSolver::rotation_gimbal_to_world(
                    feedback.target_yaw, feedback.target_pitch);
            const cv::Point3d target_in_current_gimbal = rotate(current_rotation.t(), target_in_world);
            const cv::Point3d reconstructed_world_target = rotate(current_rotation, target_in_current_gimbal);
            const auto reconstructed_solution = wit_radar::communication::AbsoluteLaserAimSolver::solve(
                reconstructed_world_target, beam_point, beam_direction);
            if (!reconstructed_solution.has_value() ||
                angular_difference(reconstructed_solution->target_yaw, expected_yaw) > 0.02F ||
                std::abs(reconstructed_solution->target_pitch - expected_pitch) > 0.02F) {
                throw std::runtime_error("Static world target produced different absolute commands.");
            }
        }
        std::cout << "Absolute laser aim: yaw=" << solution->target_yaw
                  << " pitch=" << solution->target_pitch << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Absolute laser aim test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
