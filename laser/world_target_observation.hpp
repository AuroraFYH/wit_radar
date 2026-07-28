#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

#include <opencv2/core.hpp>

namespace wit_radar {

enum class WorldTargetObservationSource {
    GlobalCylinderPnp,
    LegacyIppe,
};

struct WorldTargetObservationParameters {
    double tangential_std_floor_m = 0.01;
    double radial_std_floor_m = 0.03;
    double reprojection_std_px_floor = 0.5;
    double radial_std_per_meter = 0.01;
    double axis_residual_std_scale = 1.0;
    double confidence_reprojection_error_px = 3.0;
    double confidence_axis_residual_m = 0.05;
};

struct WorldTargetObservationInput {
    cv::Point3d position_in_camera_m;
    cv::Point3d position_in_world_m;
    cv::Matx33d camera_to_world_rotation = cv::Matx33d::eye();
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    std::chrono::steady_clock::time_point exposure_timestamp;
    std::uint32_t frame_number = 0;
    std::uint64_t camera_device_timestamp = 0;
    double average_reprojection_error_px = 0.0;
    double max_marker_axis_residual_m = 0.0;
    int marker_count = 0;
    WorldTargetObservationSource source = WorldTargetObservationSource::GlobalCylinderPnp;
};

struct WorldTargetObservation {
    cv::Point3d position_in_camera_m;
    cv::Point3d position_in_world_m;
    cv::Matx33d covariance_in_world_m2 = cv::Matx33d::eye();
    std::chrono::steady_clock::time_point exposure_timestamp;
    std::uint32_t frame_number = 0;
    std::uint64_t camera_device_timestamp = 0;
    double range_m = 0.0;
    double confidence = 0.0;
    double tangential_std_m = 0.0;
    double radial_std_m = 0.0;
    double average_reprojection_error_px = 0.0;
    double max_marker_axis_residual_m = 0.0;
    int marker_count = 0;
    WorldTargetObservationSource source = WorldTargetObservationSource::GlobalCylinderPnp;
};

class WorldTargetObservationBuilder {
public:
    explicit WorldTargetObservationBuilder(WorldTargetObservationParameters parameters = {});

    std::optional<WorldTargetObservation> build(const WorldTargetObservationInput& input) const;

private:
    WorldTargetObservationParameters parameters_;
};

}  // namespace wit_radar
