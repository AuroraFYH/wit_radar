#include "world_target_observation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wit_radar {
namespace {

bool finite_point(const cv::Point3d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool finite_matrix(const cv::Matx33d& matrix) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            if (!std::isfinite(matrix(row, column))) {
                return false;
            }
        }
    }
    return true;
}

cv::Matx33d outer_product(const cv::Vec3d& vector) {
    cv::Matx33d result;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result(row, column) = vector[row] * vector[column];
        }
    }
    return result;
}

}  // namespace

WorldTargetObservationBuilder::WorldTargetObservationBuilder(WorldTargetObservationParameters parameters)
    : parameters_(parameters) {
    if (parameters_.tangential_std_floor_m <= 0.0 || parameters_.radial_std_floor_m <= 0.0 ||
        parameters_.reprojection_std_px_floor <= 0.0 || parameters_.radial_std_per_meter < 0.0 ||
        parameters_.axis_residual_std_scale < 0.0 ||
        parameters_.confidence_reprojection_error_px <= 0.0 ||
        parameters_.confidence_axis_residual_m <= 0.0) {
        throw std::invalid_argument("World target observation parameters are invalid.");
    }
}

std::optional<WorldTargetObservation> WorldTargetObservationBuilder::build(
    const WorldTargetObservationInput& input) const {
    if (!finite_point(input.position_in_camera_m) || !finite_point(input.position_in_world_m) ||
        !finite_matrix(input.camera_to_world_rotation) || !finite_matrix(input.camera_matrix) ||
        !std::isfinite(input.average_reprojection_error_px) ||
        !std::isfinite(input.max_marker_axis_residual_m) || input.average_reprojection_error_px < 0.0 ||
        input.max_marker_axis_residual_m < 0.0 || input.marker_count <= 0) {
        return std::nullopt;
    }

    const double range_m = cv::norm(input.position_in_camera_m);
    const double focal_length_px = 0.5 * (std::abs(input.camera_matrix(0, 0)) +
                                          std::abs(input.camera_matrix(1, 1)));
    if (!std::isfinite(range_m) || range_m <= 1e-6 || !std::isfinite(focal_length_px) ||
        focal_length_px <= 1e-6) {
        return std::nullopt;
    }

    const double reprojection_std_px =
        std::max(parameters_.reprojection_std_px_floor, input.average_reprojection_error_px);
    const double tangential_std_m = std::hypot(parameters_.tangential_std_floor_m,
                                                range_m * reprojection_std_px / focal_length_px);
    const double radial_std_m = std::hypot(
        parameters_.radial_std_floor_m,
        std::hypot(range_m * parameters_.radial_std_per_meter,
                   input.max_marker_axis_residual_m * parameters_.axis_residual_std_scale));
    const cv::Vec3d camera_ray(input.position_in_camera_m.x / range_m,
                                input.position_in_camera_m.y / range_m,
                                input.position_in_camera_m.z / range_m);
    const cv::Matx33d ray_projection = outer_product(camera_ray);
    const cv::Matx33d covariance_in_camera_m2 =
        tangential_std_m * tangential_std_m * (cv::Matx33d::eye() - ray_projection) +
        radial_std_m * radial_std_m * ray_projection;
    const cv::Matx33d covariance_in_world_m2 =
        input.camera_to_world_rotation * covariance_in_camera_m2 * input.camera_to_world_rotation.t();

    const double reprojection_quality = std::exp(
        -0.5 * std::pow(input.average_reprojection_error_px /
                             parameters_.confidence_reprojection_error_px,
                         2.0));
    const double axis_quality = std::exp(-0.5 * std::pow(input.max_marker_axis_residual_m /
                                                               parameters_.confidence_axis_residual_m,
                                                           2.0));
    const double marker_quality = 1.0 - std::exp(-static_cast<double>(input.marker_count) / 4.0);

    WorldTargetObservation observation;
    observation.position_in_camera_m = input.position_in_camera_m;
    observation.position_in_world_m = input.position_in_world_m;
    observation.covariance_in_world_m2 = covariance_in_world_m2;
    observation.exposure_timestamp = input.exposure_timestamp;
    observation.frame_number = input.frame_number;
    observation.camera_device_timestamp = input.camera_device_timestamp;
    observation.range_m = range_m;
    observation.confidence = std::clamp(reprojection_quality * axis_quality * marker_quality, 0.0, 1.0);
    observation.tangential_std_m = tangential_std_m;
    observation.radial_std_m = radial_std_m;
    observation.average_reprojection_error_px = input.average_reprojection_error_px;
    observation.max_marker_axis_residual_m = input.max_marker_axis_residual_m;
    observation.marker_count = input.marker_count;
    observation.source = input.source;
    return observation;
}

}  // namespace wit_radar
