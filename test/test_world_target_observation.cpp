#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "world_target_observation.hpp"

namespace {

bool close_to(double value, double expected, double tolerance) {
    return std::abs(value - expected) <= tolerance;
}

}  // namespace

int main() {
    try {
        wit_radar::WorldTargetObservationParameters parameters;
        parameters.tangential_std_floor_m = 0.01;
        parameters.radial_std_floor_m = 0.03;
        parameters.reprojection_std_px_floor = 0.5;
        parameters.radial_std_per_meter = 0.01;
        parameters.axis_residual_std_scale = 1.0;
        parameters.confidence_reprojection_error_px = 3.0;
        parameters.confidence_axis_residual_m = 0.05;
        wit_radar::WorldTargetObservationBuilder builder(parameters);

        wit_radar::WorldTargetObservationInput input;
        input.position_in_camera_m = {0.0, 0.0, 10.0};
        input.position_in_world_m = {10.0, 0.0, 0.0};
        input.camera_to_world_rotation = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0};
        input.camera_matrix = {1000.0, 0.0, 640.0, 0.0, 1000.0, 480.0, 0.0, 0.0, 1.0};
        input.exposure_timestamp = std::chrono::steady_clock::now();
        input.frame_number = 42;
        input.camera_device_timestamp = 123456789U;
        input.average_reprojection_error_px = 2.0;
        input.max_marker_axis_residual_m = 0.02;
        input.marker_count = 8;
        const auto observation = builder.build(input);
        if (!observation.has_value()) {
            throw std::runtime_error("Observation builder rejected valid input.");
        }
        if (!close_to(observation->range_m, 10.0, 1e-9) || observation->confidence <= 0.0 ||
            observation->confidence >= 1.0 || observation->frame_number != 42 ||
            observation->camera_device_timestamp != 123456789U) {
            throw std::runtime_error("Observation metadata is invalid.");
        }
        if (!close_to(observation->covariance_in_world_m2(0, 0),
                      observation->radial_std_m * observation->radial_std_m, 1e-9) ||
            !close_to(observation->covariance_in_world_m2(2, 2),
                      observation->tangential_std_m * observation->tangential_std_m, 1e-9)) {
            throw std::runtime_error("Observation covariance was not rotated into world coordinates.");
        }

        input.marker_count = 0;
        if (builder.build(input).has_value()) {
            throw std::runtime_error("Observation builder accepted an invalid marker count.");
        }
        std::cout << "World target observation test: confidence=" << observation->confidence
                  << " range=" << observation->range_m << " m\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "World target observation test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
