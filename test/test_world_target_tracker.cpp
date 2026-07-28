#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "world_target_tracker.hpp"

namespace {

bool close_to(double value, double expected, double tolerance) {
    return std::abs(value - expected) <= tolerance;
}

wit_radar::WorldTargetObservation make_observation(const cv::Point3d& position,
                                                    std::chrono::steady_clock::time_point timestamp) {
    wit_radar::WorldTargetObservation observation;
    observation.position_in_world_m = position;
    observation.exposure_timestamp = timestamp;
    observation.covariance_in_world_m2 = cv::Matx33d::zeros();
    observation.covariance_in_world_m2(0, 0) = 0.01 * 0.01;
    observation.covariance_in_world_m2(1, 1) = 0.01 * 0.01;
    observation.covariance_in_world_m2(2, 2) = 0.02 * 0.02;
    return observation;
}

}  // namespace

int main() {
    try {
        wit_radar::WorldTargetTrackerParameters parameters;
        parameters.enabled = true;
        parameters.initialization_frames = 3;
        parameters.initialization_max_spread_m = 0.10;
        parameters.measurement_std_xy_m = 0.01;
        parameters.measurement_std_z_m = 0.02;
        parameters.acceleration_std_mps2 = 1.0;
        parameters.innovation_gate_chi2 = 16.27;
        parameters.roi_measurement_enabled = true;
        parameters.roi_measurement_std_px = 2.0;
        parameters.roi_innovation_gate_chi2 = 9.21;
        wit_radar::WorldTargetTracker tracker(parameters);
        const auto start = std::chrono::steady_clock::now();

        for (int index = 0; index < 3; ++index) {
            const auto result = tracker.update({1.0 + index * 0.001, 2.0, 3.0},
                                               start + std::chrono::milliseconds(index * 20));
            if (index < 2 && result.update != wit_radar::TargetTrackerUpdate::Initializing) {
                throw std::runtime_error("Tracker did not initialize gradually.");
            }
        }
        const auto accepted = tracker.update({1.10, 2.0, 3.0}, start + std::chrono::milliseconds(120));
        if (accepted.update != wit_radar::TargetTrackerUpdate::Accepted ||
            !accepted.estimate_in_world_m.has_value()) {
            throw std::runtime_error("Tracker rejected a plausible measurement.");
        }
        wit_radar::WorldTargetImageObservation image_observation;
        image_observation.camera_matrix = {800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0};
        image_observation.distortion_coefficients = cv::Vec<double, 5>::all(0.0);
        image_observation.pixel = {
            800.0 * accepted.estimate_in_world_m->x / accepted.estimate_in_world_m->z + 640.0,
            800.0 * accepted.estimate_in_world_m->y / accepted.estimate_in_world_m->z + 480.0};
        const auto image_accepted = tracker.update_image(image_observation, start + std::chrono::milliseconds(120));
        if (!image_accepted.image_measurement_accepted ||
            !image_accepted.estimate_in_world_m.has_value()) {
            throw std::runtime_error("Tracker rejected a consistent ROI image observation.");
        }
        image_observation.pixel = {100.0, 100.0};
        const auto image_rejected = tracker.update_image(image_observation, start + std::chrono::milliseconds(120));
        if (image_rejected.image_measurement_accepted ||
            image_rejected.update != wit_radar::TargetTrackerUpdate::Rejected) {
            throw std::runtime_error("Tracker accepted an implausible ROI image observation.");
        }
        const auto rejected = tracker.update({8.0, -4.0, 20.0}, start + std::chrono::milliseconds(140));
        if (rejected.update != wit_radar::TargetTrackerUpdate::Rejected ||
            !rejected.estimate_in_world_m.has_value()) {
            throw std::runtime_error("Tracker did not reject a large PnP outlier.");
        }
        const auto prediction = tracker.predict_ahead(0.05);
        if (!prediction.has_value() || !close_to(prediction->y, 2.0, 0.10) ||
            !close_to(prediction->z, 3.0, 0.10)) {
            throw std::runtime_error("Tracker prediction is invalid.");
        }

        wit_radar::WorldTargetTrackerParameters moving_parameters;
        moving_parameters.enabled = true;
        moving_parameters.initialization_frames = 3;
        moving_parameters.initialization_max_spread_m = 0.02;
        moving_parameters.initialization_max_speed_mps = 10.0;
        moving_parameters.initialization_velocity_std_mps = 1.0;
        moving_parameters.max_coast_ms = 200;
        moving_parameters.max_prediction_seconds = 0.4;
        moving_parameters.acceleration_std_mps2 = 1.0;
        wit_radar::WorldTargetTracker moving_tracker(moving_parameters);
        const cv::Point3d start_position{1.0, 2.0, 3.0};
        const cv::Point3d velocity{2.0, -0.5, 0.25};
        for (int index = 0; index < 3; ++index) {
            const double elapsed_seconds = 0.02 * static_cast<double>(index);
            const auto result = moving_tracker.update(make_observation(
                start_position + velocity * elapsed_seconds,
                start + std::chrono::milliseconds(index * 20)));
            if (index < 2 && result.update != wit_radar::TargetTrackerUpdate::Initializing) {
                throw std::runtime_error("Moving tracker did not initialize gradually.");
            }
        }
        const auto initialized_prediction = moving_tracker.predict_ahead(0.10);
        if (!initialized_prediction.has_value() ||
            !close_to(initialized_prediction->x, 1.28, 0.02) ||
            !close_to(initialized_prediction->y, 1.93, 0.02) ||
            !close_to(initialized_prediction->z, 3.035, 0.02)) {
            throw std::runtime_error("Moving tracker did not initialize velocity from observations.");
        }
        const auto coasting = moving_tracker.coast_to(start + std::chrono::milliseconds(120));
        if (coasting.update != wit_radar::TargetTrackerUpdate::Coasting ||
            !coasting.estimate_in_world_m.has_value() || !coasting.velocity_in_world_mps.has_value() ||
            !close_to(coasting.estimate_in_world_m->x, 1.24, 0.02)) {
            throw std::runtime_error("Moving tracker did not coast through a brief missed detection.");
        }
        const auto updated = moving_tracker.update(make_observation(
            start_position + velocity * 0.14, start + std::chrono::milliseconds(140)));
        if (updated.update != wit_radar::TargetTrackerUpdate::Accepted ||
            !updated.velocity_in_world_mps.has_value() ||
            !close_to(updated.velocity_in_world_mps->x, velocity.x, 0.25)) {
            throw std::runtime_error("Moving tracker did not maintain its velocity after correction.");
        }
        const auto lost = moving_tracker.coast_to(start + std::chrono::milliseconds(350));
        if (lost.update != wit_radar::TargetTrackerUpdate::Lost || moving_tracker.has_estimate()) {
            throw std::runtime_error("Moving tracker did not reset after the coasting timeout.");
        }
        std::cout << "World target tracker test: estimate=" << rejected.estimate_in_world_m.value()
                  << " innovation=" << rejected.innovation_chi2 << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "World target tracker test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
