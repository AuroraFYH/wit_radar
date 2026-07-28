#include "world_target_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace wit_radar {
namespace {

bool finite_point(const cv::Point3d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool finite_pixel(const cv::Point2d& pixel) {
    return std::isfinite(pixel.x) && std::isfinite(pixel.y);
}

bool valid_covariance(const cv::Matx33d& covariance) {
    cv::Mat covariance_matrix(covariance);
    cv::Mat eigenvalues;
    cv::eigen(covariance_matrix, eigenvalues);
    if (eigenvalues.rows != 3) {
        return false;
    }
    for (int index = 0; index < 3; ++index) {
        if (!std::isfinite(eigenvalues.at<double>(index)) || eigenvalues.at<double>(index) <= 1e-12) {
            return false;
        }
    }
    return true;
}

cv::Mat covariance_to_mat(const cv::Matx33d& covariance) {
    cv::Mat result(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.at<double>(row, column) = covariance(row, column);
        }
    }
    return result;
}

std::optional<cv::Point2d> project_world_point(const cv::Point3d& point_in_world_m,
                                                const WorldTargetImageObservation& observation) {
    const cv::Vec3d camera_point = observation.world_to_camera_rotation *
                                   cv::Vec3d(point_in_world_m.x, point_in_world_m.y,
                                             point_in_world_m.z) +
                                   cv::Vec3d(observation.world_to_camera_translation_m.x,
                                             observation.world_to_camera_translation_m.y,
                                             observation.world_to_camera_translation_m.z);
    if (!std::isfinite(camera_point[0]) || !std::isfinite(camera_point[1]) ||
        !std::isfinite(camera_point[2]) || camera_point[2] <= 1e-6) {
        return std::nullopt;
    }
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(std::vector<cv::Point3d>{{camera_point[0], camera_point[1], camera_point[2]}},
                      cv::Vec3d{}, cv::Vec3d{}, observation.camera_matrix,
                      observation.distortion_coefficients, image_points);
    if (image_points.size() != 1 || !finite_pixel(image_points.front())) {
        return std::nullopt;
    }
    return image_points.front();
}

}  // namespace

WorldTargetTracker::WorldTargetTracker(WorldTargetTrackerParameters parameters)
    : parameters_(parameters) {
    if (parameters_.initialization_frames <= 0 || parameters_.initialization_max_spread_m <= 0.0 ||
        parameters_.initialization_max_speed_mps <= 0.0 ||
        parameters_.initialization_velocity_std_mps <= 0.0 || parameters_.max_coast_ms < 0 ||
        parameters_.max_prediction_seconds <= 0.0 || parameters_.acceleration_std_mps2 <= 0.0 ||
        parameters_.measurement_std_xy_m <= 0.0 || parameters_.measurement_std_z_m <= 0.0 ||
        parameters_.innovation_gate_chi2 <= 0.0 || parameters_.roi_measurement_std_px <= 0.0 ||
        parameters_.roi_innovation_gate_chi2 <= 0.0) {
        throw std::invalid_argument("World target tracker parameters are invalid.");
    }
    filter_.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
    for (int axis = 0; axis < 3; ++axis) {
        filter_.measurementMatrix.at<double>(axis, axis) = 1.0;
    }
    filter_.measurementNoiseCov = cv::Mat::zeros(3, 3, CV_64F);
    filter_.measurementNoiseCov.at<double>(0, 0) =
        parameters_.measurement_std_xy_m * parameters_.measurement_std_xy_m;
    filter_.measurementNoiseCov.at<double>(1, 1) =
        parameters_.measurement_std_xy_m * parameters_.measurement_std_xy_m;
    filter_.measurementNoiseCov.at<double>(2, 2) =
        parameters_.measurement_std_z_m * parameters_.measurement_std_z_m;
}

void WorldTargetTracker::predict_to(std::chrono::steady_clock::time_point timestamp) {
    if (!initialized_ || !state_timestamp_.has_value()) {
        return;
    }
    const double dt = std::chrono::duration<double>(timestamp - state_timestamp_.value()).count();
    if (dt <= 0.0) {
        return;
    }
    configure_prediction(dt);
    filter_.statePost = filter_.predict();
    filter_.errorCovPost = filter_.errorCovPre.clone();
    state_timestamp_ = timestamp;
}

bool WorldTargetTracker::correct_linear(const cv::Mat& measurement, const cv::Mat& measurement_matrix,
                                        const cv::Mat& measurement_noise_covariance,
                                        double innovation_gate_chi2, double* innovation_chi2) {
    const cv::Mat innovation = measurement - measurement_matrix * filter_.statePost;
    const cv::Mat innovation_covariance = measurement_matrix * filter_.errorCovPost *
                                          measurement_matrix.t() + measurement_noise_covariance;
    cv::Mat inverse_innovation_covariance;
    if (cv::invert(innovation_covariance, inverse_innovation_covariance, cv::DECOMP_SVD) == 0.0) {
        return false;
    }
    const cv::Mat chi2 = innovation.t() * inverse_innovation_covariance * innovation;
    *innovation_chi2 = chi2.at<double>(0);
    if (!std::isfinite(*innovation_chi2) || *innovation_chi2 > innovation_gate_chi2) {
        return false;
    }

    const cv::Mat gain = filter_.errorCovPost * measurement_matrix.t() * inverse_innovation_covariance;
    const cv::Mat identity = cv::Mat::eye(6, 6, CV_64F);
    const cv::Mat residual_transform = identity - gain * measurement_matrix;
    filter_.statePost += gain * innovation;
    filter_.errorCovPost = residual_transform * filter_.errorCovPost * residual_transform.t() +
                           gain * measurement_noise_covariance * gain.t();
    return true;
}

void WorldTargetTracker::configure_prediction(double seconds) {
    const double dt = std::clamp(seconds, 1e-4, 1.0);
    filter_.transitionMatrix = cv::Mat::eye(6, 6, CV_64F);
    filter_.transitionMatrix.at<double>(0, 3) = dt;
    filter_.transitionMatrix.at<double>(1, 4) = dt;
    filter_.transitionMatrix.at<double>(2, 5) = dt;

    const double acceleration_variance = parameters_.acceleration_std_mps2 * parameters_.acceleration_std_mps2;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    filter_.processNoiseCov = cv::Mat::zeros(6, 6, CV_64F);
    for (int axis = 0; axis < 3; ++axis) {
        filter_.processNoiseCov.at<double>(axis, axis) = acceleration_variance * dt4 * 0.25;
        filter_.processNoiseCov.at<double>(axis, axis + 3) = acceleration_variance * dt3 * 0.5;
        filter_.processNoiseCov.at<double>(axis + 3, axis) = acceleration_variance * dt3 * 0.5;
        filter_.processNoiseCov.at<double>(axis + 3, axis + 3) = acceleration_variance * dt2;
    }
}

cv::Point3d WorldTargetTracker::state_position(const cv::Mat& state) const {
    return {state.at<double>(0), state.at<double>(1), state.at<double>(2)};
}

cv::Point3d WorldTargetTracker::state_velocity(const cv::Mat& state) const {
    return {state.at<double>(3), state.at<double>(4), state.at<double>(5)};
}

WorldTargetTrackerResult WorldTargetTracker::state_result(TargetTrackerUpdate update) const {
    WorldTargetTrackerResult result;
    result.update = update;
    if (initialized_) {
        result.estimate_in_world_m = state_position(filter_.statePost);
        result.velocity_in_world_mps = state_velocity(filter_.statePost);
    }
    return result;
}

bool WorldTargetTracker::initialize_from_samples(WorldTargetTrackerResult* result) {
    if (static_cast<int>(initialization_samples_.size()) < parameters_.initialization_frames) {
        return false;
    }
    const auto& first = initialization_samples_.front();
    const auto& last = initialization_samples_.back();
    const double time_span_seconds =
        std::chrono::duration<double>(last.timestamp - first.timestamp).count();
    if (time_span_seconds <= 1e-4) {
        return false;
    }

    double mean_time = 0.0;
    for (const InitializationSample& sample : initialization_samples_) {
        mean_time += std::chrono::duration<double>(sample.timestamp - first.timestamp).count();
    }
    mean_time /= static_cast<double>(initialization_samples_.size());
    double centered_time_square_sum = 0.0;
    for (const InitializationSample& sample : initialization_samples_) {
        const double centered_time =
            std::chrono::duration<double>(sample.timestamp - first.timestamp).count() - mean_time;
        centered_time_square_sum += centered_time * centered_time;
    }
    if (centered_time_square_sum <= 1e-8) {
        return false;
    }

    cv::Point3d mean_position{};
    for (const InitializationSample& sample : initialization_samples_) {
        mean_position += sample.position_in_world_m;
    }
    mean_position *= 1.0 / static_cast<double>(initialization_samples_.size());
    cv::Point3d velocity{};
    for (const InitializationSample& sample : initialization_samples_) {
        const double centered_time =
            std::chrono::duration<double>(sample.timestamp - first.timestamp).count() - mean_time;
        velocity += (sample.position_in_world_m - mean_position) * centered_time;
    }
    velocity *= 1.0 / centered_time_square_sum;
    if (cv::norm(velocity) > parameters_.initialization_max_speed_mps) {
        initialization_samples_ = {last};
        result->initialization_sample_count = 1;
        return false;
    }

    const double last_time = std::chrono::duration<double>(last.timestamp - first.timestamp).count();
    const cv::Point3d position_at_last = mean_position + velocity * (last_time - mean_time);
    for (const InitializationSample& sample : initialization_samples_) {
        const double sample_time =
            std::chrono::duration<double>(sample.timestamp - first.timestamp).count();
        const cv::Point3d expected_position = mean_position + velocity * (sample_time - mean_time);
        if (cv::norm(sample.position_in_world_m - expected_position) > parameters_.initialization_max_spread_m) {
            initialization_samples_ = {last};
            result->initialization_sample_count = 1;
            return false;
        }
    }

    filter_.statePost = cv::Mat::zeros(6, 1, CV_64F);
    filter_.statePost.at<double>(0) = position_at_last.x;
    filter_.statePost.at<double>(1) = position_at_last.y;
    filter_.statePost.at<double>(2) = position_at_last.z;
    filter_.statePost.at<double>(3) = velocity.x;
    filter_.statePost.at<double>(4) = velocity.y;
    filter_.statePost.at<double>(5) = velocity.z;
    filter_.errorCovPost = cv::Mat::zeros(6, 6, CV_64F);
    const cv::Mat position_covariance = covariance_to_mat(last.covariance_in_world_m2);
    position_covariance.copyTo(filter_.errorCovPost(cv::Rect(0, 0, 3, 3)));
    const double velocity_variance =
        parameters_.initialization_velocity_std_mps * parameters_.initialization_velocity_std_mps;
    for (int axis = 0; axis < 3; ++axis) {
        filter_.errorCovPost.at<double>(axis + 3, axis + 3) = velocity_variance;
    }
    initialized_ = true;
    state_timestamp_ = last.timestamp;
    last_measurement_timestamp_ = last.timestamp;
    initialization_samples_.clear();
    result->update = TargetTrackerUpdate::Accepted;
    result->estimate_in_world_m = position_at_last;
    result->velocity_in_world_mps = velocity;
    return true;
}

WorldTargetTrackerResult WorldTargetTracker::update_position(
    const cv::Point3d& measurement_in_world_m, const cv::Matx33d& covariance_in_world_m2,
    std::chrono::steady_clock::time_point timestamp) {
    if (!finite_point(measurement_in_world_m) || !valid_covariance(covariance_in_world_m2)) {
        throw std::invalid_argument("World target tracker measurement is invalid.");
    }
    if (!parameters_.enabled) {
        WorldTargetTrackerResult result;
        result.update = TargetTrackerUpdate::Disabled;
        result.estimate_in_world_m = measurement_in_world_m;
        return result;
    }

    if (initialized_ && last_measurement_timestamp_.has_value()) {
        const double measurement_gap_seconds =
            std::chrono::duration<double>(timestamp - last_measurement_timestamp_.value()).count();
        if (measurement_gap_seconds < 0.0) {
            return state_result(TargetTrackerUpdate::Rejected);
        }
        if (measurement_gap_seconds > static_cast<double>(parameters_.max_coast_ms) / 1000.0) {
            reset();
        }
    }

    WorldTargetTrackerResult result;
    if (!initialized_) {
        initialization_samples_.push_back({measurement_in_world_m, timestamp, covariance_in_world_m2});
        result.initialization_sample_count = static_cast<int>(initialization_samples_.size());
        if (!initialize_from_samples(&result)) {
            result.update = TargetTrackerUpdate::Initializing;
            return result;
        }
        return result;
    }

    predict_to(timestamp);
    cv::Mat measurement(3, 1, CV_64F);
    measurement.at<double>(0) = measurement_in_world_m.x;
    measurement.at<double>(1) = measurement_in_world_m.y;
    measurement.at<double>(2) = measurement_in_world_m.z;
    if (!correct_linear(measurement, filter_.measurementMatrix, covariance_to_mat(covariance_in_world_m2),
                        parameters_.innovation_gate_chi2, &result.innovation_chi2)) {
        result = state_result(TargetTrackerUpdate::Rejected);
        return result;
    }

    last_measurement_timestamp_ = timestamp;
    result = state_result(TargetTrackerUpdate::Accepted);
    return result;
}

WorldTargetTrackerResult WorldTargetTracker::update(const WorldTargetObservation& observation) {
    return update_position(observation.position_in_world_m, observation.covariance_in_world_m2,
                           observation.exposure_timestamp);
}

WorldTargetTrackerResult WorldTargetTracker::update(const cv::Point3d& measurement_in_world_m,
                                                     std::chrono::steady_clock::time_point timestamp) {
    cv::Matx33d covariance = cv::Matx33d::zeros();
    covariance(0, 0) = parameters_.measurement_std_xy_m * parameters_.measurement_std_xy_m;
    covariance(1, 1) = parameters_.measurement_std_xy_m * parameters_.measurement_std_xy_m;
    covariance(2, 2) = parameters_.measurement_std_z_m * parameters_.measurement_std_z_m;
    return update_position(measurement_in_world_m, covariance, timestamp);
}

WorldTargetTrackerResult WorldTargetTracker::coast_to(std::chrono::steady_clock::time_point timestamp) {
    if (!parameters_.enabled) {
        return state_result(TargetTrackerUpdate::Disabled);
    }
    if (!initialized_ || !last_measurement_timestamp_.has_value()) {
        if (!initialization_samples_.empty() &&
            timestamp - initialization_samples_.back().timestamp >
                std::chrono::milliseconds(parameters_.max_coast_ms)) {
            initialization_samples_.clear();
            WorldTargetTrackerResult result;
            result.update = TargetTrackerUpdate::Lost;
            return result;
        }
        WorldTargetTrackerResult result;
        result.update = TargetTrackerUpdate::Initializing;
        result.initialization_sample_count = static_cast<int>(initialization_samples_.size());
        return result;
    }
    const double coasting_age_seconds =
        std::chrono::duration<double>(timestamp - last_measurement_timestamp_.value()).count();
    if (coasting_age_seconds < 0.0) {
        return state_result(TargetTrackerUpdate::Rejected);
    }
    if (coasting_age_seconds > static_cast<double>(parameters_.max_coast_ms) / 1000.0) {
        reset();
        WorldTargetTrackerResult result;
        result.update = TargetTrackerUpdate::Lost;
        return result;
    }
    predict_to(timestamp);
    WorldTargetTrackerResult result = state_result(TargetTrackerUpdate::Coasting);
    result.coasting_age_seconds = coasting_age_seconds;
    return result;
}

WorldTargetTrackerResult WorldTargetTracker::update_image(
    const WorldTargetImageObservation& observation, std::chrono::steady_clock::time_point timestamp) {
    WorldTargetTrackerResult result;
    result.image_measurement_attempted = parameters_.enabled && parameters_.roi_measurement_enabled;
    if (!parameters_.enabled || !parameters_.roi_measurement_enabled) {
        result.update = TargetTrackerUpdate::Disabled;
        return result;
    }
    if (!finite_pixel(observation.pixel)) {
        throw std::invalid_argument("World target tracker image measurement must be finite.");
    }
    if (!initialized_) {
        result.update = TargetTrackerUpdate::Initializing;
        return result;
    }

    predict_to(timestamp);
    const cv::Point3d predicted_position = state_position(filter_.statePost);
    const std::optional<cv::Point2d> projected = project_world_point(predicted_position, observation);
    if (!projected.has_value()) {
        result = state_result(TargetTrackerUpdate::Rejected);
        return result;
    }
    const double focal_length_px = 0.5 * (std::abs(observation.camera_matrix(0, 0)) +
                                          std::abs(observation.camera_matrix(1, 1)));
    if (!std::isfinite(focal_length_px) || focal_length_px <= 1e-6) {
        result = state_result(TargetTrackerUpdate::Rejected);
        return result;
    }

    const cv::Vec3d camera_point = observation.world_to_camera_rotation *
                                   cv::Vec3d(predicted_position.x, predicted_position.y,
                                             predicted_position.z) +
                                   cv::Vec3d(observation.world_to_camera_translation_m.x,
                                             observation.world_to_camera_translation_m.y,
                                             observation.world_to_camera_translation_m.z);
    const double depth_m = camera_point[2];
    if (!std::isfinite(depth_m) || depth_m <= 1e-6) {
        result = state_result(TargetTrackerUpdate::Rejected);
        return result;
    }

    cv::Mat measurement(2, 1, CV_64F);
    measurement.at<double>(0) = observation.pixel.x;
    measurement.at<double>(1) = observation.pixel.y;
    cv::Mat measurement_matrix = cv::Mat::zeros(2, 6, CV_64F);
    const cv::Matx33d rotation = observation.world_to_camera_rotation;
    for (int axis = 0; axis < 3; ++axis) {
        measurement_matrix.at<double>(0, axis) =
            focal_length_px / depth_m * (rotation(0, axis) - camera_point[0] / depth_m * rotation(2, axis));
        measurement_matrix.at<double>(1, axis) =
            focal_length_px / depth_m * (rotation(1, axis) - camera_point[1] / depth_m * rotation(2, axis));
    }
    cv::Mat projected_measurement(2, 1, CV_64F);
    projected_measurement.at<double>(0) = projected->x;
    projected_measurement.at<double>(1) = projected->y;
    const cv::Mat linearized_measurement =
        measurement - projected_measurement + measurement_matrix * filter_.statePost;
    cv::Mat measurement_noise_covariance = cv::Mat::eye(2, 2, CV_64F) *
                                           parameters_.roi_measurement_std_px * parameters_.roi_measurement_std_px;
    if (!correct_linear(linearized_measurement, measurement_matrix, measurement_noise_covariance,
                        parameters_.roi_innovation_gate_chi2, &result.image_innovation_chi2)) {
        result = state_result(TargetTrackerUpdate::Rejected);
        result.image_measurement_attempted = true;
        return result;
    }

    result = state_result(TargetTrackerUpdate::Accepted);
    result.image_measurement_attempted = true;
    result.image_measurement_accepted = true;
    return result;
}

std::optional<cv::Point3d> WorldTargetTracker::predict_ahead(double seconds) const {
    if (!initialized_ || !std::isfinite(seconds) || seconds < 0.0 ||
        seconds > parameters_.max_prediction_seconds) {
        return std::nullopt;
    }
    cv::Mat transition = cv::Mat::eye(6, 6, CV_64F);
    transition.at<double>(0, 3) = seconds;
    transition.at<double>(1, 4) = seconds;
    transition.at<double>(2, 5) = seconds;
    return state_position(transition * filter_.statePost);
}

bool WorldTargetTracker::has_estimate() const noexcept {
    return initialized_;
}

void WorldTargetTracker::reset() {
    initialized_ = false;
    state_timestamp_.reset();
    last_measurement_timestamp_.reset();
    initialization_samples_.clear();
    filter_.statePost = cv::Mat::zeros(6, 1, CV_64F);
    filter_.errorCovPost = cv::Mat::eye(6, 6, CV_64F);
}

}  // namespace wit_radar
