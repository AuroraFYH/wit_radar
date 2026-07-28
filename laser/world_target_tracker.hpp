#pragma once

#include <chrono>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "world_target_observation.hpp"

namespace wit_radar {

enum class TargetTrackerUpdate {
    Disabled,
    Initializing,
    Accepted,
    Rejected,
    Coasting,
    Lost,
};

struct WorldTargetTrackerParameters {
    bool enabled = false;
    int initialization_frames = 3;
    double initialization_max_spread_m = 0.15;
    double initialization_max_speed_mps = 12.0;
    double initialization_velocity_std_mps = 2.0;
    int max_coast_ms = 200;
    double max_prediction_seconds = 0.5;
    double acceleration_std_mps2 = 3.0;
    double measurement_std_xy_m = 0.03;
    double measurement_std_z_m = 0.15;
    double innovation_gate_chi2 = 16.27;
    bool roi_measurement_enabled = false;
    double roi_measurement_std_px = 20.0;
    double roi_innovation_gate_chi2 = 9.21;
};

struct WorldTargetImageObservation {
    cv::Point2d pixel;
    cv::Matx33d camera_matrix;
    cv::Vec<double, 5> distortion_coefficients;
    cv::Matx33d world_to_camera_rotation = cv::Matx33d::eye();
    cv::Point3d world_to_camera_translation_m;
};

struct WorldTargetTrackerResult {
    TargetTrackerUpdate update = TargetTrackerUpdate::Disabled;
    std::optional<cv::Point3d> estimate_in_world_m;
    std::optional<cv::Point3d> velocity_in_world_mps;
    double innovation_chi2 = 0.0;
    int initialization_sample_count = 0;
    double coasting_age_seconds = 0.0;
    bool image_measurement_attempted = false;
    bool image_measurement_accepted = false;
    double image_innovation_chi2 = 0.0;
};

class WorldTargetTracker {
public:
    explicit WorldTargetTracker(WorldTargetTrackerParameters parameters = {});

    WorldTargetTrackerResult update(const WorldTargetObservation& observation);
    WorldTargetTrackerResult update(const cv::Point3d& measurement_in_world_m,
                                    std::chrono::steady_clock::time_point timestamp);
    WorldTargetTrackerResult coast_to(std::chrono::steady_clock::time_point timestamp);
    WorldTargetTrackerResult update_image(const WorldTargetImageObservation& observation,
                                          std::chrono::steady_clock::time_point timestamp);
    std::optional<cv::Point3d> predict_ahead(double seconds) const;
    bool has_estimate() const noexcept;
    void reset();

private:
    struct InitializationSample {
        cv::Point3d position_in_world_m;
        std::chrono::steady_clock::time_point timestamp;
        cv::Matx33d covariance_in_world_m2;
    };

    void configure_prediction(double seconds);
    void predict_to(std::chrono::steady_clock::time_point timestamp);
    WorldTargetTrackerResult update_position(const cv::Point3d& measurement_in_world_m,
                                             const cv::Matx33d& covariance_in_world_m2,
                                             std::chrono::steady_clock::time_point timestamp);
    bool initialize_from_samples(WorldTargetTrackerResult* result);
    bool correct_linear(const cv::Mat& measurement, const cv::Mat& measurement_matrix,
                        const cv::Mat& measurement_noise_covariance, double innovation_gate_chi2,
                        double* innovation_chi2);
    WorldTargetTrackerResult state_result(TargetTrackerUpdate update) const;
    cv::Point3d state_position(const cv::Mat& state) const;
    cv::Point3d state_velocity(const cv::Mat& state) const;

    WorldTargetTrackerParameters parameters_;
    cv::KalmanFilter filter_{6, 3, 0, CV_64F};
    bool initialized_ = false;
    std::optional<std::chrono::steady_clock::time_point> state_timestamp_;
    std::optional<std::chrono::steady_clock::time_point> last_measurement_timestamp_;
    std::vector<InitializationSample> initialization_samples_;
};

}  // namespace wit_radar
