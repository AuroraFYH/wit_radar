#include <chrono>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/asio/serial_port_base.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "aim_solver.hpp"
#include "cylinder_pose_solver.hpp"
#include "gimbal_arrival_time.hpp"
#include "hik_camera.hpp"
#include "keypoint_center_finder.hpp"
#include "laser_detector.hpp"
#include "robot_communicator.hpp"
#include "static_aim_compensation.hpp"
#include "world_target_observation.hpp"
#include "world_target_tracker.hpp"

namespace {

using wit_radar::communication::SerialParameters;

constexpr float kMaximumAbsoluteYawDegrees = 720.0F;
constexpr float kMaximumAbsolutePitchDegrees = 180.0F;

float normalize_degrees(float angle) {
    return std::remainder(angle, 360.0F);
}

bool valid_command_angles(const wit_radar::communication::GimbalCommand& command) {
    return std::isfinite(command.yaw) && std::isfinite(command.pitch) &&
           std::abs(command.yaw) <= kMaximumAbsoluteYawDegrees &&
           std::abs(command.pitch) <= kMaximumAbsolutePitchDegrees;
}

const char* tracker_update_name(wit_radar::TargetTrackerUpdate update) {
    switch (update) {
    case wit_radar::TargetTrackerUpdate::Disabled:
        return "disabled";
    case wit_radar::TargetTrackerUpdate::Initializing:
        return "initializing";
    case wit_radar::TargetTrackerUpdate::Accepted:
        return "accepted";
    case wit_radar::TargetTrackerUpdate::Rejected:
        return "rejected";
    case wit_radar::TargetTrackerUpdate::Coasting:
        return "coasting";
    case wit_radar::TargetTrackerUpdate::Lost:
        return "lost";
    }
    return "unknown";
}

double read_double(const cv::FileNode& config, const std::string& key, double fallback) {
    const cv::FileNode node = config[key];
    return node.empty() ? fallback : node.real();
}

int read_int(const cv::FileNode& config, const std::string& key, int fallback) {
    const cv::FileNode node = config[key];
    if (node.empty()) {
        return fallback;
    }
    int value = fallback;
    node >> value;
    return value;
}

bool read_bool(const cv::FileNode& config, const std::string& key, bool fallback) {
    const cv::FileNode node = config[key];
    if (node.empty()) {
        return fallback;
    }
    int value = fallback ? 1 : 0;
    node >> value;
    return value != 0;
}

std::string read_required_string(const cv::FileNode& config, const std::string& key,
                                 const std::string& config_path) {
    std::string value;
    config[key] >> value;
    if (value.empty()) {
        throw std::runtime_error("Missing " + key + " in configuration file: " + config_path);
    }
    return value;
}

std::string read_string(const cv::FileNode& config, const std::string& key,
                        const std::string& fallback = {}) {
    const cv::FileNode node = config[key];
    if (node.empty()) {
        return fallback;
    }
    std::string value;
    node >> value;
    return value.empty() ? fallback : value;
}

cv::Scalar read_hsv_range(const cv::FileNode& config, const std::string& key,
                          const cv::Scalar& fallback) {
    const cv::FileNode range = config[key];
    if (range.empty() || !range.isSeq() || range.size() != 3) {
        return fallback;
    }
    return {range[0].real(), range[1].real(), range[2].real()};
}

wit_radar::DeviceCenterParameters read_device_center_parameters(const cv::FileNode& config) {
    wit_radar::DeviceCenterParameters parameters;
    const cv::FileNode hsv = config["hsv"];
    const cv::FileNode components = config["components"];
    const cv::FileNode grouping = config["grouping"];
    parameters.lower_red_1 = read_hsv_range(hsv, "lower_red_1", parameters.lower_red_1);
    parameters.upper_red_1 = read_hsv_range(hsv, "upper_red_1", parameters.upper_red_1);
    parameters.lower_red_2 = read_hsv_range(hsv, "lower_red_2", parameters.lower_red_2);
    parameters.upper_red_2 = read_hsv_range(hsv, "upper_red_2", parameters.upper_red_2);
    parameters.enable_purple = read_bool(hsv, "enable_purple", parameters.enable_purple);
    parameters.lower_purple = read_hsv_range(hsv, "lower_purple", parameters.lower_purple);
    parameters.upper_purple = read_hsv_range(hsv, "upper_purple", parameters.upper_purple);
    parameters.close_kernel_size = read_int(config["morphology"], "close_kernel_size",
                                             parameters.close_kernel_size);
    parameters.min_area = read_int(components, "min_area", parameters.min_area);
    parameters.max_area_ratio =
        static_cast<float>(read_double(components, "max_area_ratio", parameters.max_area_ratio));
    parameters.min_side = static_cast<float>(read_double(components, "min_side", parameters.min_side));
    parameters.max_aspect_ratio =
        static_cast<float>(read_double(components, "max_aspect_ratio", parameters.max_aspect_ratio));
    parameters.min_fill_ratio =
        static_cast<float>(read_double(components, "min_fill_ratio", parameters.min_fill_ratio));
    parameters.link_factor =
        static_cast<float>(read_double(grouping, "link_factor", parameters.link_factor));
    parameters.min_group_points = read_int(grouping, "min_group_points", parameters.min_group_points);
    parameters.min_points_per_layer =
        read_int(grouping, "min_points_per_layer", parameters.min_points_per_layer);
    parameters.min_layer_separation_size_ratio = static_cast<float>(read_double(
        grouping, "min_layer_separation_size_ratio", parameters.min_layer_separation_size_ratio));
    parameters.min_separation_ratio =
        static_cast<float>(read_double(grouping, "min_separation_ratio", parameters.min_separation_ratio));
    parameters.max_horizontal_offset_ratio = static_cast<float>(read_double(
        grouping, "max_horizontal_offset_ratio", parameters.max_horizontal_offset_ratio));
    return parameters;
}

cv::Matx33d read_matrix(const cv::FileNode& config, const std::string& key) {
    const cv::FileNode values = config[key];
    if (values.empty() || values.size() != 9) {
        throw std::runtime_error(key + " must contain nine values.");
    }
    cv::Matx33d matrix;
    for (int index = 0; index < 9; ++index) {
        matrix(index / 3, index % 3) = values[index].real();
    }
    return matrix;
}

cv::Vec<double, 5> read_distortion(const cv::FileNode& config) {
    if (config.empty() || config.size() != 5) {
        throw std::runtime_error("camera.distortion_coefficients must contain five values.");
    }
    return {config[0].real(), config[1].real(), config[2].real(), config[3].real(), config[4].real()};
}

cv::Point3d read_point(const cv::FileNode& config, const std::string& key) {
    const cv::FileNode values = config[key];
    if (values.empty() || values.size() != 3) {
        throw std::runtime_error(key + " must contain three values.");
    }
    return {values[0].real(), values[1].real(), values[2].real()};
}

cv::Point3d apply_rotation(const cv::Matx33d& rotation, const cv::Point3d& point) {
    const cv::Vec3d transformed = rotation * cv::Vec3d(point.x, point.y, point.z);
    return {transformed[0], transformed[1], transformed[2]};
}

bool finite_point(const cv::Point3d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

std::optional<cv::Point3d> normalized(const cv::Point3d& point) {
    const double length = cv::norm(point);
    if (!finite_point(point) || length < 1e-9) {
        return std::nullopt;
    }
    return point * (1.0 / length);
}

std::optional<cv::Point2f> project_camera_point(const cv::Point3d& point,
                                                 const wit_radar::CylinderPoseParameters& parameters) {
    if (!finite_point(point) || point.z <= 0.0) {
        return std::nullopt;
    }
    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(std::vector<cv::Point3d>{point}, cv::Vec3d{}, cv::Vec3d{},
                      parameters.camera_matrix, parameters.distortion_coefficients, projected_points);
    if (projected_points.size() != 1) {
        return std::nullopt;
    }
    return cv::Point2f(static_cast<float>(projected_points.front().x),
                       static_cast<float>(projected_points.front().y));
}

void draw_projected_point(cv::Mat& image, const std::optional<cv::Point2f>& projected_point,
                          const cv::Point& roi_offset, const cv::Scalar& color,
                          const std::string& label) {
    if (!projected_point.has_value()) {
        return;
    }
    const cv::Point point(cvRound(projected_point->x) + roi_offset.x,
                          cvRound(projected_point->y) + roi_offset.y);
    cv::drawMarker(image, point, color, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    cv::circle(image, point, 5, color, 1, cv::LINE_AA);
    cv::putText(image, label, point + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                cv::LINE_AA);
}

boost::asio::serial_port_base::parity::type parse_parity(const std::string& value) {
    using Parity = boost::asio::serial_port_base::parity;
    if (value == "none") return Parity::none;
    if (value == "odd") return Parity::odd;
    if (value == "even") return Parity::even;
    throw std::runtime_error("Unsupported serial parity: " + value);
}

boost::asio::serial_port_base::stop_bits::type parse_stop_bits(const std::string& value) {
    using StopBits = boost::asio::serial_port_base::stop_bits;
    if (value == "one") return StopBits::one;
    if (value == "one_point_five") return StopBits::onepointfive;
    if (value == "two") return StopBits::two;
    throw std::runtime_error("Unsupported serial stop_bits: " + value);
}

boost::asio::serial_port_base::flow_control::type parse_flow_control(const std::string& value) {
    using FlowControl = boost::asio::serial_port_base::flow_control;
    if (value == "none") return FlowControl::none;
    if (value == "software") return FlowControl::software;
    if (value == "hardware") return FlowControl::hardware;
    throw std::runtime_error("Unsupported serial flow_control: " + value);
}

SerialParameters read_serial_parameters(const cv::FileNode& config, const std::string& config_path) {
    SerialParameters parameters;
    parameters.device_name = read_required_string(config, "device_name", config_path);
    parameters.baud_rate = static_cast<unsigned int>(read_int(config, "baud_rate", 115200));
    parameters.char_size = static_cast<unsigned int>(read_int(config, "char_size", 8));
    parameters.read_buffer_size = static_cast<std::size_t>(read_int(config, "read_buffer_size", 4096));
    parameters.parity = parse_parity(read_required_string(config, "parity", config_path));
    parameters.stop_bits = parse_stop_bits(read_required_string(config, "stop_bits", config_path));
    parameters.flow_control = parse_flow_control(read_required_string(config, "flow_control", config_path));
    return parameters;
}

void draw_marker(cv::Mat& image, const wit_radar::MarkerPose& marker, const cv::Point& offset) {
    std::vector<cv::Point> corners;
    corners.reserve(marker.corners.size());
    for (const cv::Point2f& corner : marker.corners) {
        corners.emplace_back(cvRound(corner.x) + offset.x, cvRound(corner.y) + offset.y);
    }
    const cv::Scalar color = marker.upper_ring ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 0);
    cv::polylines(image, corners, true, color, 2);
}

void draw_pnp_input_rectangle(cv::Mat& image, const cv::RotatedRect& rectangle, const cv::Point& offset) {
    std::array<cv::Point2f, 4> corners{};
    rectangle.points(corners.data());
    std::vector<cv::Point> image_corners;
    image_corners.reserve(corners.size());
    for (const cv::Point2f& corner : corners) {
        image_corners.emplace_back(cvRound(corner.x) + offset.x, cvRound(corner.y) + offset.y);
    }
    cv::polylines(image, image_corners, true, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 3) {
            throw std::runtime_error("Usage: laser_aim [config_path] [--send]");
        }
        const std::string config_path = argc > 1 ? argv[1] : "config/laser.json";
        const bool send_commands = argc == 3 && std::string(argv[2]) == "--send";
        if (argc == 3 && !send_commands) {
            throw std::runtime_error("Usage: laser_aim [config_path] [--send]");
        }

        cv::FileStorage config(config_path, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }
        const cv::FileNode camera_config = config["camera"];
        const cv::FileNode detector_config = config["detector"];
        const cv::FileNode pnp_config = config["pnp"];
        const cv::FileNode handeye_config = config["handeye"];
        const cv::FileNode laser_config = config["laser"];
        const cv::FileNode communication_config = config["communication"];
        const cv::FileNode aim_config = config["aim"];
        const cv::FileNode keypoint_center_config = config["keypoint_center"];
        const cv::FileNode tracking_config = config["tracking"];
        if (camera_config.empty() || detector_config.empty() || pnp_config.empty() || handeye_config.empty() ||
            laser_config.empty() || communication_config.empty()) {
            throw std::runtime_error("Missing required section in configuration file: " + config_path);
        }

        std::string serial_number;
        camera_config["sn"] >> serial_number;
        if (serial_number.empty()) {
            throw std::runtime_error("Missing camera.sn in configuration file: " + config_path);
        }
        const int image_width = read_int(camera_config, "image_width", 0);
        const int image_height = read_int(camera_config, "image_height", 0);
        const cv::Matx33d camera_matrix = read_matrix(camera_config, "camera_matrix");
        const cv::Vec<double, 5> distortion = read_distortion(camera_config["distortion_coefficients"]);
        const cv::Matx33d camera_to_gimbal = read_matrix(handeye_config, "R_camera2gimbal");
        const cv::Point3d camera_in_gimbal = read_point(handeye_config, "t_camera2gimbal_m");
        const cv::Point3d laser_in_camera = read_point(laser_config, "t_laser_in_camera_m");
        const cv::FileNode beam_line_config = laser_config["beam_line_in_camera"];
        const bool use_calibrated_beam_line = read_bool(beam_line_config, "enabled", false);
        cv::Point3d beam_point_in_camera = laser_in_camera;
        cv::Point3d beam_direction_in_camera{0.0, 0.0, 1.0};
        if (use_calibrated_beam_line) {
            beam_point_in_camera = read_point(beam_line_config, "point_m");
            const std::optional<cv::Point3d> direction =
                normalized(read_point(beam_line_config, "direction"));
            if (!direction.has_value()) {
                throw std::runtime_error("laser.beam_line_in_camera.direction must be non-zero.");
            }
            beam_direction_in_camera = direction.value();
        }

        wit_radar::CylinderPoseParameters pose_parameters;
        pose_parameters.camera_matrix = camera_matrix;
        pose_parameters.distortion_coefficients = distortion;
        pose_parameters.light_parameters = read_device_center_parameters(config["device_center"]);
        pose_parameters.light_square_width_m = read_double(pnp_config, "light_square_width_m", 0.0);
        pose_parameters.light_square_height_m = read_double(pnp_config, "light_square_height_m", 0.0);
        pose_parameters.emitting_face_radius_m = read_double(pnp_config, "emitting_face_radius_m", 0.0);
        pose_parameters.ring_center_separation_m = read_double(pnp_config, "ring_center_separation_m", 0.0);
        pose_parameters.min_markers_per_row = read_int(aim_config, "min_markers_per_row", 1);
        pose_parameters.max_reprojection_error_px =
            read_double(aim_config, "max_reprojection_error_px", 3.0);
        pose_parameters.max_ring_separation_error_ratio =
            read_double(aim_config, "max_ring_separation_error_ratio", 0.6);
        pose_parameters.max_marker_axis_residual_m =
            read_double(aim_config, "max_marker_axis_residual_m", 0.10);
        pose_parameters.use_global_cylinder_pnp =
            read_bool(aim_config, "use_global_cylinder_pnp", true);
        pose_parameters.allow_legacy_ippe_fallback =
            read_bool(aim_config, "allow_legacy_ippe_fallback", false);
        pose_parameters.markers_per_ring = read_int(pnp_config, "markers_per_ring", 8);
        pose_parameters.ring_alignment_deg = read_double(pnp_config, "ring_alignment_deg", 0.0);
        pose_parameters.max_global_reprojection_error_px =
            read_double(aim_config, "max_global_reprojection_error_px", 3.0);
        const double max_yaw_correction = read_double(aim_config, "max_yaw_correction_deg", 10.0);
        const double max_pitch_correction = read_double(aim_config, "max_pitch_correction_deg", 10.0);
        const double world_yaw_feedback_sign = read_double(aim_config, "world_yaw_feedback_sign", 1.0);
        const double world_pitch_feedback_sign =
            read_double(aim_config, "world_pitch_feedback_sign", -1.0);
        if (!std::isfinite(world_yaw_feedback_sign) || !std::isfinite(world_pitch_feedback_sign) ||
            std::abs(world_yaw_feedback_sign) < 1e-6 || std::abs(world_pitch_feedback_sign) < 1e-6) {
            throw std::runtime_error("aim world feedback signs must be finite and non-zero.");
        }
        wit_radar::StaticAimCompensationParameters static_compensation_parameters;
        const cv::FileNode static_compensation_config = aim_config["static_compensation"];
        static_compensation_parameters.enabled =
            read_bool(static_compensation_config, "enabled", static_compensation_parameters.enabled);
        static_compensation_parameters.range_min_m =
            read_double(static_compensation_config, "range_min_m", static_compensation_parameters.range_min_m);
        static_compensation_parameters.range_max_m =
            read_double(static_compensation_config, "range_max_m", static_compensation_parameters.range_max_m);
        static_compensation_parameters.yaw_constant_deg = read_double(
            static_compensation_config, "yaw_constant_deg", static_compensation_parameters.yaw_constant_deg);
        static_compensation_parameters.yaw_inverse_range_deg_m =
            read_double(static_compensation_config, "yaw_inverse_range_deg_m",
                        static_compensation_parameters.yaw_inverse_range_deg_m);
        static_compensation_parameters.pitch_constant_deg = read_double(
            static_compensation_config, "pitch_constant_deg", static_compensation_parameters.pitch_constant_deg);
        if (static_compensation_parameters.enabled &&
            (static_compensation_parameters.range_min_m <= 0.0 ||
             static_compensation_parameters.range_max_m < static_compensation_parameters.range_min_m ||
             !std::isfinite(static_compensation_parameters.yaw_constant_deg) ||
             !std::isfinite(static_compensation_parameters.yaw_inverse_range_deg_m) ||
             !std::isfinite(static_compensation_parameters.pitch_constant_deg))) {
            throw std::runtime_error("aim.static_compensation range is invalid.");
        }
        const int debug_log_every_n_frames = read_int(aim_config, "debug_log_every_n_frames", 30);
        if (debug_log_every_n_frames <= 0) {
            throw std::runtime_error("aim.debug_log_every_n_frames must be positive.");
        }
        wit_radar::CylinderPoseSolver pose_solver(pose_parameters);

        const std::string engine_path = read_required_string(detector_config, "engine", config_path);
        const float confidence_threshold =
            static_cast<float>(read_double(detector_config, "confidence_threshold", 0.25));
        const float nms_threshold = static_cast<float>(read_double(detector_config, "nms_threshold", 0.45));
        const float min_content_overlap =
            static_cast<float>(read_double(detector_config, "min_content_overlap", 0.70));
        const bool detector_debug = read_bool(detector_config, "debug", false);
        const int detector_debug_every_n_frames =
            read_int(detector_config, "debug_every_n_frames", 1);
        if (detector_debug_every_n_frames <= 0) {
            throw std::runtime_error("detector.debug_every_n_frames must be positive.");
        }
        const bool traditional_fallback_when_yolo_missing =
            read_bool(aim_config, "traditional_fallback_when_yolo_missing", false);
        wit_radar::LaserDetector detector(engine_path, confidence_threshold, nms_threshold,
                                          min_content_overlap, detector_debug,
                                          static_cast<unsigned int>(detector_debug_every_n_frames));

        wit_radar::KeypointCenterParameters keypoint_parameters;
        const std::string keypoint_mode = read_string(keypoint_center_config, "mode", "off");
        keypoint_parameters.enabled = read_bool(keypoint_center_config, "enabled", false);
        if (keypoint_mode == "onnx_heatmap") {
            keypoint_parameters.enabled = true;
        } else if (keypoint_mode != "off" && keypoint_mode != "yolo_roi_center") {
            throw std::runtime_error("keypoint_center.mode must be off, yolo_roi_center, or onnx_heatmap.");
        }
        keypoint_parameters.onnx_path = read_string(keypoint_center_config, "onnx");
        keypoint_parameters.input_size = {
            read_int(keypoint_center_config, "input_width", keypoint_parameters.input_size.width),
            read_int(keypoint_center_config, "input_height", keypoint_parameters.input_size.height)};
        keypoint_parameters.confidence_threshold = static_cast<float>(
            read_double(keypoint_center_config, "confidence_threshold", keypoint_parameters.confidence_threshold));
        const float keypoint_fusion_max_distance_px = static_cast<float>(
            read_double(keypoint_center_config, "fusion_max_distance_px", 8.0));
        const float keypoint_neural_weight = static_cast<float>(
            read_double(keypoint_center_config, "neural_weight", 0.5));
        if (keypoint_fusion_max_distance_px <= 0.0F || keypoint_neural_weight < 0.0F ||
            keypoint_neural_weight > 1.0F) {
            throw std::runtime_error("keypoint_center fusion parameters are invalid.");
        }
        std::unique_ptr<wit_radar::KeypointCenterFinder> keypoint_finder;
        if (keypoint_parameters.enabled) {
            keypoint_finder = std::make_unique<wit_radar::KeypointCenterFinder>(keypoint_parameters);
        }

        wit_radar::WorldTargetTrackerParameters tracker_parameters;
        tracker_parameters.enabled = read_bool(tracking_config, "enabled", false);
        tracker_parameters.initialization_frames =
            read_int(tracking_config, "initialization_frames", tracker_parameters.initialization_frames);
        tracker_parameters.initialization_max_spread_m = read_double(
            tracking_config, "initialization_max_spread_m", tracker_parameters.initialization_max_spread_m);
        tracker_parameters.initialization_max_speed_mps = read_double(
            tracking_config, "initialization_max_speed_mps", tracker_parameters.initialization_max_speed_mps);
        tracker_parameters.initialization_velocity_std_mps = read_double(
            tracking_config, "initialization_velocity_std_mps",
            tracker_parameters.initialization_velocity_std_mps);
        tracker_parameters.max_coast_ms =
            read_int(tracking_config, "max_coast_ms", tracker_parameters.max_coast_ms);
        tracker_parameters.max_prediction_seconds = read_double(
            tracking_config, "max_prediction_seconds", tracker_parameters.max_prediction_seconds);
        tracker_parameters.acceleration_std_mps2 =
            read_double(tracking_config, "acceleration_std_mps2", tracker_parameters.acceleration_std_mps2);
        tracker_parameters.measurement_std_xy_m =
            read_double(tracking_config, "measurement_std_xy_m", tracker_parameters.measurement_std_xy_m);
        tracker_parameters.measurement_std_z_m =
            read_double(tracking_config, "measurement_std_z_m", tracker_parameters.measurement_std_z_m);
        tracker_parameters.innovation_gate_chi2 =
            read_double(tracking_config, "innovation_gate_chi2", tracker_parameters.innovation_gate_chi2);
        tracker_parameters.roi_measurement_enabled =
            read_bool(tracking_config, "roi_measurement_enabled", tracker_parameters.roi_measurement_enabled);
        tracker_parameters.roi_measurement_std_px =
            read_double(tracking_config, "roi_measurement_std_px", tracker_parameters.roi_measurement_std_px);
        tracker_parameters.roi_innovation_gate_chi2 = read_double(
            tracking_config, "roi_innovation_gate_chi2", tracker_parameters.roi_innovation_gate_chi2);
        const int tracker_control_latency_ms = read_int(tracking_config, "control_latency_ms", 0);
        const int camera_frame_latency_ms = read_int(tracking_config, "camera_frame_latency_ms", 0);
        const int gimbal_state_receive_latency_ms =
            read_int(tracking_config, "gimbal_state_receive_latency_ms", 0);
        const int state_interpolation_max_gap_ms =
            read_int(tracking_config, "state_interpolation_max_gap_ms", 30);
        const int state_nearest_max_offset_ms =
            read_int(tracking_config, "state_nearest_max_offset_ms", 15);
        if (tracker_control_latency_ms < 0 || camera_frame_latency_ms < 0 ||
            gimbal_state_receive_latency_ms < 0 || state_interpolation_max_gap_ms < 0 ||
            state_nearest_max_offset_ms < 0) {
            throw std::runtime_error("tracking.control_latency_ms must not be negative.");
        }
        const cv::FileNode gimbal_motion_config = tracking_config["gimbal_motion_prediction"];
        wit_radar::communication::GimbalArrivalTimeParameters gimbal_motion_parameters;
        gimbal_motion_parameters.enabled =
            read_bool(gimbal_motion_config, "enabled", gimbal_motion_parameters.enabled);
        gimbal_motion_parameters.yaw.max_speed_deg_s = read_double(
            gimbal_motion_config, "yaw_max_speed_deg_s", gimbal_motion_parameters.yaw.max_speed_deg_s);
        gimbal_motion_parameters.yaw.max_acceleration_deg_s2 = read_double(
            gimbal_motion_config, "yaw_max_acceleration_deg_s2",
            gimbal_motion_parameters.yaw.max_acceleration_deg_s2);
        gimbal_motion_parameters.pitch.max_speed_deg_s = read_double(
            gimbal_motion_config, "pitch_max_speed_deg_s", gimbal_motion_parameters.pitch.max_speed_deg_s);
        gimbal_motion_parameters.pitch.max_acceleration_deg_s2 = read_double(
            gimbal_motion_config, "pitch_max_acceleration_deg_s2",
            gimbal_motion_parameters.pitch.max_acceleration_deg_s2);
        gimbal_motion_parameters.settle_margin_ms =
            read_double(gimbal_motion_config, "settle_margin_ms", gimbal_motion_parameters.settle_margin_ms);
        gimbal_motion_parameters.prediction_iterations = read_int(
            gimbal_motion_config, "prediction_iterations", gimbal_motion_parameters.prediction_iterations);
        gimbal_motion_parameters.convergence_ms =
            read_double(gimbal_motion_config, "convergence_ms", gimbal_motion_parameters.convergence_ms);
        wit_radar::communication::GimbalArrivalTimeEstimator gimbal_arrival_estimator(
            gimbal_motion_parameters);
        wit_radar::WorldTargetObservationParameters observation_parameters;
        observation_parameters.tangential_std_floor_m = read_double(
            tracking_config, "observation_tangential_std_floor_m",
            observation_parameters.tangential_std_floor_m);
        observation_parameters.radial_std_floor_m = read_double(
            tracking_config, "observation_radial_std_floor_m", observation_parameters.radial_std_floor_m);
        observation_parameters.reprojection_std_px_floor = read_double(
            tracking_config, "observation_reprojection_std_px_floor",
            observation_parameters.reprojection_std_px_floor);
        observation_parameters.radial_std_per_meter = read_double(
            tracking_config, "observation_radial_std_per_meter", observation_parameters.radial_std_per_meter);
        observation_parameters.axis_residual_std_scale = read_double(
            tracking_config, "observation_axis_residual_std_scale",
            observation_parameters.axis_residual_std_scale);
        observation_parameters.confidence_reprojection_error_px = read_double(
            tracking_config, "observation_confidence_reprojection_error_px",
            observation_parameters.confidence_reprojection_error_px);
        observation_parameters.confidence_axis_residual_m = read_double(
            tracking_config, "observation_confidence_axis_residual_m",
            observation_parameters.confidence_axis_residual_m);
        wit_radar::WorldTargetObservationBuilder observation_builder(observation_parameters);
        wit_radar::WorldTargetTracker world_target_tracker(tracker_parameters);

        const SerialParameters serial_parameters =
            read_serial_parameters(communication_config["serial"], config_path);
        const cv::FileNode command_config = communication_config["command"];
        wit_radar::communication::GimbalCommand command_template;
        command_template.yaw_speed = static_cast<float>(read_double(command_config, "yaw_speed", 0.0));
        command_template.pitch_speed = static_cast<float>(read_double(command_config, "pitch_speed", 0.0));
        command_template.yaw_acceleration =
            static_cast<float>(read_double(command_config, "yaw_acceleration", 0.0));
        command_template.pitch_acceleration =
            static_cast<float>(read_double(command_config, "pitch_acceleration", 0.0));
        const int send_interval_ms = read_int(command_config, "send_interval_ms", 100);
        if (send_interval_ms <= 0) {
            throw std::runtime_error("communication.command.send_interval_ms must be positive.");
        }
        const int state_timeout_ms = read_int(communication_config, "state_timeout_ms", 500);
        if (state_timeout_ms <= 0) {
            throw std::runtime_error("communication.state_timeout_ms must be positive.");
        }
        const int communication_log_interval_ms =
            read_int(communication_config, "log_interval_ms", 100);
        if (communication_log_interval_ms <= 0) {
            throw std::runtime_error("communication.log_interval_ms must be positive.");
        }

        std::mutex output_mutex;
        std::optional<std::chrono::steady_clock::time_point> last_receive_log_time;
        wit_radar::communication::RobotCommunicator communicator(
            serial_parameters, std::chrono::milliseconds(gimbal_state_receive_latency_ms));
        communicator.start([&output_mutex, &last_receive_log_time, communication_log_interval_ms](
            const wit_radar::communication::GimbalState& state) {
            const auto now = std::chrono::steady_clock::now();
            if (last_receive_log_time.has_value() &&
                now - last_receive_log_time.value() <
                    std::chrono::milliseconds(communication_log_interval_ms)) {
                return;
            }
            last_receive_log_time = now;
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "[COMM][RX] 0x02 yaw=" << state.angles.yaw
                      << " pitch=" << state.angles.pitch
                      << " timestamp_ms=" << state.device_timestamp_ms << '\n';
        });
        wit_radar::HikCamera camera;
        camera.open_by_serial_number(serial_number);

        const std::string window_name = "Laser Aim";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        std::cout << "Laser aim started in " << (send_commands ? "SEND" : "PREVIEW")
                  << " mode. Press Esc or q to exit.\n";
        std::cout << "[laser-aim] Beam model: "
                  << (use_calibrated_beam_line ? "calibrated red-laser line" : "legacy parallel fallback")
                  << '\n';
        if (keypoint_mode == "yolo_roi_center") {
            std::cout << "[laser-aim] Center comparison: YOLO ROI geometric center vs traditional center\n";
        } else if (keypoint_finder) {
            std::cout << "[laser-aim] Keypoint center model enabled: " << keypoint_parameters.onnx_path << '\n';
        }
        if (tracker_parameters.enabled) {
            std::cout << "[laser-aim] World-target EKF enabled: init="
                      << tracker_parameters.initialization_frames << " frames, gate="
                      << tracker_parameters.innovation_gate_chi2 << ", actuation latency="
                      << tracker_control_latency_ms << " ms, frame latency="
                      << camera_frame_latency_ms << " ms, state receive latency="
                      << gimbal_state_receive_latency_ms << " ms\n";
        }
        if (gimbal_motion_parameters.enabled) {
            std::cout << "[laser-aim] Gimbal motion prediction enabled: yaw=("
                      << gimbal_motion_parameters.yaw.max_speed_deg_s << " deg/s, "
                      << gimbal_motion_parameters.yaw.max_acceleration_deg_s2 << " deg/s^2) pitch=("
                      << gimbal_motion_parameters.pitch.max_speed_deg_s << " deg/s, "
                      << gimbal_motion_parameters.pitch.max_acceleration_deg_s2 << " deg/s^2) settle="
                      << gimbal_motion_parameters.settle_margin_ms << " ms\n";
        }
        if (send_commands) {
            std::cout << "[COMM] Commands require a fresh 0x02 state frame (timeout: "
                      << state_timeout_ms << " ms).\n";
        }
        if (static_compensation_parameters.enabled) {
            std::cout << "[laser-aim] Static compensation enabled: yaw="
                      << static_compensation_parameters.yaw_constant_deg << " + ("
                      << static_compensation_parameters.yaw_inverse_range_deg_m << "/range_m) deg, pitch="
                      << static_compensation_parameters.pitch_constant_deg << " deg, range=["
                      << static_compensation_parameters.range_min_m << ','
                      << static_compensation_parameters.range_max_m << "]m\n";
        }

        auto next_send_time = std::chrono::steady_clock::now();
        std::uint64_t frame_number = 0;
        bool robot_state_timed_out = true;
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            const std::optional<wit_radar::communication::GimbalState> latest_state =
                communicator.latest_state();
            const bool robot_state_fresh =
                latest_state.has_value() &&
                now - latest_state->received_at <= std::chrono::milliseconds(state_timeout_ms);
            if (robot_state_fresh && robot_state_timed_out) {
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cout << "[COMM][RX] Valid 0x02 state recovered: yaw=" << latest_state->angles.yaw
                          << " pitch=" << latest_state->angles.pitch
                          << " timestamp_ms=" << latest_state->device_timestamp_ms << '\n';
                robot_state_timed_out = false;
            } else if (!robot_state_fresh && !robot_state_timed_out) {
                const long long age_ms = latest_state.has_value()
                                             ? std::chrono::duration_cast<std::chrono::milliseconds>(
                                                   now - latest_state->received_at)
                                                   .count()
                                             : -1;
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cerr << "[COMM][RX] Robot state timeout: no valid 0x02 frame for "
                          << (age_ms >= 0 ? std::to_string(age_ms) : std::string("unknown"))
                          << " ms. Command sending is disabled.\n";
                robot_state_timed_out = true;
            }
            const wit_radar::HikCameraFrame camera_frame = camera.read_frame(1000);
            const cv::Mat& frame = camera_frame.image;
            if (!frame.empty()) {
                const auto frame_timestamp = camera_frame.received_at -
                                             std::chrono::milliseconds(camera_frame_latency_ms);
                const std::optional<wit_radar::communication::GimbalStateLookup> frame_state =
                    communicator.state_at(frame_timestamp,
                                          std::chrono::milliseconds(state_interpolation_max_gap_ms),
                                          std::chrono::milliseconds(state_nearest_max_offset_ms));
                const std::optional<wit_radar::communication::GimbalState> control_state =
                    communicator.latest_state();
                const bool control_state_fresh =
                    control_state.has_value() &&
                    camera_frame.received_at - control_state->received_at <=
                        std::chrono::milliseconds(state_timeout_ms);
                if (tracker_parameters.enabled) {
                    world_target_tracker.coast_to(frame_timestamp);
                }
                ++frame_number;
                if (image_width > 0 && image_height > 0 &&
                    (frame.cols != image_width || frame.rows != image_height)) {
                    throw std::runtime_error("Camera frame size does not match camera calibration resolution.");
                }
                cv::Mat display = frame.clone();
                std::string status = "YOLO: no ROI";
                std::vector<wit_radar::LaserDetection> detections = detector.detect(frame);
                bool using_traditional_fallback = false;
                if (detections.empty() && traditional_fallback_when_yolo_missing) {
                    detections.push_back({cv::Rect(0, 0, frame.cols, frame.rows),
                                          cv::Rect(0, 0, frame.cols, frame.rows), 0.0F, -1});
                    using_traditional_fallback = true;
                }
                if (!detections.empty()) {
                    const wit_radar::LaserDetection& detection = detections.front();
                    const cv::Scalar roi_color = using_traditional_fallback ? cv::Scalar(0, 165, 255)
                                                                            : cv::Scalar(0, 255, 0);
                    cv::rectangle(display, detection.roi, roi_color, 2);
                    if (using_traditional_fallback) {
                        cv::putText(display, "Traditional fallback ROI", detection.roi.tl() + cv::Point(8, 24),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2, cv::LINE_AA);
                    }
                    wit_radar::CylinderPoseParameters roi_pose_parameters = pose_parameters;
                    roi_pose_parameters.camera_matrix(0, 2) -= detection.roi.x;
                    roi_pose_parameters.camera_matrix(1, 2) -= detection.roi.y;
                    const wit_radar::CylinderPoseResult pose =
                        wit_radar::CylinderPoseSolver(roi_pose_parameters).solve(frame(detection.roi));
                    for (const cv::RotatedRect& rectangle : pose.pnp_input_rectangles) {
                        draw_pnp_input_rectangle(display, rectangle, detection.roi.tl());
                    }
                    wit_radar::KeypointCenterResult neural_center;
                    if (keypoint_finder) {
                        neural_center = keypoint_finder->find(frame(detection.roi));
                    } else if (keypoint_mode == "yolo_roi_center" && !using_traditional_fallback) {
                        neural_center.center = cv::Point2f((detection.roi.width - 1) * 0.5F,
                                                           (detection.roi.height - 1) * 0.5F);
                        neural_center.confidence = detection.confidence;
                        neural_center.reason = "YOLO ROI geometric center.";
                    }
                    draw_projected_point(display, pose.traditional_center_pixel, detection.roi.tl(),
                                         cv::Scalar(255, 255, 0), "Traditional");
                    draw_projected_point(display, pose.traditional_upper_center_pixel, detection.roi.tl(),
                                         cv::Scalar(255, 0, 0), "T upper");
                    draw_projected_point(display, pose.traditional_lower_center_pixel, detection.roi.tl(),
                                         cv::Scalar(255, 0, 0), "T lower");
                    if (keypoint_finder || neural_center.center.has_value()) {
                        draw_projected_point(display, neural_center.center, detection.roi.tl(),
                                             cv::Scalar(0, 165, 255), "N center");
                        draw_projected_point(display, neural_center.upper_center, detection.roi.tl(),
                                             cv::Scalar(0, 165, 255), "N upper");
                        draw_projected_point(display, neural_center.lower_center, detection.roi.tl(),
                                             cv::Scalar(0, 165, 255), "N lower");
                        if (pose.traditional_center_pixel.has_value() && neural_center.center.has_value()) {
                            const float center_distance_px = cv::norm(pose.traditional_center_pixel.value() -
                                                                       neural_center.center.value());
                            if (center_distance_px <= keypoint_fusion_max_distance_px) {
                                const cv::Point2f fused_center =
                                    pose.traditional_center_pixel.value() * (1.0F - keypoint_neural_weight) +
                                    neural_center.center.value() * keypoint_neural_weight;
                                draw_projected_point(display, fused_center, detection.roi.tl(),
                                                     cv::Scalar(0, 255, 255), "Fused");
                            } else {
                                status = "Center mismatch: traditional vs neural";
                            }
                            if (frame_number % static_cast<std::uint64_t>(debug_log_every_n_frames) == 0) {
                                std::lock_guard<std::mutex> lock(output_mutex);
                                std::cout << "[laser-aim][CENTER] traditional=("
                                          << pose.traditional_center_pixel->x << ','
                                          << pose.traditional_center_pixel->y << ") score="
                                          << pose.traditional_group_score << " neural=("
                                          << neural_center.center->x << ',' << neural_center.center->y
                                          << ") confidence=" << neural_center.confidence
                                          << " distance_px=" << center_distance_px << '\n';
                            }
                        } else if (frame_number % static_cast<std::uint64_t>(debug_log_every_n_frames) == 0) {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            std::cout << "[laser-aim][CENTER] traditional="
                                      << (pose.traditional_center_pixel.has_value() ? "OK" : "none")
                                      << " neural=" << (neural_center.center.has_value() ? "OK" : "none")
                                      << " reason=" << neural_center.reason << '\n';
                        }
                    }
                    for (const wit_radar::MarkerPose& marker : pose.markers) {
                        draw_marker(display, marker, detection.roi.tl());
                    }
                    const std::optional<cv::Point2f> upper_axis_pixel = project_camera_point(
                        pose.upper_axis_point_in_camera_m, roi_pose_parameters);
                    const std::optional<cv::Point2f> lower_axis_pixel = project_camera_point(
                        pose.lower_axis_point_in_camera_m, roi_pose_parameters);
                    draw_projected_point(display, upper_axis_pixel, detection.roi.tl(), cv::Scalar(255, 255, 0),
                                         "PnP upper");
                    draw_projected_point(display, lower_axis_pixel, detection.roi.tl(), cv::Scalar(255, 0, 255),
                                         "PnP lower");
                    if (pose.target_center_in_camera_m.has_value()) {
                        const cv::Point3d target_in_camera = pose.target_center_in_camera_m.value();
                        draw_projected_point(display, project_camera_point(target_in_camera, roi_pose_parameters),
                                             detection.roi.tl(), cv::Scalar(0, 255, 0), "PnP center");
                        const cv::Point3d target_in_gimbal =
                            apply_rotation(camera_to_gimbal, target_in_camera) + camera_in_gimbal;
                        const cv::Point3d beam_point_in_gimbal =
                            apply_rotation(camera_to_gimbal, beam_point_in_camera) + camera_in_gimbal;
                        const cv::Point3d target_direction_in_gimbal =
                            target_in_gimbal - beam_point_in_gimbal;
                        const cv::Point3d beam_direction_in_gimbal =
                            apply_rotation(camera_to_gimbal, beam_direction_in_camera);
                        if (frame_state.has_value() && control_state.has_value() && control_state_fresh) {
                            const float world_yaw = static_cast<float>(
                                world_yaw_feedback_sign * frame_state->angles.yaw);
                            const float world_pitch =
                                static_cast<float>(world_pitch_feedback_sign * frame_state->angles.pitch);
                            const cv::Matx33d frame_gimbal_to_world =
                                wit_radar::communication::AbsoluteLaserAimSolver::rotation_gimbal_to_world(
                                    world_yaw, world_pitch);
                            const cv::Point3d raw_target_in_world =
                                apply_rotation(frame_gimbal_to_world, target_in_gimbal);
                            wit_radar::WorldTargetObservationInput observation_input;
                            observation_input.position_in_camera_m = target_in_camera;
                            observation_input.position_in_world_m = raw_target_in_world;
                            observation_input.camera_to_world_rotation =
                                frame_gimbal_to_world * camera_to_gimbal;
                            observation_input.camera_matrix = camera_matrix;
                            observation_input.exposure_timestamp = frame_timestamp;
                            observation_input.frame_number = camera_frame.frame_number;
                            observation_input.camera_device_timestamp = camera_frame.device_timestamp;
                            observation_input.average_reprojection_error_px = pose.average_reprojection_error_px;
                            observation_input.max_marker_axis_residual_m = pose.max_marker_axis_residual_m;
                            observation_input.marker_count =
                                std::max(1, static_cast<int>(pose.markers.size()));
                            observation_input.source = pose.used_global_cylinder_pnp
                                                           ? wit_radar::WorldTargetObservationSource::GlobalCylinderPnp
                                                           : wit_radar::WorldTargetObservationSource::LegacyIppe;
                            const std::optional<wit_radar::WorldTargetObservation> observation =
                                observation_builder.build(observation_input);
                            if (!observation.has_value()) {
                                status = "invalid world observation";
                            } else {
                                const wit_radar::WorldTargetTrackerResult tracker_result =
                                    world_target_tracker.update(observation.value());
                            std::optional<cv::Point3d> target_for_aim = tracker_result.estimate_in_world_m;
                            std::optional<wit_radar::WorldTargetTrackerResult> roi_tracker_result;
                            if (tracker_parameters.enabled && tracker_parameters.roi_measurement_enabled &&
                                !using_traditional_fallback) {
                                const cv::Matx33d gimbal_to_camera = camera_to_gimbal.t();
                                wit_radar::WorldTargetImageObservation roi_observation;
                                roi_observation.pixel = {
                                    detection.roi.x + (detection.roi.width - 1) * 0.5,
                                    detection.roi.y + (detection.roi.height - 1) * 0.5};
                                roi_observation.camera_matrix = camera_matrix;
                                roi_observation.distortion_coefficients = distortion;
                                roi_observation.world_to_camera_rotation =
                                    gimbal_to_camera * frame_gimbal_to_world.t();
                                roi_observation.world_to_camera_translation_m =
                                    apply_rotation(gimbal_to_camera, camera_in_gimbal) * -1.0;
                                roi_tracker_result =
                                    world_target_tracker.update_image(roi_observation, frame_timestamp);
                                if (roi_tracker_result->estimate_in_world_m.has_value()) {
                                    target_for_aim = roi_tracker_result->estimate_in_world_m;
                                }
                            }
                            const auto command_timestamp = std::chrono::steady_clock::now();
                            const double base_prediction_seconds = std::max(
                                0.0, std::chrono::duration<double>(command_timestamp - frame_timestamp).count()) +
                                static_cast<double>(tracker_control_latency_ms) / 1000.0;
                            struct AimCandidate {
                                wit_radar::communication::GimbalCommand command;
                                wit_radar::StaticAimCompensationResult static_compensation;
                                double target_range_m = 0.0;
                            };
                            const auto make_aim_candidate = [&](const cv::Point3d& target_in_world)
                                -> std::optional<AimCandidate> {
                                const auto absolute_command =
                                    wit_radar::communication::AbsoluteLaserAimSolver::solve(
                                        target_in_world, beam_point_in_gimbal, beam_direction_in_gimbal);
                                if (!absolute_command.has_value()) {
                                    return std::nullopt;
                                }
                                const cv::Point3d target_in_frame_gimbal = apply_rotation(
                                    frame_gimbal_to_world.t(), target_in_world);
                                AimCandidate candidate;
                                candidate.target_range_m =
                                    cv::norm(target_in_frame_gimbal - beam_point_in_gimbal);
                                candidate.static_compensation = wit_radar::evaluate_static_aim_compensation(
                                    static_compensation_parameters, candidate.target_range_m);
                                candidate.command = command_template;
                                candidate.command.mode =
                                    wit_radar::communication::GimbalCommandMode::kTargetDetected;
                                candidate.command.yaw = static_cast<float>(
                                    absolute_command->target_yaw / world_yaw_feedback_sign);
                                candidate.command.pitch = static_cast<float>(
                                    absolute_command->target_pitch / world_pitch_feedback_sign);
                                if (candidate.static_compensation.applied) {
                                    candidate.command.yaw += static_cast<float>(
                                        candidate.static_compensation.yaw_correction_deg);
                                    candidate.command.pitch += static_cast<float>(
                                        candidate.static_compensation.pitch_correction_deg);
                                }
                                return candidate;
                            };
                            double prediction_seconds = base_prediction_seconds;
                            for (int iteration = 0;
                                 iteration < gimbal_motion_parameters.prediction_iterations &&
                                 target_for_aim.has_value();
                                 ++iteration) {
                                if (tracker_parameters.enabled) {
                                    target_for_aim = world_target_tracker.predict_ahead(prediction_seconds);
                                    if (!target_for_aim.has_value()) {
                                        break;
                                    }
                                }
                                const std::optional<AimCandidate> candidate =
                                    make_aim_candidate(target_for_aim.value());
                                if (!candidate.has_value()) {
                                    target_for_aim.reset();
                                    break;
                                }
                                const auto arrival = gimbal_arrival_estimator.estimate(
                                    control_state->angles,
                                    {candidate->command.yaw, candidate->command.pitch});
                                if (!arrival.has_value()) {
                                    target_for_aim.reset();
                                    break;
                                }
                                const double next_prediction_seconds =
                                    base_prediction_seconds + arrival->total_seconds;
                                if (std::abs(next_prediction_seconds - prediction_seconds) <=
                                    gimbal_motion_parameters.convergence_ms / 1000.0) {
                                    prediction_seconds = next_prediction_seconds;
                                    break;
                                }
                                prediction_seconds = next_prediction_seconds;
                            }
                            if (tracker_parameters.enabled && target_for_aim.has_value()) {
                                target_for_aim = world_target_tracker.predict_ahead(prediction_seconds);
                            }
                            if (target_for_aim.has_value()) {
                                const std::optional<AimCandidate> aim_candidate =
                                    make_aim_candidate(target_for_aim.value());
                                if (!aim_candidate.has_value()) {
                                    status = "World solve failed";
                                    continue;
                                }
                                const auto arrival = gimbal_arrival_estimator.estimate(
                                    control_state->angles,
                                    {aim_candidate->command.yaw, aim_candidate->command.pitch});
                                if (!arrival.has_value()) {
                                    status = "Gimbal arrival estimate failed";
                                    continue;
                                }
                                const double target_range_m = aim_candidate->target_range_m;
                                const wit_radar::StaticAimCompensationResult static_compensation =
                                    aim_candidate->static_compensation;
                                wit_radar::communication::GimbalCommand command = aim_candidate->command;
                                const float yaw_delta =
                                    normalize_degrees(command.yaw - control_state->angles.yaw);
                                const float pitch_delta = command.pitch - control_state->angles.pitch;
                                wit_radar::communication::AimSolution correction{yaw_delta, pitch_delta};
                                const bool correction_safe =
                                    std::abs(correction.target_yaw) <= max_yaw_correction &&
                                    std::abs(correction.target_pitch) <= max_pitch_correction;
                                status = cv::format("%s PnP Z=%.3fm err=%.2fpx dy=%.2f dp=%.2f",
                                                    using_traditional_fallback ? "Traditional fallback" : "YOLO ROI",
                                                    target_in_camera.z, pose.average_reprojection_error_px,
                                                    correction.target_yaw, correction.target_pitch);
                                status += pose.used_global_cylinder_pnp ? " global" : " legacy";
                                if (tracker_parameters.enabled) {
                                    const cv::Point3d estimated_velocity =
                                        tracker_result.velocity_in_world_mps.value_or(cv::Point3d{});
                                    status += cv::format(
                                        " obs=%.2f sig=(%.3f,%.3f)m v=(%.2f,%.2f,%.2f)m/s ekf=%s chi2=%.2f pred=%.0fms",
                                                         observation->confidence,
                                                         observation->tangential_std_m,
                                                         observation->radial_std_m,
                                                         estimated_velocity.x, estimated_velocity.y,
                                                         estimated_velocity.z,
                                                         tracker_update_name(tracker_result.update),
                                                         tracker_result.innovation_chi2,
                                                         prediction_seconds * 1000.0);
                                    if (roi_tracker_result.has_value()) {
                                        status += cv::format(" roi=%s chi2=%.2f",
                                                             roi_tracker_result->image_measurement_accepted ? "accepted"
                                                                                                           : "rejected",
                                                             roi_tracker_result->image_innovation_chi2);
                                    }
                                }
                                if (gimbal_motion_parameters.enabled) {
                                    status += cv::format(" arrive=(y%.0f,p%.0f,t%.0f)ms",
                                                         arrival->yaw_motion_seconds * 1000.0,
                                                         arrival->pitch_motion_seconds * 1000.0,
                                                         arrival->total_seconds * 1000.0);
                                }
                                if (static_compensation_parameters.enabled) {
                                    status += static_compensation.applied
                                                  ? cv::format(" static r=%.2fm sy=%.3f sp=%.3f", target_range_m,
                                                               static_compensation.yaw_correction_deg,
                                                               static_compensation.pitch_correction_deg)
                                                  : cv::format(" static r=%.2fm out-of-range", target_range_m);
                                }
                                const bool command_angles_valid = valid_command_angles(command);
                                status += cv::format(" frame=(%.2f,%.2f)%s%.1fms state=(%.2f,%.2f) cmd=(%.2f,%.2f)",
                                                     frame_state->angles.yaw, frame_state->angles.pitch,
                                                     frame_state->interpolated ? "i" : "n",
                                                     frame_state->nearest_sample_offset_ms,
                                                     control_state->angles.yaw, control_state->angles.pitch,
                                                     command.yaw, command.pitch);
                                if (frame_number % static_cast<std::uint64_t>(debug_log_every_n_frames) == 0) {
                                    std::lock_guard<std::mutex> lock(output_mutex);
                                    std::cout << "[laser-aim][CHAIN] PnP upper_C=("
                                              << pose.upper_axis_point_in_camera_m.x << ','
                                              << pose.upper_axis_point_in_camera_m.y << ','
                                              << pose.upper_axis_point_in_camera_m.z << ") lower_C=("
                                              << pose.lower_axis_point_in_camera_m.x << ','
                                              << pose.lower_axis_point_in_camera_m.y << ','
                                              << pose.lower_axis_point_in_camera_m.z << ")\n"
                                              << "[laser-aim][CHAIN] target_C=(" << target_in_camera.x << ','
                                              << target_in_camera.y << ',' << target_in_camera.z
                                              << ") target_G=(" << target_in_gimbal.x << ','
                                              << target_in_gimbal.y << ',' << target_in_gimbal.z
                                              << ") beam_point_G=(" << beam_point_in_gimbal.x << ','
                                              << beam_point_in_gimbal.y << ',' << beam_point_in_gimbal.z
                                              << ") target_direction_G=(" << target_direction_in_gimbal.x << ','
                                              << target_direction_in_gimbal.y << ','
                                              << target_direction_in_gimbal.z << ") beam_direction_G=("
                                              << beam_direction_in_gimbal.x << ',' << beam_direction_in_gimbal.y
                                              << ',' << beam_direction_in_gimbal.z << ")\n"
                                              << "[laser-aim][CHAIN] target_W_raw=(" << raw_target_in_world.x << ','
                                              << raw_target_in_world.y << ',' << raw_target_in_world.z
                                              << ") target_W_aim=(" << target_for_aim->x << ','
                                              << target_for_aim->y << ',' << target_for_aim->z
                                              << ") frame_pose=(" << world_yaw << ',' << world_pitch
                                              << ") frame_timing=(number=" << camera_frame.frame_number
                                              << ",device_ts=" << camera_frame.device_timestamp
                                              << ",interpolated=" << (frame_state->interpolated ? "yes" : "no")
                                              << ",nearest_offset_ms="
                                              << frame_state->nearest_sample_offset_ms
                                              << ") protocol_delta=("
                                              << correction.target_yaw << ',' << correction.target_pitch
                                              << ") feedback=(" << control_state->angles.yaw << ','
                                              << control_state->angles.pitch << ") command=(" << command.yaw << ','
                                              << command.pitch << ") static=(range=" << target_range_m
                                              << ",yaw=" << static_compensation.yaw_correction_deg
                                              << ",pitch=" << static_compensation.pitch_correction_deg
                                              << ",applied=" << (static_compensation.applied ? "yes" : "no")
                                              << ")\n";
                                    if (roi_tracker_result.has_value()) {
                                        std::cout << "[laser-aim][EKF] roi_center_full=("
                                                  << detection.roi.x + (detection.roi.width - 1) * 0.5 << ','
                                                  << detection.roi.y + (detection.roi.height - 1) * 0.5
                                                  << ") roi_update="
                                                  << (roi_tracker_result->image_measurement_accepted ? "accepted"
                                                                                                      : "rejected")
                                                  << " roi_chi2="
                                                  << roi_tracker_result->image_innovation_chi2 << '\n';
                                    }
                                }
                                const cv::Matx33d gimbal_to_camera = camera_to_gimbal.t();
                                const cv::Point3d filtered_target_in_gimbal = apply_rotation(
                                    frame_gimbal_to_world.t(), target_for_aim.value());
                                const cv::Point3d filtered_target_in_camera = apply_rotation(
                                    gimbal_to_camera, filtered_target_in_gimbal - camera_in_gimbal);
                                draw_projected_point(display,
                                                     project_camera_point(filtered_target_in_camera,
                                                                          roi_pose_parameters),
                                                     detection.roi.tl(), cv::Scalar(0, 255, 0),
                                                     "EKF center");
                                if (send_commands && control_state_fresh && correction_safe &&
                                    command_angles_valid && command_timestamp >= next_send_time) {
                                    const bool command_queued = communicator.send_command(
                                        command, [&output_mutex](const boost::system::error_code& error,
                                                                  std::size_t bytes_written) {
                                            std::lock_guard<std::mutex> lock(output_mutex);
                                            if (error) {
                                                std::cerr << "[COMM][TX] Serial write failed: "
                                                          << error.message() << '\n';
                                            } else {
                                                std::cout << "[COMM][TX] Serial write completed: "
                                                          << bytes_written << " bytes.\n";
                                            }
                                        });
                                    if (command_queued) {
                                        std::lock_guard<std::mutex> lock(output_mutex);
                                        std::cout << "[COMM][TX] 0x01 yaw=" << command.yaw
                                                  << " pitch=" << command.pitch << " queued speed=("
                                                  << command.yaw_speed << ',' << command.pitch_speed
                                                  << ") acceleration=(" << command.yaw_acceleration << ','
                                                  << command.pitch_acceleration << ") mode="
                                                  << static_cast<int>(command.mode) << '\n';
                                    } else {
                                        std::lock_guard<std::mutex> lock(output_mutex);
                                        std::cerr << "[COMM][TX] Command was not queued: serial is disconnected "
                                                     "or restarting.\n";
                                    }
                                    next_send_time =
                                        command_timestamp + std::chrono::milliseconds(send_interval_ms);
                                }
                                if (!command_angles_valid) {
                                    status += " invalid state";
                                }
                                if (!correction_safe) {
                                    status += " correction too large";
                                }
                            } else {
                                status = cv::format("EKF initializing (%d/%d)",
                                                    tracker_result.initialization_sample_count,
                                                    tracker_parameters.initialization_frames);
                            }
                            }
                        } else if (control_state.has_value() && control_state_fresh) {
                            status = "waiting for frame-time 0x02";
                        } else if (control_state.has_value()) {
                            status = "waiting for fresh 0x02";
                        } else {
                            status = "waiting for 0x02";
                        }
                    } else {
                        status = std::string(using_traditional_fallback ? "Traditional fallback PnP: " :
                                                                        "YOLO ROI PnP: ") + pose.reason;
                        if (upper_axis_pixel.has_value() && lower_axis_pixel.has_value()) {
                            const cv::Point3d rejected_center =
                                (pose.upper_axis_point_in_camera_m + pose.lower_axis_point_in_camera_m) * 0.5;
                            draw_projected_point(display,
                                                 project_camera_point(rejected_center, roi_pose_parameters),
                                                 detection.roi.tl(), cv::Scalar(0, 165, 255),
                                                 "Rejected center");
                        }
                        if (frame_number % static_cast<std::uint64_t>(debug_log_every_n_frames) == 0) {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            std::cout << "[laser-aim][PnP] markers=" << pose.markers.size()
                                      << " source="
                                      << (pose.used_global_cylinder_pnp
                                              ? "global-cylinder"
                                              : (pose.target_center_in_camera_m.has_value() ? "legacy-ippe" : "none"))
                                      << " global_hypotheses=" << pose.global_cylinder_hypothesis_count
                                      << " global_reproj=" << pose.global_cylinder_reprojection_error_px
                                      << "px"
                                      << " sets=(upper=" << pose.upper_marker_set_count
                                      << ",lower=" << pose.lower_marker_set_count << ") candidates=(upper="
                                      << pose.upper_ippe_candidate_count << ",lower="
                                      << pose.lower_ippe_candidate_count << ") compatible_pairs="
                                      << pose.compatible_ring_hypothesis_count
                                      << " closest_sep=" << pose.closest_candidate_ring_separation_m
                                      << "m closest_sep_error="
                                      << pose.closest_candidate_ring_separation_error_m
                                      << "m best_candidate_axis_residual="
                                      << pose.best_candidate_axis_residual_m << "m"
                                      << " upper_C=(" << pose.upper_axis_point_in_camera_m.x << ','
                                      << pose.upper_axis_point_in_camera_m.y << ','
                                      << pose.upper_axis_point_in_camera_m.z << ") lower_C=("
                                      << pose.lower_axis_point_in_camera_m.x << ','
                                      << pose.lower_axis_point_in_camera_m.y << ','
                                      << pose.lower_axis_point_in_camera_m.z << ") max_axis_residual="
                                      << pose.max_marker_axis_residual_m << "m reason=" << pose.reason
                                      << '\n';
                            for (std::size_t marker_index = 0;
                                 marker_index < pose.marker_candidate_diagnostics.size(); ++marker_index) {
                                const wit_radar::MarkerCandidateDiagnostics& diagnostics =
                                    pose.marker_candidate_diagnostics[marker_index];
                                std::cout << "[laser-aim][PnP][marker " << marker_index << "] ring="
                                          << (diagnostics.upper_ring ? "upper" : "lower") << " image_center=("
                                          << diagnostics.image_center.x << ',' << diagnostics.image_center.y
                                          << ") size_px=(" << diagnostics.image_size.width << ','
                                          << diagnostics.image_size.height << ") candidates="
                                          << diagnostics.candidate_count << " depth_range=("
                                          << diagnostics.minimum_depth_m << ','
                                          << diagnostics.maximum_depth_m << ")m reproj_range=("
                                          << diagnostics.minimum_reprojection_error_px << ','
                                          << diagnostics.maximum_reprojection_error_px << ")px\n";
                            }
                        }
                    }
                }
                cv::putText(display, status, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                            cv::Scalar(0, 255, 255), 2);
                cv::imshow(window_name, display);
                if (frame_number % static_cast<std::uint64_t>(debug_log_every_n_frames) == 0) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "[laser-aim] " << status << '\n';
                }
            }

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        communicator.stop();
        cv::destroyWindow(window_name);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Laser aim failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
