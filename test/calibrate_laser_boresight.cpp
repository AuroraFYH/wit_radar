#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hik_camera.hpp"

namespace {

struct CalibrationSettings {
    std::string camera_serial_number;
    int image_width = 0;
    int image_height = 0;
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    cv::Vec<double, 5> distortion_coefficients{};
    cv::Size chessboard_inner_corners{9, 6};
    double square_size_m = 0.025;
    int minimum_samples = 6;
    double minimum_depth_span_m = 2.0;
    double maximum_rms_error_m = 0.015;
    double minimum_square_size_px = 14.0;
    int stable_frame_count = 15;
    double maximum_center_stddev_m = 0.005;
    double processing_scale = 1.0;
};

struct BeamLineFit {
    cv::Point3d point_m;
    cv::Point3d direction;
    double rms_error_m = 0.0;
    double maximum_error_m = 0.0;
    double minimum_depth_m = 0.0;
    double maximum_depth_m = 0.0;
};

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

cv::Size read_chessboard_size(const cv::FileNode& config) {
    const cv::FileNode values = config["chessboard_inner_corners"];
    if (values.empty() || !values.isSeq() || values.size() != 2) {
        throw std::runtime_error("laser.boresight_calibration.chessboard_inner_corners must contain [columns, rows].");
    }
    const int columns = static_cast<int>(values[0].real());
    const int rows = static_cast<int>(values[1].real());
    if (columns < 3 || rows < 3) {
        throw std::runtime_error("Chessboard must have at least 3 by 3 inner corners.");
    }
    return {columns, rows};
}

CalibrationSettings load_settings(const std::string& config_path) {
    cv::FileStorage config(config_path, cv::FileStorage::READ);
    if (!config.isOpened()) {
        throw std::runtime_error("Unable to open configuration file: " + config_path);
    }
    const cv::FileNode camera = config["camera"];
    const cv::FileNode laser = config["laser"];
    const cv::FileNode calibration = laser["boresight_calibration"];
    if (camera.empty() || laser.empty() || calibration.empty()) {
        throw std::runtime_error("Missing camera, laser, or laser.boresight_calibration configuration.");
    }

    CalibrationSettings settings;
    camera["sn"] >> settings.camera_serial_number;
    if (settings.camera_serial_number.empty()) {
        throw std::runtime_error("Missing camera.sn in configuration file.");
    }
    settings.image_width = read_int(camera, "image_width", 0);
    settings.image_height = read_int(camera, "image_height", 0);
    settings.camera_matrix = read_matrix(camera, "camera_matrix");
    settings.distortion_coefficients = read_distortion(camera["distortion_coefficients"]);
    settings.chessboard_inner_corners = read_chessboard_size(calibration);
    settings.square_size_m = read_double(calibration, "square_size_m", 0.0);
    settings.minimum_samples = read_int(calibration, "minimum_samples", 6);
    settings.minimum_depth_span_m = read_double(calibration, "minimum_depth_span_m", 2.0);
    settings.maximum_rms_error_m = read_double(calibration, "maximum_rms_error_m", 0.015);
    settings.minimum_square_size_px = read_double(calibration, "minimum_square_size_px", 14.0);
    settings.stable_frame_count = read_int(calibration, "stable_frame_count", 15);
    settings.maximum_center_stddev_m = read_double(calibration, "maximum_center_stddev_m", 0.005);
    settings.processing_scale = read_double(calibration, "processing_scale", 1.0);
    if (settings.square_size_m <= 0.0 || settings.minimum_samples < 3 ||
        settings.minimum_depth_span_m <= 0.0 || settings.maximum_rms_error_m <= 0.0 ||
        settings.minimum_square_size_px <= 0.0 || settings.stable_frame_count < 2 ||
        settings.maximum_center_stddev_m <= 0.0 || settings.processing_scale <= 0.0 ||
        settings.processing_scale > 1.0) {
        throw std::runtime_error("Invalid laser.boresight_calibration numeric parameters.");
    }
    return settings;
}

CalibrationSettings make_processing_settings(const CalibrationSettings& raw_settings) {
    CalibrationSettings processing_settings = raw_settings;
    const double scale = raw_settings.processing_scale;
    for (int column = 0; column < 3; ++column) {
        processing_settings.camera_matrix(0, column) *= scale;
        processing_settings.camera_matrix(1, column) *= scale;
    }
    processing_settings.minimum_square_size_px *= scale;
    return processing_settings;
}

double average_square_size_px(const std::vector<cv::Point2f>& corners, const cv::Size& inner_corners) {
    if (corners.size() != static_cast<std::size_t>(inner_corners.area())) {
        return 0.0;
    }
    double total_length = 0.0;
    int edge_count = 0;
    for (int row = 0; row < inner_corners.height; ++row) {
        for (int column = 0; column < inner_corners.width; ++column) {
            const int index = row * inner_corners.width + column;
            if (column + 1 < inner_corners.width) {
                total_length += cv::norm(corners[index + 1] - corners[index]);
                ++edge_count;
            }
            if (row + 1 < inner_corners.height) {
                total_length += cv::norm(corners[index + inner_corners.width] - corners[index]);
                ++edge_count;
            }
        }
    }
    return edge_count > 0 ? total_length / static_cast<double>(edge_count) : 0.0;
}

std::pair<cv::Point3d, double> mean_and_stddev(const std::deque<cv::Point3d>& points) {
    cv::Point3d mean{};
    for (const cv::Point3d& point : points) {
        mean += point;
    }
    mean *= 1.0 / static_cast<double>(points.size());
    double squared_error_sum = 0.0;
    for (const cv::Point3d& point : points) {
        squared_error_sum += cv::norm(point - mean) * cv::norm(point - mean);
    }
    return {mean, std::sqrt(squared_error_sum / static_cast<double>(points.size()))};
}

std::vector<cv::Point3f> make_chessboard_object_points(const CalibrationSettings& settings) {
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<std::size_t>(settings.chessboard_inner_corners.area()));
    for (int row = 0; row < settings.chessboard_inner_corners.height; ++row) {
        for (int column = 0; column < settings.chessboard_inner_corners.width; ++column) {
            points.emplace_back(static_cast<float>(column * settings.square_size_m),
                                static_cast<float>(row * settings.square_size_m), 0.0F);
        }
    }
    return points;
}

cv::Point3d chessboard_center_in_object(const CalibrationSettings& settings) {
    return {(settings.chessboard_inner_corners.width - 1) * settings.square_size_m * 0.5,
            (settings.chessboard_inner_corners.height - 1) * settings.square_size_m * 0.5, 0.0};
}

std::optional<cv::Point3d> find_board_center_in_camera(const cv::Mat& frame,
                                                        const CalibrationSettings& settings,
                                                        const std::vector<cv::Point3f>& object_points,
                                                        std::vector<cv::Point2f>* image_corners,
                                                        cv::Point2f* board_center_pixel) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    const bool found = cv::findChessboardCorners(
        gray, settings.chessboard_inner_corners, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    if (!found) {
        return std::nullopt;
    }
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
    cv::Vec3d rotation_vector;
    cv::Vec3d translation_vector;
    if (!cv::solvePnP(object_points, corners, settings.camera_matrix, settings.distortion_coefficients,
                      rotation_vector, translation_vector, false, cv::SOLVEPNP_ITERATIVE)) {
        return std::nullopt;
    }
    cv::Matx33d rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);
    const cv::Point3d center = chessboard_center_in_object(settings);
    const cv::Vec3d transformed = rotation_matrix * cv::Vec3d(center.x, center.y, center.z);
    if (image_corners != nullptr) {
        *image_corners = std::move(corners);
    }
    if (board_center_pixel != nullptr) {
        std::vector<cv::Point2d> projected_center;
        cv::projectPoints(std::vector<cv::Point3d>{center}, rotation_vector, translation_vector,
                          settings.camera_matrix, settings.distortion_coefficients, projected_center);
        if (projected_center.size() == 1) {
            *board_center_pixel = cv::Point2f(static_cast<float>(projected_center.front().x),
                                               static_cast<float>(projected_center.front().y));
        }
    }
    return cv::Point3d(transformed[0] + translation_vector[0], transformed[1] + translation_vector[1],
                       transformed[2] + translation_vector[2]);
}

double dot(const cv::Point3d& first, const cv::Point3d& second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

std::optional<BeamLineFit> fit_beam_line(const std::vector<cv::Point3d>& samples,
                                          const CalibrationSettings& settings, std::string* error) {
    if (samples.size() < static_cast<std::size_t>(settings.minimum_samples)) {
        if (error != nullptr) {
            *error = "Need at least " + std::to_string(settings.minimum_samples) + " samples.";
        }
        return std::nullopt;
    }
    cv::Point3d centroid{};
    double minimum_depth = samples.front().z;
    double maximum_depth = samples.front().z;
    for (const cv::Point3d& sample : samples) {
        centroid += sample;
        minimum_depth = std::min(minimum_depth, sample.z);
        maximum_depth = std::max(maximum_depth, sample.z);
    }
    centroid *= 1.0 / static_cast<double>(samples.size());
    if (maximum_depth - minimum_depth < settings.minimum_depth_span_m) {
        if (error != nullptr) {
            *error = cv::format("Depth span %.3fm is below required %.3fm.",
                                maximum_depth - minimum_depth, settings.minimum_depth_span_m);
        }
        return std::nullopt;
    }

    cv::Matx33d covariance{};
    for (const cv::Point3d& sample : samples) {
        const cv::Point3d delta = sample - centroid;
        covariance(0, 0) += delta.x * delta.x;
        covariance(0, 1) += delta.x * delta.y;
        covariance(0, 2) += delta.x * delta.z;
        covariance(1, 0) += delta.y * delta.x;
        covariance(1, 1) += delta.y * delta.y;
        covariance(1, 2) += delta.y * delta.z;
        covariance(2, 0) += delta.z * delta.x;
        covariance(2, 1) += delta.z * delta.y;
        covariance(2, 2) += delta.z * delta.z;
    }
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    cv::eigen(cv::Mat(covariance), eigenvalues, eigenvectors);
    cv::Point3d direction{eigenvectors.at<double>(0, 0), eigenvectors.at<double>(0, 1),
                          eigenvectors.at<double>(0, 2)};
    const double direction_length = cv::norm(direction);
    if (direction_length < 1e-9) {
        if (error != nullptr) {
            *error = "Unable to determine a beam direction from the samples.";
        }
        return std::nullopt;
    }
    direction *= 1.0 / direction_length;
    if (direction.z < 0.0) {
        direction = -direction;
    }
    const cv::Point3d line_point = centroid - direction * dot(centroid, direction);

    double squared_error_sum = 0.0;
    double maximum_error = 0.0;
    for (const cv::Point3d& sample : samples) {
        const cv::Point3d normal_component = sample - line_point - direction * dot(sample - line_point, direction);
        const double error_m = cv::norm(normal_component);
        squared_error_sum += error_m * error_m;
        maximum_error = std::max(maximum_error, error_m);
    }
    BeamLineFit fit;
    fit.point_m = line_point;
    fit.direction = direction;
    fit.rms_error_m = std::sqrt(squared_error_sum / static_cast<double>(samples.size()));
    fit.maximum_error_m = maximum_error;
    fit.minimum_depth_m = minimum_depth;
    fit.maximum_depth_m = maximum_depth;
    return fit;
}

std::vector<double> line_residuals(const std::vector<cv::Point3d>& samples, const BeamLineFit& fit) {
    std::vector<double> residuals;
    residuals.reserve(samples.size());
    for (const cv::Point3d& sample : samples) {
        const cv::Point3d normal_component =
            sample - fit.point_m - fit.direction * dot(sample - fit.point_m, fit.direction);
        residuals.push_back(cv::norm(normal_component));
    }
    return residuals;
}

void write_result(const std::string& output_path, const BeamLineFit& fit,
                  const std::vector<cv::Point3d>& samples) {
    const std::filesystem::path path(output_path);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(output_path);
    if (!output.is_open()) {
        throw std::runtime_error("Unable to write calibration output: " + output_path);
    }
    output << std::fixed << std::setprecision(9);
    output << "{\n"
           << "  \"beam_line_in_camera\": {\n"
           << "    \"enabled\": 1,\n"
           << "    \"point_m\": [" << fit.point_m.x << ", " << fit.point_m.y << ", " << fit.point_m.z
           << "],\n"
           << "    \"direction\": [" << fit.direction.x << ", " << fit.direction.y << ", "
           << fit.direction.z << "]\n"
           << "  },\n"
           << "  \"quality\": {\n"
           << "    \"sample_count\": " << samples.size() << ",\n"
           << "    \"rms_line_error_m\": " << fit.rms_error_m << ",\n"
           << "    \"maximum_line_error_m\": " << fit.maximum_error_m << ",\n"
           << "    \"depth_range_m\": [" << fit.minimum_depth_m << ", " << fit.maximum_depth_m << "]\n"
           << "  }\n"
           << "}\n";
}

void draw_cross(cv::Mat& image, const cv::Point2f& point, const cv::Scalar& color, const std::string& label) {
    const cv::Point pixel(cvRound(point.x), cvRound(point.y));
    cv::drawMarker(image, pixel, color, cv::MARKER_CROSS, 24, 2, cv::LINE_AA);
    cv::putText(image, label, pixel + cv::Point(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv::LINE_AA);
}

std::optional<cv::Point2f> project_camera_point(const cv::Point3d& point,
                                                 const CalibrationSettings& settings) {
    if (point.z <= 0.0) {
        return std::nullopt;
    }
    std::vector<cv::Point2d> projected;
    cv::projectPoints(std::vector<cv::Point3d>{point}, cv::Vec3d{}, cv::Vec3d{}, settings.camera_matrix,
                      settings.distortion_coefficients, projected);
    if (projected.size() != 1) {
        return std::nullopt;
    }
    return cv::Point2f(static_cast<float>(projected.front().x), static_cast<float>(projected.front().y));
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 3) {
            throw std::runtime_error(
                "Usage: calibrate_laser_boresight [config_path] [output_json_path]");
        }
        const std::string config_path = argc > 1 ? argv[1] : "config/laser.json";
        const std::string output_path = argc > 2 ? argv[2] : "test_output/laser_beam_calibration.json";
        const CalibrationSettings settings = load_settings(config_path);
        const CalibrationSettings processing_settings = make_processing_settings(settings);
        const std::vector<cv::Point3f> object_points = make_chessboard_object_points(processing_settings);

        wit_radar::HikCamera camera;
        camera.open_by_serial_number(settings.camera_serial_number);
        const std::string window_name = "Red Laser Boresight Calibration";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        std::vector<cv::Point3d> samples;
        std::deque<cv::Point3d> stable_centers;
        std::optional<BeamLineFit> current_fit;
        std::string message = "Aim red laser dot at chessboard center, then press Space.";

        std::cout << "Red laser boresight calibration started.\n"
                  << "Processing scale: " << settings.processing_scale << " (camera output remains "
                  << settings.image_width << 'x' << settings.image_height << ").\n"
                  << "Controls: Space=capture sample, Backspace=remove last, f=fit and save, "
                     "r=reset samples, q/Esc=quit.\n";
        while (true) {
            const cv::Mat frame = camera.read(1000);
            if (frame.empty()) {
                continue;
            }
            if (settings.image_width > 0 && settings.image_height > 0 &&
                (frame.cols != settings.image_width || frame.rows != settings.image_height)) {
                throw std::runtime_error(
                    "Camera frame is " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) +
                    ", but camera calibration is configured for " + std::to_string(settings.image_width) +
                    "x" + std::to_string(settings.image_height) +
                    ". Set the Hikvision camera user set to the calibration resolution, or use matching intrinsics.");
            }
            cv::Mat processing_frame;
            if (settings.processing_scale == 1.0) {
                processing_frame = frame;
            } else {
                cv::resize(frame, processing_frame, cv::Size(), settings.processing_scale,
                           settings.processing_scale, cv::INTER_AREA);
            }
            std::vector<cv::Point2f> corners;
            cv::Point2f board_center_pixel{};
            const std::optional<cv::Point3d> board_center =
                find_board_center_in_camera(processing_frame, processing_settings, object_points, &corners,
                                            &board_center_pixel);
            const double square_size_px =
                average_square_size_px(corners, processing_settings.chessboard_inner_corners);
            const bool board_large_enough = board_center.has_value() &&
                                            square_size_px >= processing_settings.minimum_square_size_px;
            if (board_large_enough) {
                stable_centers.push_back(board_center.value());
                while (stable_centers.size() > static_cast<std::size_t>(settings.stable_frame_count)) {
                    stable_centers.pop_front();
                }
            } else {
                stable_centers.clear();
            }
            const bool has_stable_window =
                stable_centers.size() == static_cast<std::size_t>(settings.stable_frame_count);
            const auto [stable_center, center_stddev_m] = has_stable_window
                                                              ? mean_and_stddev(stable_centers)
                                                              : std::pair<cv::Point3d, double>{{}, 0.0};
            const bool board_pose_stable = has_stable_window &&
                                           center_stddev_m <= settings.maximum_center_stddev_m;
            cv::Mat display = processing_frame.clone();
            if (board_center.has_value()) {
                cv::drawChessboardCorners(display, processing_settings.chessboard_inner_corners, corners, true);
                draw_cross(display, board_center_pixel, cv::Scalar(0, 255, 0), "Board center / aim laser here");
                if (current_fit.has_value() && std::abs(current_fit->direction.z) > 1e-9) {
                    const double scale =
                        (board_center->z - current_fit->point_m.z) / current_fit->direction.z;
                    const cv::Point3d predicted_beam_point =
                        current_fit->point_m + current_fit->direction * scale;
                    const std::optional<cv::Point2f> predicted_pixel =
                        project_camera_point(predicted_beam_point, processing_settings);
                    if (predicted_pixel.has_value()) {
                        draw_cross(display, predicted_pixel.value(), cv::Scalar(0, 0, 255),
                                   "Fitted red-laser point");
                    }
                }
                cv::putText(display, cv::format("Board center C=(%.3f, %.3f, %.3f)m", board_center->x,
                                                 board_center->y, board_center->z),
                            cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2,
                            cv::LINE_AA);
            } else {
                cv::putText(display, "Chessboard not found", cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX,
                            0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
            cv::putText(display, "Samples: " + std::to_string(samples.size()) + " / " +
                                     std::to_string(settings.minimum_samples),
                        cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2,
                        cv::LINE_AA);
            const cv::Scalar quality_color = board_large_enough && board_pose_stable
                                                 ? cv::Scalar(0, 255, 0)
                                                 : cv::Scalar(0, 0, 255);
            const std::string quality = cv::format(
                "Square %.1fpx (min %.1f), stable %zu/%d, stddev %.1fmm (max %.1f)", square_size_px,
                processing_settings.minimum_square_size_px, stable_centers.size(), settings.stable_frame_count,
                center_stddev_m * 1000.0, settings.maximum_center_stddev_m * 1000.0);
            cv::putText(display, quality, cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.55, quality_color, 2,
                        cv::LINE_AA);
            cv::putText(display, message, cv::Point(20, display.rows - 25), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::imshow(window_name, display);

            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
            if (key == 'r' || key == 'R') {
                samples.clear();
                current_fit.reset();
                stable_centers.clear();
                message = "Samples reset.";
                continue;
            }
            if ((key == 8 || key == 127) && !samples.empty()) {
                samples.pop_back();
                current_fit.reset();
                message = "Removed the most recent sample.";
                std::cout << "[capture] Removed most recent sample. Remaining=" << samples.size() << '\n';
                continue;
            }
            if (key == ' ') {
                if (!board_center.has_value()) {
                    message = "Capture rejected: chessboard not found.";
                    continue;
                }
                if (!board_large_enough) {
                    message = cv::format("Capture rejected: squares %.1fpx < %.1fpx. Move board closer or use a larger board.",
                                         square_size_px, processing_settings.minimum_square_size_px);
                    continue;
                }
                if (!board_pose_stable) {
                    message = cv::format("Capture rejected: wait for %d stable frames; current stddev %.1fmm.",
                                         settings.stable_frame_count, center_stddev_m * 1000.0);
                    continue;
                }
                samples.push_back(stable_center);
                message = cv::format("Captured sample %zu at Z=%.3fm, stddev %.1fmm.", samples.size(),
                                     stable_center.z, center_stddev_m * 1000.0);
                std::cout << "[capture] " << samples.size() << " point_C=(" << stable_center.x << ','
                          << stable_center.y << ',' << stable_center.z << ") stddev=" << center_stddev_m
                          << "m square=" << square_size_px << "px\n";
                continue;
            }
            if (key == 'f' || key == 'F') {
                std::string error;
                const std::optional<BeamLineFit> fit = fit_beam_line(samples, settings, &error);
                if (!fit.has_value()) {
                    message = "Fit rejected: " + error;
                    std::cerr << "[fit] " << error << '\n';
                    continue;
                }
                const std::vector<double> residuals = line_residuals(samples, fit.value());
                for (std::size_t index = 0; index < residuals.size(); ++index) {
                    std::cout << "[fit] sample=" << (index + 1) << " Z=" << samples[index].z
                              << " residual=" << residuals[index] << "m\n";
                }
                if (fit->rms_error_m > settings.maximum_rms_error_m) {
                    message = cv::format("Fit rejected: RMS %.4fm > %.4fm. Inspect residuals; Backspace removes last.",
                                         fit->rms_error_m, settings.maximum_rms_error_m);
                    std::cerr << "[fit] RMS line error " << fit->rms_error_m << "m exceeds configured "
                              << settings.maximum_rms_error_m << "m.\n";
                    continue;
                }
                write_result(output_path, fit.value(), samples);
                current_fit = fit;
                message = "Fit saved to " + output_path + ". Copy beam_line_in_camera into config/laser.json.";
                std::cout << "[fit] point_m=[" << fit->point_m.x << ", " << fit->point_m.y << ", "
                          << fit->point_m.z << "] direction=[" << fit->direction.x << ", "
                          << fit->direction.y << ", " << fit->direction.z << "] rms="
                          << fit->rms_error_m << "m output=" << output_path << '\n';
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Laser boresight calibration failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
