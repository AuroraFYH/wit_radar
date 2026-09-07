#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hik_camera.hpp"
#include "laser_detector.hpp"

namespace {

std::string read_required_string(const cv::FileNode& config, const std::string& key,
                                 const std::string& config_path) {
    std::string value;
    config[key] >> value;
    if (value.empty()) {
        throw std::runtime_error("Missing " + key + " in configuration file: " + config_path);
    }
    return value;
}

float read_float(const cv::FileNode& config, const std::string& key, float fallback) {
    const cv::FileNode node = config[key];
    return node.empty() ? fallback : static_cast<float>(node.real());
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

cv::Scalar read_hsv_range(const cv::FileNode& config, const std::string& key,
                          const cv::Scalar& fallback) {
    const cv::FileNode range = config[key];
    if (range.empty() || !range.isSeq() || range.size() != 3) {
        return fallback;
    }
    return {range[0].real(), range[1].real(), range[2].real()};
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
    parameters.close_kernel_size =
        read_int(config["morphology"], "close_kernel_size", parameters.close_kernel_size);
    parameters.min_area = read_int(components, "min_area", parameters.min_area);
    parameters.max_area_ratio = read_float(components, "max_area_ratio", parameters.max_area_ratio);
    parameters.min_side = read_float(components, "min_side", parameters.min_side);
    parameters.max_aspect_ratio = read_float(components, "max_aspect_ratio", parameters.max_aspect_ratio);
    parameters.min_fill_ratio = read_float(components, "min_fill_ratio", parameters.min_fill_ratio);
    parameters.link_factor = read_float(grouping, "link_factor", parameters.link_factor);
    parameters.min_group_points = read_int(grouping, "min_group_points", parameters.min_group_points);
    parameters.min_points_per_layer =
        read_int(grouping, "min_points_per_layer", parameters.min_points_per_layer);
    parameters.min_layer_separation_size_ratio = read_float(
        grouping, "min_layer_separation_size_ratio", parameters.min_layer_separation_size_ratio);
    parameters.min_separation_ratio =
        read_float(grouping, "min_separation_ratio", parameters.min_separation_ratio);
    parameters.max_horizontal_offset_ratio = read_float(
        grouping, "max_horizontal_offset_ratio", parameters.max_horizontal_offset_ratio);
    return parameters;
}

unsigned int read_unsigned_int(const cv::FileNode& config, const std::string& key,
                               unsigned int fallback) {
    const cv::FileNode node = config[key];
    if (node.empty()) {
        return fallback;
    }
    int value = 0;
    node >> value;
    if (value <= 0) {
        throw std::runtime_error(key + " must be greater than zero.");
    }
    return static_cast<unsigned int>(value);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string config_path = argc > 1 ? argv[1] : "config/laser.json";
        cv::FileStorage config(config_path, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }

        std::string serial_number;
        config["camera"]["sn"] >> serial_number;
        if (serial_number.empty()) {
            throw std::runtime_error("Missing camera.sn in configuration file: " + config_path);
        }
        const wit_radar::HikCameraSettings camera_settings =
            wit_radar::read_camera_settings(config["camera"]);
        const std::string engine_path = read_required_string(config["detector"], "engine", config_path);
        const float confidence_threshold = read_float(config["detector"], "confidence_threshold", 0.25F);
        const float nms_threshold = read_float(config["detector"], "nms_threshold", 0.45F);
        const float min_content_overlap =
            read_float(config["detector"], "min_content_overlap", 0.70F);
        const bool debug_logging = read_bool(config["detector"], "debug", false);
        const unsigned int debug_every_n_frames =
            read_unsigned_int(config["detector"], "debug_every_n_frames", 1);
        const wit_radar::DeviceCenterParameters device_center_parameters =
            read_device_center_parameters(config["device_center"]);

        wit_radar::LaserDetector detector(engine_path, confidence_threshold, nms_threshold,
                                          min_content_overlap, debug_logging, debug_every_n_frames,
                                          device_center_parameters);
        wit_radar::HikCamera camera;
        camera.open_by_serial_number(serial_number, camera_settings);

        const std::string preview_window = "Laser Detection";
        const std::string roi_window = "Laser ROI";
        cv::namedWindow(preview_window, cv::WINDOW_NORMAL);
        cv::namedWindow(roi_window, cv::WINDOW_NORMAL);
        std::cout << "Laser detection started. Press Esc or q to exit.\n";

        std::uint64_t preview_frame_number = 0;
        cv::Mat no_detection_roi(240, 320, CV_8UC3, cv::Scalar::all(0));
        cv::putText(no_detection_roi, "No laser detected", cv::Point(38, 125),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(180, 180, 180), 2);

        while (true) {
            const cv::Mat frame = camera.read(1000);
            if (!frame.empty()) {
                ++preview_frame_number;
                const std::vector<wit_radar::LaserDetection> detections = detector.detect(frame);
                cv::Mat display = frame.clone();
                if (debug_logging && preview_frame_number % debug_every_n_frames == 0) {
                    std::cout << "[laser-preview][frame " << preview_frame_number << "] source="
                              << frame.cols << "x" << frame.rows << " detections="
                              << detections.size() << '\n';
                }
                if (!detections.empty()) {
                    const wit_radar::LaserDetection& best = detections.front();
                    if (debug_logging && preview_frame_number % debug_every_n_frames == 0) {
                        std::cout << "[laser-preview][frame " << preview_frame_number
                                  << "][imshow] frame(best.roi), roi=" << best.roi
                                  << " model_roi=" << best.model_roi
                                  << " score=" << best.confidence << '\n';
                    }
                    cv::rectangle(display, best.roi, cv::Scalar(0, 255, 0), 2);
                    cv::putText(display, "laser " + std::to_string(best.confidence),
                                best.roi.tl() + cv::Point(0, -8), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                cv::Scalar(0, 255, 0), 2);
                    cv::Mat roi_display = frame(best.roi).clone();
                    if (best.device_center.has_value()) {
                        const cv::Point center = best.device_center.value() -
                                                 cv::Point2f(static_cast<float>(best.roi.x),
                                                             static_cast<float>(best.roi.y));
                        cv::circle(display, best.device_center.value(), 8, cv::Scalar(255, 0, 255), -1);
                        cv::putText(display, "center", best.device_center.value() + cv::Point2f(10.0F, -10.0F),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 0, 255), 2);
                        cv::circle(roi_display, center, 8, cv::Scalar(255, 0, 255), -1);
                    }
                    cv::imshow(roi_window, roi_display);
                } else {
                    cv::imshow(roi_window, no_detection_roi);
                }
                cv::imshow(preview_window, display);
            }

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        cv::destroyWindow(roi_window);
        cv::destroyWindow(preview_window);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Laser detection failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
