#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cylinder_pose_solver.hpp"

namespace {

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

cv::Scalar read_hsv_range(const cv::FileNode& config, const std::string& key,
                          const cv::Scalar& fallback) {
    const cv::FileNode range = config[key];
    if (range.empty() || range.size() != 3) {
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

void draw_marker(cv::Mat& image, const wit_radar::MarkerPose& marker) {
    std::vector<cv::Point> corners;
    corners.reserve(marker.corners.size());
    for (const cv::Point2f& corner : marker.corners) {
        corners.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    const cv::Scalar color = marker.upper_ring ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 0);
    cv::polylines(image, corners, true, color, 2);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if ((argc < 2 || argc > 4) && argc != 6) {
            throw std::runtime_error(
                "Usage: test_cylinder_pose <roi_image> [config_path] [output_image] [roi_x roi_y]");
        }
        const std::string image_path = argv[1];
        const std::string config_path = argc > 2 ? argv[2] : "config/laser.json";
        const std::string output_path = argc > 3 ? argv[3] : "test_output/cylinder_pose.jpg";
        const int roi_x = argc == 6 ? std::stoi(argv[4]) : 0;
        const int roi_y = argc == 6 ? std::stoi(argv[5]) : 0;
        cv::FileStorage config(config_path, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }
        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Unable to read ROI image: " + image_path);
        }

        const cv::FileNode pnp = config["pnp"];
        const cv::FileNode aim = config["aim"];
        wit_radar::CylinderPoseParameters parameters;
        parameters.camera_matrix = read_matrix(config["camera"], "camera_matrix");
        parameters.camera_matrix(0, 2) -= roi_x;
        parameters.camera_matrix(1, 2) -= roi_y;
        parameters.distortion_coefficients =
            read_distortion(config["camera"]["distortion_coefficients"]);
        parameters.light_parameters = read_device_center_parameters(config["device_center"]);
        parameters.light_square_width_m = read_double(pnp, "light_square_width_m", 0.0);
        parameters.light_square_height_m = read_double(pnp, "light_square_height_m", 0.0);
        parameters.emitting_face_radius_m = read_double(pnp, "emitting_face_radius_m", 0.0);
        parameters.ring_center_separation_m = read_double(pnp, "ring_center_separation_m", 0.0);
        parameters.min_markers_per_row = read_int(aim, "min_markers_per_row", 1);
        parameters.max_reprojection_error_px =
            read_double(aim, "max_reprojection_error_px", 3.0);
        parameters.max_ring_separation_error_ratio =
            read_double(aim, "max_ring_separation_error_ratio", 0.6);
        parameters.max_marker_axis_residual_m =
            read_double(aim, "max_marker_axis_residual_m", 0.10);
        parameters.use_global_cylinder_pnp = read_bool(aim, "use_global_cylinder_pnp", true);
        parameters.markers_per_ring = read_int(pnp, "markers_per_ring", 8);
        parameters.ring_alignment_deg = read_double(pnp, "ring_alignment_deg", 0.0);
        parameters.max_global_reprojection_error_px =
            read_double(aim, "max_global_reprojection_error_px", 3.0);
        const wit_radar::CylinderPoseSolver solver(parameters);
        const wit_radar::CylinderPoseResult result = solver.solve(image);

        cv::Mat output = image.clone();
        for (const wit_radar::MarkerPose& marker : result.markers) {
            draw_marker(output, marker);
            std::cout << (marker.upper_ring ? "upper" : "lower")
                      << " marker_center=" << marker.center_in_camera_m
                      << " axis_point=" << marker.axis_point_in_camera_m
                      << " reprojection_error_px=" << marker.reprojection_error_px << '\n';
        }
        if (result.target_center_in_camera_m.has_value()) {
            const cv::Point3d target = result.target_center_in_camera_m.value();
            std::cout << "target_in_camera_m=" << target
                      << " reprojection_error_px=" << result.average_reprojection_error_px
                      << " max_axis_residual_m=" << result.max_marker_axis_residual_m
                      << " markers=" << result.markers.size() << '\n';
            cv::putText(output, cv::format("X=%.3f Y=%.3f Z=%.3fm", target.x, target.y, target.z),
                        cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        } else {
            std::cout << "PnP failed: " << result.reason << '\n';
            cv::putText(output, result.reason, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 255), 2);
        }
        if (!cv::imwrite(output_path, output)) {
            throw std::runtime_error("Unable to write output image: " + output_path);
        }
        return result.target_center_in_camera_m.has_value() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Cylinder PnP test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
