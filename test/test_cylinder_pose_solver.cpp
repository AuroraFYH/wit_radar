#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "cylinder_pose_solver.hpp"

namespace {

bool close_to(double first, double second, double tolerance) {
    return std::abs(first - second) <= tolerance;
}

void draw_marker(cv::Mat& image, const cv::Matx33d& camera_matrix, const cv::Point3d& center) {
    const double center_x = camera_matrix(0, 0) * center.x / center.z + camera_matrix(0, 2);
    const double center_y = camera_matrix(1, 1) * center.y / center.z + camera_matrix(1, 2);
    const double width = camera_matrix(0, 0) * 0.012 / center.z;
    const double height = camera_matrix(1, 1) * 0.009 / center.z;
    const cv::Rect rectangle(cvRound(center_x - width * 0.5), cvRound(center_y - height * 0.5),
                             cvRound(width), cvRound(height));
    cv::rectangle(image, rectangle, cv::Scalar(0, 0, 255), cv::FILLED);
}

void draw_global_marker(cv::Mat& image, const cv::Matx33d& camera_matrix, double radius, double theta,
                        double axis_z, const cv::Vec3d& rotation, const cv::Vec3d& translation) {
    constexpr double marker_width = 0.012;
    constexpr double marker_height = 0.008;
    const cv::Point3d center{radius * std::cos(theta), radius * std::sin(theta), axis_z};
    const cv::Point3d tangent{-std::sin(theta), std::cos(theta), 0.0};
    const cv::Point3d axis{0.0, 0.0, 1.0};
    const std::vector<cv::Point3d> corners{
        center - tangent * (marker_width * 0.5) - axis * (marker_height * 0.5),
        center + tangent * (marker_width * 0.5) - axis * (marker_height * 0.5),
        center + tangent * (marker_width * 0.5) + axis * (marker_height * 0.5),
        center - tangent * (marker_width * 0.5) + axis * (marker_height * 0.5)};
    std::vector<cv::Point2d> projected;
    cv::projectPoints(corners, rotation, translation, camera_matrix, cv::Vec<double, 5>{}, projected);
    std::vector<cv::Point> polygon;
    polygon.reserve(projected.size());
    for (const cv::Point2d& point : projected) {
        polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    cv::fillConvexPoly(image, polygon, cv::Scalar(0, 0, 255), cv::LINE_AA);
}

}  // namespace

int main() {
    try {
        const cv::Matx33d camera_matrix{900.0, 0.0, 640.0, 0.0, 900.0, 512.0, 0.0, 0.0, 1.0};
        cv::Mat image(1024, 1280, CV_8UC3, cv::Scalar::all(0));
        draw_marker(image, camera_matrix, {-0.020, -0.0335, 0.500});
        draw_marker(image, camera_matrix, {0.020, -0.0335, 0.500});
        draw_marker(image, camera_matrix, {-0.020, 0.0335, 0.500});
        draw_marker(image, camera_matrix, {0.020, 0.0335, 0.500});

        wit_radar::CylinderPoseParameters parameters;
        parameters.camera_matrix = camera_matrix;
        parameters.light_parameters.close_kernel_size = 1;
        parameters.light_parameters.min_area = 15;
        parameters.light_parameters.min_side = 2.0F;
        parameters.light_parameters.max_area_ratio = 0.01F;
        parameters.light_parameters.max_aspect_ratio = 2.0F;
        parameters.light_parameters.min_fill_ratio = 0.6F;
        parameters.light_parameters.link_factor = 10.0F;
        parameters.light_square_height_m = 0.009;
        parameters.emitting_face_radius_m = 0.025;
        parameters.ring_center_separation_m = 0.067;
        parameters.min_markers_per_row = 1;
        parameters.max_reprojection_error_px = 2.0;
        parameters.use_global_cylinder_pnp = false;
        const wit_radar::CylinderPoseSolver solver(parameters);
        const wit_radar::CylinderPoseResult result = solver.solve(image);
        if (!result.target_center_in_camera_m.has_value()) {
            throw std::runtime_error("Synthetic cylinder PnP failed: " + result.reason);
        }
        const cv::Point3d target = result.target_center_in_camera_m.value();
        if (!close_to(target.x, 0.0, 0.015) || !close_to(target.y, 0.0, 0.015) ||
            !close_to(target.z, 0.525, 0.04)) {
            throw std::runtime_error("Synthetic cylinder PnP returned an unexpected center: " +
                                     cv::format("(%.6f, %.6f, %.6f)", target.x, target.y, target.z));
        }

        std::cout << "Cylinder PnP synthetic test: target=" << target
                  << " reprojection_error=" << result.average_reprojection_error_px
                  << " px max_axis_residual=" << result.max_marker_axis_residual_m << " m\n";

        const cv::Matx33d global_camera_matrix{3000.0, 0.0, 640.0, 0.0, 3000.0, 512.0, 0.0, 0.0, 1.0};
        cv::Mat global_image(1024, 1280, CV_8UC3, cv::Scalar::all(0));
        constexpr double pi = 3.14159265358979323846;
        constexpr double radius = 0.025;
        constexpr double separation = 0.067;
        const cv::Vec3d global_rotation(pi * 0.5, 0.0, 0.0);
        const cv::Vec3d global_translation(0.040, 0.015, 1.500);
        for (const int index : {1, 2, 3, 4}) {
            const double theta = 2.0 * pi * index / 8.0;
            draw_global_marker(global_image, global_camera_matrix, radius, theta, separation * 0.5,
                               global_rotation, global_translation);
            draw_global_marker(global_image, global_camera_matrix, radius, theta, -separation * 0.5,
                               global_rotation, global_translation);
        }
        wit_radar::CylinderPoseParameters global_parameters = parameters;
        global_parameters.camera_matrix = global_camera_matrix;
        global_parameters.light_parameters.link_factor = 20.0F;
        global_parameters.use_global_cylinder_pnp = true;
        global_parameters.markers_per_ring = 8;
        global_parameters.max_global_reprojection_error_px = 2.0;
        const wit_radar::DeviceCenterResult global_group =
            wit_radar::DeviceCenterFinder(global_parameters.light_parameters).find(global_image);
        std::cout << "Global synthetic grouping: candidates=" << global_group.candidate_count
                  << " groups=" << global_group.group_count << " reason=" << global_group.reason << '\n';
        const wit_radar::CylinderPoseResult global_result =
            wit_radar::CylinderPoseSolver(global_parameters).solve(global_image);
        if (!global_result.target_center_in_camera_m.has_value()) {
            throw std::runtime_error("Synthetic global cylinder PnP failed: " + global_result.reason);
        }
        const cv::Point3d global_target = global_result.target_center_in_camera_m.value();
        if (!close_to(global_target.x, global_translation[0], 0.05) ||
            !close_to(global_target.y, global_translation[1], 0.05) ||
            !close_to(global_target.z, global_translation[2], 0.08)) {
            throw std::runtime_error("Synthetic global cylinder PnP returned an unexpected center: " +
                                     cv::format("(%.6f, %.6f, %.6f)", global_target.x, global_target.y,
                                                global_target.z));
        }
        std::cout << "Global cylinder PnP synthetic test: target=" << global_target
                  << " reprojection_error=" << global_result.average_reprojection_error_px << " px\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Cylinder PnP synthetic test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
