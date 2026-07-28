#include "cylinder_pose_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace wit_radar {
namespace {

struct RectangleCandidate {
    cv::Point2f center;
    std::array<cv::Point2f, 4> corners;
};

struct MarkerCandidateSet {
    bool upper_ring = false;
    std::vector<MarkerPose> poses;
};

struct GlobalMarkerSelection {
    std::vector<MarkerPose> upper_markers;
    std::vector<MarkerPose> lower_markers;
    cv::Point3d upper_center;
    cv::Point3d lower_center;
    double max_axis_residual_m = 0.0;
    double score = std::numeric_limits<double>::infinity();
};

struct CandidateSelectionDiagnostics {
    int compatible_ring_hypothesis_count = 0;
    double closest_ring_separation_m = -1.0;
    double closest_ring_separation_error_m = -1.0;
    double best_axis_residual_m = -1.0;
};

struct GlobalCylinderObservation {
    cv::Point2f image_center;
    cv::RotatedRect rectangle;
    bool upper_ring = false;
};

struct GlobalCylinderPose {
    cv::Point3d target_center_in_camera_m;
    cv::Point3d upper_axis_point_in_camera_m;
    cv::Point3d lower_axis_point_in_camera_m;
    std::vector<MarkerPose> markers;
    double average_reprojection_error_px = 0.0;
};

bool finite_value(double value) {
    return std::isfinite(value);
}

bool finite_point(const cv::Point3d& point) {
    return finite_value(point.x) && finite_value(point.y) && finite_value(point.z);
}

cv::Point3d mat_to_point(const cv::Mat& vector) {
    return {vector.at<double>(0), vector.at<double>(1), vector.at<double>(2)};
}

double dot(const cv::Point3d& first, const cv::Point3d& second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

cv::Point3d average_points(const std::vector<cv::Point3d>& points) {
    cv::Point3d sum{};
    for (const cv::Point3d& point : points) {
        sum += point;
    }
    return sum * (1.0 / static_cast<double>(points.size()));
}

cv::Mat make_camera_matrix(const CylinderPoseParameters& parameters) {
    cv::Mat camera_matrix(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            camera_matrix.at<double>(row, column) = parameters.camera_matrix(row, column);
        }
    }
    return camera_matrix;
}

cv::Mat make_distortion(const CylinderPoseParameters& parameters) {
    cv::Mat distortion(5, 1, CV_64F);
    for (int index = 0; index < 5; ++index) {
        distortion.at<double>(index) = parameters.distortion_coefficients[index];
    }
    return distortion;
}

cv::Point3d rotate_point(const cv::Mat& rotation, const cv::Point3d& point) {
    return {rotation.at<double>(0, 0) * point.x + rotation.at<double>(0, 1) * point.y +
                rotation.at<double>(0, 2) * point.z,
            rotation.at<double>(1, 0) * point.x + rotation.at<double>(1, 1) * point.y +
                rotation.at<double>(1, 2) * point.z,
            rotation.at<double>(2, 0) * point.x + rotation.at<double>(2, 1) * point.y +
                rotation.at<double>(2, 2) * point.z};
}

std::array<cv::Point3d, 4> marker_corners_in_cylinder(const cv::Point3d& center, double theta,
                                                       const CylinderPoseParameters& parameters) {
    const cv::Point3d tangent{-std::sin(theta), std::cos(theta), 0.0};
    const cv::Point3d axis{0.0, 0.0, 1.0};
    const double half_width = parameters.light_square_width_m * 0.5;
    const double half_height = parameters.light_square_height_m * 0.5;
    return {center - tangent * half_width - axis * half_height,
            center + tangent * half_width - axis * half_height,
            center + tangent * half_width + axis * half_height,
            center - tangent * half_width + axis * half_height};
}

std::array<cv::Point2f, 4> rectangle_corners(const cv::RotatedRect& rectangle) {
    std::array<cv::Point2f, 4> corners{};
    rectangle.points(corners.data());
    return corners;
}

std::array<cv::Point2f, 4> best_corner_order(const std::array<cv::Point2f, 4>& detected_corners,
                                              const std::array<cv::Point2f, 4>& projected_corners) {
    std::array<cv::Point2f, 4> best_order{};
    double best_error = std::numeric_limits<double>::infinity();
    for (int reversed = 0; reversed < 2; ++reversed) {
        for (int shift = 0; shift < 4; ++shift) {
            std::array<cv::Point2f, 4> candidate{};
            double error = 0.0;
            for (int index = 0; index < 4; ++index) {
                const int offset = reversed == 0 ? index : 4 - index;
                candidate[index] = detected_corners[(shift + offset) % 4];
                error += cv::norm(candidate[index] - projected_corners[index]);
            }
            if (error < best_error) {
                best_error = error;
                best_order = candidate;
            }
        }
    }
    return best_order;
}

bool refine_global_pose_with_marker_corners(
    const std::vector<GlobalCylinderObservation>& observations,
    const std::vector<std::array<cv::Point3d, 4>>& marker_corners, const cv::Mat& camera_matrix,
    const cv::Mat& distortion, cv::Mat& rotation_vector, cv::Mat& translation_vector) {
    for (int iteration = 0; iteration < 3; ++iteration) {
        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        object_points.reserve(observations.size() * 4);
        image_points.reserve(observations.size() * 4);
        for (std::size_t marker_index = 0; marker_index < observations.size(); ++marker_index) {
            std::vector<cv::Point2d> projected;
            cv::projectPoints(std::vector<cv::Point3d>(marker_corners[marker_index].begin(),
                                                        marker_corners[marker_index].end()),
                              rotation_vector, translation_vector, camera_matrix, distortion, projected);
            if (projected.size() != 4) {
                return false;
            }
            std::array<cv::Point2f, 4> projected_corners{};
            for (int corner_index = 0; corner_index < 4; ++corner_index) {
                projected_corners[corner_index] = {
                    static_cast<float>(projected[corner_index].x),
                    static_cast<float>(projected[corner_index].y)};
            }
            const std::array<cv::Point2f, 4> matched_corners =
                best_corner_order(rectangle_corners(observations[marker_index].rectangle), projected_corners);
            for (int corner_index = 0; corner_index < 4; ++corner_index) {
                const cv::Point3d& corner = marker_corners[marker_index][corner_index];
                object_points.emplace_back(static_cast<float>(corner.x), static_cast<float>(corner.y),
                                           static_cast<float>(corner.z));
                image_points.push_back(matched_corners[corner_index]);
            }
        }
        if (!cv::solvePnP(object_points, image_points, camera_matrix, distortion, rotation_vector,
                          translation_vector, true, cv::SOLVEPNP_ITERATIVE)) {
            return false;
        }
    }
    return true;
}

double corner_reprojection_error(const std::vector<GlobalCylinderObservation>& observations,
                                 const std::vector<std::array<cv::Point3d, 4>>& marker_corners,
                                 const cv::Mat& camera_matrix, const cv::Mat& distortion,
                                 const cv::Mat& rotation_vector, const cv::Mat& translation_vector) {
    double total_error = 0.0;
    for (std::size_t marker_index = 0; marker_index < observations.size(); ++marker_index) {
        std::vector<cv::Point2d> projected;
        cv::projectPoints(std::vector<cv::Point3d>(marker_corners[marker_index].begin(),
                                                    marker_corners[marker_index].end()),
                          rotation_vector, translation_vector, camera_matrix, distortion, projected);
        if (projected.size() != 4) {
            return std::numeric_limits<double>::infinity();
        }
        std::array<cv::Point2f, 4> projected_corners{};
        for (int corner_index = 0; corner_index < 4; ++corner_index) {
            projected_corners[corner_index] = {
                static_cast<float>(projected[corner_index].x),
                static_cast<float>(projected[corner_index].y)};
        }
        const std::array<cv::Point2f, 4> matched_corners =
            best_corner_order(rectangle_corners(observations[marker_index].rectangle), projected_corners);
        for (int corner_index = 0; corner_index < 4; ++corner_index) {
            total_error += cv::norm(projected_corners[corner_index] - matched_corners[corner_index]);
        }
    }
    return total_error / static_cast<double>(observations.size() * 4);
}

std::optional<GlobalCylinderPose> solve_global_cylinder_pose(
    std::vector<GlobalCylinderObservation> upper_observations,
    std::vector<GlobalCylinderObservation> lower_observations, const CylinderPoseParameters& parameters,
    int* hypothesis_count) {
    *hypothesis_count = 0;
    if (upper_observations.size() != lower_observations.size() || upper_observations.size() < 2 ||
        parameters.markers_per_ring < static_cast<int>(upper_observations.size())) {
        return std::nullopt;
    }
    std::sort(upper_observations.begin(), upper_observations.end(),
              [](const GlobalCylinderObservation& first, const GlobalCylinderObservation& second) {
                  return first.image_center.x < second.image_center.x;
              });
    std::sort(lower_observations.begin(), lower_observations.end(),
              [](const GlobalCylinderObservation& first, const GlobalCylinderObservation& second) {
                  return first.image_center.x < second.image_center.x;
              });

    constexpr double pi = 3.14159265358979323846;
    const int visible_per_ring = static_cast<int>(upper_observations.size());
    const double half_separation = parameters.ring_center_separation_m * 0.5;
    const double lower_alignment = parameters.ring_alignment_deg * pi / 180.0;
    const cv::Mat camera_matrix = make_camera_matrix(parameters);
    const cv::Mat distortion = make_distortion(parameters);
    std::optional<GlobalCylinderPose> best_pose;
    double best_error = std::numeric_limits<double>::infinity();

    for (int start_index = 0; start_index < parameters.markers_per_ring; ++start_index) {
        for (const int direction : {-1, 1}) {
            std::vector<cv::Point3f> object_points;
            std::vector<cv::Point2f> image_points;
            std::vector<GlobalCylinderObservation> observations;
            std::vector<cv::Point3d> object_points_double;
            std::vector<std::array<cv::Point3d, 4>> marker_corners;
            object_points.reserve(visible_per_ring * 2);
            image_points.reserve(visible_per_ring * 2);
            observations.reserve(visible_per_ring * 2);
            object_points_double.reserve(visible_per_ring * 2);
            marker_corners.reserve(visible_per_ring * 2);
            for (int ring_point = 0; ring_point < visible_per_ring; ++ring_point) {
                const int index = (start_index + direction * ring_point + parameters.markers_per_ring) %
                                  parameters.markers_per_ring;
                const double upper_theta = 2.0 * pi * index / parameters.markers_per_ring;
                const double lower_theta = upper_theta + lower_alignment;
                const cv::Point3d upper_point{parameters.emitting_face_radius_m * std::cos(upper_theta),
                                              parameters.emitting_face_radius_m * std::sin(upper_theta),
                                              half_separation};
                const cv::Point3d lower_point{parameters.emitting_face_radius_m * std::cos(lower_theta),
                                              parameters.emitting_face_radius_m * std::sin(lower_theta),
                                              -half_separation};
                object_points.emplace_back(static_cast<float>(upper_point.x), static_cast<float>(upper_point.y),
                                           static_cast<float>(upper_point.z));
                image_points.push_back(upper_observations[ring_point].image_center);
                observations.push_back(upper_observations[ring_point]);
                object_points_double.push_back(upper_point);
                marker_corners.push_back(marker_corners_in_cylinder(upper_point, upper_theta, parameters));
                object_points.emplace_back(static_cast<float>(lower_point.x), static_cast<float>(lower_point.y),
                                           static_cast<float>(lower_point.z));
                image_points.push_back(lower_observations[ring_point].image_center);
                observations.push_back(lower_observations[ring_point]);
                object_points_double.push_back(lower_point);
                marker_corners.push_back(marker_corners_in_cylinder(lower_point, lower_theta, parameters));
            }

            cv::Mat rotation_vector;
            cv::Mat translation_vector;
            if (!cv::solvePnP(object_points, image_points, camera_matrix, distortion, rotation_vector,
                              translation_vector, false, cv::SOLVEPNP_EPNP)) {
                continue;
            }
            cv::solvePnP(object_points, image_points, camera_matrix, distortion, rotation_vector,
                         translation_vector, true, cv::SOLVEPNP_ITERATIVE);
            if (!refine_global_pose_with_marker_corners(observations, marker_corners, camera_matrix, distortion,
                                                        rotation_vector, translation_vector)) {
                continue;
            }
            const cv::Point3d target_center = mat_to_point(translation_vector);
            if (!finite_point(target_center) || target_center.z <= 0.0) {
                continue;
            }
            ++*hypothesis_count;
            const double average_error = corner_reprojection_error(
                observations, marker_corners, camera_matrix, distortion, rotation_vector, translation_vector);
            if (average_error >= best_error) {
                continue;
            }

            cv::Mat rotation_matrix;
            cv::Rodrigues(rotation_vector, rotation_matrix);
            GlobalCylinderPose pose;
            pose.target_center_in_camera_m = target_center;
            pose.upper_axis_point_in_camera_m =
                rotate_point(rotation_matrix, {0.0, 0.0, half_separation}) + target_center;
            pose.lower_axis_point_in_camera_m =
                rotate_point(rotation_matrix, {0.0, 0.0, -half_separation}) + target_center;
            pose.average_reprojection_error_px = average_error;
            pose.markers.reserve(observations.size());
            for (std::size_t point_index = 0; point_index < observations.size(); ++point_index) {
                const cv::Point3d point_in_camera =
                    rotate_point(rotation_matrix, object_points_double[point_index]) + target_center;
                const double theta = std::atan2(object_points_double[point_index].y,
                                                object_points_double[point_index].x);
                const cv::Point3d normal_in_camera =
                    rotate_point(rotation_matrix, {std::cos(theta), std::sin(theta), 0.0});
                std::array<cv::Point2f, 4> corners{};
                observations[point_index].rectangle.points(corners.data());
                MarkerPose marker;
                marker.corners.assign(corners.begin(), corners.end());
                marker.center_in_camera_m = point_in_camera;
                marker.outward_normal_in_camera = normal_in_camera;
                marker.axis_point_in_camera_m = observations[point_index].upper_ring
                                                    ? pose.upper_axis_point_in_camera_m
                                                    : pose.lower_axis_point_in_camera_m;
                std::vector<cv::Point2d> projected_marker;
                cv::projectPoints(std::vector<cv::Point3d>(marker_corners[point_index].begin(),
                                                            marker_corners[point_index].end()),
                                  rotation_vector, translation_vector, camera_matrix, distortion,
                                  projected_marker);
                std::array<cv::Point2f, 4> projected_corners{};
                for (int corner_index = 0; corner_index < 4; ++corner_index) {
                    projected_corners[corner_index] = {
                        static_cast<float>(projected_marker[corner_index].x),
                        static_cast<float>(projected_marker[corner_index].y)};
                }
                const std::array<cv::Point2f, 4> matched_corners =
                    best_corner_order(corners, projected_corners);
                double marker_error = 0.0;
                for (int corner_index = 0; corner_index < 4; ++corner_index) {
                    marker_error += cv::norm(projected_corners[corner_index] - matched_corners[corner_index]);
                }
                marker.reprojection_error_px = marker_error * 0.25;
                marker.upper_ring = observations[point_index].upper_ring;
                pose.markers.push_back(std::move(marker));
            }
            best_error = average_error;
            best_pose = std::move(pose);
        }
    }
    return best_pose;
}

std::vector<RectangleCandidate> find_red_rectangles(const cv::Mat& bgr_roi,
                                                     const DeviceCenterParameters& parameters) {
    cv::Mat hsv;
    cv::cvtColor(bgr_roi, hsv, cv::COLOR_BGR2HSV);

    cv::Mat low_hue_mask;
    cv::Mat high_hue_mask;
    cv::inRange(hsv, parameters.lower_red_1, parameters.upper_red_1, low_hue_mask);
    cv::inRange(hsv, parameters.lower_red_2, parameters.upper_red_2, high_hue_mask);

    cv::Mat mask;
    cv::bitwise_or(low_hue_mask, high_hue_mask, mask);
    if (parameters.enable_purple) {
        cv::Mat purple_mask;
        cv::inRange(hsv, parameters.lower_purple, parameters.upper_purple, purple_mask);
        cv::bitwise_or(mask, purple_mask, mask);
    }
    const int kernel_size = parameters.close_kernel_size % 2 == 0
                                ? parameters.close_kernel_size + 1
                                : parameters.close_kernel_size;
    if (kernel_size > 1) {
        const cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int label_count =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    const double max_area = static_cast<double>(bgr_roi.total()) * parameters.max_area_ratio;

    std::vector<RectangleCandidate> candidates;
    for (int label_id = 1; label_id < label_count; ++label_id) {
        const int area = stats.at<int>(label_id, cv::CC_STAT_AREA);
        if (area < parameters.min_area || static_cast<double>(area) > max_area) {
            continue;
        }

        cv::Mat component_mask = labels == label_id;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) {
            continue;
        }
        const auto contour = std::max_element(
            contours.begin(), contours.end(), [](const std::vector<cv::Point>& first,
                                                  const std::vector<cv::Point>& second) {
                return cv::contourArea(first) < cv::contourArea(second);
            });
        const cv::RotatedRect rectangle = cv::minAreaRect(*contour);
        const double short_side = std::min(rectangle.size.width, rectangle.size.height);
        const double long_side = std::max(rectangle.size.width, rectangle.size.height);
        const double aspect_ratio = short_side > 0.0 ? long_side / short_side
                                                     : std::numeric_limits<double>::infinity();
        const double rectangle_area = rectangle.size.area();
        const double fill_ratio = rectangle_area > 0.0 ? static_cast<double>(area) / rectangle_area : 0.0;
        if (short_side < parameters.min_side || aspect_ratio > parameters.max_aspect_ratio ||
            fill_ratio < parameters.min_fill_ratio) {
            continue;
        }

        std::array<cv::Point2f, 4> corners{};
        rectangle.points(corners.data());
        candidates.push_back(
            {{static_cast<float>(centroids.at<double>(label_id, 0)),
              static_cast<float>(centroids.at<double>(label_id, 1))},
             corners});
    }
    return candidates;
}

std::vector<MarkerPose> solve_marker_pose_candidates(const RectangleCandidate& candidate,
                                                     const CylinderPoseParameters& parameters) {
    const float half_width = static_cast<float>(parameters.light_square_width_m * 0.5);
    const float half_height = static_cast<float>(parameters.light_square_height_m * 0.5);
    const std::vector<cv::Point3f> object_corners{{-half_width, -half_height, 0.0F},
                                                   {half_width, -half_height, 0.0F},
                                                   {half_width, half_height, 0.0F},
                                                   {-half_width, half_height, 0.0F}};
    cv::Mat camera_matrix(3, 3, CV_64F);
    cv::Mat distortion(5, 1, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            camera_matrix.at<double>(row, column) = parameters.camera_matrix(row, column);
        }
    }
    for (int index = 0; index < 5; ++index) {
        distortion.at<double>(index) = parameters.distortion_coefficients[index];
    }

    std::vector<MarkerPose> poses;
    for (int reversed = 0; reversed < 2; ++reversed) {
        for (int shift = 0; shift < 4; ++shift) {
            std::vector<cv::Point2f> image_corners;
            image_corners.reserve(4);
            for (int index = 0; index < 4; ++index) {
                const int offset = reversed == 0 ? index : 4 - index;
                image_corners.push_back(candidate.corners[(shift + offset) % 4]);
            }
            const double image_width_edge = cv::norm(image_corners[1] - image_corners[0]);
            const double image_height_edge = cv::norm(image_corners[2] - image_corners[1]);
            if (image_width_edge < image_height_edge) {
                continue;
            }

            std::vector<cv::Mat> rotation_vectors;
            std::vector<cv::Mat> translation_vectors;
            if (cv::solvePnPGeneric(object_corners, image_corners, camera_matrix, distortion,
                                    rotation_vectors, translation_vectors, false, cv::SOLVEPNP_IPPE) <= 0) {
                continue;
            }
            for (std::size_t solution_index = 0; solution_index < rotation_vectors.size(); ++solution_index) {
                const cv::Point3d center = mat_to_point(translation_vectors[solution_index]);
                if (!finite_point(center) || center.z <= 0.0) {
                    continue;
                }

                std::vector<cv::Point2f> projected_corners;
                cv::projectPoints(object_corners, rotation_vectors[solution_index],
                                  translation_vectors[solution_index], camera_matrix, distortion,
                                  projected_corners);
                double total_error = 0.0;
                for (int index = 0; index < 4; ++index) {
                    total_error += cv::norm(projected_corners[index] - image_corners[index]);
                }
                const double error = total_error / 4.0;

                cv::Mat rotation_matrix;
                cv::Rodrigues(rotation_vectors[solution_index], rotation_matrix);
                cv::Point3d normal{rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 2),
                                   rotation_matrix.at<double>(2, 2)};
                if (dot(normal, center) > 0.0) {
                    normal = -normal;
                }
                const cv::Point3d axis_point = center - normal * parameters.emitting_face_radius_m;
                if (!finite_point(normal) || !finite_point(axis_point)) {
                    continue;
                }
                poses.push_back({image_corners, center, normal, axis_point, error, false});
            }
        }
    }
    return poses;
}

std::vector<MarkerPose> select_layer_poses(const std::vector<MarkerCandidateSet>& marker_sets,
                                           const cv::Point3d& center) {
    std::vector<MarkerPose> selected;
    selected.reserve(marker_sets.size());
    for (const MarkerCandidateSet& marker_set : marker_sets) {
        if (marker_set.poses.empty()) {
            continue;
        }
        const auto best = std::min_element(
            marker_set.poses.begin(), marker_set.poses.end(), [&center](const MarkerPose& first,
                                                                         const MarkerPose& second) {
                const double first_score = cv::norm(first.axis_point_in_camera_m - center) +
                                           first.reprojection_error_px * 1e-3;
                const double second_score = cv::norm(second.axis_point_in_camera_m - center) +
                                            second.reprojection_error_px * 1e-3;
                return first_score < second_score;
            });
        selected.push_back(*best);
    }
    return selected;
}

std::optional<GlobalMarkerSelection> select_consistent_marker_poses(
    const std::vector<MarkerCandidateSet>& upper_sets, const std::vector<MarkerCandidateSet>& lower_sets,
    const CylinderPoseParameters& parameters, CandidateSelectionDiagnostics* diagnostics) {
    std::optional<GlobalMarkerSelection> best_selection;
    const double maximum_separation = parameters.ring_center_separation_m *
                                      (1.0 + parameters.max_ring_separation_error_ratio);
    for (const MarkerCandidateSet& upper_set : upper_sets) {
        for (const MarkerPose& upper_hypothesis : upper_set.poses) {
            for (const MarkerCandidateSet& lower_set : lower_sets) {
                for (const MarkerPose& lower_hypothesis : lower_set.poses) {
                    const double separation = cv::norm(upper_hypothesis.axis_point_in_camera_m -
                                                        lower_hypothesis.axis_point_in_camera_m);
                    const double separation_error =
                        std::abs(separation - parameters.ring_center_separation_m);
                    if (diagnostics != nullptr &&
                        (diagnostics->closest_ring_separation_error_m < 0.0 ||
                         separation_error < diagnostics->closest_ring_separation_error_m)) {
                        diagnostics->closest_ring_separation_m = separation;
                        diagnostics->closest_ring_separation_error_m = separation_error;
                    }
                    if (separation > maximum_separation) {
                        continue;
                    }
                    if (diagnostics != nullptr) {
                        ++diagnostics->compatible_ring_hypothesis_count;
                    }
                    cv::Point3d upper_center = upper_hypothesis.axis_point_in_camera_m;
                    cv::Point3d lower_center = lower_hypothesis.axis_point_in_camera_m;
                    std::vector<MarkerPose> selected_upper;
                    std::vector<MarkerPose> selected_lower;
                    for (int iteration = 0; iteration < 2; ++iteration) {
                        selected_upper = select_layer_poses(upper_sets, upper_center);
                        selected_lower = select_layer_poses(lower_sets, lower_center);
                        std::vector<cv::Point3d> upper_points;
                        std::vector<cv::Point3d> lower_points;
                        upper_points.reserve(selected_upper.size());
                        lower_points.reserve(selected_lower.size());
                        for (const MarkerPose& marker : selected_upper) {
                            upper_points.push_back(marker.axis_point_in_camera_m);
                        }
                        for (const MarkerPose& marker : selected_lower) {
                            lower_points.push_back(marker.axis_point_in_camera_m);
                        }
                        upper_center = average_points(upper_points);
                        lower_center = average_points(lower_points);
                    }

                    const double refined_separation = cv::norm(upper_center - lower_center);
                    const double refined_separation_error =
                        std::abs(refined_separation - parameters.ring_center_separation_m);
                    if (refined_separation > maximum_separation) {
                        continue;
                    }
                    GlobalMarkerSelection selection;
                    selection.upper_markers = std::move(selected_upper);
                    selection.lower_markers = std::move(selected_lower);
                    selection.upper_center = upper_center;
                    selection.lower_center = lower_center;
                    double squared_residual_sum = 0.0;
                    for (const MarkerPose& marker : selection.upper_markers) {
                        const double residual = cv::norm(marker.axis_point_in_camera_m - upper_center);
                        selection.max_axis_residual_m = std::max(selection.max_axis_residual_m, residual);
                        squared_residual_sum += residual * residual;
                    }
                    for (const MarkerPose& marker : selection.lower_markers) {
                        const double residual = cv::norm(marker.axis_point_in_camera_m - lower_center);
                        selection.max_axis_residual_m = std::max(selection.max_axis_residual_m, residual);
                        squared_residual_sum += residual * residual;
                    }
                    if (diagnostics != nullptr &&
                        (diagnostics->best_axis_residual_m < 0.0 ||
                         selection.max_axis_residual_m < diagnostics->best_axis_residual_m)) {
                        diagnostics->best_axis_residual_m = selection.max_axis_residual_m;
                    }
                    if (selection.max_axis_residual_m > parameters.max_marker_axis_residual_m) {
                        continue;
                    }
                    selection.score = squared_residual_sum +
                                      refined_separation_error * refined_separation_error *
                                          static_cast<double>(selection.upper_markers.size() +
                                                              selection.lower_markers.size());
                    if (!best_selection.has_value() || selection.score < best_selection->score) {
                        best_selection = std::move(selection);
                    }
                }
            }
        }
    }
    return best_selection;
}

}  // namespace

CylinderPoseSolver::CylinderPoseSolver(CylinderPoseParameters parameters)
    : parameters_(std::move(parameters)) {
    if (parameters_.light_square_width_m <= 0.0 || parameters_.light_square_height_m <= 0.0 ||
        parameters_.emitting_face_radius_m <= 0.0 || parameters_.ring_center_separation_m <= 0.0 ||
        parameters_.min_markers_per_row <= 0 || parameters_.max_reprojection_error_px <= 0.0 ||
        parameters_.max_ring_separation_error_ratio < 0.0 ||
        parameters_.max_marker_axis_residual_m <= 0.0) {
        throw std::invalid_argument("Cylinder PnP parameters must be positive.");
    }
}

CylinderPoseResult CylinderPoseSolver::solve(const cv::Mat& bgr_roi) const {
    CylinderPoseResult result;
    if (bgr_roi.empty()) {
        result.reason = "PnP input ROI is empty.";
        return result;
    }

    DeviceCenterResult grouped_lights;
    try {
        grouped_lights = DeviceCenterFinder(parameters_.light_parameters).find(bgr_roi);
    } catch (const cv::Exception& error) {
        result.reason = "Red-square grouping failed: " + std::string(error.what());
        return result;
    }
    result.traditional_center_pixel = grouped_lights.center;
    result.traditional_group_score = grouped_lights.group_score;
    if (grouped_lights.center.has_value()) {
        result.traditional_upper_center_pixel = grouped_lights.upper_center;
        result.traditional_lower_center_pixel = grouped_lights.lower_center;
    }
    if (!grouped_lights.center.has_value()) {
        result.reason = "Red-square grouping failed: " + grouped_lights.reason;
        return result;
    }
    result.pnp_input_rectangles = grouped_lights.selected_rectangles;
    if (parameters_.use_global_cylinder_pnp &&
        grouped_lights.selected_centers.size() == grouped_lights.selected_rectangles.size()) {
        std::vector<GlobalCylinderObservation> upper_observations;
        std::vector<GlobalCylinderObservation> lower_observations;
        for (std::size_t index = 0; index < grouped_lights.selected_centers.size(); ++index) {
            const cv::Point2f center = grouped_lights.selected_centers[index];
            const bool upper_ring = cv::norm(center - grouped_lights.upper_center) <=
                                    cv::norm(center - grouped_lights.lower_center);
            GlobalCylinderObservation observation{center, grouped_lights.selected_rectangles[index], upper_ring};
            if (upper_ring) {
                upper_observations.push_back(observation);
            } else {
                lower_observations.push_back(observation);
            }
        }
        result.upper_marker_set_count = static_cast<int>(upper_observations.size());
        result.lower_marker_set_count = static_cast<int>(lower_observations.size());
        int global_hypothesis_count = 0;
        const std::optional<GlobalCylinderPose> global_pose = solve_global_cylinder_pose(
            upper_observations, lower_observations, parameters_, &global_hypothesis_count);
        result.global_cylinder_hypothesis_count = global_hypothesis_count;
        if (global_pose.has_value()) {
            result.global_cylinder_reprojection_error_px = global_pose->average_reprojection_error_px;
            if (global_pose->average_reprojection_error_px <= parameters_.max_global_reprojection_error_px) {
                result.target_center_in_camera_m = global_pose->target_center_in_camera_m;
                result.upper_axis_point_in_camera_m = global_pose->upper_axis_point_in_camera_m;
                result.lower_axis_point_in_camera_m = global_pose->lower_axis_point_in_camera_m;
                result.markers = global_pose->markers;
                result.average_reprojection_error_px = global_pose->average_reprojection_error_px;
                result.max_marker_axis_residual_m = 0.0;
                result.used_global_cylinder_pnp = true;
                result.reason = "Cylinder center solved from global double-ring PnP.";
                return result;
            }
        }
        if (!parameters_.allow_legacy_ippe_fallback) {
            result.reason = global_pose.has_value()
                                ? cv::format("Global cylinder PnP reprojection error %.3fpx exceeds %.3fpx.",
                                             global_pose->average_reprojection_error_px,
                                             parameters_.max_global_reprojection_error_px)
                                : "Global cylinder PnP could not form a valid double-ring pose.";
            return result;
        }
    }
    std::vector<RectangleCandidate> rectangles;
    rectangles.reserve(grouped_lights.selected_rectangles.size());
    for (const cv::RotatedRect& rectangle : grouped_lights.selected_rectangles) {
        RectangleCandidate candidate;
        candidate.center = rectangle.center;
        rectangle.points(candidate.corners.data());
        rectangles.push_back(candidate);
    }
    if (rectangles.size() < static_cast<std::size_t>(parameters_.min_markers_per_row * 2)) {
        result.reason = "Not enough red light squares for upper and lower PnP rows.";
        return result;
    }

    std::vector<MarkerCandidateSet> upper_marker_sets;
    std::vector<MarkerCandidateSet> lower_marker_sets;
    for (const RectangleCandidate& rectangle : rectangles) {
        std::vector<MarkerPose> marker_candidates;
        try {
            marker_candidates = solve_marker_pose_candidates(rectangle, parameters_);
        } catch (const cv::Exception& error) {
            result.reason = "Light-square PnP failed: " + std::string(error.what());
            return result;
        }
        marker_candidates.erase(std::remove_if(marker_candidates.begin(), marker_candidates.end(),
                                                [this](const MarkerPose& marker) {
                                                    return marker.reprojection_error_px >
                                                           parameters_.max_reprojection_error_px;
                                                }),
                                marker_candidates.end());
        if (marker_candidates.empty()) {
            continue;
        }
        const bool upper_ring = cv::norm(rectangle.center - grouped_lights.upper_center) <=
                                cv::norm(rectangle.center - grouped_lights.lower_center);
        for (MarkerPose& marker : marker_candidates) {
            marker.upper_ring = upper_ring;
        }
        MarkerCandidateDiagnostics marker_diagnostics;
        marker_diagnostics.image_center = rectangle.center;
        const cv::RotatedRect input_rectangle(rectangle.corners[0], rectangle.corners[1],
                                              rectangle.corners[2]);
        marker_diagnostics.image_size = input_rectangle.size;
        marker_diagnostics.upper_ring = upper_ring;
        marker_diagnostics.candidate_count = static_cast<int>(marker_candidates.size());
        for (const MarkerPose& marker : marker_candidates) {
            if (marker_diagnostics.minimum_depth_m < 0.0 ||
                marker.center_in_camera_m.z < marker_diagnostics.minimum_depth_m) {
                marker_diagnostics.minimum_depth_m = marker.center_in_camera_m.z;
            }
            if (marker.center_in_camera_m.z > marker_diagnostics.maximum_depth_m) {
                marker_diagnostics.maximum_depth_m = marker.center_in_camera_m.z;
            }
            if (marker_diagnostics.minimum_reprojection_error_px < 0.0 ||
                marker.reprojection_error_px < marker_diagnostics.minimum_reprojection_error_px) {
                marker_diagnostics.minimum_reprojection_error_px = marker.reprojection_error_px;
            }
            if (marker.reprojection_error_px > marker_diagnostics.maximum_reprojection_error_px) {
                marker_diagnostics.maximum_reprojection_error_px = marker.reprojection_error_px;
            }
        }
        result.marker_candidate_diagnostics.push_back(marker_diagnostics);
        if (upper_ring) {
            upper_marker_sets.push_back({true, std::move(marker_candidates)});
        } else {
            lower_marker_sets.push_back({false, std::move(marker_candidates)});
        }
    }

    result.upper_marker_set_count = static_cast<int>(upper_marker_sets.size());
    result.lower_marker_set_count = static_cast<int>(lower_marker_sets.size());
    for (const MarkerCandidateSet& marker_set : upper_marker_sets) {
        result.upper_ippe_candidate_count += static_cast<int>(marker_set.poses.size());
    }
    for (const MarkerCandidateSet& marker_set : lower_marker_sets) {
        result.lower_ippe_candidate_count += static_cast<int>(marker_set.poses.size());
    }
    if (upper_marker_sets.size() < static_cast<std::size_t>(parameters_.min_markers_per_row) ||
        lower_marker_sets.size() < static_cast<std::size_t>(parameters_.min_markers_per_row)) {
        result.reason = "PnP did not retain enough valid light squares in both rows.";
        return result;
    }
    CandidateSelectionDiagnostics diagnostics;
    const std::optional<GlobalMarkerSelection> selection =
        select_consistent_marker_poses(upper_marker_sets, lower_marker_sets, parameters_, &diagnostics);
    result.compatible_ring_hypothesis_count = diagnostics.compatible_ring_hypothesis_count;
    result.closest_candidate_ring_separation_m = diagnostics.closest_ring_separation_m;
    result.closest_candidate_ring_separation_error_m = diagnostics.closest_ring_separation_error_m;
    result.best_candidate_axis_residual_m = diagnostics.best_axis_residual_m;
    if (!selection.has_value()) {
        result.reason = cv::format("PnP IPPE candidates cannot form consistent rings (max axis residual %.3fm).",
                                   parameters_.max_marker_axis_residual_m);
        return result;
    }
    result.upper_axis_point_in_camera_m = selection->upper_center;
    result.lower_axis_point_in_camera_m = selection->lower_center;
    result.max_marker_axis_residual_m = selection->max_axis_residual_m;
    result.markers = selection->upper_markers;
    result.markers.insert(result.markers.end(), selection->lower_markers.begin(),
                          selection->lower_markers.end());
    const double measured_separation =
        cv::norm(result.upper_axis_point_in_camera_m - result.lower_axis_point_in_camera_m);
    if (std::abs(measured_separation - parameters_.ring_center_separation_m) >
        parameters_.ring_center_separation_m * parameters_.max_ring_separation_error_ratio) {
        result.reason = cv::format("PnP upper/lower ring separation %.4fm differs from configured %.4fm.",
                                   measured_separation, parameters_.ring_center_separation_m);
        return result;
    }

    result.target_center_in_camera_m =
        (result.upper_axis_point_in_camera_m + result.lower_axis_point_in_camera_m) * 0.5;
    double total_error = 0.0;
    for (const MarkerPose& marker : result.markers) {
        total_error += marker.reprojection_error_px;
    }
    result.average_reprojection_error_px = total_error / static_cast<double>(result.markers.size());
    result.reason = "Cylinder center solved from globally consistent IPPE candidates.";
    return result;
}

}  // namespace wit_radar
