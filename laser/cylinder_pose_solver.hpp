#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "device_center_finder.hpp"

namespace wit_radar {

struct CylinderPoseParameters {
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    cv::Vec<double, 5> distortion_coefficients{};
    DeviceCenterParameters light_parameters{};
    double light_square_width_m = 0.012;
    double light_square_height_m = 0.008;
    double emitting_face_radius_m = 0.025;
    double ring_center_separation_m = 0.042;
    int min_markers_per_row = 1;
    double max_reprojection_error_px = 3.0;
    double max_ring_separation_error_ratio = 0.6;
    double max_marker_axis_residual_m = 0.10;
    bool use_global_cylinder_pnp = true;
    bool allow_legacy_ippe_fallback = false;
    int markers_per_ring = 8;
    double ring_alignment_deg = 0.0;
    double max_global_reprojection_error_px = 3.0;
};

struct MarkerPose {
    std::vector<cv::Point2f> corners;
    cv::Point3d center_in_camera_m;
    cv::Point3d outward_normal_in_camera;
    cv::Point3d axis_point_in_camera_m;
    double reprojection_error_px = 0.0;
    bool upper_ring = false;
};

struct MarkerCandidateDiagnostics {
    cv::Point2f image_center;
    cv::Size2f image_size;
    bool upper_ring = false;
    int candidate_count = 0;
    double minimum_depth_m = -1.0;
    double maximum_depth_m = -1.0;
    double minimum_reprojection_error_px = -1.0;
    double maximum_reprojection_error_px = -1.0;
};

struct CylinderPoseResult {
    std::optional<cv::Point3d> target_center_in_camera_m;
    std::optional<cv::Point2f> traditional_center_pixel;
    std::optional<cv::Point2f> traditional_upper_center_pixel;
    std::optional<cv::Point2f> traditional_lower_center_pixel;
    float traditional_group_score = 0.0F;
    std::vector<cv::RotatedRect> pnp_input_rectangles;
    std::vector<MarkerCandidateDiagnostics> marker_candidate_diagnostics;
    std::vector<MarkerPose> markers;
    cv::Point3d upper_axis_point_in_camera_m;
    cv::Point3d lower_axis_point_in_camera_m;
    double average_reprojection_error_px = 0.0;
    double max_marker_axis_residual_m = 0.0;
    int upper_marker_set_count = 0;
    int lower_marker_set_count = 0;
    int upper_ippe_candidate_count = 0;
    int lower_ippe_candidate_count = 0;
    int compatible_ring_hypothesis_count = 0;
    double closest_candidate_ring_separation_m = -1.0;
    double closest_candidate_ring_separation_error_m = -1.0;
    double best_candidate_axis_residual_m = -1.0;
    int global_cylinder_hypothesis_count = 0;
    double global_cylinder_reprojection_error_px = -1.0;
    bool used_global_cylinder_pnp = false;
    std::string reason;
};

class CylinderPoseSolver {
public:
    explicit CylinderPoseSolver(CylinderPoseParameters parameters);

    CylinderPoseResult solve(const cv::Mat& bgr_roi) const;

private:
    CylinderPoseParameters parameters_;
};

}  // namespace wit_radar
