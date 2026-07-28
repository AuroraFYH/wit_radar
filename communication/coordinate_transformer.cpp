#include "coordinate_transformer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace wit_radar::communication {
namespace {

bool finite_matrix(const cv::Matx33d& matrix) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            if (!std::isfinite(matrix(row, column))) {
                return false;
            }
        }
    }
    return true;
}

bool finite_point(const cv::Point3d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

cv::Point3d apply_rotation(const cv::Matx33d& rotation, const cv::Point3d& point) {
    const cv::Vec3d transformed = rotation * cv::Vec3d(point.x, point.y, point.z);
    return {transformed[0], transformed[1], transformed[2]};
}

}  // namespace

CoordinateTransformer::CoordinateTransformer(HandEyeCalibration calibration)
    : calibration_(std::move(calibration)) {
    if (!finite_matrix(calibration_.R_gimbal2imubody) ||
        !finite_matrix(calibration_.R_camera2gimbal)) {
        throw std::invalid_argument("Hand-eye rotation matrices must contain finite values.");
    }
}

std::optional<cv::Matx33d> CoordinateTransformer::quaternion_to_rotation(
    const ImuOrientationQuaternion& quaternion) {
    const double norm = std::sqrt(static_cast<double>(quaternion.w) * quaternion.w +
                                  static_cast<double>(quaternion.x) * quaternion.x +
                                  static_cast<double>(quaternion.y) * quaternion.y +
                                  static_cast<double>(quaternion.z) * quaternion.z);
    if (!std::isfinite(norm) || norm < 1e-8) {
        return std::nullopt;
    }

    const double w = quaternion.w / norm;
    const double x = quaternion.x / norm;
    const double y = quaternion.y / norm;
    const double z = quaternion.z / norm;
    return cv::Matx33d(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

std::optional<CoordinateTransformResult> CoordinateTransformer::transform(
    const cv::Point3d& target_in_camera_m, const ImuOrientationQuaternion& imu_orientation) const {
    if (!finite_point(target_in_camera_m)) {
        return std::nullopt;
    }
    const std::optional<cv::Matx33d> R_imubody2world = quaternion_to_rotation(imu_orientation);
    if (!R_imubody2world.has_value()) {
        return std::nullopt;
    }

    // This is exactly the rotation chain used by Tongji's Solver::set_R_gimbal2world().
    const cv::Matx33d R_gimbal2world = calibration_.R_gimbal2imubody.t() *
                                       R_imubody2world.value() * calibration_.R_gimbal2imubody;
    const cv::Point3d target_in_gimbal =
        apply_rotation(calibration_.R_camera2gimbal, target_in_camera_m) +
        cv::Point3d(calibration_.t_camera2gimbal_m[0], calibration_.t_camera2gimbal_m[1],
                    calibration_.t_camera2gimbal_m[2]);
    const cv::Point3d target_in_world = apply_rotation(R_gimbal2world, target_in_gimbal);
    return CoordinateTransformResult{target_in_gimbal, target_in_world, R_gimbal2world};
}

const HandEyeCalibration& CoordinateTransformer::calibration() const noexcept {
    return calibration_;
}

HandEyeCalibration load_handeye_calibration(const cv::FileNode& handeye_config) {
    if (handeye_config.empty()) {
        throw std::invalid_argument("Missing handeye configuration.");
    }

    auto read_matrix = [&](const std::string& key) {
        const cv::FileNode values = handeye_config[key];
        if (values.empty() || values.size() != 9) {
            throw std::invalid_argument("handeye." + key + " must contain nine row-major values.");
        }
        cv::Matx33d matrix;
        for (int index = 0; index < 9; ++index) {
            matrix(index / 3, index % 3) = values[index].real();
        }
        return matrix;
    };

    const cv::FileNode translation_values = handeye_config["t_camera2gimbal_m"];
    if (translation_values.empty() || translation_values.size() != 3) {
        throw std::invalid_argument("handeye.t_camera2gimbal_m must contain [x, y, z] in meters.");
    }

    HandEyeCalibration calibration;
    calibration.R_gimbal2imubody = read_matrix("R_gimbal2imubody");
    calibration.R_camera2gimbal = read_matrix("R_camera2gimbal");
    calibration.t_camera2gimbal_m = {translation_values[0].real(), translation_values[1].real(),
                                     translation_values[2].real()};
    return calibration;
}

}  // namespace wit_radar::communication
