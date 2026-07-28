#pragma once

#include <optional>

#include <opencv2/core.hpp>

namespace wit_radar::communication {

struct ImuOrientationQuaternion {
    float w = 1.0F;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

// Naming and multiplication order follow the Tongji-style auto-aim solver:
// p_gimbal = R_camera2gimbal * p_camera + t_camera2gimbal.
struct HandEyeCalibration {
    cv::Matx33d R_gimbal2imubody = cv::Matx33d::eye();
    cv::Matx33d R_camera2gimbal = cv::Matx33d::eye();
    cv::Vec3d t_camera2gimbal_m{0.0, 0.0, 0.0};
};

struct CoordinateTransformResult {
    cv::Point3d target_in_gimbal_m;
    cv::Point3d target_in_world_m;
    cv::Matx33d R_gimbal2world;
};

class CoordinateTransformer {
public:
    explicit CoordinateTransformer(HandEyeCalibration calibration);

    static std::optional<cv::Matx33d> quaternion_to_rotation(
        const ImuOrientationQuaternion& quaternion);

    std::optional<CoordinateTransformResult> transform(const cv::Point3d& target_in_camera_m,
                                                        const ImuOrientationQuaternion& imu_orientation) const;

    const HandEyeCalibration& calibration() const noexcept;

private:
    HandEyeCalibration calibration_;
};

HandEyeCalibration load_handeye_calibration(const cv::FileNode& handeye_config);

}  // namespace wit_radar::communication
