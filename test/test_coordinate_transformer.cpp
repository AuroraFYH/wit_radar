#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "coordinate_transformer.hpp"

namespace {

bool close_to(double first, double second, double tolerance = 1e-6) {
    return std::abs(first - second) <= tolerance;
}

}  // namespace

int main() {
    try {
        wit_radar::communication::HandEyeCalibration calibration;
        calibration.R_camera2gimbal = cv::Matx33d::eye();
        calibration.R_gimbal2imubody = cv::Matx33d::eye();
        calibration.t_camera2gimbal_m = {0.1, -0.2, 0.3};
        const wit_radar::communication::CoordinateTransformer transformer(calibration);

        const auto identity = transformer.transform({1.0, 2.0, 3.0}, {1.0F, 0.0F, 0.0F, 0.0F});
        if (!identity.has_value() || !close_to(identity->target_in_gimbal_m.x, 1.1) ||
            !close_to(identity->target_in_gimbal_m.y, 1.8) ||
            !close_to(identity->target_in_gimbal_m.z, 3.3) ||
            !close_to(identity->target_in_world_m.x, 1.1) ||
            !close_to(identity->target_in_world_m.y, 1.8) ||
            !close_to(identity->target_in_world_m.z, 3.3)) {
            throw std::runtime_error("Identity coordinate transform test failed.");
        }

        calibration.t_camera2gimbal_m = {0.0, 0.0, 0.0};
        const wit_radar::communication::CoordinateTransformer rotation_transformer(calibration);
        const float half_sqrt_two = static_cast<float>(std::sqrt(0.5));
        const auto rotated = rotation_transformer.transform(
            {1.0, 0.0, 0.0}, {half_sqrt_two, 0.0F, 0.0F, half_sqrt_two});
        if (!rotated.has_value() || !close_to(rotated->target_in_world_m.x, 0.0, 1e-5) ||
            !close_to(rotated->target_in_world_m.y, 1.0, 1e-5) ||
            !close_to(rotated->target_in_world_m.z, 0.0, 1e-5)) {
            throw std::runtime_error("Quaternion world rotation test failed.");
        }

        cv::FileStorage config("/tmp/wit_radar_handeye_test.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        config << "handeye" << "{";
        config << "R_gimbal2imubody" << std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
        config << "R_camera2gimbal" << std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
        config << "t_camera2gimbal_m" << std::vector<double>{0.1, 0.2, 0.3};
        config << "}";
        config.release();
        cv::FileStorage read_config("/tmp/wit_radar_handeye_test.json", cv::FileStorage::READ);
        const auto loaded = wit_radar::communication::load_handeye_calibration(read_config["handeye"]);
        if (!close_to(loaded.t_camera2gimbal_m[0], 0.1) ||
            !close_to(loaded.t_camera2gimbal_m[1], 0.2) ||
            !close_to(loaded.t_camera2gimbal_m[2], 0.3)) {
            throw std::runtime_error("Hand-eye configuration load test failed.");
        }

        std::cout << "Coordinate transformer: camera->gimbal translation and quaternion world rotation passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Coordinate transformer test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
