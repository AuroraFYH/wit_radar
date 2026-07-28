#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace cv {
class Mat;
}

namespace wit_radar {

struct DeviceCenterParameters {
    cv::Scalar lower_red_1{0, 80, 120};
    cv::Scalar upper_red_1{15, 255, 255};
    cv::Scalar lower_red_2{165, 80, 120};
    cv::Scalar upper_red_2{179, 255, 255};
    bool enable_purple = true;
    cv::Scalar lower_purple{125, 60, 80};
    cv::Scalar upper_purple{165, 255, 255};
    int close_kernel_size = 3;
    int min_area = 30;
    float max_area_ratio = 0.03F;
    float min_side = 3.0F;
    float max_aspect_ratio = 10.0F;
    float min_fill_ratio = 0.08F;
    float link_factor = 5.0F;
    int min_group_points = 4;
    int min_points_per_layer = 2;
    float min_layer_separation_size_ratio = 1.0F;
    float min_separation_ratio = 0.8F;
    float max_horizontal_offset_ratio = 0.65F;
};

struct DeviceCenterResult {
    std::optional<cv::Point2f> center;
    cv::Point2f upper_center{};
    cv::Point2f lower_center{};
    int candidate_count = 0;
    int group_count = 0;
    float group_score = 0.0F;
    std::vector<cv::RotatedRect> selected_rectangles;
    std::vector<cv::Point2f> selected_centers;
    std::string reason;
};

class DeviceCenterFinder {
public:
    explicit DeviceCenterFinder(DeviceCenterParameters parameters = {});
    DeviceCenterResult find(const cv::Mat& bgr_roi) const;

private:
    DeviceCenterParameters parameters_;
};

}  // namespace wit_radar
