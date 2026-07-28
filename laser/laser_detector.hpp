#pragma once

#include <optional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "device_center_finder.hpp"

namespace wit_radar {

struct LaserDetection {
    cv::Rect roi;
    cv::Rect model_roi;
    float confidence = 0.0F;
    int class_id = -1;
    std::optional<cv::Point2f> device_center;
    int light_candidate_count = 0;
    int light_group_count = 0;
    float light_group_score = 0.0F;
    std::string device_center_reason;
};

class LaserDetector {
public:
    LaserDetector(const std::string& engine_path, float confidence_threshold = 0.25F,
                  float nms_threshold = 0.45F, float min_content_overlap = 0.70F,
                  bool debug_logging = false,
                  unsigned int debug_every_n_frames = 1,
                  DeviceCenterParameters device_center_parameters = {});
    ~LaserDetector();

    LaserDetector(const LaserDetector&) = delete;
    LaserDetector& operator=(const LaserDetector&) = delete;

    std::vector<LaserDetection> detect(const cv::Mat& bgr_image);
    cv::Size input_size() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace wit_radar
