#pragma once

#include <memory>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

namespace wit_radar {

struct KeypointCenterParameters {
    bool enabled = false;
    std::string onnx_path;
    cv::Size input_size{192, 192};
    float confidence_threshold = 0.35F;
};

struct KeypointCenterResult {
    std::optional<cv::Point2f> center;
    std::optional<cv::Point2f> upper_center;
    std::optional<cv::Point2f> lower_center;
    float upper_confidence = 0.0F;
    float lower_confidence = 0.0F;
    float confidence = 0.0F;
    std::string reason;
};

class KeypointCenterFinder {
public:
    explicit KeypointCenterFinder(KeypointCenterParameters parameters);
    ~KeypointCenterFinder();

    KeypointCenterFinder(const KeypointCenterFinder&) = delete;
    KeypointCenterFinder& operator=(const KeypointCenterFinder&) = delete;

    bool enabled() const noexcept;
    KeypointCenterResult find(const cv::Mat& bgr_roi);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace wit_radar
