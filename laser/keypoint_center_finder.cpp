#include "keypoint_center_finder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace wit_radar {
namespace {

float probability_from_peak(float value) {
    if (value >= 0.0F && value <= 1.0F) {
        return value;
    }
    const float bounded_value = std::clamp(value, -30.0F, 30.0F);
    return 1.0F / (1.0F + std::exp(-bounded_value));
}

std::optional<cv::Point2f> heatmap_peak(const cv::Mat& heatmap, cv::Size roi_size, float* confidence) {
    double maximum_value = 0.0;
    cv::Point maximum_location;
    cv::minMaxLoc(heatmap, nullptr, &maximum_value, nullptr, &maximum_location);
    *confidence = probability_from_peak(static_cast<float>(maximum_value));
    if (!std::isfinite(*confidence)) {
        *confidence = 0.0F;
        return std::nullopt;
    }

    const int left = std::max(0, maximum_location.x - 1);
    const int right = std::min(heatmap.cols - 1, maximum_location.x + 1);
    const int top = std::max(0, maximum_location.y - 1);
    const int bottom = std::min(heatmap.rows - 1, maximum_location.y + 1);
    double minimum_value = maximum_value;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            minimum_value = std::min(minimum_value, static_cast<double>(heatmap.at<float>(y, x)));
        }
    }
    const bool probability_heatmap = minimum_value >= 0.0 && maximum_value <= 1.0;
    double total_weight = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            const float value = heatmap.at<float>(y, x);
            const double weight = probability_heatmap
                                      ? std::max(1e-6, static_cast<double>(value) - minimum_value)
                                      : std::exp(std::clamp(static_cast<double>(value - maximum_value), -30.0,
                                                            0.0));
            total_weight += weight;
            weighted_x += weight * static_cast<double>(x);
            weighted_y += weight * static_cast<double>(y);
        }
    }
    if (total_weight <= 0.0) {
        return std::nullopt;
    }
    const double peak_x = weighted_x / total_weight;
    const double peak_y = weighted_y / total_weight;
    return cv::Point2f(static_cast<float>((peak_x + 0.5) * roi_size.width / heatmap.cols - 0.5),
                       static_cast<float>((peak_y + 0.5) * roi_size.height / heatmap.rows - 0.5));
}

}  // namespace

class KeypointCenterFinder::Impl {
public:
    explicit Impl(KeypointCenterParameters parameters) : parameters_(std::move(parameters)) {
        if (!parameters_.enabled) {
            return;
        }
        if (parameters_.onnx_path.empty()) {
            throw std::invalid_argument("Keypoint center ONNX path is required when enabled.");
        }
        if (parameters_.input_size.width <= 0 || parameters_.input_size.height <= 0 ||
            parameters_.confidence_threshold < 0.0F || parameters_.confidence_threshold > 1.0F) {
            throw std::invalid_argument("Keypoint center parameters are invalid.");
        }
        network_ = cv::dnn::readNetFromONNX(parameters_.onnx_path);
        if (network_.empty()) {
            throw std::runtime_error("Unable to load keypoint center ONNX model: " + parameters_.onnx_path);
        }
    }

    KeypointCenterResult find(const cv::Mat& bgr_roi) {
        KeypointCenterResult result;
        if (!parameters_.enabled) {
            result.reason = "Keypoint model disabled.";
            return result;
        }
        if (bgr_roi.empty()) {
            result.reason = "Keypoint input ROI is empty.";
            return result;
        }

        const cv::Mat input = cv::dnn::blobFromImage(bgr_roi, 1.0 / 255.0, parameters_.input_size,
                                                      cv::Scalar(), true, false, CV_32F);
        network_.setInput(input);
        cv::Mat output = network_.forward();
        if (output.type() != CV_32F || output.dims != 4 || output.size[0] != 1 || output.size[1] < 2 ||
            output.size[2] <= 0 || output.size[3] <= 0) {
            result.reason = "Keypoint model output must have shape [1, >=2, heatmap_h, heatmap_w].";
            return result;
        }

        const int heatmap_height = output.size[2];
        const int heatmap_width = output.size[3];
        cv::Mat upper_heatmap(heatmap_height, heatmap_width, CV_32F, output.ptr<float>(0, 0));
        cv::Mat lower_heatmap(heatmap_height, heatmap_width, CV_32F, output.ptr<float>(0, 1));
        result.upper_center = heatmap_peak(upper_heatmap, bgr_roi.size(), &result.upper_confidence);
        result.lower_center = heatmap_peak(lower_heatmap, bgr_roi.size(), &result.lower_confidence);
        if (!result.upper_center.has_value() || !result.lower_center.has_value() ||
            result.upper_confidence < parameters_.confidence_threshold ||
            result.lower_confidence < parameters_.confidence_threshold) {
            result.upper_center.reset();
            result.lower_center.reset();
            result.confidence = std::min(result.upper_confidence, result.lower_confidence);
            result.reason = "Keypoint heatmap confidence is below threshold.";
            return result;
        }

        result.center = (result.upper_center.value() + result.lower_center.value()) * 0.5F;
        result.confidence = std::min(result.upper_confidence, result.lower_confidence);
        result.reason = "Keypoint upper/lower centers solved.";
        return result;
    }

    bool enabled() const noexcept {
        return parameters_.enabled;
    }

private:
    KeypointCenterParameters parameters_;
    cv::dnn::Net network_;
};

KeypointCenterFinder::KeypointCenterFinder(KeypointCenterParameters parameters)
    : impl_(std::make_unique<Impl>(std::move(parameters))) {}

KeypointCenterFinder::~KeypointCenterFinder() = default;

bool KeypointCenterFinder::enabled() const noexcept {
    return impl_->enabled();
}

KeypointCenterResult KeypointCenterFinder::find(const cv::Mat& bgr_roi) {
    return impl_->find(bgr_roi);
}

}  // namespace wit_radar
