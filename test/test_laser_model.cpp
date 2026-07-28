#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "laser_detector.hpp"

namespace {

std::string read_required_string(const cv::FileNode& config, const std::string& key,
                                 const std::string& config_path) {
    std::string value;
    config[key] >> value;
    if (value.empty()) {
        throw std::runtime_error("Missing " + key + " in configuration file: " + config_path);
    }
    return value;
}

float read_float(const cv::FileNode& config, const std::string& key, float fallback) {
    const cv::FileNode node = config[key];
    return node.empty() ? fallback : static_cast<float>(node.real());
}

cv::Mat make_letterboxed_image(const cv::Mat& image, const cv::Size& model_size) {
    const float scale = std::min(static_cast<float>(model_size.width) / image.cols,
                                 static_cast<float>(model_size.height) / image.rows);
    const int resized_width = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    const int pad_x = (model_size.width - resized_width) / 2;
    const int pad_y = (model_size.height - resized_height) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat letterboxed(model_size, CV_8UC3, cv::Scalar::all(114));
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    return letterboxed;
}

void draw_detections(cv::Mat& image, const std::vector<wit_radar::LaserDetection>& detections,
                     bool use_model_roi, const cv::Scalar& color) {
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const wit_radar::LaserDetection& detection = detections[index];
        const cv::Rect& rectangle = use_model_roi ? detection.model_roi : detection.roi;
        cv::rectangle(image, rectangle, color, 2);
        const std::string label = std::to_string(index) + ": " +
                                  cv::format("%.3f", detection.confidence);
        cv::putText(image, label, rectangle.tl() + cv::Point(0, -6), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2);
        if (!use_model_roi && detection.device_center.has_value()) {
            cv::circle(image, detection.device_center.value(), 7, cv::Scalar(255, 0, 255), -1);
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2 || argc > 4) {
            throw std::runtime_error(
                "Usage: test_laser_model <image_path> [config_path] [output_directory]");
        }

        const std::string image_path = argv[1];
        const std::string config_path = argc > 2 ? argv[2] : "config/laser.json";
        const std::filesystem::path output_directory =
            argc > 3 ? argv[3] : "test_output/laser_model";

        cv::FileStorage config(config_path, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }
        const std::string engine_path = read_required_string(config["detector"], "engine", config_path);
        const float confidence_threshold = read_float(config["detector"], "confidence_threshold", 0.25F);
        const float nms_threshold = read_float(config["detector"], "nms_threshold", 0.45F);
        const float min_content_overlap =
            read_float(config["detector"], "min_content_overlap", 0.70F);

        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Unable to read image: " + image_path);
        }

        wit_radar::LaserDetector detector(engine_path, confidence_threshold, nms_threshold,
                                          min_content_overlap);
        const std::vector<wit_radar::LaserDetection> detections = detector.detect(image);

        std::filesystem::create_directories(output_directory);
        cv::Mat model_input = make_letterboxed_image(image, detector.input_size());
        cv::Mat source_with_roi = image.clone();
        draw_detections(model_input, detections, true, cv::Scalar(0, 255, 0));
        draw_detections(source_with_roi, detections, false, cv::Scalar(0, 0, 255));

        const std::filesystem::path model_input_path = output_directory / "01_model_input_boxes.jpg";
        const std::filesystem::path source_roi_path = output_directory / "02_source_roi_boxes.jpg";
        if (!cv::imwrite(model_input_path.string(), model_input) ||
            !cv::imwrite(source_roi_path.string(), source_with_roi)) {
            throw std::runtime_error("Unable to write diagnostic images to: " + output_directory.string());
        }

        for (std::size_t index = 0; index < detections.size(); ++index) {
            const std::filesystem::path roi_path =
                output_directory / ("03_roi_" + std::to_string(index) + ".jpg");
            if (!cv::imwrite(roi_path.string(), image(detections[index].roi))) {
                throw std::runtime_error("Unable to write ROI image: " + roi_path.string());
            }
        }

        std::cout << "Detections: " << detections.size() << '\n'
                  << "Model-space boxes: " << model_input_path << '\n'
                  << "Source-space ROI boxes: " << source_roi_path << '\n';
        for (std::size_t index = 0; index < detections.size(); ++index) {
            const wit_radar::LaserDetection& detection = detections[index];
            std::cout << "ROI " << index << ": source=" << detection.roi
                      << ", model=" << detection.model_roi << ", confidence=" << std::fixed
                      << std::setprecision(3) << detection.confidence << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Laser model test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
