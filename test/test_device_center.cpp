#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "device_center_finder.hpp"

int main(int argc, char* argv[]) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::runtime_error("Usage: test_device_center <roi_image_path> [output_image_path]");
        }

        const std::string image_path = argv[1];
        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Unable to read image: " + image_path);
        }

        wit_radar::DeviceCenterFinder finder;
        const wit_radar::DeviceCenterResult result = finder.find(image);
        std::cout << "light_candidates=" << result.candidate_count << " groups=" << result.group_count
                  << " score=" << result.group_score << " result=" << result.reason << '\n';
        if (!result.center.has_value()) {
            return EXIT_SUCCESS;
        }

        std::cout << "center=" << result.center.value() << " upper=" << result.upper_center
                  << " lower=" << result.lower_center << '\n';
        if (argc == 3) {
            cv::Mat annotated = image.clone();
            cv::line(annotated, result.upper_center, result.lower_center, cv::Scalar(0, 255, 255), 2);
            cv::circle(annotated, result.upper_center, 6, cv::Scalar(0, 255, 255), -1);
            cv::circle(annotated, result.lower_center, 6, cv::Scalar(0, 255, 255), -1);
            cv::circle(annotated, result.center.value(), 8, cv::Scalar(255, 0, 255), -1);
            if (!cv::imwrite(argv[2], annotated)) {
                throw std::runtime_error("Unable to write output image: " + std::string(argv[2]));
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Device center test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
