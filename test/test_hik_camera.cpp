#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "hik_camera.hpp"

int main(int argc, char* argv[]) {
    try {
        const std::string config_path = argc > 1 ? argv[1] : "test/test.yaml";
        cv::FileStorage config(config_path, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }

        std::string serial_number;
        config["camera"]["sn"] >> serial_number;
        if (serial_number.empty()) {
            throw std::runtime_error("Missing camera.sn in configuration file: " + config_path);
        }

        wit_radar::HikCamera camera;
        camera.open_by_serial_number(serial_number);

        const std::string window_name = "Hikvision Camera Preview";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        std::cout << "Camera " << serial_number
                  << " preview started. Press Esc or q to exit.\n";

        while (true) {
            const cv::Mat image = camera.read(1000);
            if (!image.empty()) {
                cv::imshow(window_name, image);
            }

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        cv::destroyWindow(window_name);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Hikvision capture test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
