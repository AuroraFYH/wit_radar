#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include <opencv2/core/mat.hpp>

namespace wit_radar {

struct HikCameraFrame {
    cv::Mat image;
    std::chrono::steady_clock::time_point received_at;
    std::uint32_t frame_number = 0;
    std::uint64_t device_timestamp = 0;
    std::int64_t host_timestamp = 0;
};

class HikCamera {
public:
    HikCamera() = default;
    ~HikCamera();

    HikCamera(const HikCamera&) = delete;
    HikCamera& operator=(const HikCamera&) = delete;
    HikCamera(HikCamera&&) = delete;
    HikCamera& operator=(HikCamera&&) = delete;

    void open(std::size_t device_index = 0);
    void open_by_serial_number(const std::string& serial_number);
    void close() noexcept;
    bool is_open() const noexcept;
    cv::Mat read(unsigned int timeout_ms = 1000);
    HikCameraFrame read_frame(unsigned int timeout_ms = 1000);

private:
    void open_device(void* device_info);

    void* handle_ = nullptr;
    bool opened_ = false;
    bool grabbing_ = false;
    bool sdk_initialized_ = false;
};

}  // namespace wit_radar
