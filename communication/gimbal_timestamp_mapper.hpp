#pragma once

#include <chrono>
#include <cstdint>

namespace wit_radar::communication {

class GimbalTimestampMapper {
public:
    std::chrono::steady_clock::time_point map(
        std::uint32_t device_timestamp_ms,
        std::chrono::steady_clock::time_point received_at,
        std::chrono::milliseconds receive_latency);

private:
    bool initialized_ = false;
    std::uint32_t last_device_timestamp_ms_ = 0;
    std::uint64_t unwrapped_device_timestamp_ms_ = 0;
    std::chrono::steady_clock::time_point host_epoch_;
};

}  // namespace wit_radar::communication
