#include "gimbal_timestamp_mapper.hpp"

#include <chrono>
#include <cstdint>

namespace wit_radar::communication {

namespace {

constexpr auto kMaxMappingDrift = std::chrono::seconds(2);

}  // namespace

std::chrono::steady_clock::time_point GimbalTimestampMapper::map(
    std::uint32_t device_timestamp_ms,
    std::chrono::steady_clock::time_point received_at,
    std::chrono::milliseconds receive_latency) {
    const auto expected_sampled_at = received_at - receive_latency;
    if (!initialized_) {
        initialized_ = true;
        last_device_timestamp_ms_ = device_timestamp_ms;
        unwrapped_device_timestamp_ms_ = device_timestamp_ms;
        host_epoch_ = expected_sampled_at - std::chrono::milliseconds(unwrapped_device_timestamp_ms_);
        return expected_sampled_at;
    }

    const std::int32_t elapsed_ms =
        static_cast<std::int32_t>(device_timestamp_ms - last_device_timestamp_ms_);
    if (elapsed_ms < 0) {
        initialized_ = false;
        return map(device_timestamp_ms, received_at, receive_latency);
    }

    last_device_timestamp_ms_ = device_timestamp_ms;
    unwrapped_device_timestamp_ms_ += static_cast<std::uint32_t>(elapsed_ms);
    const auto mapped_sampled_at = host_epoch_ + std::chrono::milliseconds(unwrapped_device_timestamp_ms_);
    const auto mapping_drift = mapped_sampled_at - expected_sampled_at;
    if (mapping_drift > kMaxMappingDrift || mapping_drift < -kMaxMappingDrift) {
        host_epoch_ = expected_sampled_at -
                      std::chrono::milliseconds(unwrapped_device_timestamp_ms_);
        return expected_sampled_at;
    }
    return mapped_sampled_at;
}

}  // namespace wit_radar::communication
