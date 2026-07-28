#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

#include "gimbal_state_history.hpp"
#include "gimbal_timestamp_mapper.hpp"
#include "robot_protocol.hpp"
#include "serial_driver.hpp"

namespace wit_radar::communication {

struct GimbalState {
    GimbalAngles angles;
    std::uint32_t device_timestamp_ms = 0;
    std::vector<std::uint8_t> raw_packet;
    std::chrono::steady_clock::time_point sampled_at;
    std::chrono::steady_clock::time_point received_at;
};

class RobotCommunicator {
public:
    using StateHandler = std::function<void(const GimbalState&)>;
    using CommandWriteHandler = SerialDriver::WriteHandler;

    explicit RobotCommunicator(
        SerialParameters parameters,
        std::chrono::milliseconds state_receive_latency = std::chrono::milliseconds(0));
    ~RobotCommunicator();

    RobotCommunicator(const RobotCommunicator&) = delete;
    RobotCommunicator& operator=(const RobotCommunicator&) = delete;

    void start(StateHandler handler = {});
    void stop() noexcept;
    std::optional<GimbalState> latest_state() const;
    std::optional<GimbalStateLookup> state_at(
        std::chrono::steady_clock::time_point query_time,
        std::chrono::milliseconds max_interpolation_gap,
        std::chrono::milliseconds max_nearest_sample_offset) const;
    bool send_command(const GimbalCommand& command, CommandWriteHandler handler = {});

private:
    void consume_bytes(std::vector<std::uint8_t> bytes);

    SerialDriver serial_driver_;
    mutable std::mutex mutex_;
    std::vector<std::uint8_t> receive_buffer_;
    std::optional<GimbalState> latest_state_;
    GimbalStateHistory state_history_;
    GimbalTimestampMapper timestamp_mapper_;
    std::chrono::milliseconds state_receive_latency_;
    StateHandler state_handler_;
};

}  // namespace wit_radar::communication
