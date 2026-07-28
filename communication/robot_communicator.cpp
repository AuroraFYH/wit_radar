#include "robot_communicator.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace wit_radar::communication {
namespace {

bool has_implausible_angle_jump(const GimbalState& previous, const GimbalAngles& current,
                                std::chrono::steady_clock::time_point now) {
    const double elapsed_seconds =
        std::chrono::duration<double>(now - previous.received_at).count();
    const double yaw_difference =
        std::abs(std::remainder(static_cast<double>(current.yaw - previous.angles.yaw), 360.0));
    const double pitch_difference = std::abs(static_cast<double>(current.pitch - previous.angles.pitch));
    const double maximum_yaw_difference = std::max(20.0, 720.0 * elapsed_seconds);
    const double maximum_pitch_difference = std::max(15.0, 360.0 * elapsed_seconds);
    return yaw_difference > maximum_yaw_difference || pitch_difference > maximum_pitch_difference;
}

}  // namespace

RobotCommunicator::RobotCommunicator(SerialParameters parameters,
                                     std::chrono::milliseconds state_receive_latency)
    : serial_driver_(std::move(parameters)), state_receive_latency_(state_receive_latency) {}

RobotCommunicator::~RobotCommunicator() {
    stop();
}

void RobotCommunicator::start(StateHandler handler) {
    state_handler_ = std::move(handler);
    serial_driver_.start([this](std::vector<std::uint8_t> bytes) { consume_bytes(std::move(bytes)); });
}

void RobotCommunicator::stop() noexcept {
    serial_driver_.stop();
}

std::optional<GimbalState> RobotCommunicator::latest_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_state_;
}

std::optional<GimbalStateLookup> RobotCommunicator::state_at(
    std::chrono::steady_clock::time_point query_time,
    std::chrono::milliseconds max_interpolation_gap,
    std::chrono::milliseconds max_nearest_sample_offset) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_history_.lookup(query_time, max_interpolation_gap, max_nearest_sample_offset);
}

bool RobotCommunicator::send_command(const GimbalCommand& command, CommandWriteHandler handler) {
    return serial_driver_.write(encode_gimbal_command(command), std::move(handler));
}

void RobotCommunicator::consume_bytes(std::vector<std::uint8_t> bytes) {
    std::vector<GimbalState> received_states;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        receive_buffer_.insert(receive_buffer_.end(), bytes.begin(), bytes.end());
        while (!receive_buffer_.empty()) {
            const auto header =
                std::find(receive_buffer_.begin(), receive_buffer_.end(), kRobotStatePacketId);
            if (header == receive_buffer_.end()) {
                receive_buffer_.clear();
                break;
            }
            receive_buffer_.erase(receive_buffer_.begin(), header);
            if (receive_buffer_.size() < kRobotStatePacketSize) {
                break;
            }
            const std::optional<GimbalStateData> state_data =
                parse_gimbal_state(receive_buffer_.data(), kRobotStatePacketSize);
            if (!state_data.has_value()) {
                receive_buffer_.erase(receive_buffer_.begin());
                continue;
            }
            std::vector<std::uint8_t> raw_packet(
                receive_buffer_.begin(),
                receive_buffer_.begin() + static_cast<std::ptrdiff_t>(kRobotStatePacketSize));
            const auto received_at = std::chrono::steady_clock::now();
            if (latest_state_.has_value() &&
                has_implausible_angle_jump(latest_state_.value(), state_data->angles, received_at)) {
                receive_buffer_.erase(receive_buffer_.begin(),
                                      receive_buffer_.begin() + static_cast<std::ptrdiff_t>(kRobotStatePacketSize));
                continue;
            }
            const auto sampled_at = timestamp_mapper_.map(
                state_data->timestamp_ms, received_at, state_receive_latency_);
            GimbalState state{
                state_data->angles,
                state_data->timestamp_ms,
                std::move(raw_packet),
                sampled_at,
                received_at,
            };
            latest_state_ = state;
            state_history_.add(state.angles, sampled_at);
            received_states.push_back(state);
            receive_buffer_.erase(receive_buffer_.begin(),
                                  receive_buffer_.begin() + static_cast<std::ptrdiff_t>(kRobotStatePacketSize));
        }
        if (receive_buffer_.size() > kRobotStatePacketSize * 4) {
            receive_buffer_.erase(receive_buffer_.begin(),
                                  receive_buffer_.end() - static_cast<std::ptrdiff_t>(kRobotStatePacketSize));
        }
    }
    for (const GimbalState& state : received_states) {
        if (state_handler_) {
            state_handler_(state);
        }
    }
}

}  // namespace wit_radar::communication
