#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "aim_solver.hpp"
#include "robot_protocol.hpp"

namespace {

void write_float(std::vector<std::uint8_t>& bytes, std::size_t offset, float value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void write_uint32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

float read_float(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

}  // namespace

int main() {
    try {
        std::vector<std::uint8_t> state_bytes(wit_radar::communication::kRobotStatePacketSize, 0);
        state_bytes[0] = wit_radar::communication::kRobotStatePacketId;
        write_float(state_bytes, 1, 8.0F);
        write_float(state_bytes, 5, -4.0F);
        write_uint32(state_bytes, 9, 123456U);
        const auto state = wit_radar::communication::parse_gimbal_state(state_bytes.data(), state_bytes.size());
        if (!state.has_value() || state->angles.yaw != 8.0F || state->angles.pitch != -4.0F ||
            state->timestamp_ms != 123456U) {
            throw std::runtime_error("Yaw/pitch gimbal state parse failed.");
        }
        write_float(state_bytes, 1, std::numeric_limits<float>::max());
        if (wit_radar::communication::parse_gimbal_state(state_bytes.data(), state_bytes.size()).has_value()) {
            throw std::runtime_error("Out-of-range gimbal state was accepted.");
        }

        const wit_radar::communication::GimbalCommand command{
            .yaw = 20.0F,
            .pitch = -10.0F,
            .yaw_speed = 5.0F,
            .pitch_speed = 4.0F,
            .yaw_acceleration = 8.0F,
            .pitch_acceleration = 7.0F,
            .mode = wit_radar::communication::GimbalCommandMode::kTargetDetected,
        };
        const auto command_bytes = wit_radar::communication::encode_gimbal_command(command);
        if (command_bytes.size() != wit_radar::communication::kGimbalCommandPacketSize ||
            command_bytes[0] != wit_radar::communication::kGimbalCommandPacketId ||
            read_float(command_bytes, 1) != command.yaw || read_float(command_bytes, 5) != command.pitch ||
            read_float(command_bytes, 9) != command.yaw_speed ||
            read_float(command_bytes, 13) != command.pitch_speed ||
            read_float(command_bytes, 17) != command.yaw_acceleration ||
            read_float(command_bytes, 21) != command.pitch_acceleration ||
            command_bytes[25] != static_cast<std::uint8_t>(command.mode)) {
            throw std::runtime_error("Gimbal command encoding failed.");
        }

        const auto forward = wit_radar::communication::WorldTargetSolver::solve({10.0, 0.0, 0.0}, {});
        const auto left = wit_radar::communication::WorldTargetSolver::solve({0.0, 10.0, 0.0}, {});
        const auto up = wit_radar::communication::WorldTargetSolver::solve({10.0, 0.0, 10.0}, {});
        if (!forward.has_value() || !left.has_value() || !up.has_value() ||
            std::abs(forward->target_yaw) > 1e-4F || std::abs(forward->target_pitch) > 1e-4F ||
            std::abs(left->target_yaw - 90.0F) > 1e-4F || std::abs(up->target_pitch - 45.0F) > 1e-4F) {
            throw std::runtime_error("WorldTargetSolver angle convention test failed.");
        }

        std::cout << "Yaw/pitch state bytes=" << state_bytes.size()
                  << " command packet bytes=" << command_bytes.size() << '\n'
                  << "World target solver: forward=(0,0), left=(90,0), up=(0,45) degrees\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Robot protocol test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
