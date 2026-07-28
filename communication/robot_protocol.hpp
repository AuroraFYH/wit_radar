#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace wit_radar::communication {

struct GimbalAngles {
    float yaw = 0.0F;
    float pitch = 0.0F;
};

struct GimbalStateData {
    GimbalAngles angles;
    std::uint32_t timestamp_ms = 0;
};

enum class GimbalCommandMode : std::uint8_t {
    kNoTarget = 0,
    kTargetDetected = 1,
};

struct GimbalCommand {
    float yaw = 0.0F;
    float pitch = 0.0F;
    float yaw_speed = 0.0F;
    float pitch_speed = 0.0F;
    float yaw_acceleration = 0.0F;
    float pitch_acceleration = 0.0F;
    GimbalCommandMode mode = GimbalCommandMode::kNoTarget;
};

constexpr std::uint8_t kRobotStatePacketId = 0x02;
constexpr std::uint8_t kGimbalCommandPacketId = 0x01;
constexpr std::size_t kRobotStatePacketSize = 1 + sizeof(float) * 2 + sizeof(std::uint32_t);
constexpr std::size_t kGimbalCommandPacketSize = 1 + sizeof(float) * 6 + sizeof(std::uint8_t);

std::optional<GimbalStateData> parse_gimbal_state(const std::uint8_t* data, std::size_t size);

std::vector<std::uint8_t> encode_gimbal_command(const GimbalCommand& command);

}  // namespace wit_radar::communication
