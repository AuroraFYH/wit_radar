#include "robot_protocol.hpp"

#include <cmath>
#include <cstring>
#include <type_traits>

namespace wit_radar::communication {
namespace {

#pragma pack(push, 1)
struct ReceiveRobotWirePacket {
    std::uint8_t header;
    float yaw;
    float pitch;
    std::uint32_t timestamp_ms;
};

struct SendRobotWirePacket {
    std::uint8_t header;
    float yaw;
    float pitch;
    float yaw_speed;
    float pitch_speed;
    float yaw_acceleration;
    float pitch_acceleration;
    std::uint8_t mode;
};
#pragma pack(pop)

static_assert(sizeof(ReceiveRobotWirePacket) == kRobotStatePacketSize);
static_assert(sizeof(SendRobotWirePacket) == kGimbalCommandPacketSize);
static_assert(std::is_trivially_copyable_v<ReceiveRobotWirePacket>);
static_assert(std::is_trivially_copyable_v<SendRobotWirePacket>);

constexpr float kMaximumAbsoluteYawDegrees = 720.0F;
constexpr float kMaximumAbsolutePitchDegrees = 180.0F;

}  // namespace

std::optional<GimbalStateData> parse_gimbal_state(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size != kRobotStatePacketSize || data[0] != kRobotStatePacketId) {
        return std::nullopt;
    }

    ReceiveRobotWirePacket packet{};
    std::memcpy(&packet, data, sizeof(packet));
    if (!std::isfinite(packet.yaw) || !std::isfinite(packet.pitch) ||
        std::abs(packet.yaw) > kMaximumAbsoluteYawDegrees ||
        std::abs(packet.pitch) > kMaximumAbsolutePitchDegrees) {
        return std::nullopt;
    }
    return GimbalStateData{{packet.yaw, packet.pitch}, packet.timestamp_ms};
}

std::vector<std::uint8_t> encode_gimbal_command(const GimbalCommand& command) {
    const SendRobotWirePacket packet{
        .header = kGimbalCommandPacketId,
        .yaw = command.yaw,
        .pitch = command.pitch,
        .yaw_speed = command.yaw_speed,
        .pitch_speed = command.pitch_speed,
        .yaw_acceleration = command.yaw_acceleration,
        .pitch_acceleration = command.pitch_acceleration,
        .mode = static_cast<std::uint8_t>(command.mode),
    };

    std::vector<std::uint8_t> bytes(sizeof(packet));
    std::memcpy(bytes.data(), &packet, sizeof(packet));
    return bytes;
}

}  // namespace wit_radar::communication
