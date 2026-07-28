#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "robot_communicator.hpp"

namespace {

std::string read_required_string(const cv::FileNode& node, const std::string& key,
                                 const std::string& config_path) {
    std::string value;
    node[key] >> value;
    if (value.empty()) {
        throw std::runtime_error("Missing " + key + " in configuration file: " + config_path);
    }
    return value;
}

int read_int(const cv::FileNode& node, const std::string& key, int fallback) {
    const cv::FileNode value = node[key];
    if (value.empty()) {
        return fallback;
    }
    int result = fallback;
    value >> result;
    return result;
}

float read_float(const cv::FileNode& node, const std::string& key, float fallback) {
    const cv::FileNode value = node[key];
    return value.empty() ? fallback : static_cast<float>(value.real());
}

std::string format_bytes(const std::vector<std::uint8_t>& bytes) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        if (index != 0) {
            output << ' ';
        }
        output << "0x" << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return output.str();
}

boost::asio::serial_port_base::parity::type parse_parity(const std::string& value) {
    if (value == "none") {
        return boost::asio::serial_port_base::parity::none;
    }
    if (value == "even") {
        return boost::asio::serial_port_base::parity::even;
    }
    if (value == "odd") {
        return boost::asio::serial_port_base::parity::odd;
    }
    throw std::runtime_error("serial.parity must be none, even, or odd.");
}

boost::asio::serial_port_base::stop_bits::type parse_stop_bits(const std::string& value) {
    if (value == "one") {
        return boost::asio::serial_port_base::stop_bits::one;
    }
    if (value == "one_point_five") {
        return boost::asio::serial_port_base::stop_bits::onepointfive;
    }
    if (value == "two") {
        return boost::asio::serial_port_base::stop_bits::two;
    }
    throw std::runtime_error("serial.stop_bits must be one, one_point_five, or two.");
}

boost::asio::serial_port_base::flow_control::type parse_flow_control(const std::string& value) {
    if (value == "none") {
        return boost::asio::serial_port_base::flow_control::none;
    }
    if (value == "software") {
        return boost::asio::serial_port_base::flow_control::software;
    }
    if (value == "hardware") {
        return boost::asio::serial_port_base::flow_control::hardware;
    }
    throw std::runtime_error("serial.flow_control must be none, software, or hardware.");
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string config_path = argc > 1 ? argv[1] : "config/laser.json";
        cv::FileStorage config(config_path, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }
        const cv::FileNode communication = config["communication"];
        if (communication.empty()) {
            throw std::runtime_error("Missing communication in configuration file: " + config_path);
        }
        const cv::FileNode serial = communication["serial"];
        wit_radar::communication::SerialParameters serial_parameters;
        serial_parameters.device_name = read_required_string(serial, "device_name", config_path);
        serial_parameters.baud_rate = static_cast<unsigned int>(read_int(serial, "baud_rate", 115200));
        serial_parameters.char_size = static_cast<unsigned int>(read_int(serial, "char_size", 8));
        serial_parameters.read_buffer_size =
            static_cast<std::size_t>(read_int(serial, "read_buffer_size", 4096));
        serial_parameters.parity = parse_parity(read_required_string(serial, "parity", config_path));
        serial_parameters.stop_bits = parse_stop_bits(read_required_string(serial, "stop_bits", config_path));
        serial_parameters.flow_control =
            parse_flow_control(read_required_string(serial, "flow_control", config_path));

        std::optional<wit_radar::communication::GimbalCommand> command;
        int send_interval_ms = 100;
        if (argc == 4) {
            const cv::FileNode command_config = communication["command"];
            wit_radar::communication::GimbalCommand target;
            target.yaw = std::stof(argv[2]);
            target.pitch = std::stof(argv[3]);
            target.yaw_speed = read_float(command_config, "yaw_speed", 0.0F);
            target.pitch_speed = read_float(command_config, "pitch_speed", 0.0F);
            target.yaw_acceleration = read_float(command_config, "yaw_acceleration", 0.0F);
            target.pitch_acceleration = read_float(command_config, "pitch_acceleration", 0.0F);
            send_interval_ms = read_int(command_config, "send_interval_ms", 100);
            if (send_interval_ms <= 0) {
                throw std::runtime_error("communication.command.send_interval_ms must be positive.");
            }
            command = target;
        } else if (argc != 1 && argc != 2) {
            throw std::runtime_error("Usage: test_robot_communication [config_path] [target_yaw target_pitch]");
        }

        wit_radar::communication::RobotCommunicator communicator(std::move(serial_parameters));
        std::mutex output_mutex;
        std::atomic<bool> received_state{false};
        auto last_receive_log_time = std::chrono::steady_clock::time_point{};
        communicator.start([&](const wit_radar::communication::GimbalState& state) {
            received_state = true;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_receive_log_time < std::chrono::seconds(1)) {
                return;
            }
            last_receive_log_time = now;
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "[RX] header=0x02 yaw=" << state.angles.yaw
                      << " pitch=" << state.angles.pitch
                      << " timestamp_ms=" << state.device_timestamp_ms << " | bytes: "
                      << format_bytes(state.raw_packet) << '\n';
        });

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "Listening for robot state. Press Ctrl+C to exit.";
        }
        if (command.has_value()) {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "\n[TX] Waiting for the first 0x02 state frame, then sending every "
                      << send_interval_ms << " ms.\n";
        }

        auto next_send_time = std::chrono::steady_clock::now();
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (command.has_value() && received_state.load() && now >= next_send_time) {
                const std::vector<std::uint8_t> command_bytes =
                    wit_radar::communication::encode_gimbal_command(command.value());
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "[TX] header=0x01 yaw=" << command->yaw << " pitch=" << command->pitch
                              << " yaw_speed=" << command->yaw_speed
                              << " pitch_speed=" << command->pitch_speed
                              << " yaw_acceleration=" << command->yaw_acceleration
                              << " pitch_acceleration=" << command->pitch_acceleration
                              << " mode=" << static_cast<int>(command->mode)
                              << " | bytes: " << format_bytes(command_bytes) << '\n';
                }
                const bool command_queued = communicator.send_command(
                    command.value(), [&](const boost::system::error_code& error, std::size_t bytes_written) {
                        std::lock_guard<std::mutex> lock(output_mutex);
                        if (error) {
                            std::cerr << "[TX] Serial write failed: " << error.message() << '\n';
                        } else {
                            std::cout << "[TX] Serial write completed: " << bytes_written << " bytes.\n";
                        }
                    });
                if (!command_queued) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cerr << "[TX] Command was not queued.\n";
                }
                next_send_time = now + std::chrono::milliseconds(send_interval_ms);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch (const std::exception& error) {
        std::cerr << "Robot communication test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
