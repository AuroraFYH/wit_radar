#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <boost/asio.hpp>

namespace wit_radar::communication {

struct SerialParameters {
    std::string device_name;
    unsigned int baud_rate = 115200;
    unsigned int char_size = 8;
    boost::asio::serial_port_base::parity::type parity =
        boost::asio::serial_port_base::parity::none;
    boost::asio::serial_port_base::stop_bits::type stop_bits =
        boost::asio::serial_port_base::stop_bits::one;
    boost::asio::serial_port_base::flow_control::type flow_control =
        boost::asio::serial_port_base::flow_control::none;
    std::size_t read_buffer_size = 4096;
};

class SerialDriver {
public:
    using ReadHandler = std::function<void(std::vector<std::uint8_t>)>;
    using WriteHandler = std::function<void(const boost::system::error_code&, std::size_t)>;

    explicit SerialDriver(SerialParameters parameters);
    ~SerialDriver();

    SerialDriver(const SerialDriver&) = delete;
    SerialDriver& operator=(const SerialDriver&) = delete;

    void start(ReadHandler handler);
    void stop() noexcept;
    bool write(std::vector<std::uint8_t> data, WriteHandler handler = {});
    bool running() const noexcept;

private:
    void run_io();
    bool open_port();
    void close_port() noexcept;
    void start_read(std::uint64_t generation);
    void do_write(std::uint64_t generation);
    void fail_pending_writes(const boost::system::error_code& error);
    void report_error(const boost::system::error_code& error, const char* operation) const;

    struct PendingWrite {
        std::vector<std::uint8_t> data;
        WriteHandler handler;
    };

    SerialParameters parameters_;
    boost::asio::io_context io_;
    boost::asio::serial_port port_;
    std::vector<std::uint8_t> read_buffer_;
    std::deque<PendingWrite> write_queue_;
    ReadHandler read_handler_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::atomic<std::uint64_t> generation_{0};
    std::thread io_thread_;
};

}  // namespace wit_radar::communication
