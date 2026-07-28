#include "serial_driver.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace wit_radar::communication {

SerialDriver::SerialDriver(SerialParameters parameters)
    : parameters_(std::move(parameters)), port_(io_), read_buffer_(parameters_.read_buffer_size) {
    if (parameters_.device_name.empty() || parameters_.read_buffer_size == 0) {
        throw std::invalid_argument("SerialDriver requires a device name and a non-empty read buffer.");
    }
}

SerialDriver::~SerialDriver() {
    stop();
}

void SerialDriver::start(ReadHandler handler) {
    if (!handler) {
        throw std::invalid_argument("SerialDriver requires a read handler.");
    }
    if (running_.exchange(true)) {
        return;
    }
    read_handler_ = std::move(handler);
    io_thread_ = std::thread([this] { run_io(); });
}

void SerialDriver::stop() noexcept {
    if (!running_.exchange(false)) {
        return;
    }
    boost::asio::post(io_, [this] {
        boost::system::error_code error;
        port_.cancel(error);
        port_.close(error);
    });
    io_.stop();
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

bool SerialDriver::write(std::vector<std::uint8_t> data, WriteHandler handler) {
    if (data.empty() || !running() || !connected_.load()) {
        return false;
    }
    boost::asio::post(io_, [this, data = std::move(data), handler = std::move(handler)]() mutable {
        if (!running()) {
            return;
        }
        const bool idle = write_queue_.empty();
        write_queue_.push_back(PendingWrite{std::move(data), std::move(handler)});
        if (idle && port_.is_open()) {
            do_write(generation_.load());
        }
    });
    return true;
}

bool SerialDriver::running() const noexcept {
    return running_.load();
}

void SerialDriver::run_io() {
    while (running()) {
        if (!open_port()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        const std::uint64_t generation = ++generation_;
        std::fill(read_buffer_.begin(), read_buffer_.end(), 0);
        start_read(generation);
        io_.run();
        io_.restart();
        close_port();
        if (running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
}

bool SerialDriver::open_port() {
    boost::system::error_code error;
    port_.open(parameters_.device_name, error);
    if (error) {
        report_error(error, "open");
        return false;
    }
    port_.set_option(boost::asio::serial_port_base::baud_rate(parameters_.baud_rate), error);
    port_.set_option(boost::asio::serial_port_base::character_size(parameters_.char_size), error);
    port_.set_option(boost::asio::serial_port_base::parity(parameters_.parity), error);
    port_.set_option(boost::asio::serial_port_base::stop_bits(parameters_.stop_bits), error);
    port_.set_option(boost::asio::serial_port_base::flow_control(parameters_.flow_control), error);
    if (error) {
        report_error(error, "configure");
        close_port();
        return false;
    }
    connected_.store(true);
    std::cout << "Serial connected: " << parameters_.device_name << '\n';
    return true;
}

void SerialDriver::close_port() noexcept {
    connected_.store(false);
    boost::system::error_code error;
    port_.cancel(error);
    port_.close(error);
}

void SerialDriver::start_read(std::uint64_t generation) {
    port_.async_read_some(boost::asio::buffer(read_buffer_),
                          [this, generation](const boost::system::error_code& error, std::size_t size) {
                              if (generation != generation_.load()) {
                                  return;
                              }
                              if (error) {
                                  if (error != boost::asio::error::operation_aborted) {
                                      report_error(error, "read");
                                      fail_pending_writes(error);
                                  }
                                  io_.stop();
                                  return;
                              }
                              if (size > 0 && read_handler_) {
                                  read_handler_(std::vector<std::uint8_t>(read_buffer_.begin(),
                                                                          read_buffer_.begin() + size));
                              }
                              start_read(generation);
                          });
}

void SerialDriver::do_write(std::uint64_t generation) {
    if (write_queue_.empty()) {
        return;
    }
    boost::asio::async_write(port_, boost::asio::buffer(write_queue_.front().data),
                             [this, generation](const boost::system::error_code& error, std::size_t) {
                                 if (generation != generation_.load()) {
                                     return;
                                 }
                                 if (error) {
                                     if (error != boost::asio::error::operation_aborted) {
                                         report_error(error, "write");
                                     }
                                     fail_pending_writes(error);
                                     io_.stop();
                                     return;
                                 }
                                 WriteHandler handler = std::move(write_queue_.front().handler);
                                 const std::size_t bytes_written = write_queue_.front().data.size();
                                 write_queue_.pop_front();
                                 if (handler) {
                                     handler({}, bytes_written);
                                 }
                                 do_write(generation);
                             });
}

void SerialDriver::fail_pending_writes(const boost::system::error_code& error) {
    while (!write_queue_.empty()) {
        WriteHandler handler = std::move(write_queue_.front().handler);
        write_queue_.pop_front();
        if (handler) {
            handler(error, 0);
        }
    }
}

void SerialDriver::report_error(const boost::system::error_code& error, const char* operation) const {
    if (error && error != boost::asio::error::operation_aborted) {
        std::cerr << "Serial " << operation << " error: " << error.message() << '\n';
    }
}

}  // namespace wit_radar::communication
