#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "gimbal_timestamp_mapper.hpp"

int main() {
    try {
        using namespace std::chrono_literals;
        const auto start = std::chrono::steady_clock::now();
        wit_radar::communication::GimbalTimestampMapper mapper;
        const auto first = mapper.map(1000U, start + 15ms, 5ms);
        const auto second = mapper.map(1012U, start + 27ms, 5ms);
        if (first != start + 10ms || second != start + 22ms) {
            throw std::runtime_error("Device timestamp host-time mapping failed.");
        }

        wit_radar::communication::GimbalTimestampMapper wrap_mapper;
        const auto before_wrap = wrap_mapper.map(0xFFFFFFFDU, start + 5ms, 0ms);
        const auto after_wrap = wrap_mapper.map(2U, start + 10ms, 0ms);
        if (after_wrap - before_wrap != 5ms) {
            throw std::runtime_error("Device timestamp wrap handling failed.");
        }

        std::cout << "Gimbal timestamp mapper test passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Gimbal timestamp mapper test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
