#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "gimbal_state_history.hpp"

namespace {

bool close_to(double value, double expected, double tolerance = 1e-5) {
    return std::abs(value - expected) <= tolerance;
}

}  // namespace

int main() {
    try {
        using namespace std::chrono_literals;
        const auto start = std::chrono::steady_clock::now();
        wit_radar::communication::GimbalStateHistory history(500ms);
        history.add({179.0F, -2.0F}, start);
        history.add({-179.0F, 2.0F}, start + 20ms);

        const auto interpolated = history.lookup(start + 10ms, 30ms, 5ms);
        if (!interpolated.has_value() || !interpolated->interpolated ||
            !close_to(std::abs(interpolated->angles.yaw), 180.0) ||
            !close_to(interpolated->angles.pitch, 0.0)) {
            throw std::runtime_error("Gimbal state history interpolation failed.");
        }
        const auto nearest = history.lookup(start + 27ms, 5ms, 10ms);
        if (!nearest.has_value() || nearest->interpolated || !close_to(nearest->angles.yaw, -179.0) ||
            !close_to(nearest->nearest_sample_offset_ms, -7.0)) {
            throw std::runtime_error("Gimbal state history nearest lookup failed.");
        }
        if (history.lookup(start + 50ms, 5ms, 10ms).has_value()) {
            throw std::runtime_error("Gimbal state history accepted an overly stale sample.");
        }
        std::cout << "Gimbal state history test passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Gimbal state history test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
