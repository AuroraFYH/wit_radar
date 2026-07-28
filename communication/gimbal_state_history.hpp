#pragma once

#include <chrono>
#include <deque>
#include <optional>

#include "robot_protocol.hpp"

namespace wit_radar::communication {

struct GimbalStateLookup {
    GimbalAngles angles;
    bool interpolated = false;
    std::chrono::steady_clock::time_point source_time;
    double nearest_sample_offset_ms = 0.0;
};

class GimbalStateHistory {
public:
    explicit GimbalStateHistory(std::chrono::milliseconds retention = std::chrono::seconds(2));

    void add(const GimbalAngles& angles, std::chrono::steady_clock::time_point received_at);
    std::optional<GimbalStateLookup> lookup(
        std::chrono::steady_clock::time_point query_time,
        std::chrono::milliseconds max_interpolation_gap,
        std::chrono::milliseconds max_nearest_sample_offset) const;

private:
    struct Sample {
        GimbalAngles angles;
        std::chrono::steady_clock::time_point received_at;
    };

    void discard_expired(std::chrono::steady_clock::time_point newest_time);

    std::chrono::milliseconds retention_;
    std::deque<Sample> samples_;
};

}  // namespace wit_radar::communication
