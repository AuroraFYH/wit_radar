#include "gimbal_state_history.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wit_radar::communication {

GimbalStateHistory::GimbalStateHistory(std::chrono::milliseconds retention)
    : retention_(retention) {
    if (retention_ <= std::chrono::milliseconds::zero()) {
        throw std::invalid_argument("Gimbal state history retention must be positive.");
    }
}

void GimbalStateHistory::add(const GimbalAngles& angles,
                             std::chrono::steady_clock::time_point received_at) {
    if (!std::isfinite(angles.yaw) || !std::isfinite(angles.pitch)) {
        throw std::invalid_argument("Gimbal state history received non-finite angles.");
    }
    samples_.push_back({angles, received_at});
    discard_expired(received_at);
}

std::optional<GimbalStateLookup> GimbalStateHistory::lookup(
    std::chrono::steady_clock::time_point query_time,
    std::chrono::milliseconds max_interpolation_gap,
    std::chrono::milliseconds max_nearest_sample_offset) const {
    if (samples_.empty() || max_interpolation_gap < std::chrono::milliseconds::zero() ||
        max_nearest_sample_offset < std::chrono::milliseconds::zero()) {
        return std::nullopt;
    }

    const auto upper = std::lower_bound(
        samples_.begin(), samples_.end(), query_time,
        [](const Sample& sample, std::chrono::steady_clock::time_point time) {
            return sample.received_at < time;
        });
    if (upper != samples_.end() && upper->received_at == query_time) {
        return GimbalStateLookup{upper->angles, false, upper->received_at, 0.0};
    }
    if (upper != samples_.begin() && upper != samples_.end()) {
        const auto lower = std::prev(upper);
        const auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(
            upper->received_at - lower->received_at);
        if (gap <= max_interpolation_gap && gap > std::chrono::milliseconds::zero()) {
            const double fraction = std::chrono::duration<double>(query_time - lower->received_at).count() /
                                    std::chrono::duration<double>(upper->received_at - lower->received_at).count();
            const float yaw_delta = std::remainder(upper->angles.yaw - lower->angles.yaw, 360.0F);
            return GimbalStateLookup{
                {lower->angles.yaw + yaw_delta * static_cast<float>(fraction),
                 lower->angles.pitch + (upper->angles.pitch - lower->angles.pitch) *
                                           static_cast<float>(fraction)},
                true,
                query_time,
                0.0};
        }
    }

    const Sample* nearest = nullptr;
    if (upper != samples_.end()) {
        nearest = &*upper;
    }
    if (upper != samples_.begin()) {
        const Sample& previous = *std::prev(upper);
        if (nearest == nullptr || std::abs(std::chrono::duration<double, std::milli>(
                                      previous.received_at - query_time).count()) <
                                   std::abs(std::chrono::duration<double, std::milli>(
                                      nearest->received_at - query_time).count())) {
            nearest = &previous;
        }
    }
    if (nearest == nullptr) {
        return std::nullopt;
    }
    const double offset_ms =
        std::chrono::duration<double, std::milli>(nearest->received_at - query_time).count();
    if (std::abs(offset_ms) > static_cast<double>(max_nearest_sample_offset.count())) {
        return std::nullopt;
    }
    return GimbalStateLookup{nearest->angles, false, nearest->received_at, offset_ms};
}

void GimbalStateHistory::discard_expired(std::chrono::steady_clock::time_point newest_time) {
    const auto oldest_allowed = newest_time - retention_;
    while (!samples_.empty() && samples_.front().received_at < oldest_allowed) {
        samples_.pop_front();
    }
}

}  // namespace wit_radar::communication
