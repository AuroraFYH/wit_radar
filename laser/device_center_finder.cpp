#include "device_center_finder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace wit_radar {
namespace {

struct LightCandidate {
    cv::Point2f center;
    float width = 0.0F;
    float height = 0.0F;
    cv::RotatedRect rectangle;
};

struct DeviceGroup {
    std::vector<int> member_indices;
    cv::Point2f upper_center{};
    cv::Point2f lower_center{};
    cv::Point2f center{};
    float score = -1000.0F;
    bool valid = false;
    std::string reason;
};

float candidate_size(const LightCandidate& candidate) {
    return std::max(candidate.width, candidate.height);
}

float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    const float upper = values[middle];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle - 1), values.end());
    return (values[middle - 1] + upper) * 0.5F;
}

std::vector<LightCandidate> find_light_candidates(const cv::Mat& bgr_roi,
                                                  const DeviceCenterParameters& parameters) {
    cv::Mat hsv;
    cv::cvtColor(bgr_roi, hsv, cv::COLOR_BGR2HSV);

    cv::Mat lower_red_mask;
    cv::Mat upper_red_mask;
    cv::inRange(hsv, parameters.lower_red_1, parameters.upper_red_1, lower_red_mask);
    cv::inRange(hsv, parameters.lower_red_2, parameters.upper_red_2, upper_red_mask);

    cv::Mat red_mask;
    cv::bitwise_or(lower_red_mask, upper_red_mask, red_mask);

    cv::Mat light_mask = red_mask;
    if (parameters.enable_purple) {
        cv::Mat purple_mask;
        cv::inRange(hsv, parameters.lower_purple, parameters.upper_purple, purple_mask);
        cv::bitwise_or(light_mask, purple_mask, light_mask);
    }

    const int close_kernel_size = parameters.close_kernel_size % 2 == 0
                                      ? parameters.close_kernel_size + 1
                                      : parameters.close_kernel_size;
    if (close_kernel_size > 1) {
        const cv::Mat close_kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_kernel_size, close_kernel_size));
        cv::morphologyEx(light_mask, light_mask, cv::MORPH_CLOSE, close_kernel);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int label_count =
        cv::connectedComponentsWithStats(light_mask, labels, stats, centroids, 8, CV_32S);

    const float max_area = static_cast<float>(bgr_roi.total()) * parameters.max_area_ratio;

    std::vector<LightCandidate> candidates;
    for (int label_id = 1; label_id < label_count; ++label_id) {
        const int area = stats.at<int>(label_id, cv::CC_STAT_AREA);
        if (area < parameters.min_area || static_cast<float>(area) > max_area) {
            continue;
        }

        cv::Mat component_mask = labels == label_id;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) {
            continue;
        }
        const auto contour = std::max_element(
            contours.begin(), contours.end(), [](const std::vector<cv::Point>& first,
                                                  const std::vector<cv::Point>& second) {
                return cv::contourArea(first) < cv::contourArea(second);
            });
        const cv::RotatedRect rectangle = cv::minAreaRect(*contour);
        const float width = rectangle.size.width;
        const float height = rectangle.size.height;
        const float short_side = std::min(width, height);
        const float long_side = std::max(width, height);
        const float aspect_ratio = short_side > 0.0F ? long_side / short_side
                                                     : std::numeric_limits<float>::infinity();
        const float rectangle_area = width * height;
        const float fill_ratio = rectangle_area > 0.0F ? static_cast<float>(area) / rectangle_area : 0.0F;
        if (short_side < parameters.min_side || aspect_ratio > parameters.max_aspect_ratio ||
            fill_ratio < parameters.min_fill_ratio) {
            continue;
        }

        candidates.push_back({cv::Point2f(static_cast<float>(centroids.at<double>(label_id, 0)),
                                           static_cast<float>(centroids.at<double>(label_id, 1))),
                              width, height, rectangle});
    }
    return candidates;
}

std::vector<std::vector<int>> build_spatial_groups(const std::vector<LightCandidate>& candidates,
                                                    float& median_size,
                                                    const DeviceCenterParameters& parameters) {
    if (candidates.empty()) {
        median_size = 0.0F;
        return {};
    }

    std::vector<float> sizes;
    sizes.reserve(candidates.size());
    for (const LightCandidate& candidate : candidates) {
        sizes.push_back(candidate_size(candidate));
    }
    median_size = median(sizes);

    std::vector<std::vector<int>> adjacency(candidates.size());
    for (std::size_t first_index = 0; first_index < candidates.size(); ++first_index) {
        for (std::size_t second_index = first_index + 1; second_index < candidates.size(); ++second_index) {
            const float distance = cv::norm(candidates[first_index].center - candidates[second_index].center);
            const float local_size = std::max(
                median_size, 0.5F * (sizes[first_index] + sizes[second_index]));
            if (distance <= parameters.link_factor * local_size) {
                adjacency[first_index].push_back(static_cast<int>(second_index));
                adjacency[second_index].push_back(static_cast<int>(first_index));
            }
        }
    }

    std::vector<bool> visited(candidates.size(), false);
    std::vector<std::vector<int>> groups;
    for (std::size_t start_index = 0; start_index < candidates.size(); ++start_index) {
        if (visited[start_index]) {
            continue;
        }
        std::vector<int> group;
        std::vector<int> stack{static_cast<int>(start_index)};
        visited[start_index] = true;
        while (!stack.empty()) {
            const int current_index = stack.back();
            stack.pop_back();
            group.push_back(current_index);
            for (const int neighbor_index : adjacency[current_index]) {
                if (!visited[neighbor_index]) {
                    visited[neighbor_index] = true;
                    stack.push_back(neighbor_index);
                }
            }
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

std::optional<std::vector<int>> split_into_two_y_layers(const std::vector<float>& y_values) {
    if (y_values.size() < 2) {
        return std::nullopt;
    }
    const auto [minimum, maximum] = std::minmax_element(y_values.begin(), y_values.end());
    float first_center = *minimum;
    float second_center = *maximum;
    if (std::abs(second_center - first_center) < 1e-6F) {
        return std::nullopt;
    }

    std::vector<int> labels(y_values.size(), -1);
    for (int iteration = 0; iteration < 30; ++iteration) {
        std::vector<int> new_labels(y_values.size());
        int first_count = 0;
        int second_count = 0;
        float first_sum = 0.0F;
        float second_sum = 0.0F;
        for (std::size_t value_index = 0; value_index < y_values.size(); ++value_index) {
            const int label = std::abs(y_values[value_index] - second_center) <
                                      std::abs(y_values[value_index] - first_center)
                                  ? 1
                                  : 0;
            new_labels[value_index] = label;
            if (label == 0) {
                first_sum += y_values[value_index];
                ++first_count;
            } else {
                second_sum += y_values[value_index];
                ++second_count;
            }
        }
        if (first_count == 0 || second_count == 0) {
            return std::nullopt;
        }
        const float new_first_center = first_sum / first_count;
        const float new_second_center = second_sum / second_count;
        const bool labels_changed = new_labels != labels;
        const bool centers_changed = std::abs(new_first_center - first_center) > 1e-4F ||
                                     std::abs(new_second_center - second_center) > 1e-4F;
        labels = std::move(new_labels);
        first_center = new_first_center;
        second_center = new_second_center;
        if (!labels_changed && !centers_changed) {
            break;
        }
    }
    return labels;
}

cv::Point2f average_center(const std::vector<int>& indices,
                           const std::vector<LightCandidate>& candidates) {
    cv::Point2f center{};
    for (const int index : indices) {
        center += candidates[index].center;
    }
    return center * (1.0F / static_cast<float>(indices.size()));
}

float y_standard_deviation(const std::vector<int>& indices,
                           const std::vector<LightCandidate>& candidates) {
    const cv::Point2f center = average_center(indices, candidates);
    float sum_squared_distance = 0.0F;
    for (const int index : indices) {
        const float distance = candidates[index].center.y - center.y;
        sum_squared_distance += distance * distance;
    }
    return std::sqrt(sum_squared_distance / static_cast<float>(indices.size()));
}

DeviceGroup analyze_group(const std::vector<int>& member_indices,
                          const std::vector<LightCandidate>& candidates, float median_size,
                          const DeviceCenterParameters& parameters) {
    DeviceGroup group;
    group.member_indices = member_indices;
    if (static_cast<int>(member_indices.size()) < parameters.min_group_points) {
        group.reason = "POINTS_TOO_FEW";
        return group;
    }

    std::vector<float> y_values;
    y_values.reserve(member_indices.size());
    for (const int index : member_indices) {
        y_values.push_back(candidates[index].center.y);
    }
    const std::optional<std::vector<int>> layer_labels = split_into_two_y_layers(y_values);
    if (!layer_labels.has_value()) {
        group.reason = "LAYER_SPLIT_FAILED";
        return group;
    }

    std::vector<int> first_layer;
    std::vector<int> second_layer;
    for (std::size_t local_index = 0; local_index < member_indices.size(); ++local_index) {
        (layer_labels.value()[local_index] == 0 ? first_layer : second_layer)
            .push_back(member_indices[local_index]);
    }
    const float first_y = average_center(first_layer, candidates).y;
    const float second_y = average_center(second_layer, candidates).y;
    std::vector<int>& upper_indices = first_y <= second_y ? first_layer : second_layer;
    std::vector<int>& lower_indices = first_y <= second_y ? second_layer : first_layer;
    if (static_cast<int>(upper_indices.size()) < parameters.min_points_per_layer) {
        group.reason = "UPPER_POINTS_TOO_FEW";
        return group;
    }
    if (static_cast<int>(lower_indices.size()) < parameters.min_points_per_layer) {
        group.reason = "LOWER_POINTS_TOO_FEW";
        return group;
    }

    group.upper_center = average_center(upper_indices, candidates);
    group.lower_center = average_center(lower_indices, candidates);
    group.center = (group.upper_center + group.lower_center) * 0.5F;
    const cv::Point2f axis_vector = group.lower_center - group.upper_center;
    const float layer_separation = cv::norm(axis_vector);
    const float within_layer_spread = std::max(y_standard_deviation(upper_indices, candidates),
                                               y_standard_deviation(lower_indices, candidates));
    const float separation_ratio = layer_separation / (within_layer_spread + std::max(median_size, 1.0F));

    float minimum_x = candidates[member_indices.front()].center.x;
    float maximum_x = minimum_x;
    for (const int index : member_indices) {
        minimum_x = std::min(minimum_x, candidates[index].center.x);
        maximum_x = std::max(maximum_x, candidates[index].center.x);
    }
    const float total_x_span = std::max({maximum_x - minimum_x, median_size, 1.0F});
    const float horizontal_offset_ratio =
        std::abs(group.upper_center.x - group.lower_center.x) / total_x_span;
    const float count_balance = std::abs(static_cast<float>(upper_indices.size()) -
                                         static_cast<float>(lower_indices.size())) /
                                static_cast<float>(member_indices.size());
    group.score = static_cast<float>(member_indices.size()) + 4.0F * separation_ratio -
                  3.0F * horizontal_offset_ratio - 1.5F * count_balance;

    if (layer_separation < parameters.min_layer_separation_size_ratio * median_size) {
        group.reason = "LAYER_TOO_CLOSE";
        return group;
    }
    if (separation_ratio < parameters.min_separation_ratio) {
        group.reason = "LAYER_NOT_CLEAR";
        return group;
    }
    if (horizontal_offset_ratio > parameters.max_horizontal_offset_ratio) {
        group.reason = "CENTER_OFFSET_LARGE";
        return group;
    }
    group.valid = true;
    group.reason = "OK";
    return group;
}

}  // namespace

DeviceCenterFinder::DeviceCenterFinder(DeviceCenterParameters parameters)
    : parameters_(std::move(parameters)) {
    if (parameters_.close_kernel_size < 1 || parameters_.min_area < 1 ||
        parameters_.max_area_ratio <= 0.0F || parameters_.min_side <= 0.0F ||
        parameters_.max_aspect_ratio <= 0.0F || parameters_.min_fill_ratio < 0.0F ||
        parameters_.link_factor <= 0.0F || parameters_.min_group_points < 2 ||
        parameters_.min_points_per_layer < 1 || parameters_.min_layer_separation_size_ratio < 0.0F ||
        parameters_.min_separation_ratio < 0.0F || parameters_.max_horizontal_offset_ratio < 0.0F) {
        throw std::invalid_argument("DeviceCenterFinder received invalid parameters.");
    }
}

DeviceCenterResult DeviceCenterFinder::find(const cv::Mat& bgr_roi) const {
    DeviceCenterResult result;
    if (bgr_roi.empty() || bgr_roi.type() != CV_8UC3) {
        result.reason = "INVALID_ROI";
        return result;
    }

    const std::vector<LightCandidate> candidates = find_light_candidates(bgr_roi, parameters_);
    result.candidate_count = static_cast<int>(candidates.size());
    float median_size = 0.0F;
    const std::vector<std::vector<int>> groups =
        build_spatial_groups(candidates, median_size, parameters_);
    result.group_count = static_cast<int>(groups.size());

    std::optional<DeviceGroup> best_group;
    for (const std::vector<int>& group_indices : groups) {
        DeviceGroup group = analyze_group(group_indices, candidates, median_size, parameters_);
        if (group.valid && (!best_group.has_value() || group.score > best_group->score)) {
            best_group = std::move(group);
        }
    }
    if (!best_group.has_value()) {
        result.reason = "NO_VALID_DOUBLE_LAYER_GROUP";
        return result;
    }

    result.center = best_group->center;
    result.upper_center = best_group->upper_center;
    result.lower_center = best_group->lower_center;
    result.group_score = best_group->score;
    result.selected_rectangles.reserve(best_group->member_indices.size());
    result.selected_centers.reserve(best_group->member_indices.size());
    for (const int index : best_group->member_indices) {
        result.selected_rectangles.push_back(candidates[index].rectangle);
        result.selected_centers.push_back(candidates[index].center);
    }
    result.reason = "OK";
    return result;
}

}  // namespace wit_radar
