#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "hik_camera.hpp"

namespace {

struct Parameters {
    cv::Scalar lower_red_1{0, 80, 120};
    cv::Scalar upper_red_1{15, 255, 255};
    cv::Scalar lower_red_2{165, 80, 120};
    cv::Scalar upper_red_2{179, 255, 255};
    bool enable_purple = true;
    cv::Scalar lower_purple{125, 60, 80};
    cv::Scalar upper_purple{165, 255, 255};
    int close_kernel_size = 3;
    int min_area = 30;
    float max_area_ratio = 0.03F;
    float min_side = 3.0F;
    float max_aspect_ratio = 10.0F;
    float min_fill_ratio = 0.08F;
    float link_factor = 5.0F;
    int min_group_points = 4;
    int min_points_per_layer = 2;
    float min_layer_separation_size_ratio = 1.0F;
    float min_separation_ratio = 0.8F;
    float max_horizontal_offset_ratio = 0.65F;
};

struct Component {
    int label_id = 0;
    cv::Point2f center{};
    int area = 0;
    cv::RotatedRect rectangle;
    float aspect_ratio = 0.0F;
    float fill_ratio = 0.0F;
    bool accepted = false;
    std::string reason;
};

struct DeviceGroup {
    std::vector<int> member_indices;
    std::vector<int> upper_indices;
    std::vector<int> lower_indices;
    cv::Point2f upper_center{};
    cv::Point2f lower_center{};
    cv::Point2f center{};
    float separation = 0.0F;
    float separation_ratio = 0.0F;
    float horizontal_offset_ratio = 1.0F;
    float score = -1000.0F;
    bool valid = false;
    std::string reason;
};

struct PipelineResult {
    cv::Mat original;
    cv::Mat gray;
    cv::Mat hue;
    cv::Mat saturation;
    cv::Mat value;
    cv::Mat lower_red_mask;
    cv::Mat upper_red_mask;
    cv::Mat red_mask;
    cv::Mat purple_mask;
    cv::Mat light_mask;
    cv::Mat closed_mask;
    cv::Mat all_components;
    cv::Mat accepted_components;
    cv::Mat spatial_groups;
    cv::Mat best_device_center;
    std::vector<Component> components;
    std::vector<Component> candidates;
    std::vector<std::vector<int>> groups;
    std::optional<DeviceGroup> best_group;
    float median_component_size = 0.0F;
    std::string failure_reason;
};

std::string read_string(const cv::FileNode& node, const std::string& key,
                        const std::string& fallback = {}) {
    const cv::FileNode value_node = node[key];
    if (value_node.empty()) {
        return fallback;
    }
    std::string value;
    value_node >> value;
    return value;
}

int read_int(const cv::FileNode& node, const std::string& key, int fallback) {
    const cv::FileNode value_node = node[key];
    if (value_node.empty()) {
        return fallback;
    }
    int value = fallback;
    value_node >> value;
    return value;
}

bool read_bool(const cv::FileNode& node, const std::string& key, bool fallback) {
    const cv::FileNode value_node = node[key];
    if (value_node.empty()) {
        return fallback;
    }
    int value = fallback ? 1 : 0;
    value_node >> value;
    return value != 0;
}

float read_float(const cv::FileNode& node, const std::string& key, float fallback) {
    const cv::FileNode value_node = node[key];
    return value_node.empty() ? fallback : static_cast<float>(value_node.real());
}

cv::Scalar read_hsv_range(const cv::FileNode& node, const std::string& key,
                          const cv::Scalar& fallback) {
    const cv::FileNode range_node = node[key];
    if (range_node.empty() || !range_node.isSeq() || range_node.size() != 3) {
        return fallback;
    }
    return {range_node[0].real(), range_node[1].real(), range_node[2].real()};
}

Parameters load_parameters(const cv::FileNode& config) {
    Parameters parameters;
    const cv::FileNode hsv = config["hsv"];
    const cv::FileNode components = config["components"];
    const cv::FileNode grouping = config["grouping"];
    parameters.lower_red_1 = read_hsv_range(hsv, "lower_red_1", parameters.lower_red_1);
    parameters.upper_red_1 = read_hsv_range(hsv, "upper_red_1", parameters.upper_red_1);
    parameters.lower_red_2 = read_hsv_range(hsv, "lower_red_2", parameters.lower_red_2);
    parameters.upper_red_2 = read_hsv_range(hsv, "upper_red_2", parameters.upper_red_2);
    parameters.enable_purple = read_bool(hsv, "enable_purple", parameters.enable_purple);
    parameters.lower_purple = read_hsv_range(hsv, "lower_purple", parameters.lower_purple);
    parameters.upper_purple = read_hsv_range(hsv, "upper_purple", parameters.upper_purple);
    parameters.close_kernel_size = read_int(config["morphology"], "close_kernel_size", parameters.close_kernel_size);
    parameters.min_area = read_int(components, "min_area", parameters.min_area);
    parameters.max_area_ratio = read_float(components, "max_area_ratio", parameters.max_area_ratio);
    parameters.min_side = read_float(components, "min_side", parameters.min_side);
    parameters.max_aspect_ratio = read_float(components, "max_aspect_ratio", parameters.max_aspect_ratio);
    parameters.min_fill_ratio = read_float(components, "min_fill_ratio", parameters.min_fill_ratio);
    parameters.link_factor = read_float(grouping, "link_factor", parameters.link_factor);
    parameters.min_group_points = read_int(grouping, "min_group_points", parameters.min_group_points);
    parameters.min_points_per_layer = read_int(grouping, "min_points_per_layer", parameters.min_points_per_layer);
    parameters.min_layer_separation_size_ratio =
        read_float(grouping, "min_layer_separation_size_ratio", parameters.min_layer_separation_size_ratio);
    parameters.min_separation_ratio = read_float(grouping, "min_separation_ratio", parameters.min_separation_ratio);
    parameters.max_horizontal_offset_ratio =
        read_float(grouping, "max_horizontal_offset_ratio", parameters.max_horizontal_offset_ratio);
    if (parameters.close_kernel_size < 1 || parameters.min_area < 1 || parameters.link_factor <= 0.0F ||
        parameters.min_group_points < 2 || parameters.min_points_per_layer < 1) {
        throw std::runtime_error("Traditional-center configuration contains an invalid positive-value parameter.");
    }
    return parameters;
}

float component_size(const Component& component) {
    return std::max(component.rectangle.size.width, component.rectangle.size.height);
}

float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5F : values[middle];
}

std::vector<Component> analyze_components(const cv::Mat& closed_mask, const Parameters& parameters) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int label_count =
        cv::connectedComponentsWithStats(closed_mask, labels, stats, centroids, 8, CV_32S);
    const float max_area = static_cast<float>(closed_mask.total()) * parameters.max_area_ratio;

    std::vector<Component> components;
    for (int label_id = 1; label_id < label_count; ++label_id) {
        Component component;
        component.label_id = label_id;
        component.area = stats.at<int>(label_id, cv::CC_STAT_AREA);
        component.center = {static_cast<float>(centroids.at<double>(label_id, 0)),
                            static_cast<float>(centroids.at<double>(label_id, 1))};

        cv::Mat component_mask = labels == label_id;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) {
            component.reason = "NO_CONTOUR";
            components.push_back(std::move(component));
            continue;
        }
        const auto contour = std::max_element(
            contours.begin(), contours.end(), [](const std::vector<cv::Point>& first,
                                                  const std::vector<cv::Point>& second) {
                return cv::contourArea(first) < cv::contourArea(second);
            });
        component.rectangle = cv::minAreaRect(*contour);
        const float short_side = std::min(component.rectangle.size.width, component.rectangle.size.height);
        const float long_side = std::max(component.rectangle.size.width, component.rectangle.size.height);
        component.aspect_ratio = short_side > 0.0F ? long_side / short_side
                                                    : std::numeric_limits<float>::infinity();
        const float rectangle_area = component.rectangle.size.area();
        component.fill_ratio = rectangle_area > 0.0F ? static_cast<float>(component.area) / rectangle_area : 0.0F;

        if (component.area < parameters.min_area) {
            component.reason = "AREA_SMALL";
        } else if (static_cast<float>(component.area) > max_area) {
            component.reason = "AREA_LARGE";
        } else if (short_side < parameters.min_side) {
            component.reason = "SIDE_SMALL";
        } else if (component.aspect_ratio > parameters.max_aspect_ratio) {
            component.reason = "TOO_LONG";
        } else if (component.fill_ratio < parameters.min_fill_ratio) {
            component.reason = "FILL_LOW";
        } else {
            component.accepted = true;
            component.reason = "OK";
        }
        components.push_back(std::move(component));
    }
    return components;
}

std::vector<Component> accepted_components(const std::vector<Component>& components) {
    std::vector<Component> accepted;
    for (const Component& component : components) {
        if (component.accepted) {
            accepted.push_back(component);
        }
    }
    return accepted;
}

std::vector<cv::Point> rectangle_points(const cv::RotatedRect& rectangle) {
    std::array<cv::Point2f, 4> points_float{};
    rectangle.points(points_float.data());
    std::vector<cv::Point> points;
    points.reserve(points_float.size());
    for (const cv::Point2f& point : points_float) {
        points.push_back(point);
    }
    return points;
}

cv::Mat draw_components(const cv::Mat& image, const std::vector<Component>& components,
                        bool accepted_only) {
    cv::Mat visualization = image.clone();
    for (const Component& component : components) {
        if (accepted_only && !component.accepted) {
            continue;
        }
        const cv::Scalar color = component.accepted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        const std::vector<cv::Point> points = rectangle_points(component.rectangle);
        if (points.size() == 4) {
            cv::polylines(visualization, points, true, color, 2);
        }
        cv::circle(visualization, component.center, 3, color, -1);
        cv::putText(visualization, std::to_string(component.label_id) + " " + component.reason,
                    component.center + cv::Point2f(5.0F, -5.0F), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    }
    return visualization;
}

std::vector<std::vector<int>> build_groups(const std::vector<Component>& candidates,
                                           float& median_size, const Parameters& parameters) {
    if (candidates.empty()) {
        median_size = 0.0F;
        return {};
    }
    std::vector<float> sizes;
    sizes.reserve(candidates.size());
    for (const Component& candidate : candidates) {
        sizes.push_back(component_size(candidate));
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
        std::vector<int> stack{static_cast<int>(start_index)};
        std::vector<int> group;
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

std::optional<std::vector<int>> split_y_layers(const std::vector<float>& y_values) {
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

cv::Point2f average_center(const std::vector<int>& indices, const std::vector<Component>& candidates) {
    cv::Point2f center{};
    for (const int index : indices) {
        center += candidates[index].center;
    }
    return center * (1.0F / static_cast<float>(indices.size()));
}

float y_standard_deviation(const std::vector<int>& indices, const std::vector<Component>& candidates) {
    const float mean_y = average_center(indices, candidates).y;
    float sum_squared_distance = 0.0F;
    for (const int index : indices) {
        const float distance = candidates[index].center.y - mean_y;
        sum_squared_distance += distance * distance;
    }
    return std::sqrt(sum_squared_distance / static_cast<float>(indices.size()));
}

DeviceGroup analyze_group(const std::vector<int>& member_indices, const std::vector<Component>& candidates,
                          float median_size, const Parameters& parameters) {
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
    const std::optional<std::vector<int>> labels = split_y_layers(y_values);
    if (!labels.has_value()) {
        group.reason = "LAYER_SPLIT_FAILED";
        return group;
    }

    std::vector<int> first_layer;
    std::vector<int> second_layer;
    for (std::size_t local_index = 0; local_index < member_indices.size(); ++local_index) {
        (labels.value()[local_index] == 0 ? first_layer : second_layer).push_back(member_indices[local_index]);
    }
    if (average_center(first_layer, candidates).y <= average_center(second_layer, candidates).y) {
        group.upper_indices = std::move(first_layer);
        group.lower_indices = std::move(second_layer);
    } else {
        group.upper_indices = std::move(second_layer);
        group.lower_indices = std::move(first_layer);
    }
    if (static_cast<int>(group.upper_indices.size()) < parameters.min_points_per_layer ||
        static_cast<int>(group.lower_indices.size()) < parameters.min_points_per_layer) {
        group.reason = "LAYER_POINTS_TOO_FEW";
        return group;
    }

    group.upper_center = average_center(group.upper_indices, candidates);
    group.lower_center = average_center(group.lower_indices, candidates);
    group.center = (group.upper_center + group.lower_center) * 0.5F;
    group.separation = cv::norm(group.lower_center - group.upper_center);
    const float within_layer_spread = std::max(y_standard_deviation(group.upper_indices, candidates),
                                               y_standard_deviation(group.lower_indices, candidates));
    group.separation_ratio = group.separation / (within_layer_spread + std::max(median_size, 1.0F));

    float minimum_x = candidates[member_indices.front()].center.x;
    float maximum_x = minimum_x;
    for (const int index : member_indices) {
        minimum_x = std::min(minimum_x, candidates[index].center.x);
        maximum_x = std::max(maximum_x, candidates[index].center.x);
    }
    const float total_x_span = std::max({maximum_x - minimum_x, median_size, 1.0F});
    group.horizontal_offset_ratio = std::abs(group.upper_center.x - group.lower_center.x) / total_x_span;
    const float count_balance = std::abs(static_cast<float>(group.upper_indices.size()) -
                                         static_cast<float>(group.lower_indices.size())) /
                                static_cast<float>(member_indices.size());
    group.score = static_cast<float>(member_indices.size()) + 4.0F * group.separation_ratio -
                  3.0F * group.horizontal_offset_ratio - 1.5F * count_balance;

    if (group.separation < parameters.min_layer_separation_size_ratio * median_size) {
        group.reason = "LAYER_TOO_CLOSE";
    } else if (group.separation_ratio < parameters.min_separation_ratio) {
        group.reason = "LAYER_NOT_CLEAR";
    } else if (group.horizontal_offset_ratio > parameters.max_horizontal_offset_ratio) {
        group.reason = "CENTER_OFFSET_LARGE";
    } else {
        group.valid = true;
        group.reason = "OK";
    }
    return group;
}

cv::Mat draw_groups(const cv::Mat& image, const std::vector<std::vector<int>>& groups,
                    const std::vector<Component>& candidates) {
    const std::array<cv::Scalar, 6> colors = {cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                               cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255),
                                               cv::Scalar(255, 255, 0), cv::Scalar(0, 128, 255)};
    cv::Mat visualization = image.clone();
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const cv::Scalar color = colors[group_index % colors.size()];
        std::vector<cv::Point> centers;
        for (const int candidate_index : groups[group_index]) {
            const Component& candidate = candidates[candidate_index];
            cv::polylines(visualization, rectangle_points(candidate.rectangle), true, color, 2);
            cv::circle(visualization, candidate.center, 4, color, -1);
            cv::putText(visualization, "G" + std::to_string(group_index),
                        candidate.center + cv::Point2f(5.0F, -5.0F), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            centers.push_back(candidate.center);
        }
        if (!centers.empty()) {
            const cv::Rect bounds = cv::boundingRect(centers);
            cv::rectangle(visualization, bounds, color, 1);
        }
    }
    return visualization;
}

cv::Mat draw_best_group(const cv::Mat& image, const std::vector<Component>& candidates,
                        const DeviceGroup& group) {
    cv::Mat visualization = image.clone();
    for (const int candidate_index : group.member_indices) {
        cv::polylines(visualization, rectangle_points(candidates[candidate_index].rectangle), true,
                      cv::Scalar(0, 255, 0), 2);
    }
    cv::line(visualization, group.upper_center, group.lower_center, cv::Scalar(0, 255, 255), 3);
    cv::circle(visualization, group.upper_center, 7, cv::Scalar(0, 255, 255), -1);
    cv::circle(visualization, group.lower_center, 7, cv::Scalar(0, 255, 255), -1);
    cv::circle(visualization, group.center, 9, cv::Scalar(255, 0, 255), -1);
    cv::putText(visualization, "center", group.center + cv::Point2f(12.0F, -12.0F),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 0, 255), 2);
    cv::putText(visualization, "score=" + cv::format("%.2f", group.score), cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
    return visualization;
}

void save_image(const std::filesystem::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Unable to save image: " + path.string());
    }
}

PipelineResult process_frame(const cv::Mat& image, const Parameters& parameters) {
    PipelineResult result;
    result.original = image.clone();
    cv::cvtColor(image, result.gray, cv::COLOR_BGR2GRAY);
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    result.hue = hsv_channels[0];
    result.saturation = hsv_channels[1];
    result.value = hsv_channels[2];
    cv::inRange(hsv, parameters.lower_red_1, parameters.upper_red_1, result.lower_red_mask);
    cv::inRange(hsv, parameters.lower_red_2, parameters.upper_red_2, result.upper_red_mask);
    cv::bitwise_or(result.lower_red_mask, result.upper_red_mask, result.red_mask);
    result.purple_mask = cv::Mat::zeros(result.red_mask.size(), result.red_mask.type());
    if (parameters.enable_purple) {
        cv::inRange(hsv, parameters.lower_purple, parameters.upper_purple, result.purple_mask);
    }
    cv::bitwise_or(result.red_mask, result.purple_mask, result.light_mask);

    const int normalized_kernel_size = parameters.close_kernel_size % 2 == 0
                                           ? parameters.close_kernel_size + 1
                                           : parameters.close_kernel_size;
    result.closed_mask = result.light_mask.clone();
    if (normalized_kernel_size > 1) {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(normalized_kernel_size, normalized_kernel_size));
        cv::morphologyEx(result.light_mask, result.closed_mask, cv::MORPH_CLOSE, kernel);
    }

    result.components = analyze_components(result.closed_mask, parameters);
    result.candidates = accepted_components(result.components);
    result.groups = build_groups(result.candidates, result.median_component_size, parameters);
    std::string first_group_failure_reason;
    for (const std::vector<int>& group_indices : result.groups) {
        DeviceGroup group = analyze_group(group_indices, result.candidates,
                                          result.median_component_size, parameters);
        if (group.valid && (!result.best_group.has_value() || group.score > result.best_group->score)) {
            result.best_group = std::move(group);
        } else if (first_group_failure_reason.empty()) {
            first_group_failure_reason = group.reason;
        }
    }
    if (!result.best_group.has_value()) {
        if (result.candidates.empty()) {
            result.failure_reason = "NO_ACCEPTED_COMPONENTS";
        } else if (result.groups.empty()) {
            result.failure_reason = "NO_SPATIAL_GROUP";
        } else {
            result.failure_reason = "NO_VALID_DOUBLE_LAYER_GROUP";
            if (!first_group_failure_reason.empty()) {
                result.failure_reason += ":" + first_group_failure_reason;
            }
        }
    }

    result.all_components = draw_components(image, result.components, false);
    result.accepted_components = draw_components(image, result.components, true);
    result.spatial_groups = draw_groups(image, result.groups, result.candidates);
    result.best_device_center = result.best_group.has_value()
                                    ? draw_best_group(image, result.candidates, result.best_group.value())
                                    : image.clone();
    return result;
}

void save_pipeline(const PipelineResult& result, const std::filesystem::path& output_directory) {
    std::filesystem::create_directories(output_directory);
    save_image(output_directory / "01_original.jpg", result.original);
    save_image(output_directory / "02_gray.jpg", result.gray);
    save_image(output_directory / "03_hsv_hue.png", result.hue);
    save_image(output_directory / "04_hsv_saturation.png", result.saturation);
    save_image(output_directory / "05_hsv_value.png", result.value);
    save_image(output_directory / "06_red_mask_low_hue.png", result.lower_red_mask);
    save_image(output_directory / "07_red_mask_high_hue.png", result.upper_red_mask);
    save_image(output_directory / "08_red_mask_combined.png", result.red_mask);
    save_image(output_directory / "09_purple_mask.png", result.purple_mask);
    save_image(output_directory / "10_light_mask_combined.png", result.light_mask);
    save_image(output_directory / "11_closed_mask.png", result.closed_mask);
    save_image(output_directory / "12_all_components.jpg", result.all_components);
    save_image(output_directory / "13_accepted_components.jpg", result.accepted_components);
    save_image(output_directory / "14_spatial_groups.jpg", result.spatial_groups);
    save_image(output_directory / "15_best_device_center.jpg", result.best_device_center);
}

void write_report(const PipelineResult& result, const std::string& image_path,
                  const std::filesystem::path& output_directory) {
    std::ofstream report(output_directory / "report.txt");
    report << "image=" << image_path << '\n'
           << "components=" << result.components.size() << '\n'
           << "accepted_candidates=" << result.candidates.size() << '\n'
           << "spatial_groups=" << result.groups.size() << '\n'
           << "median_component_size=" << result.median_component_size << '\n';
    if (result.best_group.has_value()) {
        report << "result=OK\n"
               << "center_x=" << result.best_group->center.x << '\n'
               << "center_y=" << result.best_group->center.y << '\n'
               << "upper_center=" << result.best_group->upper_center << '\n'
               << "lower_center=" << result.best_group->lower_center << '\n'
               << "score=" << result.best_group->score << '\n';
    } else {
        report << "result=" << result.failure_reason << '\n';
    }
}

void show_pipeline(const PipelineResult& result) {
    const std::array<std::pair<const char*, const cv::Mat*>, 15> windows = {
        std::pair{"01 Original", &result.original},
        std::pair{"02 Gray", &result.gray},
        std::pair{"03 HSV Hue", &result.hue},
        std::pair{"04 HSV Saturation", &result.saturation},
        std::pair{"05 HSV Value", &result.value},
        std::pair{"06 Red Mask Low Hue", &result.lower_red_mask},
        std::pair{"07 Red Mask High Hue", &result.upper_red_mask},
        std::pair{"08 Red Mask Combined", &result.red_mask},
        std::pair{"09 Purple Mask", &result.purple_mask},
        std::pair{"10 Light Mask Combined", &result.light_mask},
        std::pair{"11 Closed Mask", &result.closed_mask},
        std::pair{"12 All Components", &result.all_components},
        std::pair{"13 Accepted Components", &result.accepted_components},
        std::pair{"14 Spatial Groups", &result.spatial_groups},
        std::pair{"15 Best Device Center", &result.best_device_center},
    };
    for (const auto& [name, image] : windows) {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, *image);
    }
}

void print_result(const PipelineResult& result, const std::string& prefix,
                  std::uint64_t success_streak = 0, std::uint64_t failure_streak = 0) {
    std::cout << prefix << " components=" << result.components.size()
              << " accepted=" << result.candidates.size() << " groups=" << result.groups.size();
    if (result.best_group.has_value()) {
        std::cout << " center=" << result.best_group->center
                  << " score=" << result.best_group->score;
    } else {
        std::cout << " center=none reason=" << result.failure_reason;
    }
    if (success_streak > 0 || failure_streak > 0) {
        std::cout << " streak=(success=" << success_streak << ", failure=" << failure_streak << ')';
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        bool camera_override = false;
        std::vector<std::string> positional_arguments;
        for (int argument_index = 1; argument_index < argc; ++argument_index) {
            const std::string argument = argv[argument_index];
            if (argument == "--camera") {
                camera_override = true;
            } else {
                positional_arguments.push_back(argument);
            }
        }
        if (positional_arguments.size() > 3) {
            throw std::runtime_error(
                "Usage: test_traditional_center [config_path] [image_path] [output_directory] [--camera]");
        }
        const std::string config_path =
            !positional_arguments.empty() ? positional_arguments[0] : "config/test.json";
        cv::FileStorage config_file(config_path, cv::FileStorage::READ);
        if (!config_file.isOpened()) {
            throw std::runtime_error("Unable to open configuration file: " + config_path);
        }
        const cv::FileNode config = config_file["traditional_center"];
        if (config.empty()) {
            throw std::runtime_error("Missing traditional_center in configuration file: " + config_path);
        }
        const Parameters parameters = load_parameters(config);
        const std::string input_mode = camera_override ? "camera" : read_string(config, "input_mode", "image");
        if (input_mode != "image" && input_mode != "camera") {
            throw std::runtime_error("traditional_center.input_mode must be image or camera.");
        }

        if (input_mode == "camera") {
            if (positional_arguments.size() > 1) {
                throw std::runtime_error("Image and output paths are not used in camera mode.");
            }
            std::string serial_number;
            config_file["camera"]["sn"] >> serial_number;
            if (serial_number.empty()) {
                throw std::runtime_error("Missing camera.sn in configuration file: " + config_path);
            }
            const wit_radar::HikCameraSettings camera_settings =
                wit_radar::read_camera_settings(config_file["camera"]);
            const int wait_key_ms = read_int(config, "camera_wait_key_ms", 1);
            const int log_every_n_frames = std::max(1, read_int(config, "log_every_n_frames", 30));
            const bool save_lost_frames = read_bool(config, "save_lost_frames", true);
            const std::filesystem::path lost_frame_output_directory =
                read_string(config, "lost_frame_output_dir", "test_output/traditional_center_lost");
            wit_radar::HikCamera camera;
            camera.open_by_serial_number(serial_number, camera_settings);
            std::cout << "Traditional center camera preview started. Press Esc or q to exit.\n";
            std::uint64_t frame_number = 0;
            std::uint64_t success_streak = 0;
            std::uint64_t failure_streak = 0;
            bool previous_frame_found_center = false;
            bool have_previous_frame_result = false;
            while (true) {
                const cv::Mat frame = camera.read(1000);
                if (!frame.empty()) {
                    ++frame_number;
                    const PipelineResult result = process_frame(frame, parameters);
                    show_pipeline(result);
                    const bool found_center = result.best_group.has_value();
                    if (found_center) {
                        ++success_streak;
                        failure_streak = 0;
                    } else {
                        ++failure_streak;
                        success_streak = 0;
                    }
                    if (have_previous_frame_result && found_center != previous_frame_found_center) {
                        print_result(result, found_center ? "[traditional][recovered frame " +
                                                               std::to_string(frame_number) + "]"
                                                         : "[traditional][lost frame " +
                                                               std::to_string(frame_number) + "]",
                                     success_streak, failure_streak);
                        if (!found_center && save_lost_frames) {
                            const std::filesystem::path snapshot_directory =
                                lost_frame_output_directory / ("frame_" + std::to_string(frame_number));
                            save_pipeline(result, snapshot_directory);
                            write_report(result, "camera_frame_" + std::to_string(frame_number),
                                         snapshot_directory);
                            std::cout << "[traditional][lost] Saved diagnostic images to "
                                      << snapshot_directory << '\n';
                        }
                    } else if (frame_number % static_cast<std::uint64_t>(log_every_n_frames) == 0) {
                        print_result(result, "[traditional][frame " + std::to_string(frame_number) + "]",
                                     success_streak, failure_streak);
                    }
                    previous_frame_found_center = found_center;
                    have_previous_frame_result = true;
                }
                const int key = cv::waitKey(wait_key_ms);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
            }
            cv::destroyAllWindows();
            return EXIT_SUCCESS;
        }

        const std::string image_path = positional_arguments.size() > 1
                                           ? positional_arguments[1]
                                           : read_string(config, "image");
        const std::filesystem::path output_directory = positional_arguments.size() > 2
                                                           ? positional_arguments[2]
                                                           : read_string(config, "output_dir", "test_output/traditional_center");
        if (image_path.empty()) {
            throw std::runtime_error("Provide an image path or set traditional_center.image in " + config_path);
        }

        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Unable to read image: " + image_path);
        }
        const PipelineResult result = process_frame(image, parameters);
        save_pipeline(result, output_directory);
        write_report(result, image_path, output_directory);
        print_result(result, "[traditional][image]");
        std::cout << "Output: " << output_directory << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Traditional center test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
