#include "laser_detector.hpp"
#include "device_center_finder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

namespace wit_radar {
namespace {

constexpr unsigned char letterbox_padding_value = 114;

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

TrtLogger trt_logger;

std::runtime_error make_error(const std::string& message) {
    return std::runtime_error("Laser detector: " + message);
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw make_error(std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

std::size_t volume(const nvinfer1::Dims& shape) {
    std::size_t result = 1;
    for (int index = 0; index < shape.nbDims; ++index) {
        if (shape.d[index] <= 0) {
            throw make_error("engine has an unresolved tensor shape.");
        }
        result *= static_cast<std::size_t>(shape.d[index]);
    }
    return result;
}

std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(std::uint16_t);
    default:
        throw make_error("only FP32 and FP16 engine tensors are supported.");
    }
}

std::vector<char> unwrap_ultralytics_engine(std::vector<char> serialized_engine) {
    constexpr std::size_t metadata_size_bytes = sizeof(std::uint32_t);
    if (serialized_engine.size() <= metadata_size_bytes) {
        return serialized_engine;
    }

    std::uint32_t metadata_size = 0;
    std::memcpy(&metadata_size, serialized_engine.data(), metadata_size_bytes);
    const std::size_t engine_offset = metadata_size_bytes + static_cast<std::size_t>(metadata_size);
    if (engine_offset >= serialized_engine.size() || serialized_engine[metadata_size_bytes] != '{') {
        return serialized_engine;
    }

    return {serialized_engine.begin() + static_cast<std::ptrdiff_t>(engine_offset),
            serialized_engine.end()};
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    std::uint32_t mantissa = bits & 0x7fffffU;

    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
    }
    if (exponent > 142U) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (exponent < 113U) {
        if (exponent < 103U) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000U;
        const unsigned int shift = 126U - exponent;
        return static_cast<std::uint16_t>(sign | ((mantissa + (1U << (shift - 1U))) >> shift));
    }
    return static_cast<std::uint16_t>(sign | ((exponent - 112U) << 10U) |
                                      ((mantissa + 0x1000U) >> 13U));
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value) & 0x8000U) << 16U;
    std::uint32_t exponent = (static_cast<std::uint32_t>(value) >> 10U) & 0x1fU;
    std::uint32_t mantissa = static_cast<std::uint32_t>(value) & 0x03ffU;
    std::uint32_t bits = 0;

    if (exponent == 0U) {
        if (mantissa != 0U) {
            exponent = 113U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            bits = sign | (exponent << 23U) | ((mantissa & 0x03ffU) << 13U);
        } else {
            bits = sign;
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }

    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float intersection_over_union(const cv::Rect& first, const cv::Rect& second) {
    const cv::Rect intersection = first & second;
    const float intersection_area = static_cast<float>(intersection.area());
    const float union_area = static_cast<float>(first.area() + second.area()) - intersection_area;
    return union_area > 0.0F ? intersection_area / union_area : 0.0F;
}

std::vector<LaserDetection> non_maximum_suppression(std::vector<LaserDetection> detections,
                                                     float threshold) {
    std::sort(detections.begin(), detections.end(), [](const LaserDetection& first,
                                                       const LaserDetection& second) {
        return first.confidence > second.confidence;
    });

    std::vector<LaserDetection> selected;
    for (const LaserDetection& candidate : detections) {
        const bool overlaps_selected = std::any_of(
            selected.begin(), selected.end(), [&](const LaserDetection& selected_detection) {
                return candidate.class_id == selected_detection.class_id &&
                       intersection_over_union(candidate.roi, selected_detection.roi) > threshold;
            });
        if (!overlaps_selected) {
            selected.push_back(candidate);
        }
    }
    return selected;
}

}  // namespace

class LaserDetector::Impl {
public:
    Impl(const std::string& engine_path, float confidence_threshold, float nms_threshold,
         float min_content_overlap, bool debug_logging, unsigned int debug_every_n_frames,
         DeviceCenterParameters device_center_parameters)
        : confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold),
          min_content_overlap_(min_content_overlap), debug_logging_(debug_logging),
          debug_every_n_frames_(debug_every_n_frames),
          device_center_finder_(std::move(device_center_parameters)) {
        if (confidence_threshold_ < 0.0F || confidence_threshold_ > 1.0F ||
            nms_threshold_ < 0.0F || nms_threshold_ > 1.0F ||
            min_content_overlap_ < 0.0F || min_content_overlap_ > 1.0F) {
            throw make_error("thresholds must be between 0 and 1.");
        }
        if (debug_every_n_frames_ == 0) {
            throw make_error("debug_every_n_frames must be greater than zero.");
        }

        std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
        if (!engine_file) {
            throw make_error("unable to open engine file: " + engine_path);
        }
        const std::streamsize size = engine_file.tellg();
        if (size <= 0) {
            throw make_error("engine file is empty: " + engine_path);
        }
        engine_file.seekg(0);
        std::vector<char> serialized_engine(static_cast<std::size_t>(size));
        if (!engine_file.read(serialized_engine.data(), size)) {
            throw make_error("unable to read engine file: " + engine_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(trt_logger));
        if (!runtime_) {
            throw make_error("failed to create TensorRT runtime.");
        }
        serialized_engine = unwrap_ultralytics_engine(std::move(serialized_engine));
        engine_.reset(runtime_->deserializeCudaEngine(serialized_engine.data(),
                                                       serialized_engine.size()));
        if (!engine_) {
            throw make_error("failed to deserialize engine. Re-export it for this GPU and TensorRT version.");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw make_error("failed to create TensorRT execution context.");
        }

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (name == nullptr) {
                throw make_error("engine contains an unnamed IO tensor.");
            }
            const auto mode = engine_->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name_.empty()) {
                    throw make_error("only one input tensor is supported.");
                }
                input_name_ = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                if (!output_name_.empty()) {
                    throw make_error("only one output tensor is supported.");
                }
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw make_error("engine must have one input and one output tensor.");
        }

        input_shape_ = engine_->getTensorShape(input_name_.c_str());
        output_shape_ = context_->getTensorShape(output_name_.c_str());
        if (input_shape_.nbDims != 4 || input_shape_.d[0] != 1 || input_shape_.d[1] != 3) {
            throw make_error("expected a static NCHW input with shape [1, 3, H, W].");
        }
        input_height_ = input_shape_.d[2];
        input_width_ = input_shape_.d[3];
        if (input_height_ <= 0 || input_width_ <= 0) {
            throw make_error("engine input dimensions must be static and positive.");
        }
        if (output_shape_.nbDims != 3 || output_shape_.d[0] != 1 || output_shape_.d[1] < 5) {
            throw make_error("expected a YOLO output with shape [1, 4 + classes, anchors].");
        }

        input_type_ = engine_->getTensorDataType(input_name_.c_str());
        output_type_ = engine_->getTensorDataType(output_name_.c_str());
        input_elements_ = volume(input_shape_);
        output_elements_ = volume(output_shape_);
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        check_cuda(cudaMalloc(&device_input_, input_elements_ * element_size(input_type_)), "cudaMalloc(input)");
        try {
            check_cuda(cudaMalloc(&device_output_, output_elements_ * element_size(output_type_)),
                       "cudaMalloc(output)");
            if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
                !context_->setTensorAddress(output_name_.c_str(), device_output_)) {
                throw make_error("failed to bind TensorRT input or output buffers.");
            }
        } catch (...) {
            cudaFree(device_input_);
            device_input_ = nullptr;
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
            throw;
        }
    }

    ~Impl() {
        if (device_output_ != nullptr) {
            cudaFree(device_output_);
        }
        if (device_input_ != nullptr) {
            cudaFree(device_input_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    std::vector<LaserDetection> detect(const cv::Mat& bgr_image) {
        if (bgr_image.empty() || bgr_image.type() != CV_8UC3) {
            throw make_error("detect expects a non-empty CV_8UC3 BGR image.");
        }

        const std::uint64_t frame_number = ++frame_count_;
        const bool write_debug_log =
            debug_logging_ && frame_number % debug_every_n_frames_ == 0;

        const float scale = std::min(static_cast<float>(input_width_) / bgr_image.cols,
                                     static_cast<float>(input_height_) / bgr_image.rows);
        const int resized_width = std::max(1, static_cast<int>(std::round(bgr_image.cols * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::round(bgr_image.rows * scale)));
        const int pad_x = (input_width_ - resized_width) / 2;
        const int pad_y = (input_height_ - resized_height) / 2;
        if (write_debug_log) {
            std::cout << "[laser][frame " << frame_number << "][preprocess] source="
                      << bgr_image.cols << "x" << bgr_image.rows << " model=" << input_width_
                      << "x" << input_height_ << " scale=" << scale << " resized="
                      << resized_width << "x" << resized_height << " pad=(" << pad_x << ", "
                      << pad_y << ")\n";
        }
        const cv::Rect valid_model_region(pad_x, pad_y, resized_width, resized_height);

        cv::Mat resized;
        cv::resize(bgr_image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
                   cv::INTER_LINEAR);
        cv::Mat letterboxed(input_height_, input_width_, CV_8UC3,
                            cv::Scalar::all(letterbox_padding_value));
        resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
        cv::cvtColor(letterboxed, letterboxed, cv::COLOR_BGR2RGB);

        cv::Mat normalized;
        letterboxed.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
        std::vector<float> input_fp32(input_elements_);
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < input_height_; ++row) {
                const cv::Vec3f* pixels = normalized.ptr<cv::Vec3f>(row);
                for (int column = 0; column < input_width_; ++column) {
                    input_fp32[static_cast<std::size_t>(channel) * input_height_ * input_width_ +
                               static_cast<std::size_t>(row) * input_width_ + column] =
                        pixels[column][channel];
                }
            }
        }

        if (input_type_ == nvinfer1::DataType::kFLOAT) {
            check_cuda(cudaMemcpyAsync(device_input_, input_fp32.data(),
                                       input_fp32.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync(input)");
        } else {
            std::vector<std::uint16_t> input_fp16(input_elements_);
            std::transform(input_fp32.begin(), input_fp32.end(), input_fp16.begin(), float_to_half);
            check_cuda(cudaMemcpyAsync(device_input_, input_fp16.data(),
                                       input_fp16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync(input)");
        }

        if (!context_->enqueueV3(stream_)) {
            throw make_error("TensorRT inference enqueue failed.");
        }

        std::vector<float> output(output_elements_);
        if (output_type_ == nvinfer1::DataType::kFLOAT) {
            check_cuda(cudaMemcpyAsync(output.data(), device_output_, output.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync(output)");
        } else {
            std::vector<std::uint16_t> output_fp16(output_elements_);
            check_cuda(cudaMemcpyAsync(output_fp16.data(), device_output_,
                                       output_fp16.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync(output)");
            check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            std::transform(output_fp16.begin(), output_fp16.end(), output.begin(), half_to_float);
            return add_device_centers(
                decode(output, scale, pad_x, pad_y, valid_model_region, bgr_image.size(), frame_number,
                       write_debug_log),
                bgr_image, frame_number, write_debug_log);
        }
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        return add_device_centers(
            decode(output, scale, pad_x, pad_y, valid_model_region, bgr_image.size(), frame_number,
                   write_debug_log),
            bgr_image, frame_number, write_debug_log);
    }

    cv::Size input_size() const noexcept {
        return {input_width_, input_height_};
    }

private:
    std::vector<LaserDetection> add_device_centers(std::vector<LaserDetection> detections,
                                                   const cv::Mat& bgr_image,
                                                   std::uint64_t frame_number,
                                                   bool write_debug_log) const {
        for (LaserDetection& detection : detections) {
            const DeviceCenterResult result = device_center_finder_.find(bgr_image(detection.roi));
            detection.light_candidate_count = result.candidate_count;
            detection.light_group_count = result.group_count;
            detection.light_group_score = result.group_score;
            detection.device_center_reason = result.reason;
            if (result.center.has_value()) {
                detection.device_center = result.center.value() + cv::Point2f(
                    static_cast<float>(detection.roi.x), static_cast<float>(detection.roi.y));
            }
            if (write_debug_log) {
                std::cout << "[laser][frame " << frame_number << "][device-center] roi="
                          << detection.roi << " light_candidates=" << result.candidate_count
                          << " groups=" << result.group_count << " score=" << result.group_score
                          << " result=" << result.reason;
                if (detection.device_center.has_value()) {
                    std::cout << " roi_center=" << result.center.value()
                              << " source_center=" << detection.device_center.value();
                }
                std::cout << '\n';
            }
        }
        return detections;
    }

    std::vector<LaserDetection> decode(const std::vector<float>& output, float scale, int pad_x,
                                       int pad_y, const cv::Rect& valid_model_region,
                                       const cv::Size& image_size,
                                       std::uint64_t frame_number, bool write_debug_log) const {
        const int attributes = output_shape_.d[1];
        const int anchors = output_shape_.d[2];
        std::vector<LaserDetection> detections;
        detections.reserve(static_cast<std::size_t>(anchors));

        for (int anchor = 0; anchor < anchors; ++anchor) {
            int class_id = -1;
            float confidence = 0.0F;
            for (int class_index = 4; class_index < attributes; ++class_index) {
                const float score = output[static_cast<std::size_t>(class_index) * anchors + anchor];
                if (score > confidence) {
                    confidence = score;
                    class_id = class_index - 4;
                }
            }
            if (confidence < confidence_threshold_) {
                continue;
            }

            const float center_x = output[anchor];
            const float center_y = output[anchors + anchor];
            const float width = output[2 * anchors + anchor];
            const float height = output[3 * anchors + anchor];
            const int model_left = static_cast<int>(std::floor(center_x - width * 0.5F));
            const int model_top = static_cast<int>(std::floor(center_y - height * 0.5F));
            const int model_right = static_cast<int>(std::ceil(center_x + width * 0.5F));
            const int model_bottom = static_cast<int>(std::ceil(center_y + height * 0.5F));
            const cv::Rect model_roi =
                cv::Rect(model_left, model_top, model_right - model_left, model_bottom - model_top) &
                cv::Rect(0, 0, input_width_, input_height_);
            const float content_overlap = model_roi.area() > 0
                                              ? static_cast<float>((model_roi & valid_model_region).area()) /
                                                    model_roi.area()
                                              : 0.0F;
            if (content_overlap < min_content_overlap_) {
                if (write_debug_log) {
                    std::cout << "[laser][frame " << frame_number << "][filter] anchor=" << anchor
                              << " rejected: model_roi=" << model_roi
                              << " content_overlap=" << content_overlap << " threshold="
                              << min_content_overlap_ << '\n';
                }
                continue;
            }
            const int left = static_cast<int>(std::floor((center_x - width * 0.5F - pad_x) / scale));
            const int top = static_cast<int>(std::floor((center_y - height * 0.5F - pad_y) / scale));
            const int right = static_cast<int>(std::ceil((center_x + width * 0.5F - pad_x) / scale));
            const int bottom = static_cast<int>(std::ceil((center_y + height * 0.5F - pad_y) / scale));
            const cv::Rect source_roi_unclamped(left, top, right - left, bottom - top);
            const cv::Rect roi = source_roi_unclamped &
                                 cv::Rect(0, 0, image_size.width, image_size.height);
            if (model_roi.area() > 0 && roi.area() > 0) {
                detections.push_back({roi, model_roi, confidence, class_id});
                if (write_debug_log && detections.size() <= 20) {
                    std::cout << "[laser][frame " << frame_number << "][decode] anchor="
                              << anchor << " score=" << confidence << " raw_xywh=(" << center_x
                              << ", " << center_y << ", " << width << ", " << height
                              << ") model_roi=" << model_roi
                              << " content_overlap=" << content_overlap
                              << " source_roi_unclamped=" << source_roi_unclamped
                              << " source_roi=" << roi << '\n';
                }
            }
        }
        const std::size_t candidates_before_nms = detections.size();
        std::vector<LaserDetection> selected = non_maximum_suppression(std::move(detections), nms_threshold_);
        if (write_debug_log) {
            std::cout << "[laser][frame " << frame_number << "][nms] candidates="
                      << candidates_before_nms << " selected=" << selected.size() << '\n';
            for (std::size_t index = 0; index < selected.size(); ++index) {
                std::cout << "[laser][frame " << frame_number << "][selected " << index
                          << "] model_roi=" << selected[index].model_roi
                          << " source_roi=" << selected[index].roi
                          << " score=" << selected[index].confidence << '\n';
            }
        }
        return selected;
    }

    float confidence_threshold_;
    float nms_threshold_;
    float min_content_overlap_;
    bool debug_logging_ = false;
    unsigned int debug_every_n_frames_ = 1;
    std::uint64_t frame_count_ = 0;
    DeviceCenterFinder device_center_finder_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string input_name_;
    std::string output_name_;
    nvinfer1::Dims input_shape_{};
    nvinfer1::Dims output_shape_{};
    nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_type_ = nvinfer1::DataType::kFLOAT;
    int input_width_ = 0;
    int input_height_ = 0;
    std::size_t input_elements_ = 0;
    std::size_t output_elements_ = 0;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

LaserDetector::LaserDetector(const std::string& engine_path, float confidence_threshold,
                             float nms_threshold, float min_content_overlap, bool debug_logging,
                             unsigned int debug_every_n_frames,
                             DeviceCenterParameters device_center_parameters)
    : impl_(std::make_unique<Impl>(engine_path, confidence_threshold, nms_threshold,
                                   min_content_overlap,
                                   debug_logging, debug_every_n_frames,
                                   std::move(device_center_parameters))) {}

LaserDetector::~LaserDetector() = default;

std::vector<LaserDetection> LaserDetector::detect(const cv::Mat& bgr_image) {
    return impl_->detect(bgr_image);
}

cv::Size LaserDetector::input_size() const noexcept {
    return impl_->input_size();
}

}  // namespace wit_radar
