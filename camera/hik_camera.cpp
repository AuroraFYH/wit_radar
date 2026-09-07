#include "hik_camera.hpp"

#include <algorithm>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>

#include "MvCameraControl.h"

namespace wit_radar {
namespace {

std::mutex sdk_mutex;
std::size_t sdk_reference_count = 0;

std::runtime_error make_sdk_error(const char* operation, int result) {
    std::ostringstream message;
    message << operation << " failed, MVS error code: 0x" << std::hex
            << static_cast<unsigned int>(result);
    return std::runtime_error(message.str());
}

void initialize_sdk() {
    std::lock_guard<std::mutex> lock(sdk_mutex);
    if (sdk_reference_count == 0) {
        const int result = MV_CC_Initialize();
        if (result != MV_OK) {
            throw make_sdk_error("MV_CC_Initialize", result);
        }
    }
    ++sdk_reference_count;
}

void finalize_sdk() noexcept {
    std::lock_guard<std::mutex> lock(sdk_mutex);
    if (sdk_reference_count == 0) {
        return;
    }

    --sdk_reference_count;
    if (sdk_reference_count == 0) {
        MV_CC_Finalize();
    }
}

template <std::size_t Size>
std::string make_string(const unsigned char (&value)[Size]) {
    std::size_t length = 0;
    while (length < Size && value[length] != '\0') {
        ++length;
    }
    return {reinterpret_cast<const char*>(value), length};
}

std::string serial_number(const MV_CC_DEVICE_INFO& device) {
    switch (device.nTLayerType) {
    case MV_GIGE_DEVICE:
    case MV_GENTL_GIGE_DEVICE:
        return make_string(device.SpecialInfo.stGigEInfo.chSerialNumber);
    case MV_USB_DEVICE:
        return make_string(device.SpecialInfo.stUsb3VInfo.chSerialNumber);
    case MV_CAMERALINK_DEVICE:
        return make_string(device.SpecialInfo.stCamLInfo.chSerialNumber);
    case MV_GENTL_CAMERALINK_DEVICE:
        return make_string(device.SpecialInfo.stCMLInfo.chSerialNumber);
    case MV_GENTL_CXP_DEVICE:
        return make_string(device.SpecialInfo.stCXPInfo.chSerialNumber);
    case MV_GENTL_XOF_DEVICE:
        return make_string(device.SpecialInfo.stXoFInfo.chSerialNumber);
    case MV_GENTL_VIR_DEVICE:
        return make_string(device.SpecialInfo.stVirInfo.chSerialNumber);
    default:
        return {};
    }
}

void set_float_value(void* handle, const char* key, double value) {
    MVCC_FLOATVALUE range{};
    const int range_result = MV_CC_GetFloatValue(handle, key, &range);
    if (range_result != MV_OK) {
        throw make_sdk_error(key, range_result);
    }

    const double clamped = std::clamp(value, static_cast<double>(range.fMin),
                                      static_cast<double>(range.fMax));
    const int set_result = MV_CC_SetFloatValue(handle, key, static_cast<float>(clamped));
    if (set_result != MV_OK) {
        throw make_sdk_error(key, set_result);
    }
}

void set_enum_value_by_string(void* handle, const char* key, const char* value) {
    const int result = MV_CC_SetEnumValueByString(handle, key, value);
    if (result != MV_OK) {
        throw make_sdk_error(key, result);
    }
}

MV_CC_DEVICE_INFO_LIST enumerate_devices() {
    MV_CC_DEVICE_INFO_LIST devices{};
    constexpr unsigned int transport_layers =
        MV_GIGE_DEVICE | MV_USB_DEVICE | MV_CAMERALINK_DEVICE |
        MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE |
        MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE | MV_GENTL_VIR_DEVICE;
    const int result = MV_CC_EnumDevices(transport_layers, &devices);
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_EnumDevices", result);
    }
    return devices;
}

}  // namespace

HikCamera::~HikCamera() {
    close();
}

HikCameraSettings read_camera_settings(const cv::FileNode& camera_config) {
    HikCameraSettings settings;
    if (camera_config.empty()) {
        return settings;
    }

    const cv::FileNode gain = camera_config["gain_db"];
    if (!gain.empty()) {
        const double value = gain.real();
        if (value >= 0.0) {
            settings.gain_db = value;
        }
    }

    const cv::FileNode exposure = camera_config["exposure_time_us"];
    if (!exposure.empty()) {
        const double value = exposure.real();
        if (value > 0.0) {
            settings.exposure_time_us = value;
        }
    }

    return settings;
}

void HikCamera::open(std::size_t device_index, const HikCameraSettings& settings) {
    close();
    initialize_sdk();
    sdk_initialized_ = true;

    try {
        const MV_CC_DEVICE_INFO_LIST devices = enumerate_devices();
        if (device_index >= devices.nDeviceNum || devices.pDeviceInfo[device_index] == nullptr) {
            throw std::runtime_error("Requested Hikvision camera index is unavailable.");
        }

        open_device(devices.pDeviceInfo[device_index], settings);
    } catch (...) {
        close();
        throw;
    }
}

void HikCamera::open_by_serial_number(const std::string& requested_serial_number,
                                      const HikCameraSettings& settings) {
    if (requested_serial_number.empty()) {
        throw std::invalid_argument("Camera serial number must not be empty.");
    }

    close();
    initialize_sdk();
    sdk_initialized_ = true;

    try {
        const MV_CC_DEVICE_INFO_LIST devices = enumerate_devices();
        for (unsigned int index = 0; index < devices.nDeviceNum; ++index) {
            MV_CC_DEVICE_INFO* device = devices.pDeviceInfo[index];
            if (device != nullptr && serial_number(*device) == requested_serial_number) {
                open_device(device, settings);
                return;
            }
        }

        throw std::runtime_error("Hikvision camera with serial number '" +
                                 requested_serial_number + "' was not found.");
    } catch (...) {
        close();
        throw;
    }
}

void HikCamera::open_device(void* device_info, const HikCameraSettings& settings) {
    auto* device = static_cast<MV_CC_DEVICE_INFO*>(device_info);
    int result = MV_CC_CreateHandle(&handle_, device);
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_CreateHandle", result);
    }

    result = MV_CC_OpenDevice(handle_);
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_OpenDevice", result);
    }
    opened_ = true;

    if (device->nTLayerType == MV_GIGE_DEVICE) {
        const int packet_size = MV_CC_GetOptimalPacketSize(handle_);
        if (packet_size > 0) {
            result = MV_CC_SetIntValueEx(handle_, "GevSCPSPacketSize", packet_size);
            if (result != MV_OK) {
                throw make_sdk_error("MV_CC_SetIntValueEx(GevSCPSPacketSize)", result);
            }
        }
    }

    if (settings.gain_db.has_value()) {
        set_enum_value_by_string(handle_, "GainAuto", "Off");
        set_float_value(handle_, "Gain", settings.gain_db.value());
    }

    if (settings.exposure_time_us.has_value()) {
        set_enum_value_by_string(handle_, "ExposureAuto", "Off");
        set_float_value(handle_, "ExposureTime", settings.exposure_time_us.value());
    }

    result = MV_CC_SetEnumValueByString(handle_, "AcquisitionMode", "Continuous");
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_SetEnumValueByString(AcquisitionMode)", result);
    }

    result = MV_CC_SetEnumValueByString(handle_, "TriggerMode", "Off");
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_SetEnumValueByString(TriggerMode)", result);
    }

    result = MV_CC_StartGrabbing(handle_);
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_StartGrabbing", result);
    }
    grabbing_ = true;
}

void HikCamera::close() noexcept {
    if (handle_ != nullptr && grabbing_) {
        MV_CC_StopGrabbing(handle_);
        grabbing_ = false;
    }
    if (handle_ != nullptr && opened_) {
        MV_CC_CloseDevice(handle_);
        opened_ = false;
    }
    if (handle_ != nullptr) {
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }
    if (sdk_initialized_) {
        finalize_sdk();
        sdk_initialized_ = false;
    }
}

bool HikCamera::is_open() const noexcept {
    return handle_ != nullptr && opened_ && grabbing_;
}

cv::Mat HikCamera::read(unsigned int timeout_ms) {
    return read_frame(timeout_ms).image;
}

HikCameraFrame HikCamera::read_frame(unsigned int timeout_ms) {
    if (!is_open()) {
        throw std::runtime_error("Camera is not open.");
    }

    MV_FRAME_OUT frame{};
    const int result = MV_CC_GetImageBuffer(handle_, &frame, timeout_ms);
    if (result == MV_E_NODATA) {
        return {};
    }
    if (result != MV_OK) {
        throw make_sdk_error("MV_CC_GetImageBuffer", result);
    }
    const auto received_at = std::chrono::steady_clock::now();

    cv::Mat image;
    try {
        const unsigned int width = frame.stFrameInfo.nExtendWidth;
        const unsigned int height = frame.stFrameInfo.nExtendHeight;
        if (width == 0 || height == 0 || frame.pBufAddr == nullptr) {
            throw std::runtime_error("Camera returned an invalid image frame.");
        }

        image = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
        MV_CC_PIXEL_CONVERT_PARAM_EX conversion{};
        conversion.nWidth = width;
        conversion.nHeight = height;
        conversion.enSrcPixelType = frame.stFrameInfo.enPixelType;
        conversion.pSrcData = frame.pBufAddr;
        conversion.nSrcDataLen = frame.stFrameInfo.nFrameLen;
        conversion.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        conversion.pDstBuffer = image.data;
        conversion.nDstBufferSize = static_cast<unsigned int>(image.total() * image.elemSize());

        const int convert_result = MV_CC_ConvertPixelTypeEx(handle_, &conversion);
        if (convert_result != MV_OK) {
            throw make_sdk_error("MV_CC_ConvertPixelTypeEx", convert_result);
        }
    } catch (...) {
        MV_CC_FreeImageBuffer(handle_, &frame);
        throw;
    }

    const int free_result = MV_CC_FreeImageBuffer(handle_, &frame);
    if (free_result != MV_OK) {
        throw make_sdk_error("MV_CC_FreeImageBuffer", free_result);
    }
    return {std::move(image), received_at, frame.stFrameInfo.nFrameNum,
            (static_cast<std::uint64_t>(frame.stFrameInfo.nDevTimeStampHigh) << 32U) |
                frame.stFrameInfo.nDevTimeStampLow,
            frame.stFrameInfo.nHostTimeStamp};
}

}  // namespace wit_radar
