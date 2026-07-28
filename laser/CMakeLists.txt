find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATH_SUFFIXES x86_64-linux-gnu
    REQUIRED)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    REQUIRED)

find_path(CUDA_RUNTIME_INCLUDE_DIR
    NAMES cuda_runtime_api.h
    PATHS /usr/local/cuda/targets/x86_64-linux/include
    REQUIRED)

find_library(CUDA_RUNTIME_LIBRARY
    NAMES cudart
    PATHS /usr/local/cuda/targets/x86_64-linux/lib
    REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS calib3d core dnn imgproc highgui video)

add_library(laser_detector laser_detector.cpp device_center_finder.cpp cylinder_pose_solver.cpp
    keypoint_center_finder.cpp static_aim_compensation.cpp world_target_observation.cpp
    world_target_tracker.cpp)

target_include_directories(laser_detector
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_RUNTIME_INCLUDE_DIR}")

target_link_libraries(laser_detector
    PUBLIC ${OpenCV_LIBS}
    PRIVATE "${TENSORRT_LIBRARY}" "${CUDA_RUNTIME_LIBRARY}")

add_executable(test_laser test_laser.cpp)

target_link_libraries(test_laser PRIVATE laser_detector hik_camera ${OpenCV_LIBS})

add_executable(laser_aim laser_aim.cpp)

target_link_libraries(laser_aim PRIVATE laser_detector hik_camera robot_communication ${OpenCV_LIBS})

set_target_properties(test_laser PROPERTIES
    BUILD_RPATH "/opt/MVS/lib/64"
    INSTALL_RPATH "/opt/MVS/lib/64")

set_target_properties(laser_aim PROPERTIES
    BUILD_RPATH "/opt/MVS/lib/64"
    INSTALL_RPATH "/opt/MVS/lib/64")
