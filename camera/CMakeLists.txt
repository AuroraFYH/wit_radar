set(MVS_ROOT "/opt/MVS" CACHE PATH "Hikrobot MVS SDK root directory")

find_path(MVS_INCLUDE_DIR
    NAMES MvCameraControl.h
    PATHS "${MVS_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED)

find_library(MVS_CAMERA_CONTROL_LIBRARY
    NAMES MvCameraControl
    PATHS "${MVS_ROOT}/lib/64"
    NO_DEFAULT_PATH
    REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core)

add_library(hik_camera hik_camera.cpp)

target_include_directories(hik_camera
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${MVS_INCLUDE_DIR}")

target_link_libraries(hik_camera
    PUBLIC ${OpenCV_LIBS}
    PRIVATE "${MVS_CAMERA_CONTROL_LIBRARY}")

set_target_properties(hik_camera PROPERTIES
    BUILD_RPATH "${MVS_ROOT}/lib/64"
    INSTALL_RPATH "${MVS_ROOT}/lib/64")
