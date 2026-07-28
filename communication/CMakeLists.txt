find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED COMPONENTS core)

add_library(robot_communication
    gimbal_arrival_time.cpp
    gimbal_state_history.cpp
    gimbal_timestamp_mapper.cpp
    serial_driver.cpp
    robot_protocol.cpp
    robot_communicator.cpp
    coordinate_transformer.cpp
    aim_solver.cpp)

target_include_directories(robot_communication
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(robot_communication
    PUBLIC Boost::system ${OpenCV_LIBS})
