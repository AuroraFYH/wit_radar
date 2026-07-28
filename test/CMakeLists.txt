find_package(OpenCV REQUIRED COMPONENTS calib3d highgui imgcodecs imgproc)

add_executable(test_hik_camera test_hik_camera.cpp)

target_link_libraries(test_hik_camera PRIVATE hik_camera ${OpenCV_LIBS})

add_executable(test_laser_model test_laser_model.cpp)

target_link_libraries(test_laser_model PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(test_device_center test_device_center.cpp)

target_link_libraries(test_device_center PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(test_traditional_center test_traditional_center.cpp)

target_link_libraries(test_traditional_center PRIVATE hik_camera ${OpenCV_LIBS})

add_executable(test_robot_protocol test_robot_protocol.cpp)

target_link_libraries(test_robot_protocol PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_gimbal_state_history test_gimbal_state_history.cpp)

target_link_libraries(test_gimbal_state_history PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_gimbal_timestamp_mapper test_gimbal_timestamp_mapper.cpp)

target_link_libraries(test_gimbal_timestamp_mapper PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_gimbal_arrival_time test_gimbal_arrival_time.cpp)

target_link_libraries(test_gimbal_arrival_time PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_absolute_laser_aim test_absolute_laser_aim.cpp)

target_link_libraries(test_absolute_laser_aim PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_robot_communication test_robot_communication.cpp)

target_link_libraries(test_robot_communication PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_coordinate_transformer test_coordinate_transformer.cpp)

target_link_libraries(test_coordinate_transformer PRIVATE robot_communication ${OpenCV_LIBS})

add_executable(test_cylinder_pose_solver test_cylinder_pose_solver.cpp)

target_link_libraries(test_cylinder_pose_solver PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(test_static_aim_compensation test_static_aim_compensation.cpp)

target_link_libraries(test_static_aim_compensation PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(test_cylinder_pose test_cylinder_pose.cpp)

target_link_libraries(test_cylinder_pose PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(test_world_target_tracker test_world_target_tracker.cpp)

target_link_libraries(test_world_target_tracker PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(test_world_target_observation test_world_target_observation.cpp)

target_link_libraries(test_world_target_observation PRIVATE laser_detector ${OpenCV_LIBS})

add_executable(calibrate_laser_boresight calibrate_laser_boresight.cpp)

target_link_libraries(calibrate_laser_boresight PRIVATE hik_camera ${OpenCV_LIBS})
