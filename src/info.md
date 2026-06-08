[INFO] [1719312000.000000000] [wire_andpipe_detection_node]: WireAndPipeDetectionNode initialized
[INFO] [1719312000.000000000] [wire_andpipe_detection_node]:   Camera: /rgb_camera_front/compressed
[INFO] [1719312000.000000000] [wire_andpipe_detection_node]:   Laser: /front_scan
[INFO] [1719312000.000000000] [wire_andpipe_detection_node]:   Model: /opt/ros/humble/share/wireandpipe_detection_cpp/wire_pipe_det.onnx
[INFO] [1719312000.000000000] [wire_andpipe_detection_node]:   FOV: h=60.00° v=36.87°
[INFO] [1719312000.000000000] [wire_andpipe_detection_node]:   Distance threshold: 1.00m
[INFO] [1719312000.001000000] [wire_andpipe_detection_node]: [OBB YOLO] Model loaded: /opt/ros/humble/share/wireandpipe_detection_cpp/wire_pipe_det.onnx
[INFO] [1719312000.001000000] [wire_andpipe_detection_node]: [OBB YOLO] GPU backend enabled (ONNX Runtime CUDA EP)
[INFO] [1719312000.500000000] [wire_andpipe_detection_node]: globalPlanCallback: received path with 247 poses
[INFO] [1719312000.510000000] [wire_andpipe_detection_node]: localPosesCallback: received 38 poses
[INFO] [1719312000.600000000] [wire_andpipe_detection_node]: imageCallback triggered
[INFO] [1719312000.605000000] [wire_andpipe_detection_node]: [imageCallback] Decoded image 1280x720
[INFO] [1719312000.648000000] [wire_andpipe_detection_node]: [OBB YOLO] First inference raw output:
[INFO] [1719312000.648000000] [wire_andpipe_detection_node]: [OBB YOLO]   output_shape = [1, 7, 8400]
[INFO] [1719312000.648000000] [wire_andpipe_detection_node]: [OBB YOLO]   total=58800 non_zero=41623 (70.8%) min=-8.4312 max=642.1094 mean=1.8734
[INFO] [1719312000.648000000] [wire_andpipe_detection_node]: [OBB YOLO]   sample[0..19]: 3.2451 5.1289 7.0312 8.9141 10.8125 12.7031 14.5938 16.4844 18.3750 20.2656 22.1562 24.0469 25.9375 27.8281 29.7188 31.6094 33.5000 35.3906 37.2812 39.1719
[INFO] [1719312000.650000000] [wire_andpipe_detection_node]: Waiting to detect wire/water_pipe...
[INFO] [1719312000.700000000] [wire_andpipe_detection_node]: Waiting to detect wire/water_pipe...
[INFO] [1719312002.100000000] [wire_andpipe_detection_node]: Wire/Water pipe detected for the first time
[INFO] [1719312002.100000000] [wire_andpipe_detection_node]: [distance] pinhole=2.47m  laser=2.31m
[INFO] [1719312002.200000000] [wire_andpipe_detection_node]: [global_path] obstacle on global path
[WARN] [1719312002.200000000] [wire_andpipe_detection_node]: [timer] Obstacle detected on path
[WARN] [1719312002.200000000] [wire_andpipe_detection_node]: Wire/Pipe obstacle warning, id: 1
[INFO] [1719312002.300000000] [wire_andpipe_detection_node]: [distance] pinhole=2.35m  laser=2.19m
[INFO] [1719312002.400000000] [wire_andpipe_detection_node]: [distance] pinhole=2.21m  laser=2.08m
[INFO] [1719312002.500000000] [wire_andpipe_detection_node]: [distance] pinhole=2.09m  laser=1.95m
[INFO] [1719312002.600000000] [wire_andpipe_detection_node]: [distance] pinhole=1.98m  laser=1.84m
[INFO] [1719312002.700000000] [wire_andpipe_detection_node]: [distance] pinhole=1.87m  laser=1.73m
[INFO] [1719312002.800000000] [wire_andpipe_detection_node]: [distance] pinhole=1.76m  laser=1.62m
[INFO] [1719312002.900000000] [wire_andpipe_detection_node]: [distance] pinhole=1.65m  laser=1.51m
[INFO] [1719312003.000000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.100000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.200000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.300000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.400000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.500000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.600000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.700000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.800000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312003.900000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312004.000000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312004.100000000] [wire_andpipe_detection_node]: [distance] pinhole=1.54m  laser=1.40m
[INFO] [1719312005.300000000] [wire_andpipe_detection_node]: [global_path] no match: min_dist=1.87m threshold=1.00m
[INFO] [1719312005.300000000] [wire_andpipe_detection_node]: [timer] Path clear, obstacle no longer in range
[INFO] [1719312005.400000000] [wire_andpipe_detection_node]: Wire/Pipe obstacle warning cleared, id: 1
[INFO] [1719312005.500000000] [wire_andpipe_detection_node]: [distance] pinhole=3.12m  laser=2.98m
[INFO] [1719312005.600000000] [wire_andpipe_detection_node]: [distance] pinhole=3.45m  laser=3.31m
[INFO] [1719312006.000000000] [wire_andpipe_detection_node]: globalPlanCallback: received path with 189 poses
[INFO] [1719312006.010000000] [wire_andpipe_detection_node]: localPosesCallback: received 34 poses
[INFO] [1719312007.200000000] [wire_andpipe_detection_node]: [distance] pinhole=1.82m  laser=1.69m
[INFO] [1719312007.200000000] [wire_andpipe_detection_node]: [local_path] obstacle on local path
[WARN] [1719312007.200000000] [wire_andpipe_detection_node]: [timer] Obstacle detected on path
[WARN] [1719312007.200000000] [wire_andpipe_detection_node]: Wire/Pipe obstacle warning, id: 2
[INFO] [1719312007.300000000] [wire_andpipe_detection_node]: [distance] pinhole=1.74m  laser=1.61m
[INFO] [1719312007.400000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312007.500000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312007.600000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312007.700000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312007.800000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312007.900000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312008.000000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312008.100000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312008.200000000] [wire_andpipe_detection_node]: [distance] pinhole=1.68m  laser=1.55m
[INFO] [1719312009.400000000] [wire_andpipe_detection_node]: [distance] pinhole=2.89m  laser=2.76m
[INFO] [1719312009.500000000] [wire_andpipe_detection_node]: [local_path] no match: min_dist=2.14m threshold=1.00m
[INFO] [1719312009.500000000] [wire_andpipe_detection_node]: [timer] Path clear, obstacle no longer in range
[INFO] [1719312010.400000000] [wire_andpipe_detection_node]: Wire/Pipe obstacle warning cleared, id: 2
[INFO] [1719312010.500000000] [wire_andpipe_detection_node]: globalPlanCallback: received path with 152 poses
[INFO] [1719312010.510000000] [wire_andpipe_detection_node]: localPosesCallback: received 30 poses
[INFO] [1719312011.000000000] [wire_andpipe_detection_node]: [distance] pinhole=4.21m  laser=4.05m
[INFO] [1719312011.500000000] [wire_andpipe_detection_node]: [distance] pinhole=5.03m  laser=4.88m
[INFO] [1719312012.000000000] [wire_andpipe_detection_node]: [distance] pinhole=5.67m  laser=5.51m