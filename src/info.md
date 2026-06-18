ros2 run wireandpipe_detection_cpp wireandpipe_detection_node \
    --ros-args -p filter_above_horizon:=false

ros2 topic pub --once /is_pipes_and_wires_in_path std_msgs/msg/Bool "{data: true}" \
  --qos-durability transient_local --qos-reliability reliable    