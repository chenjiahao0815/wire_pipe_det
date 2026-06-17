#!/usr/bin/env python3
"""
2.py

功能：获取机器人在 map 下的当前位姿，并向前方直发 5 个路径点：
      - 全局路径：teb_global_plan（nav_msgs/Path）
      - 局部路径：teb_poses（geometry_msgs/PoseArray）

每个点间隔 1m，方向与机器人当前航向一致。
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException


class SimplePathPublisher(Node):
    def __init__(self):
        super().__init__('simple_path_publisher')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('num_points', 5)
        self.declare_parameter('step_distance', 1.0)
        self.declare_parameter('publish_rate_hz', 1.0)

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value
        self.step_distance = self.get_parameter('step_distance').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        # QoS：Path / PoseArray 用 transient_local 方便 late-joiner 也能收到最新路径
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_global_plan = self.create_publisher(Path, 'teb_global_plan', qos)
        self.pub_local_poses = self.create_publisher(PoseArray, 'teb_poses', qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0 / publish_rate, self.publish_paths)

        self.get_logger().info(
            f'SimplePathPublisher started: map={self.map_frame}, base={self.base_frame}, '
            f'num_points={self.num_points}, step={self.step_distance}m'
        )

    def publish_paths(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)

        now = self.get_clock().now().to_msg()

        path = Path()
        path.header.stamp = now
        path.header.frame_id = self.map_frame

        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = self.map_frame

        for i in range(1, self.num_points + 1):
            d = i * self.step_distance
            px = x + d * math.cos(yaw)
            py = y + d * math.sin(yaw)
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)

            # Path 用的 PoseStamped
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = 0.0
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)

            # PoseArray 用的 Pose
            p = Pose()
            p.position.x = px
            p.position.y = py
            p.position.z = 0.0
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = qz
            p.orientation.w = qw
            pose_array.poses.append(p)

        self.pub_global_plan.publish(path)
        self.pub_local_poses.publish(pose_array)

        self.get_logger().info(
            f'Published {self.num_points} path points ahead from ({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.1f}°',
            throttle_duration_sec=5.0
        )

    @staticmethod
    def _quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = SimplePathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
