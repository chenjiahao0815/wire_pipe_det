#!/usr/bin/env python3
"""
Wire/Pipe Detection Simulation Test (v2 - TF-aware)

关键设计:
  - 不发布 map->odom / odom->base_link, 避免与机器人真实定位 TF 冲突
  - 通过 TF 读取机器人当前 map 系位姿 (rx, ry, ryaw)
  - 路径以机器人当前位置为起点, 沿机器人朝向延伸 5m
  - 激光在前方 0.8m 模拟障碍 (远小于 1.0m 阈值)
  - 这样无论机器人在地图哪里, 障碍物都会落在路径上, 必触发报警

用法:
  python3 test_wireandpipe_sim.py [image_path]
  默认在当前目录找 *.png/*.jpg

注意:
  如果识别到的目标框完全在图像上半部 (y < 360), 节点的 filter_above_horizon
  会过滤掉, 此时启动节点时加 -p filter_above_horizon:=false
"""

import sys
import glob
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener

import cv2

# 调试: True = 把节点发布的标注图存成 annotated_wire_pipe.jpg
SAVE_ANNOTATED_IMAGE = False

# 障碍物到激光雷达的距离 (米). 必须 < 1.0m (节点默认 distance_threshold)
LASER_OBSTACLE_DIST = 0.8


class WirePipeTestNode(Node):
    def __init__(self, image_path: str):
        super().__init__('wire_pipe_test_node')
        self.get_logger().info(f'Test node starting, image: {image_path}')

        # ---- 读取并编码图像 ----
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        img = cv2.resize(img, (1280, 720))
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError('cv2.imencode failed')
        self.compressed_data = enc.tobytes()
        self.get_logger().info(f'Image encoded: {len(self.compressed_data)} bytes, 1280x720')

        # ---- Publishers ----
        self.pub_image = self.create_publisher(
            CompressedImage, '/rgb_camera_front/compressed', 10)
        self.pub_laser = self.create_publisher(
            LaserScan, '/front_scan', 10)
        self.pub_global_plan = self.create_publisher(
            Path, 'teb_global_plan', 10)
        self.pub_local_poses = self.create_publisher(
            PoseArray, 'teb_poses', 10)

        # ---- 静态 TF: 仅 base_link -> camera/laser ----
        # 不再发布 map->odom 和 odom->base_link, 让机器人真实定位负责这部分
        self.static_tf = StaticTransformBroadcaster(self)
        self.publish_static_transforms()

        # ---- TF buffer: 用于读取 map->base_link ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Subscriber: 报警 ----
        qos_alert = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            Bool, '/is_pipes_and_wires_in_path',
            self.avoiding_callback, qos_alert)
        self.warning_received = False

        # ---- Subscriber: 调试用标注图 ----
        self.annotated_saved = False
        if SAVE_ANNOTATED_IMAGE:
            self.create_subscription(
                Image, '/rgb_camera_front/annotated_image_wire',
                self.annotated_callback, 5)

        self.frame_count = 0
        self.tf_ready = False
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('All publishers ready, waiting for map->base_link TF...')

    # ------------------------------------------------------------------
    # 静态 TF: 只发相机和激光相对 base_link 的外参
    # ------------------------------------------------------------------
    def publish_static_transforms(self):
        now = self.get_clock().now().to_msg()
        transforms = []

        # base_link -> front_camera_color_frame (相机在前 0.3m, 高 0.5m)
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = 'front_camera_color_frame'
        t1.transform.translation.x = 0.3
        t1.transform.translation.z = 0.5
        t1.transform.rotation.w = 1.0
        transforms.append(t1)

        # base_link -> front_laser_frame (激光在前 0.2m, 高 0.3m)
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'base_link'
        t2.child_frame_id = 'front_laser_frame'
        t2.transform.translation.x = 0.2
        t2.transform.translation.z = 0.3
        t2.transform.rotation.w = 1.0
        transforms.append(t2)

        self.static_tf.sendTransform(transforms)
        self.get_logger().info(
            'Static TF published: base_link -> camera/laser '
            '(map/odom TFs left to robot localization)')

    # ------------------------------------------------------------------
    # 从 TF 读取机器人在 map 系下的当前位姿
    # ------------------------------------------------------------------
    def get_robot_pose(self):
        try:
            tr = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            x = tr.transform.translation.x
            y = tr.transform.translation.y
            q = tr.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(
                f'TF lookup map->base_link failed: {e}',
                throttle_duration_sec=2.0)
            return None

    # ------------------------------------------------------------------
    # Timer 10Hz: 发布图像、激光、路径
    # ------------------------------------------------------------------
    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        self.frame_count += 1

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, ryaw = pose
        if not self.tf_ready:
            self.tf_ready = True
            self.get_logger().info(
                f'>>> Robot pose acquired: x={rx:.2f} y={ry:.2f} '
                f'yaw={math.degrees(ryaw):.1f}deg, start publishing data')
        cosy = math.cos(ryaw)
        siny = math.sin(ryaw)

        # ---- 图像 ----
        img_msg = CompressedImage()
        img_msg.header.stamp = now
        img_msg.header.frame_id = 'front_camera_color_frame'
        img_msg.format = 'jpeg'
        img_msg.data = self.compressed_data
        self.pub_image.publish(img_msg)

        # ---- 激光: 正前方 ±15° 范围内 0.8m 处放障碍 ----
        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = 'front_laser_frame'
        scan.angle_min = -math.pi / 2.0
        scan.angle_max = math.pi / 2.0
        N = 360
        scan.angle_increment = (scan.angle_max - scan.angle_min) / (N - 1)
        scan.scan_time = 0.1
        scan.range_min = 0.1
        scan.range_max = 10.0
        ranges = []
        for i in range(N):
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) < math.radians(15.0):
                ranges.append(LASER_OBSTACLE_DIST)
            else:
                ranges.append(8.0)
        scan.ranges = ranges
        self.pub_laser.publish(scan)

        # ---- 全局路径: 机器人当前位置沿朝向前进 5m, 50 个点 ----
        path = Path()
        path.header.stamp = now
        path.header.frame_id = 'map'
        for i in range(50):
            s = i * 0.1
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = 'map'
            ps.pose.position.x = rx + s * cosy
            ps.pose.position.y = ry + s * siny
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_global_plan.publish(path)

        # ---- 局部路径: 同样起点, 前进 2m ----
        pa = PoseArray()
        pa.header.stamp = now
        pa.header.frame_id = 'map'
        for i in range(20):
            s = i * 0.1
            p = Pose()
            p.position.x = rx + s * cosy
            p.position.y = ry + s * siny
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pub_local_poses.publish(pa)

        # ---- 状态日志 ----
        if self.frame_count % 50 == 0:
            status = 'WARNING TRIGGERED' if self.warning_received else 'waiting...'
            self.get_logger().info(
                f'Frame #{self.frame_count} pose=({rx:.1f},{ry:.1f},'
                f'{math.degrees(ryaw):.0f}deg) status: {status}')

    # ------------------------------------------------------------------
    # 报警回调
    # ------------------------------------------------------------------
    def avoiding_callback(self, msg: Bool):
        if msg.data and not self.warning_received:
            self.get_logger().warn(
                '>>> ALERT RECEIVED: /is_pipes_and_wires_in_path = True <<<')
            self.warning_received = True
        elif not msg.data and self.warning_received:
            self.get_logger().info(
                '>>> ALERT CLEARED: /is_pipes_and_wires_in_path = False <<<')
            self.warning_received = False

    # ------------------------------------------------------------------
    # 调试: 保存节点发布的标注图
    # ------------------------------------------------------------------
    def annotated_callback(self, msg: Image):
        if self.annotated_saved or msg.encoding != 'bgr8':
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8) \
                    .reshape((msg.height, msg.width, 3))
            cv2.imwrite('annotated_wire_pipe.jpg', arr)
            self.get_logger().info('Saved annotated_wire_pipe.jpg')
            self.annotated_saved = True
        except Exception as e:
            self.get_logger().warn(f'save annotated failed: {e}')


def find_image():
    for pat in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        files = glob.glob(pat)
        if files:
            return files[0]
    return None


def main():
    rclpy.init()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = find_image()
        if image_path is None:
            print('ERROR: No image found. Usage: python3 test.py <image>')
            rclpy.shutdown()
            return

    try:
        node = WirePipeTestNode(image_path)
    except RuntimeError as e:
        print(f'ERROR: {e}')
        rclpy.shutdown()
        return

    print('')
    print('=' * 60)
    print('  Wire/Pipe Detection Simulation Test (v2)')
    print('=' * 60)
    print(f'  Image:        {image_path}')
    print(f'  Image topic:  /rgb_camera_front/compressed')
    print(f'  Laser topic:  /front_scan  (obstacle @ {LASER_OBSTACLE_DIST}m)')
    print(f'  Global path:  teb_global_plan  (robot pose -> +5m)')
    print(f'  Local path:   teb_poses        (robot pose -> +2m)')
    print(f'  Alert topic:  /is_pipes_and_wires_in_path')
    print('-' * 60)
    print('  Prerequisites:')
    print('    1. Robot localization is publishing map->base_link TF')
    print('    2. C++ node started, ideally with:')
    print('       ros2 run wireandpipe_detection_cpp wireandpipe_detection_node \\')
    print('         --ros-args -p filter_above_horizon:=false')
    print('       (only needed if your detection box is in upper image half)')
    print('=' * 60)
    print('  Ctrl+C to stop')
    print('')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('\nTest node shut down.')


if __name__ == '__main__':
    main()