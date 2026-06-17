#!/usr/bin/env python3
"""
Wire/Pipe Detection 模拟测试脚本
用法: python3 test_wireandpipe_sim.py [图片路径]
默认在当前目录找 *.png 或 *.jpg 图片

功能:
  1. 发布压缩图像到 /rgb_camera_front/compressed
  2. 发布假激光雷达数据到 /front_scan（正前方 1.5m 有障碍物）
  3. 发布全局路径到 teb_global_plan（直线前进）
  4. 发布局部路径到 teb_poses（直线前进）
  5. 发布 TF: map -> odom -> base_link -> front_camera_color_frame
  6. 订阅 /is_pipes_and_wires_in_path 确认报警触发
"""

import sys
import glob
import time
import math
import numpy as np

# ---- 调试开关 ----
# 改成 True 即可保存 C++ 节点发布的标注图像到 annotated_wire_pipe.jpg
SAVE_ANNOTATED_IMAGE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

import cv2


class WirePipeTestNode(Node):
    def __init__(self, image_path: str):
        super().__init__('wire_pipe_test_node')
        self.get_logger().info(f'Test node starting, image: {image_path}')

        # ---- 读取图片并编码为 JPEG ----
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f'Failed to read image: {image_path}')
            raise RuntimeError(f'Cannot read image: {image_path}')

        # 缩放到 1280x720（与节点默认参数匹配）
        img = cv2.resize(img, (1280, 720))
        success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError('cv2.imencode failed')
        self.compressed_data = encoded.tobytes()
        self.get_logger().info(f'Image encoded: {len(self.compressed_data)} bytes, 1280x720')

        # ---- Publishers ----
        # 压缩图像
        self.pub_image = self.create_publisher(
            CompressedImage, '/rgb_camera_front/compressed', 10)

        # 激光雷达
        self.pub_laser = self.create_publisher(
            LaserScan, '/front_scan', 10)

        # 全局路径
        self.pub_global_plan = self.create_publisher(
            Path, 'teb_global_plan', 10)

        # 局部路径
        self.pub_local_poses = self.create_publisher(
            PoseArray, 'teb_poses', 10)

        # ---- TF Broadcasters ----
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)

        # 发布静态 TF
        self.publish_static_transforms()

        # ---- Subscriber: 监听报警结果 ----
        qos_transient = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_avoiding = self.create_subscription(
            Bool, '/is_pipes_and_wires_in_path', self.avoiding_callback, qos_transient)
        self.warning_received = False

        # ---- Subscriber: 保存 C++ 节点发布的标注图像（调试用） ----
        self.save_annotated = SAVE_ANNOTATED_IMAGE
        if self.save_annotated:
            self.sub_annotated = self.create_subscription(
                Image, '/rgb_camera_front/annotated_image_wire', self.annotated_callback, 5)
        self.annotated_saved = False

        # ---- Timer: 10Hz 循环发布 ----
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.frame_count = 0

        self.get_logger().info('All publishers ready, starting to publish...')

    def publish_static_transforms(self):
        """发布静态 TF 链: map -> odom -> base_link -> front_camera_color_frame"""
        now = self.get_clock().now().to_msg()
        transforms = []

        # map -> odom (identity)
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'map'
        t1.child_frame_id = 'odom'
        t1.transform.rotation.w = 1.0
        transforms.append(t1)

        # odom -> base_link (identity, 机器人在原点)
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'odom'
        t2.child_frame_id = 'base_link'
        t2.transform.rotation.w = 1.0
        transforms.append(t2)

        # base_link -> front_camera_color_frame
        # 相机在 base_link 前方 0.3m, 高 0.5m
        t3 = TransformStamped()
        t3.header.stamp = now
        t3.header.frame_id = 'base_link'
        t3.child_frame_id = 'front_camera_color_frame'
        t3.transform.translation.x = 0.3
        t3.transform.translation.z = 0.5
        t3.transform.rotation.w = 1.0
        transforms.append(t3)

        # base_link -> front_laser_frame (激光雷达)
        t4 = TransformStamped()
        t4.header.stamp = now
        t4.header.frame_id = 'base_link'
        t4.child_frame_id = 'front_laser_frame'
        t4.transform.translation.x = 0.2
        t4.transform.translation.z = 0.3
        t4.transform.rotation.w = 1.0
        transforms.append(t4)

        self.static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info('Static TF published: map->odom->base_link->camera/laser')

    def timer_callback(self):
        """10Hz 循环发布所有模拟数据"""
        now = self.get_clock().now().to_msg()
        self.frame_count += 1

        # ---- 动态 TF (保持 base_link 在原点不动) ----
        t_dyn = TransformStamped()
        t_dyn.header.stamp = now
        t_dyn.header.frame_id = 'odom'
        t_dyn.child_frame_id = 'base_link'
        t_dyn.transform.rotation.w = 1.0
        self.dynamic_tf_broadcaster.sendTransform(t_dyn)

        # ---- 压缩图像 ----
        img_msg = CompressedImage()
        img_msg.header.stamp = now
        img_msg.header.frame_id = 'front_camera_color_frame'
        img_msg.format = 'jpeg'
        img_msg.data = self.compressed_data
        self.pub_image.publish(img_msg)

        # ---- 激光雷达 ----
        # 模拟前方 1.5m 有障碍物（角度 -5° 到 +5° 范围内）
        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = 'front_laser_frame'
        scan.angle_min = -math.pi / 2.0        # -90°
        scan.angle_max = math.pi / 2.0         # +90°
        num_rays = 360
        scan.angle_increment = (scan.angle_max - scan.angle_min) / (num_rays - 1)
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.1
        scan.range_max = 10.0

        ranges = []
        for i in range(num_rays):
            angle = scan.angle_min + i * scan.angle_increment
            # 正前方 ±5° 范围内放 1.5m 的障碍物
            if abs(angle) < math.radians(5.0):
                ranges.append(1.5)
            else:
                ranges.append(8.0)  # 远处无障碍
        scan.ranges = ranges
        self.pub_laser.publish(scan)

        # ---- 全局路径 (从机器人前方直线延伸 5m) ----
        path = Path()
        path.header.stamp = now
        path.header.frame_id = 'map'
        for i in range(50):
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(i) * 0.1  # 0m ~ 5m
            ps.pose.position.y = 0.0
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_global_plan.publish(path)

        # ---- 局部路径 (从机器人前方 2m 内) ----
        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = 'map'
        for i in range(20):
            p = Pose()
            p.position.x = float(i) * 0.1  # 0m ~ 2m
            p.position.y = 0.0
            p.position.z = 0.0
            p.orientation.w = 1.0
            pose_array.poses.append(p)
        self.pub_local_poses.publish(pose_array)

        # ---- 日志 ----
        if self.frame_count == 1:
            self.get_logger().info('First frame published')
        if self.frame_count % 50 == 0:
            status = 'WARNING TRIGGERED' if self.warning_received else 'waiting...'
            self.get_logger().info(
                f'Frame #{self.frame_count} published, alert status: {status}')

    def avoiding_callback(self, msg: Bool):
        if msg.data and not self.warning_received:
            self.get_logger().warn(
                '>>> ALERT RECEIVED: /is_pipes_and_wires_in_path = True <<<')
            self.warning_received = True
        elif not msg.data and self.warning_received:
            self.get_logger().info(
                '>>> ALERT CLEARED: /is_pipes_and_wires_in_path = False <<<')
            self.warning_received = False

    def annotated_callback(self, msg: Image):
        """把节点发布的标注图存下来，方便查看检测框和激光测距结果"""
        if not SAVE_ANNOTATED_IMAGE or self.annotated_saved:
            return
        if msg.encoding != 'bgr8':
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            out_path = 'annotated_wire_pipe.jpg'
            cv2.imwrite(out_path, img)
            self.get_logger().info(f'Saved annotated image to: {out_path}')
            self.annotated_saved = True
        except Exception as e:
            self.get_logger().warn(f'Failed to save annotated image: {e}')


def find_image():
    """在当前目录找一张图片"""
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            return files[0]
    return None


def main():
    rclpy.init()

    # 确定图片路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = find_image()
        if image_path is None:
            print('ERROR: No image found in current directory.')
            print('Usage: python3 1.py [image_path]')
            print('  Supported formats: png, jpg, jpeg')
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
    print('  Wire/Pipe Detection Simulation Test')
    print('=' * 60)
    print(f'  Image:        {image_path}')
    print(f'  Image topic:  /rgb_camera_front/compressed')
    print(f'  Laser topic:  /front_scan')
    print(f'  Global path:  teb_global_plan')
    print(f'  Local path:   teb_poses')
    print(f'  Alert topic:  /is_pipes_and_wires_in_path')
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