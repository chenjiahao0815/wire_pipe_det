import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy, ReliabilityPolicy, QoSProfile, HistoryPolicy
# garbage_cord_markers
import tf2_geometry_msgs
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from capella_ros_msg.msg import GarbageDetect
from visualization_msgs.msg import Marker, MarkerArray

import torch

import cv2
import time
import math
import numpy as np
import os
from collections import deque
from functools import partial

from ultralytics import YOLO

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class GarbageDetectionDemonstration(Node):
    def __init__(self):
        super().__init__('garbage_detection_demo')
        
        # 相机编号列表  show
        #self.camera_ids = [1, 2, 4]
        self.camera_ids = [1]
        self.callback_group_list = []
        self.subscription_list = []
        
        # 每个相机独立的数据存储
        self.camera_states = {}
        
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        
        self.model = YOLO(os.path.join(os.path.dirname(__file__), '/capella/lib/python3.10/site-packages/garbage_detection_node/2-24.pt'))   

        self.get_logger().info("Model initialization successful !")
        
        qos_ = QoSProfile(depth=1)
        qos_.reliability = ReliabilityPolicy.BEST_EFFORT
        
        # 为每个摄像头初始化数据存储和订阅
        for camera_id in self.camera_ids:
            # 初始化相机状态
            self.camera_states[camera_id] = {
                'K': None,
                'depth_data': None,
                'depth_x': None,
                'depth_y': None,
                'depth_width': None,
                'depth_height': None,
                'depth_frame_id': None,
                'frame_skip_counter': 0,  # 跳帧计数器
            }
            
            # 为每个摄像头创建独立的回调组
            group_color = MutuallyExclusiveCallbackGroup()
            group_depth = MutuallyExclusiveCallbackGroup()
            group_info = MutuallyExclusiveCallbackGroup()
            self.callback_group_list.extend([group_color, group_depth, group_info])
            
            # topic 名称
            color_topic = f'/camera{camera_id}/color/image_raw'
            depth_topic = f'/camera{camera_id}/depth/image_raw'
            camera_info_topic = f'/camera{camera_id}/depth/camera_info'
            
            # 订阅color图像、深度图像和CameraInfo
            self.subscription_list.append(
                self.create_subscription(
                    Image,
                    color_topic,
                    partial(self.color_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_color,
                )
            )
            
            self.subscription_list.append(
                self.create_subscription(
                    Image,
                    depth_topic,
                    partial(self.depth_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_depth,
                )
            )
            
            self.subscription_list.append(
                self.create_subscription(
                    CameraInfo,
                    camera_info_topic,
                    partial(self.camera_info_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_info,
                )
            )
        
        self.pose_publisher = self.create_publisher(GarbageDetect, '/garbage_cord', 1)
        
        # 垃圾历史点可视化发布器
        self.garbage_markers_pub = self.create_publisher(MarkerArray, '/garbage_cord_markers', 10)

        # 各摄像头 YOLO 检测框标注图像发布器
        self.annotated_image_publishers = {}
        for _cam_id in self.camera_ids:
            self.annotated_image_publishers[_cam_id] = self.create_publisher(
                Image, f'/camera{_cam_id}/annotated_image', 10
            )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.horizontal_fov = 86

        # 已发布的垃圾坐标列表
        self._published_garbage_xy_map: list[tuple[float, float]] = []
        
        # 可视化的垃圾点，用一个列表来装
        self._visualization_garbage_list = deque(maxlen=6)
        # 垃圾可视化
        self.garbage_visualization_timer = self.create_timer(0.5, self._update_garbage_visualization)
        self.declare_parameter('unique_distance_threshold_m', 0.15)
        self._unique_distance_threshold_m = float(self.get_parameter('unique_distance_threshold_m').value)
        


    def _is_new_garbage_xy_map(self, x_m: float, y_m: float) -> bool:
        for px, py in self._published_garbage_xy_map:
            if (x_m - px) ** 2 + (y_m - py) ** 2 <= self._unique_distance_threshold_m ** 2:
                return False
        return True


    def camera_info_callback(self, msg, camera_id: int):
        self.camera_states[camera_id]['K'] = msg.k
        

    def depth_callback(self, msg, camera_id: int):
        state = self.camera_states[camera_id]
        if len(msg.data) > 0:
            depth_width = msg.width
            depth_height = msg.height
            state['depth_width'] = depth_width
            state['depth_height'] = depth_height
            state['depth_frame_id'] = msg.header.frame_id
            
            state['depth_data'] = (
                torch.from_numpy(np.array(msg.data))
                .to(self.device)
                .to(torch.float32)
                .reshape((depth_height, depth_width, -1))
            )
            
            # 深度相机中每个像素的x和y
            if (
                state['depth_x'] is None
                or state['depth_y'] is None
                or state['depth_x'].shape[0] != depth_height
                or state['depth_x'].shape[1] != depth_width
            ):
                x = torch.arange(depth_width, device=self.device).repeat(depth_height, 1)
                y = torch.arange(depth_height, device=self.device).unsqueeze(1).repeat(1, depth_width)
                state['depth_x'] = x
                state['depth_y'] = y
        else:
            state['depth_data'] = None
            self.get_logger().info(f"[camera{camera_id}] Depth cameras do not have depth data !") 


    def color_callback(self, msg, camera_id: int):
        # 跳帧：每5帧推理一次，减少CPU占用
        state = self.camera_states[camera_id]
        state['frame_skip_counter'] += 1
        if state['frame_skip_counter'] % 5 != 0:
            return

        if len(msg.data) > 0:
            color_time = msg.header.stamp
            color_width = msg.width
            color_height = msg.height
            
            # color_data_bgr = np.array(msg.data).reshape((color_height, color_width, 3))
            # color_data = cv2.cvtColor(color_data_bgr, cv2.COLOR_BGR2RGB)
            # annotated_bgr = color_data_bgr.copy()


            color_data_rgb = np.array(msg.data).reshape((color_height, color_width, 3))
            color_data = color_data_rgb  # YOLO 用 RGB
            annotated_bgr = cv2.cvtColor(color_data_rgb, cv2.COLOR_RGB2BGR)
            results = self.model.predict(color_data, conf=0.5, classes=[0,1,2,3,4], show=False, verbose=False)
            any_garbage_detected = False
            for result in results:
                boxes = result.boxes.xyxy.tolist()
                classes = result.boxes.cls.tolist()
                confidences = result.boxes.conf.tolist()
                # 获取每个检测框的坐标（格式为 [左上角x, 左上角y, 右下角x, 右下角y]），转换为列表。
                # 获取每个检测框对应的类别索引，转换为列表。
                
                # 绘制 YOLO 检测框到标注图像
                if len(boxes) > 0:
                    any_garbage_detected = True
                    _cls_colors = {
                        0: (0, 255, 0),    # green
                        1: (255, 128, 0),  # orange
                        2: (0, 128, 255),  # light-blue
                        3: (0, 0, 255),    # red
                        4: (255, 0, 255),  # magenta
                    }
                    for _i in range(len(boxes)):
                        _bx1 = int(boxes[_i][0])
                        _by1 = int(boxes[_i][1])
                        _bx2 = int(boxes[_i][2])
                        _by2 = int(boxes[_i][3])
                        _cls = int(classes[_i])
                        _conf = confidences[_i] if _i < len(confidences) else 0.0
                        _color = _cls_colors.get(_cls, (0, 0, 255))
                        cv2.rectangle(annotated_bgr, (_bx1, _by1), (_bx2, _by2), _color, 2)
                        _label = f'cls:{_cls} {_conf:.2f}'
                        cv2.putText(
                            annotated_bgr, _label,
                            (_bx1, max(15, _by1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _color, 1
                        )

                if state['K'] is not None and state['depth_data'] is not None:
                    depth_image = self.get_depth_xyz(state['depth_data'], state['K'], state['depth_x'], state['depth_y'])

                    if len(boxes) > 0:
                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
                            # 跳过污水（YOLO类别3）
                            if int(classes[i]) in [0, 1, 3]:
                                continue
                            class_id = 2

                            center_x = int((x1 + x2) // 2)
                            center_y = int((y1 + y2) // 2)
                            # self.get_logger().info(f"center_y:{center_y}")
                            # 计算检测框的宽度和高度
                            dx = abs(x1-x2)
                            dy = abs(y1-y2)
                            # self.get_logger().info(f"dx: {dx}, dy: {dy}")
                            # 深度图像和彩色图像的分辨率可能会不同，乘以这些比例，确保在深度图像中获取正确的对应区域的深度数据
                            if state['depth_width'] is None or state['depth_height'] is None:
                                continue
                            x_scale = state['depth_width'] / color_width
                            y_scale = state['depth_height'] / color_height
                            
                            if center_y >= int(color_height * 0):
                                if dy <= 20:
                                    depth_center = self.get_valid_depth_in_bbox(
                                        depth_image, x1, y1-5, x2, y2+5, x_scale, y_scale,
                                        depth_width=state['depth_width'],
                                        depth_height=state['depth_height']
                                    )
                                else:
                                    depth_center = self.get_valid_depth_in_bbox(
                                        depth_image, x1, y1, x2, y2, x_scale, y_scale,
                                        depth_width=state['depth_width'],
                                        depth_height=state['depth_height']
                                    )

                                camera_center_x, camera_center_y, camera_center_z = depth_center[:3]

                                if camera_center_x == 0 and camera_center_y == 0 and camera_center_z == 0:
                                    empty_pose = PoseStamped()
                                    empty_pose.header.frame_id = 'base_link'
                                    empty_pose.pose.position.x = 0.0
                                    empty_pose.pose.position.y = 0.0
                                    empty_pose.pose.position.z = 0.0

                                    empty_pose.pose.orientation.x = 0.0
                                    empty_pose.pose.orientation.y = 0.0
                                    empty_pose.pose.orientation.z = 0.0
                                    empty_pose.pose.orientation.w = 0.0

                                    pose_msg_no_bbox = GarbageDetect()
                                    pose_msg_no_bbox.pose = empty_pose
                                    pose_msg_no_bbox.class_id = -1
                                    self.pose_publisher.publish(pose_msg_no_bbox)
                                    self.get_logger().info(f"[camera{camera_id}] No garbage detected (no depth data) !", throttle_duration_sec=3.0)

                                else:
                                    pose_msg = PoseStamped()
                                    # 从相机状态字典中获取frame_id
                                    pose_msg.header.frame_id = state.get('depth_frame_id') or msg.header.frame_id
                                    pose_msg.header.stamp = color_time
                                    pose_msg.pose.position.x = float(camera_center_x)
                                    pose_msg.pose.position.y = float(camera_center_y)
                                    pose_msg.pose.position.z = float(camera_center_z)

                                    pose_msg.pose.orientation.x = 0.0
                                    pose_msg.pose.orientation.y = 0.0
                                    pose_msg.pose.orientation.z = 0.0
                                    pose_msg.pose.orientation.w = 1.0

                                    try:
                                        transform = self.tf_buffer.lookup_transform('base_link', pose_msg.header.frame_id,
                                                                                    rclpy.time.Time(),
                                                                                    rclpy.duration.Duration(seconds=0.5))
                                        # 将相机坐标系的位姿转换为base_link坐标系
                                        pose_in_base_link = tf2_geometry_msgs.do_transform_pose_stamped(pose_msg, transform)

                                        # 将base_link坐标系的位姿转换为map坐标系）
                                        transform_map = self.tf_buffer.lookup_transform(
                                            'map',
                                            'base_link',
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.5)
                                        )
                                        pose_in_map = tf2_geometry_msgs.do_transform_pose_stamped(pose_in_base_link, transform_map)
                                        pose_in_map.header.frame_id = 'map'
                                        
                                        theta = np.arctan2(pose_in_base_link.pose.position.y, pose_in_base_link.pose.position.x)
                                        theta_deg = np.rad2deg(theta)
                                        w = np.cos(theta / 2)
                                        z = np.sin(theta / 2)

                                        # 获取边界框四个角点在base_link坐标系中的坐标
                                        bbox_corner_points = self.get_bbox_corner_points_in_base_link(
                                            depth_image, x1, y1, x2, y2, x_scale, y_scale, state['K'],
                                            color_time, state.get('depth_frame_id') or msg.header.frame_id,
                                            depth_width=state['depth_width'],
                                            depth_height=state['depth_height'],
                                            transform=transform
                                        )

                                        # 只发布干垃圾 (class_id == 2)，湿垃圾 (class_id == 1) 不发布
                                        if class_id == 2:
                                            pose_msg_no_bbox = GarbageDetect()
                                            pose_msg_no_bbox.pose = pose_in_base_link
                                            pose_msg_no_bbox.pose.pose.orientation.x = 0.0
                                            pose_msg_no_bbox.pose.pose.orientation.y = 0.0
                                            pose_msg_no_bbox.pose.pose.orientation.z = float(z)
                                            pose_msg_no_bbox.pose.pose.orientation.w = float(w)
                                            pose_msg_no_bbox.class_id = class_id
                                            pose_msg_no_bbox.bbox_corner_points = bbox_corner_points
                                            self.pose_publisher.publish(pose_msg_no_bbox)

                                        # 计数 +1
                                        is_new = self._is_new_garbage_xy_map(
                                            pose_in_map.pose.position.x,
                                            pose_in_map.pose.position.y,
                                        )
                                        if is_new and class_id != -1:
                                            self._published_garbage_xy_map.append(
                                                (pose_in_map.pose.position.x, pose_in_map.pose.position.y)
                                            )
                                            self._visualization_garbage_list.append(
                                                (pose_in_map.pose.position.x, pose_in_map.pose.position.y)
                                            )

                                        # 构建角点坐标字符串
                                        corners_str = ""
                                        if bbox_corner_points:
                                            corners_list = [(p.x, p.y, p.z) for p in bbox_corner_points]
                                            corners_str = f", corners: {corners_list}"
                                        else:
                                            corners_str = ", corners: []"

                                        self.get_logger().info(
                                            f"[camera{camera_id}] id=%d, base_link(x=%.3f, y=%.3f), map(x=%.3f, y=%.3f), yaw=%.2fdeg{corners_str}" % (
                                                class_id,
                                                pose_in_base_link.pose.position.x,
                                                pose_in_base_link.pose.position.y,
                                                pose_in_map.pose.position.x,
                                                pose_in_map.pose.position.y,
                                                theta_deg,
                                            )
                                        )

                                    except (LookupException, ConnectivityException, ExtrapolationException) as e:
                                        self.get_logger().warn(f"[camera{camera_id}] Transform lookup failed: {str(e)}")
                                                                

         
            # 有检测框时发布标注图像
            if any_garbage_detected and camera_id in self.annotated_image_publishers:
                annotated_msg = Image()
                annotated_msg.header = msg.header
                annotated_msg.height = color_height
                annotated_msg.width = color_width
                annotated_msg.encoding = 'bgr8'
                annotated_msg.is_bigendian = 0
                annotated_msg.step = color_width * 3
                annotated_msg.data = annotated_bgr.tobytes()
                self.annotated_image_publishers[camera_id].publish(annotated_msg)

        else:
            self.get_logger().info("Depth cameras do not have color data !")



    def get_valid_depth_in_bbox(self, depth_image, x1, y1, x2, y2, x_scale, y_scale, depth_width=None, depth_height=None):
        # center_x = (x1 + x2) / 2
        # center_y = (y1 + y2) / 2
        
        # width = (x2 - x1) / 2
        # height = (y2 - y1) / 2
        
        # new_x1 = center_x - width / 2
        # new_y1 = center_y - height / 2
        # new_x2 = center_x + width / 2
        # new_y2 = center_y + height / 2

        # new_x1, new_x2 = int(new_x1 * x_scale), int(new_x2 * x_scale)
        # new_y1, new_y2 = int(new_y1 * y_scale), int(new_y2 * y_scale)

        # new_x1, new_x2 = max(0, new_x1), min(self.depth_width, new_x2)
        # new_y1, new_y2 = max(0, new_y1), min(self.depth_height, new_y2)
        
        # depth_roi = depth_image[new_y1:new_y2, new_x1:new_x2].cpu().numpy().astype(float)
        
        x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
        y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
        
        # 使用传入的depth_width和depth_height参数
        if depth_width is not None and depth_height is not None:
            x1, x2 = max(0, x1), min(depth_width, x2)
            y1, y2 = max(0, y1), min(depth_height, y2)
        else:
            # 向后兼容：如果没有传入参数，使用self属性
            x1, x2 = max(0, x1), min(self.depth_width, x2)
            y1, y2 = max(0, y1), min(self.depth_height, y2)
        
        depth_roi = depth_image[y1:y2, x1:x2].cpu().numpy().astype(float)

        # 过滤有效深度值
        valid_depths = depth_roi[(depth_roi[:, :, 2] > 0) & (depth_roi[:, :, 2] < 2.1)]

        if len(valid_depths) > 0:
            # 计算平均值
            avg_depth = valid_depths.mean(axis=0)
            return avg_depth
        else:
            return np.array([0.0, 0.0, 0.0])


    def get_bbox_corner_points_in_base_link(self, depth_image, x1, y1, x2, y2, x_scale, y_scale, K, 
                                            color_time, frame_id, depth_width=None, depth_height=None, transform=None):

        # 四个角点的像素坐标
        corner_pixels = [
            (int(x1), int(y1)),      # 左上
            (int(x2), int(y1)),      # 右上
            (int(x1), int(y2)),      # 左下
            (int(x2), int(y2)),      # 右下
        ]
        
        bbox_corner_points = []
        
        for pixel_x, pixel_y in corner_pixels:
            # 转换到深度图像坐标
            depth_x = int(pixel_x * x_scale)
            depth_y = int(pixel_y * y_scale)
            
            # 边界检查
            if depth_width is not None and depth_height is not None:
                depth_x = max(0, min(depth_x, depth_width - 1))
                depth_y = max(0, min(depth_y, depth_height - 1))
            else:
                depth_x = max(0, min(depth_x, self.depth_width - 1))
                depth_y = max(0, min(depth_y, self.depth_height - 1))
            
            # 获取该点的深度信息
            depth_point = depth_image[depth_y, depth_x].cpu().numpy()
            
            # 检查深度值是否有效
            if depth_point[2] > 0 and depth_point[2] < 2.1:
                # 使用该点的深度坐标
                camera_x = depth_point[0]
                camera_y = depth_point[1]
                camera_z = depth_point[2]
            else:
                # 如果单点深度无效，从周围区域获取平均深度
                radius = 2
                y_start = max(0, depth_y - radius)
                y_end = min(depth_height if depth_height is not None else self.depth_height, depth_y + radius + 1)
                x_start = max(0, depth_x - radius)
                x_end = min(depth_width if depth_width is not None else self.depth_width, depth_x + radius + 1)
                
                roi = depth_image[y_start:y_end, x_start:x_end]
                valid_depths = roi[(roi[:, :, 2] > 0) & (roi[:, :, 2] < 2.1)]
                
                if len(valid_depths) > 0:
                    avg_depth = valid_depths.mean(axis=0)
                    camera_x = avg_depth[0]
                    camera_y = avg_depth[1]
                    camera_z = avg_depth[2]
                else:
                    # 如果仍然无法获取有效深度，跳过此角点
                    continue
            
            # 创建相机坐标系中的点
            corner_pose = PoseStamped()
            corner_pose.header.frame_id = frame_id
            corner_pose.header.stamp = color_time
            corner_pose.pose.position.x = float(camera_x)
            corner_pose.pose.position.y = float(camera_y)
            corner_pose.pose.position.z = float(camera_z)
            corner_pose.pose.orientation.x = 0.0
            corner_pose.pose.orientation.y = 0.0
            corner_pose.pose.orientation.z = 0.0
            corner_pose.pose.orientation.w = 1.0
            
            # 转换到base_link坐标系
            if transform is not None:
                corner_in_base_link = tf2_geometry_msgs.do_transform_pose_stamped(corner_pose, transform)
                point = Point()
                point.x = float(corner_in_base_link.pose.position.x)
                point.y = float(corner_in_base_link.pose.position.y)
                point.z = float(corner_in_base_link.pose.position.z)
                bbox_corner_points.append(point)
        
        return bbox_corner_points

                    
    def get_depth_xyz(self, depth_data, K, x, y):
        # 处理深度图像数据，将其转换为三维坐标系中的点云数据
        depth_z_ = (depth_data[:,:,1]*256 + depth_data[:,:,0])
        depth_x_ = (x - K[2]) / K[0] * depth_z_
        depth_y_ = (y - K[5]) / K[4] * depth_z_
        
        depth_z_ *= 0.001
        depth_x_ *= 0.001
        depth_y_ *= 0.001
        
        depth_image = torch.stack([depth_x_, depth_y_, depth_z_], axis=2)
            
        return depth_image


    def _update_garbage_visualization(self):
 
        marker_array = MarkerArray()
 
        for i, (x, y) in enumerate(self._visualization_garbage_list):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'garbage_history'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # 位置
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.1  # 圆柱体高度的一半
            marker.pose.orientation.w = 1.0
            
            # 尺寸
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # 颜色：红色，半透明
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime.sec = 0  # 永久显示
            
            marker_array.markers.append(marker)
        
        # 如果可视化列表为空，发布一个删除所有的标记
        if len(self._visualization_garbage_list) == 0:
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = 'garbage_history'
            delete_marker.id = 0
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
        
        self.garbage_markers_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    garbage_detection_demonstration = GarbageDetectionDemonstration()

    multi_executor = MultiThreadedExecutor(num_threads=4)
    multi_executor.add_node(garbage_detection_demonstration)
    multi_executor.spin()
    garbage_detection_demonstration.destroy_node()
    multi_executor.shutdown()
