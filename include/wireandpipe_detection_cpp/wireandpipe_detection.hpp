#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// OBB 检测结果：包含旋转角度
struct ObbBBox
{
    float cx{0.0F};       // 中心 x（像素坐标）
    float cy{0.0F};       // 中心 y（像素坐标）
    float width{0.0F};    // box 宽度（像素）
    float height{0.0F};   // box 高度（像素）
    float angle{0.0F};    // 旋转角度（弧度，取决于模型输出）
};

struct ObbTrackItem
{
    int class_id{0};
    float confidence{0.0F};
    ObbBBox bbox;
};

// OBB YOLO tracker 基类 —— 后续根据实际 .pt 模型实现具体子类
class ObbYoloTracker
{
public:
    virtual ~ObbYoloTracker() = default;
    virtual std::vector<ObbTrackItem> track(const cv::Mat &frame) = 0;
};

class WireAndPipeDetectionNode : public rclcpp::Node
{
public:
    WireAndPipeDetectionNode();

private:
    // 图像回调结果
    struct DetectionResult
    {
        bool detected{false};
        builtin_interfaces::msg::Time stamp;
        // 障碍物在 camera 坐标系内的 3D 估算坐标
        // x=前, y=左, z=上
        std::vector<geometry_msgs::msg::Point> obstacles_camera;
        std::string camera_frame_id;
    };

    // ---- 回调 ----
    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    bool runYoloTrack(const cv::Mat &frame, std::vector<ObbTrackItem> &tracks);
    void globalPlanCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void timerCallback();

    // ---- util ----
    cv::Mat decodeCompressedImage(
        const sensor_msgs::msg::CompressedImage::SharedPtr &msg,
        rclcpp::Time &out_stamp);

    std::vector<geometry_msgs::msg::Point> downsamplePath(
        const std::vector<geometry_msgs::msg::Point> &nav_points,
        double min_distance,
        double lookahead_distance);

    bool checkObstacleOnGlobalPath(
        const std::vector<geometry_msgs::msg::PointStamped> &obstacles,
        const std::vector<geometry_msgs::msg::Point> &downsampled_points,
        double threshold,
        double *hit_distance = nullptr);

    bool checkObstacleOnLocalPath(
        const std::vector<geometry_msgs::msg::PointStamped> &obstacles,
        const std::vector<geometry_msgs::msg::Point> &downsampled_points,
        double threshold,
        double *hit_distance = nullptr);

    // =========================================================================
    // [METHOD 1] 相机内参 + 物理尺寸估算距离（pinhole 模型）
    // 返回 camera 坐标系下的 3D 坐标 (x=前, y=左, z=上)
    // =========================================================================
    geometry_msgs::msg::Point estimate3DFromObb(
        const ObbBBox &obb, int class_id) const;

    // =========================================================================
    // [METHOD 2] 激光雷达测距（AABB 水平角度范围 + 最近激光点）
    // 返回距离（米），失败返回 NaN
    // =========================================================================
    float estimateDistanceFromLaser(
        const ObbBBox &obb, int img_width) const;

    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

    // ---- callback groups ----
    rclcpp::CallbackGroup::SharedPtr global_plan_callback_group_;
    rclcpp::CallbackGroup::SharedPtr local_poses_callback_group_;
    rclcpp::CallbackGroup::SharedPtr laser_callback_group_;

    // ---- 订阅 ----
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_camera_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_global_plan_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_local_poses_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_laser_;

    // ---- 发布 ----
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_avoiding_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_annotated_image_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_obstacle_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_downsampled_path_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_raw_path_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_wire_pipe_3d_;

    // ---- TF ----
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::TimerBase::SharedPtr timer_;

    // ---- 路径缓存 ----
    nav_msgs::msg::Path::SharedPtr last_global_plan_;
    geometry_msgs::msg::PoseArray::SharedPtr last_local_poses_;

    std::mutex detection_mutex_;
    DetectionResult detection_result_;

    // 参数
    // 相机
    double fx_{640.0};
    double fy_{640.0};
    double cx_{640.0};
    double cy_{360.0};
    int camera_width_{1280};
    int camera_height_{720};
    double h_fov_rad_{1.0472};
    double v_fov_rad_{0.6435};
    double camera_x_{0.0};
    double camera_y_{0.0};
    double camera_z_{1.0};
    double camera_pitch_{0.0};
    double camera_yaw_{0.0};
    double camera_roll_{0.0};

    // YOLO
    std::string yolo_model_path_;
    double conf_threshold_{0.5};
    int wire_class_id_{0};
    int water_pipe_class_id_{1};

    // 路径
    double global_search_distance_{5.0};
    double local_search_distance_{5.0};

    // 避让
    double obstacle_distance_threshold_{1.0};
    double avoid_hold_seconds_{3.0};

    // 物理尺寸
    double wire_typical_width_{0.03};
    double wire_typical_height_{0.01};
    double water_pipe_typical_width_{0.04};
    double water_pipe_typical_height_{0.03};

    // 跳帧
    int yolo_frame_skip_{1};

    // 可视化
    bool publish_annotated_image_{true};

    // ---- 状态 ----
    bool has_recent_detection_{false};
    bool last_published_avoiding_{false};
    bool last_global_match_{false};
    bool last_local_match_{false};
    rclcpp::Time last_obstacle_detect_time_;
    uint64_t warning_event_id_{0};

    bool global_plan_dirty_{false};
    bool local_poses_dirty_{false};
    std::vector<geometry_msgs::msg::Point> cached_ds_global_;
    std::vector<geometry_msgs::msg::Point> cached_ds_local_;

    std::shared_ptr<ObbYoloTracker> yolo_;

    // ---- 激光队列 ----
    struct LaserQueueItem {
        double ts;
        sensor_msgs::msg::LaserScan::SharedPtr scan;
    };
    std::deque<LaserQueueItem> laser_queue_;
    std::mutex laser_queue_mutex_;
    static constexpr size_t laser_queue_max_size_{10};

    // 激光测距用的缓存（在 imageCallback 中计算，供 estimateDistanceFromLaser 使用）
    struct LaserCamPt {
        float xc;         // 相机系 X（前）
        float yc;         // 相机系 Y（左）
        float r;          // 原始距离
        float theta_l;    // 原始角度
        float cam_angle;  // 相机系方位角
    };
    mutable std::vector<LaserCamPt> cached_laser_cam_pts_;
    mutable sensor_msgs::msg::LaserScan::SharedPtr cached_laser_scan_;
    mutable bool cached_laser_valid_{false};

    // 日志去重
    bool waiting_detect_log_printed_{false};
    bool first_obstacle_detected_logged_{false};
    rclcpp::Time last_waiting_log_time_{0, 0, RCL_ROS_TIME};

    // timer 用于只在状态变化时打印日志
    bool last_trigger_state_{false};
};
