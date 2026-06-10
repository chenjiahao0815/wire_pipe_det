#include "wireandpipe_detection_cpp/wireandpipe_detection.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>

using namespace std::chrono_literals;

namespace
{
// =============================================================================
// OrtObbYoloTracker: ONNX Runtime OBB YOLO tracker
// YOLOv11 OBB 导出 ONNX 的输出格式支持两种:
//   Format A (per-class scores): [1, 6+nc, N] = [1, 8, 8400]
//     每列: cx, cy, w, h, angle, obj_conf, class_0, class_1
//   Format B (single class logit): [1, 7, N] = [1, 7, 8400]
//     每列: cx, cy, w, h, angle, obj_conf, cls_logit
//     ultralytics 默认导出格式 (imgsz=640 simplify opset=12)
// nc = 2: ["wire", "water_pipe"]
// 代码会动态检测通道数并选择相应的解析模式。
// =============================================================================
class OrtObbYoloTracker : public ObbYoloTracker
{
public:
    OrtObbYoloTracker(const std::string &model_path,
                      const rclcpp::Logger &logger,
                      int num_classes = 2)
        : logger_(logger), steady_clock_(RCL_SYSTEM_TIME),
          env_(ORT_LOGGING_LEVEL_WARNING, "yolo_obb"), num_classes_(num_classes)
    {
        try {
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // CUDA EP
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);

            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

            auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
            input_name_ = input_name_ptr.get();
            auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
            output_name_ = output_name_ptr.get();

            yolo_enabled_ = true;
            RCLCPP_INFO(logger_, "[OBB YOLO] Model loaded: %s", model_path.c_str());
            RCLCPP_INFO(logger_, "[OBB YOLO] GPU backend enabled (ONNX Runtime CUDA EP)");

        } catch (const std::exception &e) {
            RCLCPP_ERROR(logger_, "[OBB YOLO] Exception while loading model: %s", e.what());
            yolo_enabled_ = false;
        }
    }

    std::vector<ObbTrackItem> track(const cv::Mat &frame) override
    {
        std::vector<ObbTrackItem> tracks;
        if (!yolo_enabled_ || frame.empty()) return tracks;

        constexpr int input_size = 640;
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                               cv::Size(input_size, input_size),
                               cv::Scalar(), true, false);

        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        size_t input_tensor_size = 1 * 3 * 640 * 640;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, (float*)blob.data, input_tensor_size, input_shape.data(), input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        static bool first_inference_logged = false;
        if (!first_inference_logged) {
            RCLCPP_INFO(logger_, "[OBB YOLO] First inference raw output:");
            RCLCPP_INFO(logger_, "[OBB YOLO]   output_shape = [%ld, %ld, %ld]",
                (long)output_shape[0], (long)output_shape[1], (long)output_shape[2]);

            int total = 1;
            for (auto d : output_shape) total *= static_cast<int>(d);
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            double sum = 0.0;
            int non_zero = 0;
            for (int i = 0; i < total; ++i) {
                float v = output_data[i];
                if (v != 0.0f) ++non_zero;
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
                sum += v;
            }
            float mean_val = total > 0 ? static_cast<float>(sum / total) : 0.0f;
            RCLCPP_INFO(logger_, "[OBB YOLO]   total=%d non_zero=%d (%.1f%%) min=%.4f max=%.4f mean=%.4f",
                total, non_zero, 100.0f * non_zero / std::max(total, 1), min_val, max_val, mean_val);
            if (total >= 20) {
                RCLCPP_INFO(logger_, "[OBB YOLO]   sample[0..19]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                    output_data[0], output_data[1], output_data[2], output_data[3], output_data[4],
                    output_data[5], output_data[6], output_data[7], output_data[8], output_data[9],
                    output_data[10], output_data[11], output_data[12], output_data[13], output_data[14],
                    output_data[15], output_data[16], output_data[17], output_data[18], output_data[19]);
            }
            first_inference_logged = true;
        }

        // 解析 YOLOv11 OBB 输出: [1, D, N] 或转置形式 [1, N, D]
        // D = channels_per_det (6 + nc per-class 模式) 或 7 (single class logit 模式)
        // const int channels_per_class = 6 + num_classes_;  // unused, 8 for nc=2
        int rows = static_cast<int>(output_shape[1]);   // D or N
        int cols = static_cast<int>(output_shape[2]);   // N or D

        // 通道顺序: [cx, cy, w, h, cls0, cls1, angle]
        // cls 已经 sigmoid 后，直接比较；angle 在最后一列
        // 如果 shape 是 [1, D, N]，需要转置为 [N, D]
        bool need_transpose = (rows <= 20 && cols > rows && rows != 1);
        cv::Mat det;
        if (need_transpose) {
            cv::Mat raw(rows, cols, CV_32F, output_data);
            cv::transpose(raw, det);  // det shape: [N, D]
        } else if (cols <= 20 && rows > cols && cols != 1) {
            // shape 已经是 [1, N, D]，直接取 [N, D]
            det = cv::Mat(rows, cols, CV_32F, output_data).clone();
        } else {
            RCLCPP_WARN_THROTTLE(logger_, steady_clock_, 2000,
                "[OBB YOLO] Unexpected output shape: [%ld,%ld,%ld], cannot determine D/N",
                (long)output_shape[0], (long)output_shape[1], (long)output_shape[2]);
            return tracks;
        }

        const float x_scale = static_cast<float>(frame.cols) / static_cast<float>(input_size);
        const float y_scale = static_cast<float>(frame.rows) / static_cast<float>(input_size);

        const int num_detections = det.rows;
        const int det_channels = det.cols;
        if (det_channels < 6) return tracks;

        std::vector<cv::Rect> boxes;          // 轴对齐外接矩形用于 NMS (cv::Rect = Rect_<int>)
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<ObbBBox> obb_boxes;      // OBB 原始信息

        for (int i = 0; i < num_detections; ++i) {
            const float *row = det.ptr<float>(i);
            if (!row) continue;

            // 通道顺序: [cx, cy, w, h, cls0, cls1, angle]
            // cls 已经是 sigmoid 后的值，直接比较即可
            const int class_offset = 4;
            const int num_class_channels = det_channels - class_offset - 1;  // 最后一列是 angle

            int best_class = -1;
            float best_score = 0.0F;
            for (int c = 0; c < num_class_channels; ++c) {
                const float score = row[class_offset + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_score < 0.15F || best_class < 0) continue;

            const float cx  = row[0] * x_scale;
            const float cy  = row[1] * y_scale;
            const float w   = row[2] * x_scale;
            const float h   = row[3] * y_scale;
            const float angle = row[det_channels - 1];  // 最后一列是 angle

            // 轴对齐外接矩形用于 NMS
            const float half_diag = std::sqrt(w * w + h * h) * 0.5F;
            cv::Rect aabbox(
                static_cast<int>(cx - half_diag),
                static_cast<int>(cy - half_diag),
                static_cast<int>(half_diag * 2.0F),
                static_cast<int>(half_diag * 2.0F));

            boxes.push_back(aabbox);
            confidences.push_back(best_score);
            class_ids.push_back(best_class);

            ObbBBox obb;
            obb.cx = cx;
            obb.cy = cy;
            obb.width = w;
            obb.height = h;
            obb.angle = angle;
            obb_boxes.push_back(obb);
        }

        // NMS（用轴对齐外接矩形）
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.15F, 0.65F, indices);
        }

        tracks.reserve(indices.size());
        for (int idx : indices) {
            ObbTrackItem item;
            item.class_id   = class_ids[idx];
            item.confidence = confidences[idx];
            item.bbox       = obb_boxes[idx];
            tracks.push_back(item);
        }

        return tracks;
    }

private:
    rclcpp::Logger logger_;
    rclcpp::Clock steady_clock_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    bool yolo_enabled_{false};
    int num_classes_{2};
};
}  // namespace

// =============================================================================
// WireAndPipeDetectionNode 实现
// =============================================================================

WireAndPipeDetectionNode::WireAndPipeDetectionNode()
    : Node("wire_andpipe_detection_node")
{
    // ---- 声明参数 ----
    // 摄像头
    this->declare_parameter<std::string>("camera_topic", "/rgb_camera_front/compressed");
    this->declare_parameter<std::string>("rgb_camera_frame", "front_camera_color_frame");
    this->declare_parameter<double>("fx", 640.0);
    this->declare_parameter<double>("fy", 640.0);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 360.0);
    this->declare_parameter<int>("camera_width", 1280);
    this->declare_parameter<int>("camera_height", 720);
    this->declare_parameter<double>("h_fov_rad", 1.0472);
    this->declare_parameter<double>("v_fov_rad", 0.6435);
    this->declare_parameter<double>("camera_x", 0.0);
    this->declare_parameter<double>("camera_y", 0.0);
    this->declare_parameter<double>("camera_z", 1.0);
    this->declare_parameter<double>("camera_pitch", 0.0);
    this->declare_parameter<double>("camera_yaw", 0.0);
    this->declare_parameter<double>("camera_roll", 0.0);

    // YOLO
    this->declare_parameter<std::string>("yolo_model_path", "");
    this->declare_parameter<int>("wire_class_id", 0);
    this->declare_parameter<int>("water_pipe_class_id", 1);
    this->declare_parameter<double>("conf_threshold", 0.5);

    // 路径
    this->declare_parameter<std::string>("global_plan_topic", "teb_global_plan");
    this->declare_parameter<std::string>("local_poses_topic", "teb_poses");
    this->declare_parameter<double>("global_search_distance", 5.0);
    this->declare_parameter<double>("local_search_distance", 5.0);

    // 避让
    this->declare_parameter<double>("pedestrian_distance_threshold", 1.0);
    this->declare_parameter<double>("avoid_hold_seconds", 3.0);

    // 物理尺寸
    this->declare_parameter<double>("wire_typical_width", 0.03);
    this->declare_parameter<double>("wire_typical_height", 0.01);
    this->declare_parameter<double>("water_pipe_typical_width", 0.04);
    this->declare_parameter<double>("water_pipe_typical_height", 0.03);

    // 跳帧
    this->declare_parameter<int>("yolo_frame_skip", 1);

    // 可视化
    this->declare_parameter<bool>("publish_annotated_image", true);

    // ---- 读取参数 ----
    fx_ = this->get_parameter("fx").as_double();
    fy_ = this->get_parameter("fy").as_double();
    cx_ = this->get_parameter("cx").as_double();
    cy_ = this->get_parameter("cy").as_double();
    camera_width_ = this->get_parameter("camera_width").as_int();
    camera_height_ = this->get_parameter("camera_height").as_int();
    h_fov_rad_ = this->get_parameter("h_fov_rad").as_double();
    v_fov_rad_ = this->get_parameter("v_fov_rad").as_double();
    camera_x_ = this->get_parameter("camera_x").as_double();
    camera_y_ = this->get_parameter("camera_y").as_double();
    camera_z_ = this->get_parameter("camera_z").as_double();
    camera_pitch_ = this->get_parameter("camera_pitch").as_double();
    camera_yaw_ = this->get_parameter("camera_yaw").as_double();
    camera_roll_ = this->get_parameter("camera_roll").as_double();

    wire_class_id_ = this->get_parameter("wire_class_id").as_int();
    water_pipe_class_id_ = this->get_parameter("water_pipe_class_id").as_int();
    conf_threshold_ = this->get_parameter("conf_threshold").as_double();

    global_search_distance_ = this->get_parameter("global_search_distance").as_double();
    local_search_distance_ = this->get_parameter("local_search_distance").as_double();
    obstacle_distance_threshold_ = this->get_parameter("pedestrian_distance_threshold").as_double();
    avoid_hold_seconds_ = this->get_parameter("avoid_hold_seconds").as_double();

    wire_typical_width_ = this->get_parameter("wire_typical_width").as_double();
    wire_typical_height_ = this->get_parameter("wire_typical_height").as_double();
    water_pipe_typical_width_ = this->get_parameter("water_pipe_typical_width").as_double();
    water_pipe_typical_height_ = this->get_parameter("water_pipe_typical_height").as_double();

    yolo_frame_skip_ = this->get_parameter("yolo_frame_skip").as_int();
    publish_annotated_image_ = this->get_parameter("publish_annotated_image").as_bool();

    // ---- 模型路径 ----
    std::string yolo_model_path = this->get_parameter("yolo_model_path").as_string();
    if (yolo_model_path.empty()) {
        yolo_model_path = ament_index_cpp::get_package_share_directory("wireandpipe_detection_cpp")
                        + "/wire_pipe_det.onnx";
    }
    yolo_ = std::make_shared<OrtObbYoloTracker>(yolo_model_path, this->get_logger(), 2);

    // ---- callback groups ----
    global_plan_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    local_poses_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    laser_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // ---- QoS ----
    auto qos_transient_local = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();

    // ---- 发布者 ----
    pub_avoiding_ = this->create_publisher<std_msgs::msg::Bool>(
        "/is_pipes_and_wires_in_path", qos_transient_local);
    pub_annotated_image_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/rgb_camera_front/annotated_image", 10);
    pub_obstacle_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/obstacles/map_markers", 10);
    pub_downsampled_path_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/downsampled_path/markers", 10);
    pub_raw_path_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/raw_path/markers", 10);
    pub_wire_pipe_3d_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/wire_pipe/markers_3d", 10);

    // ---- 订阅者 ----
    const std::string camera_topic = this->get_parameter("camera_topic").as_string();
    sub_camera_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        camera_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
            this->imageCallback(msg);
        });

    this->declare_parameter<std::string>("scan_topic_name_front", "/front_scan");
    const std::string scan_topic = this->get_parameter("scan_topic_name_front").as_string();
    auto laser_sub_options = rclcpp::SubscriptionOptions();
    laser_sub_options.callback_group = laser_callback_group_;
    sub_laser_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
            this->laserCallback(msg);
        },
        laser_sub_options);

    auto global_plan_sub_options = rclcpp::SubscriptionOptions();
    global_plan_sub_options.callback_group = global_plan_callback_group_;
    sub_global_plan_ = this->create_subscription<nav_msgs::msg::Path>(
        this->get_parameter("global_plan_topic").as_string(), 10,
        [this](const nav_msgs::msg::Path::SharedPtr msg) {
            this->globalPlanCallback(msg);
        }, global_plan_sub_options);

    auto local_poses_sub_options = rclcpp::SubscriptionOptions();
    local_poses_sub_options.callback_group = local_poses_callback_group_;
    sub_local_poses_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        this->get_parameter("local_poses_topic").as_string(), 10,
        [this](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
            this->localPosesCallback(msg);
        }, local_poses_sub_options);

    // ---- TF ----
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ---- Timer  ----
    timer_ = this->create_wall_timer(100ms, [this]() { this->timerCallback(); });

    RCLCPP_INFO(this->get_logger(), "WireAndPipeDetectionNode initialized");
    RCLCPP_INFO(this->get_logger(), "  Camera: %s", camera_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Laser: %s", scan_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Model: %s", yolo_model_path.c_str());
    RCLCPP_INFO(this->get_logger(), "  FOV: h=%.2f° v=%.2f°",
        h_fov_rad_ * 180.0 / M_PI, v_fov_rad_ * 180.0 / M_PI);
    RCLCPP_INFO(this->get_logger(), "  Distance threshold: %.2fm", obstacle_distance_threshold_);
}

// ---------------------------------------------------------------------------
// 图像解压
// ---------------------------------------------------------------------------
cv::Mat WireAndPipeDetectionNode::decodeCompressedImage(
    const sensor_msgs::msg::CompressedImage::SharedPtr &msg,
    rclcpp::Time &out_stamp)
{
    out_stamp = msg->header.stamp;
    cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (img.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[decode] cv::imdecode failed");
    }
    return img;
}

// ---------------------------------------------------------------------------
// 利用相机内参 + 物体物理尺寸估算 camera 坐标系下的 3D 坐标
// 参考 vehicle_information_tracker.py 的 estimate_distance 逻辑
// ---------------------------------------------------------------------------
geometry_msgs::msg::Point WireAndPipeDetectionNode::estimate3DFromObb(
    const ObbBBox &obb, int class_id) const
{
    // 获取典型物理尺寸
    double real_width = 0.03;   // 默认
    double real_height = 0.01;
    if (class_id == wire_class_id_) {
        real_width = wire_typical_width_;
        real_height = wire_typical_height_;
    } else if (class_id == water_pipe_class_id_) {
        real_width = water_pipe_typical_width_;
        real_height = water_pipe_typical_height_;
    }
    (void)real_width;

    // 使用 OBB 的 height来估算距离
    // distance = (real_size * img_height) / (2 * bbox_pixel_size * tan(v_fov / 2))
    const double bbox_pixel_h = static_cast<double>(obb.height);
    double distance = (real_height * static_cast<double>(camera_height_))
                      / (2.0 * bbox_pixel_h * std::tan(v_fov_rad_ / 2.0));

    // 距离下限保护
    distance = std::max(distance, 0.1);

    // 水平方向：根据 cx 计算偏离角
    const double theta_x = (0.5 - static_cast<double>(obb.cx) / static_cast<double>(camera_width_))
                           * h_fov_rad_;
    const double x_cam = distance * std::cos(theta_x);
    const double y_cam = distance * std::sin(theta_x);
    const double z_cam = 0.0;  // 地平面近似

    geometry_msgs::msg::Point pt;
    pt.x = x_cam;   // 前
    pt.y = y_cam;   // 左
    pt.z = z_cam;   // 上
    return pt;
}

// ---------------------------------------------------------------------------
// 图像回调：YOLO 推理 + 标注 + 距离估计
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::imageCallback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    static bool callback_logged_once = false;
    if (!callback_logged_once) {
        RCLCPP_INFO(this->get_logger(), "imageCallback triggered");
        callback_logged_once = true;
    }
    if (!msg || msg->data.empty()) return;

    // 跳帧
    static int frame_skip_counter = 0;
    if (yolo_frame_skip_ > 1 && ++frame_skip_counter % yolo_frame_skip_ != 0) return;

    // 解码
    rclcpp::Time img_stamp;
    cv::Mat frame = decodeCompressedImage(msg, img_stamp);
    if (frame.empty()) return;

    static bool decoded_logged_once = false;
    if (!decoded_logged_once) {
        RCLCPP_INFO(this->get_logger(), "[imageCallback] Decoded image %dx%d",
            frame.cols, frame.rows);
        decoded_logged_once = true;
    }

    // ---- 激光雷达预处理（参考 avoiding_pedestrians.cpp 逻辑） ----
    // 获取最新激光扫描
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan;
    {
        std::lock_guard<std::mutex> lock(laser_queue_mutex_);
        if (!laser_queue_.empty()) {
            latest_scan = laser_queue_.back().scan;
        }
    }

    // 获取激光雷达到相机坐标系的 TF 变换
    const std::string rgb_camera_frame = this->get_parameter("rgb_camera_frame").as_string();
    geometry_msgs::msg::TransformStamped tf_cam_laser;
    bool have_tf = false;
    if (latest_scan && !rgb_camera_frame.empty()) {
        try {
            tf_cam_laser = tf_buffer_->lookupTransform(
                rgb_camera_frame,
                latest_scan->header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            have_tf = true;
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[TF] lookupTransform failed: %s (cam='%s', laser='%s')",
                e.what(), rgb_camera_frame.c_str(), latest_scan->header.frame_id.c_str());
        }
    }

    // 将所有有效激光点批量变换到相机坐标系，排序后缓存
    cached_laser_valid_ = false;
    cached_laser_cam_pts_.clear();
    if (latest_scan && have_tf) {
        const auto &q = tf_cam_laser.transform.rotation;
        const auto &t = tf_cam_laser.transform.translation;
        const float qx = static_cast<float>(q.x);
        const float qy = static_cast<float>(q.y);
        const float qz = static_cast<float>(q.z);
        const float qw = static_cast<float>(q.w);
        const float R00 = 1.0F - 2.0F*(qy*qy + qz*qz);
        const float R01 = 2.0F*(qx*qy - qz*qw);
        const float R10 = 2.0F*(qx*qy + qz*qw);
        const float R11 = 1.0F - 2.0F*(qx*qx + qz*qz);
        const float tx = static_cast<float>(t.x);
        const float ty = static_cast<float>(t.y);
        const int n = static_cast<int>(latest_scan->ranges.size());
        cached_laser_cam_pts_.reserve(n);
        for (int i = 0; i < n; ++i) {
            const float r = latest_scan->ranges[i];
            if (!std::isfinite(r) || r < latest_scan->range_min || r > latest_scan->range_max) continue;
            const float theta_l = latest_scan->angle_min + i * latest_scan->angle_increment;
            const float xl = r * std::cos(theta_l);
            const float yl = r * std::sin(theta_l);
            const float xc = R00*xl + R01*yl + tx;
            const float yc = R10*xl + R11*yl + ty;
            const float cam_angle = std::atan2(yc, xc);
            cached_laser_cam_pts_.push_back({xc, yc, r, theta_l, cam_angle});
        }
        std::sort(cached_laser_cam_pts_.begin(), cached_laser_cam_pts_.end(),
            [](const LaserCamPt &a, const LaserCamPt &b) { return a.cam_angle < b.cam_angle; });
        cached_laser_scan_ = latest_scan;
        cached_laser_valid_ = true;
    }

    // YOLO OBB 检测
    std::vector<ObbTrackItem> tracks;
    if (!runYoloTrack(frame, tracks)) return;

    // 统计满足阈值的目标
    int obstacle_count = 0;
    for (const auto &t : tracks) {
        if (t.confidence >= static_cast<float>(conf_threshold_)) ++obstacle_count;
    }

    if (obstacle_count == 0) {
        if (!waiting_detect_log_printed_) {
            const auto now_time = this->now();
            const double elapsed = (now_time - last_waiting_log_time_).seconds();
            if (elapsed >= 10.0) {
                RCLCPP_INFO(this->get_logger(), "Waiting to detect wire/water_pipe...");
                waiting_detect_log_printed_ = true;
                last_waiting_log_time_ = now_time;
            }
        }
    } else {
        // 检测到目标，重置状态允许下次丢失时再打印一次
        waiting_detect_log_printed_ = false;
        if (!first_obstacle_detected_logged_) {
            RCLCPP_INFO(this->get_logger(), "Wire/Water pipe detected for the first time");
            first_obstacle_detected_logged_ = true;
        }
    }

    // 标注图像
    cv::Mat annotated = frame.clone();
    DetectionResult result;
    result.detected = false;
    result.stamp = msg->header.stamp;
    result.camera_frame_id = this->get_parameter("rgb_camera_frame").as_string();

    for (const auto &det : tracks) {
        if (det.confidence < static_cast<float>(conf_threshold_)) continue;

        // ---- 两种距离估算 ----
        // 1) 相机内参 + 物理尺寸
        geometry_msgs::msg::Point pt_cam = estimate3DFromObb(det.bbox, det.class_id);
        const float dist_pinhole = std::hypot(pt_cam.x, pt_cam.y);

        // 2) 激光雷达测距（AABB 水平角度范围 + 最近激光点）
        const float dist_laser = estimateDistanceFromLaser(det.bbox, frame.cols);

        // 打印两种距离对比（2 秒一次，防刷屏）
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[distance] pinhole=%.2fm  laser=%.2fm",
            dist_pinhole,
            std::isfinite(dist_laser) ? dist_laser : -1.0f);

        result.obstacles_camera.push_back(pt_cam);
        result.detected = true;

        // 绘制 OBB（简化：画轴对齐矩形 + 角度标注）
        const int x = std::max(0, std::min(static_cast<int>(det.bbox.cx - det.bbox.width * 0.5F), annotated.cols - 1));
        const int y = std::max(0, std::min(static_cast<int>(det.bbox.cy - det.bbox.height * 0.5F), annotated.rows - 1));
        const int w = std::max(1, std::min(static_cast<int>(det.bbox.width), annotated.cols - x));
        const int h = std::max(1, std::min(static_cast<int>(det.bbox.height), annotated.rows - y));

        cv::Scalar color = (det.class_id == wire_class_id_)
            ? cv::Scalar(255, 0, 0)    // 电线 → 蓝色
            : cv::Scalar(0, 255, 0);   // 水管 → 绿色
        cv::rectangle(annotated, cv::Rect(x, y, w, h), color, 2);

        // 类别名
        const std::string cls_name = (det.class_id == wire_class_id_) ? "wire" : "water_pipe";

        std::ostringstream line1;
        line1 << cls_name << " conf:" << std::fixed << std::setprecision(2) << det.confidence;

        std::ostringstream line2;
        line2 << "dist:" << std::fixed << std::setprecision(2)
              << std::hypot(pt_cam.x, pt_cam.y) << "m";

        std::ostringstream line3;
        line3 << "angle:" << std::fixed << std::setprecision(2)
              << det.bbox.angle * 180.0 / M_PI << "deg";

        const double font_scale = 0.6;
        const int thickness = 2;
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const int line_gap = 20;
        const int text_y = std::max(20, y - 5);

        cv::putText(annotated, line1.str(), cv::Point(x, text_y),
                    font, font_scale, cv::Scalar(0, 255, 0), thickness);
        cv::putText(annotated, line2.str(), cv::Point(x, text_y + line_gap),
                    font, font_scale, cv::Scalar(0, 255, 255), thickness);
        cv::putText(annotated, line3.str(), cv::Point(x, text_y + line_gap * 2),
                    font, font_scale, cv::Scalar(255, 165, 0), thickness);
    }

    // 发布水管/电线 3D LINE_LIST 可视化（在 camera 坐标系）
    if (pub_wire_pipe_3d_) {
        visualization_msgs::msg::MarkerArray wp_markers;
        int wp_id = 0;

        // DELETEALL per namespace
        visualization_msgs::msg::Marker del_wire;
        del_wire.header.frame_id = result.camera_frame_id;
        del_wire.header.stamp = msg->header.stamp;
        del_wire.ns = "wire_3d";
        del_wire.id = wp_id++;
        del_wire.action = visualization_msgs::msg::Marker::DELETEALL;
        wp_markers.markers.push_back(del_wire);

        visualization_msgs::msg::Marker del_pipe;
        del_pipe.header.frame_id = result.camera_frame_id;
        del_pipe.header.stamp = msg->header.stamp;
        del_pipe.ns = "water_pipe_3d";
        del_pipe.id = wp_id++;
        del_pipe.action = visualization_msgs::msg::Marker::DELETEALL;
        wp_markers.markers.push_back(del_pipe);

        // 收集端点，按类别分开
        std::vector<geometry_msgs::msg::Point> wire_pts, pipe_pts;

        for (const auto &det : tracks) {
            if (det.confidence < static_cast<float>(conf_threshold_)) continue;

            const geometry_msgs::msg::Point pt_cam = estimate3DFromObb(det.bbox, det.class_id);
            const double distance = std::hypot(pt_cam.x, pt_cam.y);
            (void)distance;
            const double angle = static_cast<double>(det.bbox.angle);

            // 方向向量：在相机坐标系 YZ 平面内
            const double dir_y = -std::cos(angle);
            const double dir_z = -std::sin(angle);

            // 物理半长 ~ 0.3m，仅用于可视化方向
            constexpr double viz_half_len = 0.30;

            geometry_msgs::msg::Point p1, p2;
            p1.x = pt_cam.x; p1.y = pt_cam.y + dir_y * viz_half_len; p1.z = pt_cam.z + dir_z * viz_half_len;
            p2.x = pt_cam.x; p2.y = pt_cam.y - dir_y * viz_half_len; p2.z = pt_cam.z - dir_z * viz_half_len;

            if (det.class_id == wire_class_id_) {
                wire_pts.push_back(p1);
                wire_pts.push_back(p2);
            } else {
                pipe_pts.push_back(p1);
                pipe_pts.push_back(p2);
            }
        }

        // 发布电线 LINE_LIST（蓝色）
        if (!wire_pts.empty()) {
            visualization_msgs::msg::Marker wire_m;
            wire_m.header.frame_id = result.camera_frame_id;
            wire_m.header.stamp = msg->header.stamp;
            wire_m.ns = "wire_3d";
            wire_m.id = wp_id++;
            wire_m.type = visualization_msgs::msg::Marker::LINE_LIST;
            wire_m.action = visualization_msgs::msg::Marker::ADD;
            wire_m.scale.x = 0.03;
            wire_m.color.r = 0.0F; wire_m.color.g = 0.4F; wire_m.color.b = 1.0F; wire_m.color.a = 0.9F;
            wire_m.points = std::move(wire_pts);
            wp_markers.markers.push_back(wire_m);
        }

        // 发布水管 LINE_LIST（绿色）
        if (!pipe_pts.empty()) {
            visualization_msgs::msg::Marker pipe_m;
            pipe_m.header.frame_id = result.camera_frame_id;
            pipe_m.header.stamp = msg->header.stamp;
            pipe_m.ns = "water_pipe_3d";
            pipe_m.id = wp_id++;
            pipe_m.type = visualization_msgs::msg::Marker::LINE_LIST;
            pipe_m.action = visualization_msgs::msg::Marker::ADD;
            pipe_m.scale.x = 0.04;
            pipe_m.color.r = 0.0F; pipe_m.color.g = 0.9F; pipe_m.color.b = 0.0F; pipe_m.color.a = 0.9F;
            pipe_m.points = std::move(pipe_pts);
            wp_markers.markers.push_back(pipe_m);
        }

        pub_wire_pipe_3d_->publish(wp_markers);
    }

    // 发布标注图像（每帧都发布，确保 rqt 持续显示）
    if (pub_annotated_image_ && publish_annotated_image_) {
        auto annotated_msg = sensor_msgs::msg::Image();
        annotated_msg.header = msg->header;
        annotated_msg.height = static_cast<uint32_t>(annotated.rows);
        annotated_msg.width = static_cast<uint32_t>(annotated.cols);
        annotated_msg.encoding = sensor_msgs::image_encodings::BGR8;
        annotated_msg.is_bigendian = 0;
        annotated_msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(
            annotated.cols * annotated.elemSize());
        const size_t image_bytes = static_cast<size_t>(annotated_msg.step)
                                   * static_cast<size_t>(annotated.rows);
        annotated_msg.data.assign(annotated.data, annotated.data + image_bytes);
        pub_annotated_image_->publish(annotated_msg);
    }

    {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        detection_result_ = std::move(result);
    }
}

bool WireAndPipeDetectionNode::runYoloTrack(const cv::Mat &frame,
                                            std::vector<ObbTrackItem> &tracks)
{
    if (!yolo_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "YOLO tracker is not initialized.");
        return false;
    }
    try {
        tracks = yolo_->track(frame);
        return true;
    } catch (const std::exception &e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "YOLO inference failed: %s", e.what());
        return false;
    } catch (...) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "YOLO inference failed with unknown exception.");
        return false;
    }
}

// ---------------------------------------------------------------------------
// 路径回调
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::globalPlanCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
    if (!msg) return;
    last_global_plan_ = msg;
    global_plan_dirty_ = true;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "globalPlanCallback: received path with %zu poses",
            msg->poses.size());
        once = false;
    }

    // 发布原始全局路径 LINE_STRIP 可视化
    if (pub_raw_path_markers_ && !msg->poses.empty()) {
        visualization_msgs::msg::MarkerArray raw_markers;
        int mid = 0;

        // DELETEALL
        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp = msg->header.stamp;
        del.ns = "raw_global_path";
        del.id = mid++;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        raw_markers.markers.push_back(del);

        // LINE_STRIP
        visualization_msgs::msg::Marker line;
        line.header.frame_id = "map";
        line.header.stamp = msg->header.stamp;
        line.ns = "raw_global_path";
        line.id = mid++;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.05;  // 线宽
        line.color.r = 0.2F; line.color.g = 0.6F; line.color.b = 1.0F; line.color.a = 0.7F;
        line.points.reserve(msg->poses.size());
        for (const auto& ps : msg->poses) {
            line.points.push_back(ps.pose.position);
        }
        raw_markers.markers.push_back(line);

        pub_raw_path_markers_->publish(raw_markers);
    }
}

void WireAndPipeDetectionNode::localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    if (!msg) return;
    last_local_poses_ = msg;
    local_poses_dirty_ = true;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "localPosesCallback: received %zu poses",
            msg->poses.size());
        once = false;
    }

    // 发布原始局部路径 LINE_STRIP 可视化
    if (pub_raw_path_markers_ && !msg->poses.empty()) {
        visualization_msgs::msg::MarkerArray raw_markers;
        int mid = 0;

        visualization_msgs::msg::Marker del;
        del.header.frame_id = msg->header.frame_id;
        del.header.stamp = msg->header.stamp;
        del.ns = "raw_local_path";
        del.id = mid++;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        raw_markers.markers.push_back(del);

        visualization_msgs::msg::Marker line;
        line.header.frame_id = msg->header.frame_id;
        line.header.stamp = msg->header.stamp;
        line.ns = "raw_local_path";
        line.id = mid++;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.03;  // 线宽
        line.color.r = 1.0F; line.color.g = 0.65F; line.color.b = 0.0F; line.color.a = 0.7F;  // 橙色
        line.points.reserve(msg->poses.size());
        for (const auto& ps : msg->poses) {
            line.points.push_back(ps.position);
        }
        raw_markers.markers.push_back(line);

        pub_raw_path_markers_->publish(raw_markers);
    }
}

// ---------------------------------------------------------------------------
// Timer 回调：路径障碍物检查 + 避让信号发布
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::timerCallback()
{
    bool trigger_now = false;
    double timer_global_hit_dist = 0.0;
    double timer_local_hit_dist = 0.0;
    bool timer_on_global = false;
    bool timer_on_local = false;
    const auto now_stamp = this->now();

    if (last_global_plan_ || last_local_poses_) {
        const double threshold = obstacle_distance_threshold_;
        const bool has_global_path = last_global_plan_ && !last_global_plan_->poses.empty();
        const bool has_local_path = last_local_poses_ && !last_local_poses_->poses.empty();

        // 下采样路径
        if (global_plan_dirty_ && has_global_path) {
            std::vector<geometry_msgs::msg::Point> global_points;
            global_points.reserve(last_global_plan_->poses.size());
            for (const auto &pose_stamped : last_global_plan_->poses) {
                global_points.push_back(pose_stamped.pose.position);
            }
            cached_ds_global_ = downsamplePath(global_points, threshold, global_search_distance_);
            global_plan_dirty_ = false;
        }
        if (local_poses_dirty_ && has_local_path) {
            std::vector<geometry_msgs::msg::Point> local_points;
            local_points.reserve(last_local_poses_->poses.size());
            for (const auto &pose : last_local_poses_->poses) {
                local_points.push_back(pose.position);
            }
            cached_ds_local_ = downsamplePath(local_points, threshold, local_search_distance_);
            local_poses_dirty_ = false;
        }

        const auto &ds_global = cached_ds_global_;
        const auto &ds_local = cached_ds_local_;

        // 可视化下采样路径点
        if (pub_downsampled_path_markers_) {
            visualization_msgs::msg::MarkerArray ds_marker_array;
            int marker_id = 0;

            visualization_msgs::msg::Marker delete_all;
            delete_all.header.frame_id = "map";
            delete_all.header.stamp = now_stamp;
            delete_all.ns = "downsampled_path";
            delete_all.id = marker_id++;
            delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
            ds_marker_array.markers.push_back(delete_all);

            // 全局路径点（蓝色）
            for (const auto &pt : ds_global) {
                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp = now_stamp;
                m.ns = "downsampled_path";
                m.id = marker_id++;
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose.position = pt;
                m.pose.position.z = 0.05;
                m.pose.orientation.w = 1.0;
                m.scale.x = 0.12; m.scale.y = 0.12; m.scale.z = 0.12;
                m.color.r = 0.0F; m.color.g = 0.0F; m.color.b = 1.0F; m.color.a = 0.8F;
                ds_marker_array.markers.push_back(m);
            }
            // 局部路径点（黄色）
            for (const auto &pt : ds_local) {
                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp = now_stamp;
                m.ns = "downsampled_path";
                m.id = marker_id++;
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose.position = pt;
                m.pose.position.z = 0.05;
                m.pose.orientation.w = 1.0;
                m.scale.x = 0.12; m.scale.y = 0.12; m.scale.z = 0.12;
                m.color.r = 1.0F; m.color.g = 1.0F; m.color.b = 0.0F; m.color.a = 0.8F;
                ds_marker_array.markers.push_back(m);
            }
            pub_downsampled_path_markers_->publish(ds_marker_array);
        }

        // 获取检测结果
        DetectionResult det;
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            det = detection_result_;
        }

        if (det.detected && !det.obstacles_camera.empty()) {
            // 将 camera 坐标系下的障碍物转换到 map
            const std::string rgb_camera_frame = det.camera_frame_id;
            std::vector<geometry_msgs::msg::PointStamped> obstacles_map;
            obstacles_map.reserve(det.obstacles_camera.size());

            for (const auto &pt_cam : det.obstacles_camera) {
                geometry_msgs::msg::PointStamped p_in;
                p_in.header.frame_id = rgb_camera_frame;
                p_in.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
                p_in.point = pt_cam;
                try {
                    geometry_msgs::msg::PointStamped p_map;
                    tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.2));
                    obstacles_map.push_back(p_map);
                } catch (const tf2::TransformException &e) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                        "[TF] camera->map transform failed: %s", e.what());
                }
            }

            if (!obstacles_map.empty()) {
                // 发布可视化
                visualization_msgs::msg::MarkerArray marker_array;
                marker_array.markers.reserve(obstacles_map.size() + 1);

                visualization_msgs::msg::Marker delete_all;
                delete_all.header.frame_id = "map";
                delete_all.header.stamp = now_stamp;
                delete_all.ns = "obstacles";
                delete_all.id = 0;
                delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
                marker_array.markers.push_back(delete_all);

                for (size_t i = 0; i < obstacles_map.size(); ++i) {
                    const auto &p = obstacles_map[i];
                    visualization_msgs::msg::Marker marker;
                    marker.header.frame_id = "map";
                    marker.header.stamp = now_stamp;
                    marker.ns = "obstacles";
                    marker.id = static_cast<int>(i) + 1;
                    marker.type = visualization_msgs::msg::Marker::SPHERE;
                    marker.action = visualization_msgs::msg::Marker::ADD;
                    marker.pose.position = p.point;
                    marker.pose.orientation.w = 1.0;
                    marker.scale.x = 0.25; marker.scale.y = 0.25; marker.scale.z = 0.25;
                    marker.color.r = 1.0F; marker.color.g = 0.3F; marker.color.b = 0.0F;
                    marker.color.a = 0.9F;
                    marker_array.markers.push_back(marker);
                }
                if (pub_obstacle_markers_) pub_obstacle_markers_->publish(marker_array);

                // 判断是否在路径上
                timer_on_global = has_global_path && !ds_global.empty()
                    ? checkObstacleOnGlobalPath(obstacles_map, ds_global, threshold, &timer_global_hit_dist)
                    : false;
                timer_on_local = has_local_path && !ds_local.empty()
                    ? checkObstacleOnLocalPath(obstacles_map, ds_local, threshold, &timer_local_hit_dist)
                    : false;

                trigger_now = (timer_on_global || timer_on_local);
            }
        }
    }

    if (trigger_now && !last_trigger_state_) {
        const double report_dist = timer_on_global ? timer_global_hit_dist : timer_local_hit_dist;
        RCLCPP_WARN(this->get_logger(),
            "[timer] Obstacle detected on path, distance to path: %.2fm", report_dist);
    } else if (!trigger_now && last_trigger_state_) {
        RCLCPP_INFO(this->get_logger(),
            "[timer] Path clear, obstacle no longer in range");
    }
    last_trigger_state_ = trigger_now;

    // 避让状态管理
    if (trigger_now) {
        has_recent_detection_ = true;
        last_obstacle_detect_time_ = now_stamp;
    }

    bool is_avoiding = false;
    if (has_recent_detection_) {
        const double dt = (now_stamp - last_obstacle_detect_time_).seconds();
        if (dt < avoid_hold_seconds_) {
            is_avoiding = true;
        } else {
            has_recent_detection_ = false;
            is_avoiding = false;
        }
    }

    // 状态变化时发布
    if (is_avoiding && !last_published_avoiding_) {
        auto out = std_msgs::msg::Bool();
        out.data = true;
        pub_avoiding_->publish(out);
        ++warning_event_id_;
        RCLCPP_WARN(this->get_logger(), "Wire/Pipe obstacle warning, id: %llu",
            static_cast<unsigned long long>(warning_event_id_));
        last_published_avoiding_ = true;
    } else if (!is_avoiding && last_published_avoiding_) {
        auto out = std_msgs::msg::Bool();
        out.data = false;
        pub_avoiding_->publish(out);
        RCLCPP_INFO(this->get_logger(), "Wire/Pipe obstacle warning cleared, id: %llu",
            static_cast<unsigned long long>(warning_event_id_));
        last_published_avoiding_ = false;
    }
}

// ---------------------------------------------------------------------------
// 路径下采样
// ---------------------------------------------------------------------------
std::vector<geometry_msgs::msg::Point> WireAndPipeDetectionNode::downsamplePath(
    const std::vector<geometry_msgs::msg::Point> &nav_points,
    double min_distance,
    double lookahead_distance)
{
    std::vector<geometry_msgs::msg::Point> out_points;
    if (nav_points.empty()) return out_points;

    if (nav_points.size() >= 2) {
        const double end_dx = nav_points.back().x - nav_points.front().x;
        const double end_dy = nav_points.back().y - nav_points.front().y;
        const double end_dist = std::sqrt(end_dx * end_dx + end_dy * end_dy);
        if (end_dist < lookahead_distance) return nav_points;
    }

    out_points.push_back(nav_points.front());
    if (nav_points.size() == 1) return out_points;

    const double sample_dist_sq = min_distance * min_distance;
    size_t last_keep_idx = 0;
    double accumulated_len = 0.0;

    for (size_t i = 1; i < nav_points.size(); ++i) {
        const double seg_dx = nav_points[i].x - nav_points[i - 1].x;
        const double seg_dy = nav_points[i].y - nav_points[i - 1].y;
        const double seg_len = std::sqrt(seg_dx * seg_dx + seg_dy * seg_dy);
        if (accumulated_len + seg_len > lookahead_distance) break;
        accumulated_len += seg_len;

        const double dx = nav_points[i].x - nav_points[last_keep_idx].x;
        const double dy = nav_points[i].y - nav_points[last_keep_idx].y;
        const double dist_sq = dx * dx + dy * dy;
        if (dist_sq >= sample_dist_sq) {
            out_points.push_back(nav_points[i]);
            last_keep_idx = i;
        }
    }
    return out_points;
}

// ---------------------------------------------------------------------------
// 障碍物-路径碰撞检查
// ---------------------------------------------------------------------------
bool WireAndPipeDetectionNode::checkObstacleOnGlobalPath(
    const std::vector<geometry_msgs::msg::PointStamped> &obstacles,
    const std::vector<geometry_msgs::msg::Point> &downsampled_points,
    double threshold,
    double *hit_distance)
{
    if (obstacles.empty() || downsampled_points.empty()) return false;

    const double threshold_sq = threshold * threshold;
    double min_dist_global = std::numeric_limits<double>::max();
    for (const auto &obs : obstacles) {
        for (const auto &pt : downsampled_points) {
            const double dx = obs.point.x - pt.x;
            const double dy = obs.point.y - pt.y;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_global) min_dist_global = dist_sq;
            if (dist_sq <= threshold_sq) {
                if (hit_distance) *hit_distance = std::sqrt(dist_sq);
                if (!last_global_match_) {
                    RCLCPP_INFO(this->get_logger(), "[global_path] obstacle on global path");
                    last_global_match_ = true;
                }
                return true;
            }
        }
    }
    if (last_global_match_) {
        RCLCPP_INFO(this->get_logger(),
            "[global_path] no match: min_dist=%.2fm threshold=%.2fm",
            std::sqrt(min_dist_global), threshold);
        last_global_match_ = false;
    }
    return false;
}

bool WireAndPipeDetectionNode::checkObstacleOnLocalPath(
    const std::vector<geometry_msgs::msg::PointStamped> &obstacles,
    const std::vector<geometry_msgs::msg::Point> &downsampled_points,
    double threshold,
    double *hit_distance)
{
    if (obstacles.empty() || downsampled_points.empty()) return false;

    const double threshold_sq = threshold * threshold;
    double min_dist_local = std::numeric_limits<double>::max();
    for (const auto &obs : obstacles) {
        for (const auto &pt : downsampled_points) {
            const double dx = obs.point.x - pt.x;
            const double dy = obs.point.y - pt.y;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_local) min_dist_local = dist_sq;
            if (dist_sq <= threshold_sq) {
                if (hit_distance) *hit_distance = std::sqrt(dist_sq);
                if (!last_local_match_) {
                    RCLCPP_INFO(this->get_logger(), "[local_path] obstacle on local path");
                    last_local_match_ = true;
                }
                return true;
            }
        }
    }
    if (last_local_match_) {
        RCLCPP_INFO(this->get_logger(),
            "[local_path] no match: min_dist=%.2fm threshold=%.2fm",
            std::sqrt(min_dist_local), threshold);
        last_local_match_ = false;
    }
    return false;
}

// =========================================================================
// [METHOD 2] 激光雷达测距（AABB 水平角度范围 + 最近激光点）
// 利用图像内已知的 h_fov，将 OBB 的 x 范围映射为水平角度范围，
// 在 cached_laser_cam_pts_ 中二分查找落入该角度范围的激光点，取最近距离。
// 返回距离（米），失败返回 NaN
// =========================================================================
float WireAndPipeDetectionNode::estimateDistanceFromLaser(
    const ObbBBox &obb, int img_width) const
{
    if (!cached_laser_valid_ || cached_laser_cam_pts_.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // AABB: x_min = cx - w/2,  x_max = cx + w/2
    // 像素 x → 水平角度 θ = (0.5 - x / img_w) * h_fov
    const float theta_min = (0.5F - (obb.cx + obb.width * 0.5F) / static_cast<float>(img_width))
                            * static_cast<float>(h_fov_rad_);
    const float theta_max = (0.5F - (obb.cx - obb.width * 0.5F) / static_cast<float>(img_width))
                            * static_cast<float>(h_fov_rad_);

    // 确保 theta_min <= theta_max（因为 x 越大 → θ 越小）
    const float lo = std::min(theta_min, theta_max);
    const float hi = std::max(theta_min, theta_max);

    // 二分搜索
    auto it_lo = std::lower_bound(cached_laser_cam_pts_.begin(), cached_laser_cam_pts_.end(), lo,
        [](const LaserCamPt &lp, float val) { return lp.cam_angle < val; });
    auto it_hi = std::upper_bound(it_lo, cached_laser_cam_pts_.end(), hi,
        [](float val, const LaserCamPt &lp) { return val < lp.cam_angle; });

    float best_dist = std::numeric_limits<float>::max();
    for (auto it = it_lo; it != it_hi; ++it) {
        const float dist = std::hypot(it->xc, it->yc);
        if (dist > 0.15F && dist < best_dist) {
            best_dist = dist;
        }
    }

    if (best_dist >= std::numeric_limits<float>::max()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return best_dist;
}

// ---------------------------------------------------------------------------
// 激光雷达回调
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::laserCallback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (!msg) return;

    const double ts = static_cast<double>(msg->header.stamp.sec)
                    + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    std::lock_guard<std::mutex> lock(laser_queue_mutex_);
    laser_queue_.emplace_back(LaserQueueItem{ts, msg});

    while (laser_queue_.size() > laser_queue_max_size_) {
        laser_queue_.pop_front();
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WireAndPipeDetectionNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
