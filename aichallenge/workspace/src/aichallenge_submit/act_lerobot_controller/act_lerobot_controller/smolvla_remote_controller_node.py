#!/usr/bin/env python3
import base64
import json
import time
import urllib.error
import urllib.request
from collections import deque

import cv2
import numpy as np
import rclpy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class SmolVLARemoteControllerNode(Node):
    def __init__(self):
        super().__init__("smolvla_remote_controller_node")

        self.declare_parameter("server_url", "http://127.0.0.1:8765/predict")
        self.declare_parameter("task", "drive the racing kart around the course")
        self.declare_parameter("timeout_sec", 2.0)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("image_height", 200)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("crop_top_ratio", 0.375)
        self.declare_parameter("crop_bottom_ratio", 0.0)
        self.declare_parameter("n_action_steps", 20)
        self.declare_parameter("state_mode", "vehicle_status")
        self.declare_parameter("control_mode", "ai")
        self.declare_parameter("acceleration", 0.6)
        self.declare_parameter("debug", False)
        self.declare_parameter("log_interval_sec", 5.0)

        self.server_url = self.get_parameter("server_url").value
        self.task = self.get_parameter("task").value
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.crop_top_ratio = float(self.get_parameter("crop_top_ratio").value)
        self.crop_bottom_ratio = float(self.get_parameter("crop_bottom_ratio").value)
        self.n_action_steps = int(self.get_parameter("n_action_steps").value)
        self.state_mode = self.get_parameter("state_mode").value
        self.control_mode = self.get_parameter("control_mode").value
        self.acceleration = float(self.get_parameter("acceleration").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval = float(self.get_parameter("log_interval_sec").value)

        if self.crop_top_ratio + self.crop_bottom_ratio >= 1.0:
            raise ValueError("crop_top_ratio + crop_bottom_ratio must be < 1.0")

        self.latest_odom = None
        self.latest_velocity = None
        self.action_queue = deque(maxlen=max(1, self.n_action_steps))
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, "/image_raw", self.image_callback, qos)
        self.create_subscription(Odometry, "/localization/kinematic_state", self.odom_callback, 1)
        self.create_subscription(VelocityReport, "/vehicle/status/velocity_status", self.velocity_callback, 1)
        self.pub_control = self.create_publisher(AckermannControlCommand, "/control/command/control_cmd", 1)

        self.get_logger().info(f"SmolVLARemoteControllerNode is ready. server_url={self.server_url}")

    def odom_callback(self, msg):
        self.latest_odom = msg

    def velocity_callback(self, msg):
        self.latest_velocity = msg

    def image_callback(self, msg):
        start_time = time.monotonic()
        image = self._image_msg_to_numpy(msg)
        if image is None:
            return

        try:
            if not self.action_queue:
                chunk = self._request_action_chunk(image, self._current_state())
                for action in chunk[: self.n_action_steps]:
                    self.action_queue.append(action)
            accel, steer = self.action_queue.popleft()
        except Exception as exc:
            self.get_logger().error(f"SmolVLA remote inference failed: {exc}", throttle_duration_sec=5.0)
            return

        accel = float(np.clip(accel, -1.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))
        if self.control_mode != "ai":
            accel = self.acceleration

        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = accel
        cmd.lateral.steering_tire_angle = steer
        self.pub_control.publish(cmd)

        if self.debug:
            self.inference_times.append((time.monotonic() - start_time) * 1000.0)
            self._log_performance_metrics()

    def _request_action_chunk(self, image, state):
        image = self._preprocess_image(image)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")

        payload = {
            "image_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "state": None if state is None else state.tolist(),
            "task": self.task,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.server_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach SmolVLA server at {self.server_url}: {exc}") from exc

        if "error" in result:
            raise RuntimeError(result["error"])
        chunk = np.asarray(result.get("action_chunk", result.get("action")), dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.shape[1] < 2:
            raise RuntimeError(f"invalid action shape from server: {chunk.shape}")
        return chunk[:, :2]

    def _current_state(self):
        if self.state_mode == "odometry" and self.latest_odom is not None:
            twist = self.latest_odom.twist.twist
            return np.array([twist.linear.x, twist.linear.y, twist.angular.z], dtype=np.float32)
        if self.state_mode == "vehicle_status" and self.latest_velocity is not None:
            msg = self.latest_velocity
            return np.array([msg.longitudinal_velocity, msg.heading_rate], dtype=np.float32)
        return None

    def _image_msg_to_numpy(self, msg):
        try:
            encoding = msg.encoding.lower()
            if encoding == "bgr8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if encoding == "rgb8":
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
            if encoding == "bgra8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            if encoding == "rgba8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}", throttle_duration_sec=5.0)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}", throttle_duration_sec=5.0)
        return None

    def _preprocess_image(self, image):
        if self.crop_top_ratio > 0 or self.crop_bottom_ratio > 0:
            h = image.shape[0]
            top = int(h * self.crop_top_ratio)
            bottom = h - int(h * self.crop_bottom_ratio)
            image = image[top:bottom, :, :]
        return cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)

    def _log_performance_metrics(self):
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds / 1e9 <= self.log_interval:
            return
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0.0
            self.get_logger().info(f"DEBUG: Avg remote inference: {avg_time:.2f}ms ({fps:.2f}Hz)")
            self.inference_times.clear()
        self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLARemoteControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
