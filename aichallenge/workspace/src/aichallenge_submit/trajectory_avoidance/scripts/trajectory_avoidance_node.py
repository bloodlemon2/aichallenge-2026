#!/usr/bin/env python3
"""Predict moving karts from /aichallenge/objects and laterally avoid them."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from autoware_auto_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


@dataclass
class TrackedObject:
    x: float
    y: float
    z: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class LaneletPolygon:
    polygon: List[Tuple[float, float]]
    boundaries: List[List[Tuple[float, float]]]


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if abs(denom) <= 1.0e-9:
                j = i
                continue
            x_cross = (xj - xi) * (y - yi) / denom + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_segment_distance(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1.0e-12:
        return distance(point, a)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    return distance(point, (ax + t * vx, ay + t * vy))


def polyline_distance(point: Tuple[float, float], polyline: Sequence[Tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return float("inf")
    return min(point_segment_distance(point, polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1))


class TrajectoryAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_avoidance")
        self.detection_distance = self.declare_parameter("detection_distance", 30.0).value
        self.collision_horizon = self.declare_parameter("collision_horizon", 4.0).value
        self.safety_margin = self.declare_parameter("safety_margin", 0.35).value
        self.avoidance_offset = self.declare_parameter("avoidance_offset", 1.0).value
        self.shift_start_distance = self.declare_parameter("shift_start_distance", 5.0).value
        self.shift_end_distance = self.declare_parameter("shift_end_distance", 7.0).value
        self.assumed_ego_speed = self.declare_parameter("assumed_ego_speed", 4.17).value
        self.max_object_speed = self.declare_parameter("max_object_speed", 12.0).value
        self.object_match_distance = self.declare_parameter("object_match_distance", 2.0).value
        self.object_timeout = self.declare_parameter("object_timeout", 0.5).value
        self.lanelet_map_path = self.declare_parameter("lanelet_map_path", "").value
        self.wall_margin = self.declare_parameter("wall_margin", 0.45).value
        self.clearance_tie_epsilon = self.declare_parameter("clearance_tie_epsilon", 0.05).value
        self.trajectory: Optional[Trajectory] = None
        self.odometry: Optional[Odometry] = None
        self.objects: List[TrackedObject] = []
        self.objects_stamp_ns: Optional[int] = None
        self.lanelets = self.load_lanelet_map(self.lanelet_map_path)
        trajectory_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Trajectory, "input/trajectory", self.on_trajectory, trajectory_qos)
        self.create_subscription(Odometry, "input/odometry", self.on_odometry, 10)
        self.create_subscription(Float64MultiArray, "input/objects", self.on_objects, 10)
        # Downstream controllers use the default RELIABLE QoS.  Keep this independent
        # from the BEST_EFFORT QoS required by simple_trajectory_generator above.
        self.publisher = self.create_publisher(Trajectory, "output/trajectory", 1)

    def load_lanelet_map(self, path: str) -> List[LaneletPolygon]:
        if not path:
            self.get_logger().warning("lanelet_map_path is empty; wall checks are disabled")
            return []
        if not os.path.exists(path):
            self.get_logger().warning(f"Lanelet map not found: {path}; wall checks are disabled")
            return []
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as err:
            self.get_logger().warning(f"Failed to parse lanelet map: {err}; wall checks are disabled")
            return []

        nodes = {}
        for node in root.findall("node"):
            tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in node.findall("tag")}
            if "local_x" in tags and "local_y" in tags:
                nodes[node.attrib["id"]] = (float(tags["local_x"]), float(tags["local_y"]))

        ways = {}
        for way in root.findall("way"):
            points = [nodes[nd.attrib["ref"]] for nd in way.findall("nd") if nd.attrib["ref"] in nodes]
            if len(points) >= 2:
                ways[way.attrib["id"]] = points

        lanelets: List[LaneletPolygon] = []
        for relation in root.findall("relation"):
            tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in relation.findall("tag")}
            if tags.get("type") != "lanelet" or tags.get("subtype") != "road":
                continue
            left, right = None, None
            for member in relation.findall("member"):
                if member.attrib.get("type") != "way":
                    continue
                if member.attrib.get("role") == "left":
                    left = ways.get(member.attrib["ref"])
                elif member.attrib.get("role") == "right":
                    right = ways.get(member.attrib["ref"])
            if left and right:
                lanelets.append(LaneletPolygon(left + list(reversed(right)), [left, right]))

        self.get_logger().info(f"Loaded {len(lanelets)} road lanelets for wall checks")
        return lanelets

    def on_trajectory(self, message: Trajectory) -> None:
        self.trajectory = message
        self.publish_trajectory()

    def on_odometry(self, message: Odometry) -> None:
        self.odometry = message

    def on_objects(self, message: Float64MultiArray) -> None:
        if len(message.data) % 4:
            self.get_logger().warning("input/objects must contain x, y, z, radius tuples")
            return
        now_ns = self.get_clock().now().nanoseconds
        dt = None if self.objects_stamp_ns is None else (now_ns - self.objects_stamp_ns) / 1.0e9
        previous, used, current = self.objects, [False] * len(self.objects), []
        for i in range(0, len(message.data), 4):
            obj = TrackedObject(*message.data[i:i + 4])
            closest, closest_distance = None, float(self.object_match_distance)
            for j, old in enumerate(previous):
                if not used[j]:
                    d = distance((obj.x, obj.y), (old.x, old.y))
                    if d < closest_distance:
                        closest, closest_distance = j, d
            if closest is not None and dt is not None and dt > 1.0e-4:
                old = previous[closest]
                obj.vx, obj.vy = (obj.x - old.x) / dt, (obj.y - old.y) / dt
                if math.hypot(obj.vx, obj.vy) > self.max_object_speed:
                    obj.vx, obj.vy = 0.0, 0.0
                used[closest] = True
            obj.radius = max(0.05, obj.radius)
            current.append(obj)
        self.objects, self.objects_stamp_ns = current, now_ns
        self.publish_trajectory()

    def is_ahead(self, obj: TrackedObject) -> bool:
        assert self.odometry is not None
        pose = self.odometry.pose.pose
        heading = yaw(pose.orientation)
        dx, dy = obj.x - pose.position.x, obj.y - pose.position.y
        return dx * math.cos(heading) + dy * math.sin(heading) > 0.0 and math.hypot(dx, dy) < self.detection_distance

    def nearest_index(self, trajectory: Trajectory) -> int:
        assert self.odometry is not None
        ego = self.odometry.pose.pose.position
        return min(range(len(trajectory.points)), key=lambda i: distance(
            (trajectory.points[i].pose.position.x, trajectory.points[i].pose.position.y), (ego.x, ego.y)))

    @staticmethod
    def arc_lengths(trajectory: Trajectory, begin: int) -> List[float]:
        arc = [0.0] * len(trajectory.points)
        for i in range(begin + 1, len(trajectory.points)):
            a, b = trajectory.points[i - 1].pose.position, trajectory.points[i].pose.position
            arc[i] = arc[i - 1] + distance((a.x, a.y), (b.x, b.y))
        return arc

    def clearance(self, trajectory: Trajectory, arc: Sequence[float], begin: int) -> float:
        assert self.odometry is not None
        ego_speed = max(float(self.assumed_ego_speed), abs(self.odometry.twist.twist.linear.x))
        minimum = float("inf")
        for i in range(begin, len(trajectory.points)):
            prediction_time = arc[i] / ego_speed
            if prediction_time > self.collision_horizon:
                break
            point = trajectory.points[i].pose.position
            for obj in self.objects:
                if self.is_ahead(obj):
                    predicted = (obj.x + obj.vx * prediction_time, obj.y + obj.vy * prediction_time)
                    minimum = min(minimum, distance((point.x, point.y), predicted) - obj.radius - self.safety_margin)
        return minimum

    def first_collision_distance(self, trajectory: Trajectory, arc: Sequence[float], begin: int) -> float:
        for i in range(begin, len(trajectory.points)):
            prediction_time = arc[i] / float(self.assumed_ego_speed)
            if prediction_time > self.collision_horizon:
                break
            point = trajectory.points[i].pose.position
            for obj in self.objects:
                if self.is_ahead(obj) and distance((point.x, point.y), (obj.x + obj.vx * prediction_time, obj.y + obj.vy * prediction_time)) < obj.radius + self.safety_margin:
                    return arc[i]
        return self.collision_horizon * float(self.assumed_ego_speed)

    def shifted(self, base: Trajectory, arc: Sequence[float], begin: int, collision_s: float, offset: float) -> Trajectory:
        result = copy.deepcopy(base)
        start, end = max(0.0, collision_s - self.shift_start_distance), collision_s + self.shift_end_distance
        for i in range(begin, len(result.points)):
            if start <= arc[i] < collision_s:
                amount = smoothstep((arc[i] - start) / max(1.0e-6, collision_s - start))
            elif collision_s <= arc[i] < end:
                amount = 1.0 - smoothstep((arc[i] - collision_s) / max(1.0e-6, end - collision_s))
            else:
                continue
            point, base_pose = result.points[i].pose.position, base.points[i].pose
            heading = yaw(base_pose.orientation)
            point.x += -math.sin(heading) * offset * amount
            point.y += math.cos(heading) * offset * amount
        for i in range(begin, len(result.points) - 1):
            point, next_point = result.points[i].pose.position, result.points[i + 1].pose.position
            heading = math.atan2(next_point.y - point.y, next_point.x - point.x)
            orientation = result.points[i].pose.orientation
            orientation.x, orientation.y = 0.0, 0.0
            orientation.z, orientation.w = math.sin(heading * 0.5), math.cos(heading * 0.5)
        return result

    def drivable_margin(self, point: Tuple[float, float]) -> float:
        if not self.lanelets:
            return float("inf")
        best = -float("inf")
        for lanelet in self.lanelets:
            if point_in_polygon(point, lanelet.polygon):
                margin = min(polyline_distance(point, boundary) for boundary in lanelet.boundaries)
                best = max(best, margin)
        return best

    def wall_clearance(self, trajectory: Trajectory, arc: Sequence[float], begin: int, collision_s: float) -> float:
        if not self.lanelets:
            return float("inf")
        start = max(0.0, collision_s - self.shift_start_distance)
        end = collision_s + self.shift_end_distance
        minimum = float("inf")
        checked = False
        for i in range(begin, len(trajectory.points)):
            if arc[i] < start:
                continue
            if arc[i] > end:
                break
            point = trajectory.points[i].pose.position
            minimum = min(minimum, self.drivable_margin((point.x, point.y)))
            checked = True
        return minimum if checked else float("inf")

    def publish_trajectory(self) -> None:
        if self.trajectory is None:
            return
        stale = self.objects_stamp_ns is None or self.get_clock().now().nanoseconds - self.objects_stamp_ns > self.object_timeout * 1.0e9
        if self.odometry is None or not self.objects or stale or len(self.trajectory.points) < 3:
            self.publisher.publish(self.trajectory)
            return
        begin = self.nearest_index(self.trajectory)
        arc = self.arc_lengths(self.trajectory, begin)
        base_clearance = self.clearance(self.trajectory, arc, begin)
        if base_clearance >= 0.0:
            self.publisher.publish(self.trajectory)
            return
        collision_s = self.first_collision_distance(self.trajectory, arc, begin)
        left = self.shifted(self.trajectory, arc, begin, collision_s, self.avoidance_offset)
        right = self.shifted(self.trajectory, arc, begin, collision_s, -self.avoidance_offset)
        left_clearance, right_clearance = self.clearance(left, arc, begin), self.clearance(right, arc, begin)
        left_wall_clearance = self.wall_clearance(left, arc, begin, collision_s)
        right_wall_clearance = self.wall_clearance(right, arc, begin, collision_s)
        if left_wall_clearance < self.wall_margin:
            left_clearance = -float("inf")
        if right_wall_clearance < self.wall_margin:
            right_clearance = -float("inf")
        if max(left_clearance, right_clearance) <= base_clearance:
            self.get_logger().warning("No laterally safer avoidance candidate; keeping the base trajectory")
            self.publisher.publish(self.trajectory)
            return
        if abs(left_clearance - right_clearance) <= self.clearance_tie_epsilon:
            self.publisher.publish(left if left_wall_clearance > right_wall_clearance else right)
            return
        self.publisher.publish(left if left_clearance > right_clearance else right)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryAvoidanceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
