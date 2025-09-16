#!/usr/bin/env python3
"""Exploration manager node.

Tracks explored coverage and publishes exploration waypoints until coverage
is complete. Designed to run alongside QA and world model nodes in challenge
mode so that the robot proactively scans the entire environment before
answering questions.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import rospy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from cinaps_vlm.perception.fusion import pointcloud2_to_xyz_array


@dataclass
class Waypoint:
    x: float
    y: float
    heading_deg: Optional[float] = None


class ExplorationManager:
    def __init__(self):
        rospy.init_node("exploration_manager", anonymous=True)

        self.voxel_size: float = rospy.get_param("~voxel_size", 0.6)
        self.coverage_threshold: float = rospy.get_param("~coverage_threshold", 0.88)
        self.object_settle_time: float = rospy.get_param("~object_settle_time", 6.0)
        self.progress_check_period: float = rospy.get_param("~progress_check_period", 1.0)
        self.dist_threshold: float = rospy.get_param("~waypoint_distance_threshold", 0.8)
        self.goal_timeout: float = rospy.get_param("~waypoint_timeout", 20.0)
        self.state_frame: str = rospy.get_param("~frame", "map")

        # Fixed waypoints remain available as a fallback pattern.
        wp_list = rospy.get_param("~waypoints", [])
        self._waypoints: List[Waypoint] = [self._parse_waypoint(w) for w in wp_list]
        self._use_frontier_first = rospy.get_param("~use_frontier", True)
        self._frontier_sample_limit = rospy.get_param("~frontier_sample_limit", 3000)
        self._frontier_radius: float = rospy.get_param("~frontier_cluster_radius", 1.5)
        self._frontier_revisit_time: float = rospy.get_param("~frontier_revisit_time", 20.0)
        self._coarse_mode: bool = rospy.get_param("~coarse_mode", False)
        self._coarse_stride: int = max(1, int(rospy.get_param("~coarse_frontier_stride", 3)))
        self._coarse_coverage_threshold: float = rospy.get_param("~coarse_coverage_threshold", 0.75)
        if self._coarse_mode:
            original = self.coverage_threshold
            self.coverage_threshold = min(self.coverage_threshold, self._coarse_coverage_threshold)
            rospy.loginfo(
                "Exploration coarse mode enabled: coverage threshold %.2f -> %.2f, stride=%d",
                original,
                self.coverage_threshold,
                self._coarse_stride,
            )

        # Publishers / subscribers
        self.pub_waypoint = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=1)
        self.pub_state = rospy.Publisher("/exploration/state", String, queue_size=1, latch=True)

        self.sub_traversable = rospy.Subscriber("/traversable_area", PointCloud2, self._on_traversable, queue_size=1)
        self.sub_explored = rospy.Subscriber("/explored_areas", PointCloud2, self._on_explored, queue_size=1)
        self.sub_world = rospy.Subscriber("/world_model/debug", String, self._on_world, queue_size=1)
        self.sub_state = rospy.Subscriber("/state_estimation", Odometry, self._on_pose, queue_size=5)
        self.sub_request = rospy.Subscriber("/exploration/request", String, self._on_request, queue_size=1)

        # State containers
        self._traversable_voxels: Set[Tuple[int, int, int]] = set()
        self._explored_voxels: Set[Tuple[int, int, int]] = set()
        self._last_cov_ratio: float = 0.0
        self._last_cov_update: float = rospy.Time.now().to_sec()

        self._world_count: int = 0
        self._last_world_change: float = rospy.Time.now().to_sec()

        self._vehicle_xy: Tuple[float, float] = (0.0, 0.0)
        self._current_goal_index: int = -1
        self._current_goal_stamp: float = 0.0
        self._current_goal: Optional[Waypoint] = None
        self._recent_goals: Deque[Tuple[float, float, float]] = deque(maxlen=20)
        self._unreachable_goals: Dict[Tuple[int, int], float] = {}

        self._exploration_complete = False
        self._active = False
        self._announce_state("idle")

        rospy.Timer(rospy.Duration(self.progress_check_period), self._tick)

    # -- callbacks -----------------------------------------------------------------

    def _on_traversable(self, msg: PointCloud2):
        voxels = self._cloud_to_voxels(msg)
        if voxels:
            self._traversable_voxels.update(voxels)

    def _on_explored(self, msg: PointCloud2):
        voxels = self._cloud_to_voxels(msg)
        if voxels:
            self._explored_voxels.update(voxels)
            self._last_cov_update = rospy.Time.now().to_sec()

    def _on_world(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if isinstance(data, dict):
            count = len(data)
            if count != self._world_count:
                self._world_count = count
                self._last_world_change = rospy.Time.now().to_sec()

    def _on_pose(self, msg: Odometry):
        self._vehicle_xy = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    # -- periodic ------------------------------------------------------------------

    def _tick(self, _event):
        coverage = self._compute_coverage()

        if abs(coverage - self._last_cov_ratio) > 1e-3:
            self._last_cov_ratio = coverage
            rospy.loginfo(f"Exploration coverage: {coverage:.2%}")

        if self._exploration_complete:
            return

        if not self._active:
            return

        now = rospy.Time.now().to_sec()

        if self._coverage_satisfied(coverage, now):
            self._exploration_complete = True
            self._active = False
            self._announce_state("complete")
            rospy.loginfo("Exploration complete. Holding position for QA tasks.")
            return

        self._announce_state("exploring")
        self._ensure_goal(now)

    # -- goal management -----------------------------------------------------------

    def _ensure_goal(self, now: float):
        goal = self._determine_goal(now)
        if goal is None:
            return

        gx, gy = goal.x, goal.y
        heading = goal.heading_deg
        if heading is None:
            heading = math.degrees(math.atan2(gy - self._vehicle_xy[1], gx - self._vehicle_xy[0]))

        pose = Pose2D()
        pose.x = gx
        pose.y = gy
        pose.theta = math.radians(heading)
        self.pub_waypoint.publish(pose)
        self._recent_goals.append((gx, gy, now))

    def _determine_goal(self, now: float) -> Optional[Waypoint]:
        # Try dynamic frontier goal first
        if self._use_frontier_first:
            frontier_goal = self._pick_frontier_goal(now)
            if frontier_goal is not None:
                self._current_goal = frontier_goal
                self._current_goal_stamp = now
                return self._current_goal

        # Fallback to fixed waypoint pattern
        if not self._waypoints:
            return None

        if self._current_goal is None:
            self._current_goal_index = -1
        elif self._distance_xy(self._vehicle_xy, (self._current_goal.x, self._current_goal.y)) <= self.dist_threshold:
            self._current_goal = None
        elif (now - self._current_goal_stamp) > self.goal_timeout:
            self._mark_unreachable(self._current_goal)
            self._current_goal = None

        if self._current_goal is None:
            next_index = (self._current_goal_index + 1) % len(self._waypoints)
            if next_index == 0 and self._current_goal_index != -1:
                if self._compute_coverage() >= self.coverage_threshold:
                    return None
            self._current_goal_index = next_index
            self._current_goal = self._waypoints[next_index]
            self._current_goal_stamp = now
            rospy.loginfo(
                "Dispatching fallback waypoint %d/%d at (%.2f, %.2f)",
                next_index + 1,
                len(self._waypoints),
                self._current_goal.x,
                self._current_goal.y,
            )

        return self._current_goal

    # -- coverage utilities --------------------------------------------------------

    def _coverage_satisfied(self, coverage: float, now: float) -> bool:
        if coverage < self.coverage_threshold:
            return False
        if now - self._last_world_change < self.object_settle_time:
            return False
        if now - self._last_cov_update < self.object_settle_time:
            return False
        return True

    def _compute_coverage(self) -> float:
        if not self._traversable_voxels:
            return 0.0
        covered = len(self._explored_voxels.intersection(self._traversable_voxels))
        return float(covered) / max(1, len(self._traversable_voxels))

    def _pick_frontier_goal(self, now: float) -> Optional[Waypoint]:
        if not self._traversable_voxels:
            return None

        frontiers = self._collect_frontiers()
        if not frontiers:
            return None

        clusters = self._cluster_frontiers(frontiers)
        if not clusters:
            return None

        best_wp = None
        best_score = -float('inf')
        for center, stats in clusters:
            if self._recently_visited(center, now):
                continue
            if self._is_unreachable(center, now):
                continue
            dist = self._distance_xy(self._vehicle_xy, center)
            if dist < max(self.dist_threshold, self.voxel_size * 1.0):
                continue
            coverage_gain = stats['count']
            score = coverage_gain - 0.1 * dist
            if score > best_score:
                best_score = score
                best_wp = Waypoint(x=center[0], y=center[1], heading_deg=None)

        if best_wp is None:
            return None

        rospy.loginfo(
            "Frontier cluster selected at %.2f, %.2f (score %.2f)",
            best_wp.x,
            best_wp.y,
            best_score,
        )
        return best_wp

    def _is_frontier_cell(self, cell: Tuple[int, int, int], seen: Set[Tuple[int, int, int]]) -> bool:
        cx, cy, cz = cell
        neighbors = [
            (cx + 1, cy, cz),
            (cx - 1, cy, cz),
            (cx, cy + 1, cz),
            (cx, cy - 1, cz),
            (cx, cy, cz + 1),
            (cx, cy, cz - 1),
        ]
        for n in neighbors:
            if n in seen:
                return True
        return False

    def _collect_frontiers(self) -> List[Tuple[int, int, int]]:
        seen = self._explored_voxels
        trav = self._traversable_voxels
        limit = self._frontier_sample_limit
        frontiers: List[Tuple[int, int, int]] = []
        count = 0
        stride = self._coarse_stride if self._coarse_mode else 1
        for cell in trav:
            if cell in seen:
                continue
            if self._is_frontier_cell(cell, seen):
                if self._coarse_mode and (count % stride != 0):
                    count += 1
                    continue
                frontiers.append(cell)
                count += 1
                if limit and count >= limit:
                    break
        return frontiers

    def _cluster_frontiers(
        self, frontiers: List[Tuple[int, int, int]]
    ) -> List[Tuple[Tuple[float, float], Dict[str, float]]]:
        if not frontiers:
            return []

        clusters: List[Tuple[Tuple[float, float], Dict[str, float]]] = []
        visited: Set[int] = set()
        radius2 = (self._frontier_radius / max(self.voxel_size, 1e-3)) ** 2

        for idx, cell in enumerate(frontiers):
            if idx in visited:
                continue
            queue = [idx]
            visited.add(idx)
            pts: List[Tuple[float, float]] = []
            weight = 0
            while queue:
                i = queue.pop()
                wx, wy = self._voxel_to_world(frontiers[i])
                pts.append((wx, wy))
                weight += 1
                for j in range(len(frontiers)):
                    if j in visited:
                        continue
                    if self._voxel_distance2(frontiers[i], frontiers[j]) <= radius2:
                        visited.add(j)
                        queue.append(j)

            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            clusters.append(((cx, cy), {"count": float(weight)}))
        return clusters

    def _voxel_distance2(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return float(dx * dx + dy * dy + dz * dz)

    def _recently_visited(self, center: Tuple[float, float], now: float) -> bool:
        for gx, gy, ts in self._recent_goals:
            if now - ts > self._frontier_revisit_time:
                continue
            if self._distance_xy((gx, gy), center) <= self._frontier_radius:
                return True
        return False

    def _is_unreachable(self, center: Tuple[float, float], now: float) -> bool:
        key = (int(round(center[0] / max(self.voxel_size, 1e-3))), int(round(center[1] / max(self.voxel_size, 1e-3))))
        expiry = self._unreachable_goals.get(key)
        if expiry is None:
            return False
        if now > expiry:
            del self._unreachable_goals[key]
            return False
        return True

    def _mark_unreachable(self, goal: Waypoint):
        key = (
            int(round(goal.x / max(self.voxel_size, 1e-3))),
            int(round(goal.y / max(self.voxel_size, 1e-3))),
        )
        ttl = rospy.get_param("~frontier_unreachable_ttl", 30.0)
        self._unreachable_goals[key] = rospy.Time.now().to_sec() + ttl

    def _cloud_to_voxels(self, msg: PointCloud2) -> Set[Tuple[int, int, int]]:
        try:
            xyz = pointcloud2_to_xyz_array(msg)
        except Exception:
            return set()

        voxels: Set[Tuple[int, int, int]] = set()
        vs = max(self.voxel_size, 1e-3)
        inv = 1.0 / vs
        for x, y, z in xyz:
            ix = int(math.floor(x * inv))
            iy = int(math.floor(y * inv))
            iz = int(math.floor(z * inv))
            voxels.add((ix, iy, iz))
        return voxels

    # -- helpers -------------------------------------------------------------------

    def _parse_waypoint(self, raw: Sequence[float]) -> Waypoint:
        if len(raw) == 2:
            return Waypoint(x=float(raw[0]), y=float(raw[1]), heading_deg=None)
        if len(raw) >= 3:
            return Waypoint(x=float(raw[0]), y=float(raw[1]), heading_deg=float(raw[2]))
        raise ValueError("Waypoint must have at least x and y components")

    def _distance_xy(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        return math.hypot(ax - bx, ay - by)

    def _announce_state(self, state: str):
        msg = String()
        msg.data = state
        self.pub_state.publish(msg)

    def _voxel_to_world(self, cell: Tuple[int, int, int]) -> Tuple[float, float]:
        x, y, _ = cell
        vs = self.voxel_size
        return (x + 0.5) * vs, (y + 0.5) * vs

    def _on_request(self, msg: String):
        command = msg.data.strip().lower()
        if command in {"start", "explore", "scan"}:
            self._begin_exploration()
        elif command in {"stop", "halt"}:
            self._active = False
            self._current_goal = None
            self._announce_state("idle")
        elif command in {"reset"}:
            self._reset_state()

    def _begin_exploration(self):
        now = rospy.Time.now().to_sec()
        self._exploration_complete = False
        self._active = True
        self._current_goal_index = -1
        self._current_goal = None
        self._current_goal_stamp = 0.0
        self._last_world_change = now
        self._last_cov_update = now
        self._announce_state("exploring")
        rospy.loginfo("Exploration requested; beginning waypoint patrol")

    def _reset_state(self):
        rospy.loginfo("Exploration reset request received")
        self._exploration_complete = False
        self._active = False
        self._current_goal_index = -1
        self._current_goal = None
        self._current_goal_stamp = 0.0
        self._traversable_voxels.clear()
        self._explored_voxels.clear()
        self._announce_state("idle")

    # -----------------------------------------------------------------------------

    def spin(self):
        rospy.loginfo("exploration_manager spinning")
        rospy.spin()


def main():
    manager = ExplorationManager()
    manager.spin()


if __name__ == "__main__":
    main()
