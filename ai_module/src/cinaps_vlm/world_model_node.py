#!/usr/bin/env python3
"""
ROS node: adaptive world model builder (skeleton).

Behavior:
- Try to build from /object_markers (development) if available quickly.
- Otherwise subscribe to /camera/image and /registered_scan, run OVD + fusion,
  and maintain a simple in-memory world model.

This is a minimal runnable outline; fill in TODOs as needed.
"""

import json
import math
from typing import Dict, Tuple, Optional

import rospy
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from cinaps_vlm.perception.world_model import (
    WorldModel,
    build_from_ground_truth,
    build_from_camera_lidar,
)
from cinaps_vlm.perception.fusion import pointcloud2_to_xyz_array

try:
    from cinaps_vlm.perception.ovd import OwlVitDetector
except Exception:
    OwlVitDetector = None  # type: ignore


class WorldModelNode:
    def __init__(self):
        rospy.init_node("world_model_node", anonymous=True)

        self.use_ground_truth = rospy.get_param("~use_ground_truth", False)
        self.use_markers_timeout = rospy.get_param("~object_markers_timeout", 0.5)
        self.detection_threshold = rospy.get_param("~detection_threshold", 0.3)
        self.target_objects = rospy.get_param("~target_objects", [])  # list[str]
        self.merge_distance = rospy.get_param("~merge_distance", 0.75)
        self.stale_time = rospy.get_param("~stale_object_timeout", 6.0)

        self.detector = None
        if OwlVitDetector is not None:
            try:
                self.detector = OwlVitDetector()
                rospy.loginfo("OwlVitDetector initialized")
            except Exception as e:
                rospy.logwarn(f"Failed to init OwlVitDetector: {e}")

        self.world = WorldModel()
        self.pub_debug = rospy.Publisher("/world_model/debug", String, queue_size=1, latch=True)
        self._objects_dict: Dict[str, Dict] = {}
        self._next_object_id = 0

        self.sub_control = rospy.Subscriber("/exploration/request", String, self._on_control, queue_size=1)

        # Initialize based on configuration
        if self.use_ground_truth:
            rospy.loginfo("Mode: Ground Truth - attempting to use /object_markers")
            # Try GT markers once
            try:
                markers = rospy.wait_for_message("/object_markers", MarkerArray, timeout=self.use_markers_timeout)
                rospy.loginfo("✓ Using /object_markers (development mode)")
                self.world = build_from_ground_truth(markers)
                now = rospy.Time.now().to_sec()
                gt_objects = self.world.as_dict()
                for oid, data in gt_objects.items():
                    data['last_update_time'] = now
                self._objects_dict = gt_objects
                self._publish_debug()
            except rospy.ROSException:
                rospy.logwarn("/object_markers unavailable; falling back to camera+LiDAR")
                self._setup_observation_mode()
        else:
            rospy.loginfo("Mode: Real Observation - using camera+LiDAR only")
            self._setup_observation_mode()
            
    def _setup_observation_mode(self):
        """Setup camera+LiDAR observation mode"""
        # Subscribe for continuous updates
        self.image_sub = rospy.Subscriber("/camera/image", ROSImage, self._image_cb, queue_size=1)
        self.cloud_sub = rospy.Subscriber("/registered_scan", PointCloud2, self._cloud_cb, queue_size=1)
        self._last_image = None
        self._last_cloud = None

    def _image_cb(self, msg: ROSImage):
        self._last_image = msg
        self._try_fuse_once()

    def _cloud_cb(self, msg: PointCloud2):
        self._last_cloud = msg
        self._try_fuse_once()

    def _try_fuse_once(self):
        if self._last_image is None or self._last_cloud is None:
            return
        try:
            wm = build_from_camera_lidar(
                image_msg=self._last_image,
                cloud_msg=self._last_cloud,
                detector=self.detector,
                target_objects=self.target_objects,
                detection_threshold=self.detection_threshold,
            )
            # Merge new detections with existing world model
            if wm:
                self._upsert_objects(wm.as_dict())
            self._publish_debug()
        except Exception as e:
            rospy.logerr(f"Fusion failed: {e}")

    def _publish_debug(self):
        try:
            # Use accumulated objects dict
            if hasattr(self, '_objects_dict') and self._objects_dict:
                payload = json.dumps(self._objects_dict)
            else:
                payload = json.dumps({})
                
            msg = String()
            msg.data = payload
            self.pub_debug.publish(msg)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish debug world model: {e}")
            # Emergency fallback
            try:
                empty_msg = String()
                empty_msg.data = "{}"
                self.pub_debug.publish(empty_msg)
            except:
                pass

    def _on_control(self, msg: String):
        cmd = (msg.data or "").strip().lower()
        if cmd in {"stop", "halt", "shutdown"}:
            rospy.loginfo("Exploration stop received; shutting down world model node")
            rospy.signal_shutdown("exploration stop")

    def _upsert_objects(self, detections: Dict[str, Dict]):
        """Merge new detections into the accumulated object dictionary."""
        now = rospy.Time.now().to_sec()

        for det in detections.values():
            obj_class = det.get('class', 'object')
            center = det.get('center', [0.0, 0.0, 0.0])
            size = det.get('size', [0.0, 0.0, 0.0])

            match_id, match_dist = self._find_match(obj_class, center)

            if match_id is not None:
                existing = self._objects_dict[match_id]
                existing_center = existing.get('center', center)
                existing_size = existing.get('size', size)

                blended_center = self._weighted_average(existing_center, center)
                blended_size = self._weighted_average(existing_size, size)

                existing['center'] = blended_center
                existing['size'] = blended_size
                existing['confidence'] = max(existing.get('confidence', 0.0), det.get('confidence', 0.0))
                attrs = existing.get('attributes', {}) or {}
                new_attrs = det.get('attributes', {}) or {}
                attrs.update(new_attrs)
                existing['attributes'] = attrs
                existing['last_update_time'] = now
                rospy.loginfo(f"Updated object {match_id} ({obj_class}) dist={match_dist:.2f}")
            else:
                obj_id = f"obj_{self._next_object_id}"
                self._next_object_id += 1
                det['last_update_time'] = now
                self._objects_dict[obj_id] = det
                rospy.loginfo(f"Added object {obj_id} ({obj_class})")

        # Remove stale entries
        stale_ids = [
            oid for oid, data in self._objects_dict.items()
            if now - data.get('last_update_time', now) > self.stale_time
        ]
        for oid in stale_ids:
            rospy.loginfo(f"Removing stale object {oid} ({self._objects_dict[oid].get('class', 'unknown')})")
            del self._objects_dict[oid]

        rospy.loginfo(f"Total objects in world model: {len(self._objects_dict)}")

    def _find_match(self, obj_class: str, center: Dict) -> Tuple[Optional[str], float]:
        best_id = None
        best_dist = float('inf')
        for oid, data in self._objects_dict.items():
            if data.get('class') != obj_class:
                continue
            dist = self._center_distance(data.get('center'), center)
            if dist < self.merge_distance and dist < best_dist:
                best_dist = dist
                best_id = oid
        return best_id, best_dist

    def _center_distance(self, a, b) -> float:
        try:
            ax, ay, az = float(a[0]), float(a[1]), float(a[2])
            bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        except Exception:
            return float('inf')
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    def _weighted_average(self, prev, new, alpha: float = 0.6):
        try:
            return [alpha * float(prev[i]) + (1 - alpha) * float(new[i]) for i in range(len(prev))]
        except Exception:
            try:
                return list(new)
            except Exception:
                return prev

    def spin(self):
        rospy.loginfo("world_model_node spinning")
        rospy.spin()


if __name__ == "__main__":
    try:
        node = WorldModelNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
