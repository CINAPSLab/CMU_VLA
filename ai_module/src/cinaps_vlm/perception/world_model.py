#!/usr/bin/env python3
"""
Adaptive World Model skeleton.

- Tries to build from GT markers if available (development mode)
- Falls back to camera+LiDAR sensor fusion (test/eval mode)

This is a minimal skeleton with safe defaults and TODO hooks to extend.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import rospy
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from .fusion import (
    pointcloud2_to_xyz_array,
    project_points_equirectangular,
    select_points_in_bbox,
    estimate_bbox_3d,
)

try:
    # Local OVD wrapper (optional; safe fallback if not available)
    from .ovd import OwlVitDetector
except Exception:
    OwlVitDetector = None  # type: ignore


@dataclass
class ObjectState:
    object_id: str
    class_name: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    confidence: float = 1.0
    attributes: Dict = field(default_factory=dict)
    last_update_time: float = field(default_factory=lambda: time.time())


class WorldModel:
    """In-memory world model storing 3D object states."""

    def __init__(self) -> None:
        self.objects: Dict[str, ObjectState] = {}

    def upsert(self, state: ObjectState) -> None:
        self.objects[state.object_id] = state

    def as_dict(self) -> Dict:
        return {
            oid: {
                "class": s.class_name,
                "center": list(s.center),
                "size": list(s.size),
                "confidence": s.confidence,
                "attributes": s.attributes,
                "last_update_time": s.last_update_time,
            }
            for oid, s in self.objects.items()
        }


def build_from_ground_truth(markers: MarkerArray) -> WorldModel:
    """Build world model from GT markers (development only).

    Assumptions (robust defaults):
    - marker.ns or marker.text contains class label if available
    - marker.pose.position is center
    - marker.scale is bbox size (x,y,z)
    - marker.id is unique within the array
    """
    wm = WorldModel()
    for m in markers.markers:
        class_name = getattr(m, "text", None) or m.ns or "object"
        center = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
        size = (max(m.scale.x, 1e-3), max(m.scale.y, 1e-3), max(m.scale.z, 1e-3))
        oid = f"gt_{m.id}"
        wm.upsert(ObjectState(object_id=oid, class_name=class_name, center=center, size=size, confidence=1.0))
    return wm


def build_from_camera_lidar(
    image_msg: ROSImage,
    cloud_msg: PointCloud2,
    detector: Optional["OwlVitDetector"],
    target_objects: Optional[List[str]] = None,
    detection_threshold: float = 0.3,
) -> WorldModel:
    """Build world model via camera+LiDAR fusion.

    Steps (simplified):
    - 2D detect on image
    - Project LiDAR to equirectangular image plane
    - For each 2D bbox, collect in-box points and estimate 3D bbox
    """
    wm = WorldModel()

    # Convert point cloud to Nx3
    xyz = pointcloud2_to_xyz_array(cloud_msg)

    # Image shape
    try:
        import cv2
        from cv_bridge import CvBridge

        cv_img = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
        H, W = cv_img.shape[:2]
    except Exception:
        # Fallback: assume 1920x640 if not convertible
        H, W = 640, 1920
        cv_img = None

    # 2D detection
    detections: List[Dict] = []
    if detector is not None:
        try:
            detections = detector.detect(cv_img, target_objects or [], threshold=detection_threshold)
        except Exception as e:
            rospy.logwarn(f"OVD detection failed: {e}")
    else:
        rospy.logwarn("OwlVitDetector unavailable; skipping 2D detection (empty detections)")

    # Project LiDAR to equirectangular pixels (approximate; assumes aligned frames)
    uv = project_points_equirectangular(xyz, W, H)

    # Associate points to detections and estimate 3D
    def _dominant_color_label(bgr_img, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        try:
            import cv2
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr_img.shape[1] - 1, x2), min(bgr_img.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            roi = bgr_img[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Decide achromatic colors first
            mean_s = float(s.mean())
            mean_v = float(v.mean())
            if mean_v < 50:
                return "black"
            if mean_s < 25 and mean_v > 200:
                return "white"
            if mean_s < 35:
                return "gray"
            # Hue histogram
            hist = cv2.calcHist([h], [0], None, [180], [0, 180]).flatten()
            hue_idx = int(hist.argmax())
            # Map hue to color name
            if (hue_idx <= 10) or (hue_idx >= 160):
                return "red"
            if 11 <= hue_idx <= 25:
                return "orange"
            if 26 <= hue_idx <= 35:
                return "yellow"
            if 36 <= hue_idx <= 85:
                return "green"
            if 86 <= hue_idx <= 95:
                return "cyan"
            if 96 <= hue_idx <= 130:
                return "blue"
            if 131 <= hue_idx <= 155:
                return "purple"
            return None
        except Exception:
            return None

    for i, det in enumerate(detections):
        bbox = det.get("box", {})
        xmin, ymin = int(bbox.get("xmin", 0)), int(bbox.get("ymin", 0))
        xmax, ymax = int(bbox.get("xmax", 0)), int(bbox.get("ymax", 0))
        mask = select_points_in_bbox(uv, (xmin, ymin, xmax, ymax), W, H)
        pts = xyz[mask]
        if pts.shape[0] < 5:
            # Not enough points; skip or keep a weak placeholder
            center = (0.0, 0.0, 0.0)
            size = (0.0, 0.0, 0.0)
            conf = float(det.get("score", 0.0)) * 0.1
        else:
            center, size = estimate_bbox_3d(pts)
            conf = float(det.get("score", 0.0))

        # Estimate dominant color if image available
        color_label = None
        if cv_img is not None:
            color_label = _dominant_color_label(cv_img, (xmin, ymin, xmax, ymax))

        oid = f"det_{i}_{det.get('label', 'object')}"
        wm.upsert(
            ObjectState(
                object_id=oid,
                class_name=det.get("label", "object"),
                center=center,
                size=size,
                confidence=conf,
                attributes={"source": "fusion", **({"color": color_label} if color_label else {})},
            )
        )

    return wm


def adaptive_world_model(
    target_objects: Optional[List[str]] = None,
    object_markers_timeout: float = 0.5,
    detection_threshold: float = 0.3,
) -> WorldModel:
    """Try GT markers first; fallback to camera+LiDAR fusion once.

    This is intended for one-shot construction during startup.
    Continuous update should be handled by a ROS node (see world_model_node).
    """
    # Try GT markers (development only)
    try:
        markers = rospy.wait_for_message("/object_markers", MarkerArray, timeout=object_markers_timeout)
        rospy.loginfo("Building world model from /object_markers (development mode)")
        return build_from_ground_truth(markers)
    except rospy.ROSException:
        rospy.loginfo("/object_markers unavailable; falling back to camera+LiDAR fusion (test mode)")

    # Fallback: one pair of image and point cloud
    image_msg = rospy.wait_for_message("/camera/image", ROSImage)
    cloud_msg = rospy.wait_for_message("/registered_scan", PointCloud2)

    # Lazy OVD detector init (optional)
    detector = None
    if OwlVitDetector is not None:
        try:
            detector = OwlVitDetector()
        except Exception as e:
            rospy.logwarn(f"Failed to init OwlVitDetector: {e}")

    return build_from_camera_lidar(
        image_msg=image_msg,
        cloud_msg=cloud_msg,
        detector=detector,
        target_objects=target_objects or [],
        detection_threshold=detection_threshold,
    )
