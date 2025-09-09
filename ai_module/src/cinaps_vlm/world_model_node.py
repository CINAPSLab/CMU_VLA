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

        self.use_markers_timeout = rospy.get_param("~object_markers_timeout", 0.5)
        self.detection_threshold = rospy.get_param("~detection_threshold", 0.3)
        self.target_objects = rospy.get_param("~target_objects", [])  # list[str]

        self.detector = None
        if OwlVitDetector is not None:
            try:
                self.detector = OwlVitDetector()
                rospy.loginfo("OwlVitDetector initialized")
            except Exception as e:
                rospy.logwarn(f"Failed to init OwlVitDetector: {e}")

        self.world = WorldModel()
        self.pub_debug = rospy.Publisher("/world_model/debug", String, queue_size=1, latch=True)

        # Try GT markers once
        try:
            markers = rospy.wait_for_message("/object_markers", MarkerArray, timeout=self.use_markers_timeout)
            rospy.loginfo("Using /object_markers (development mode)")
            self.world = build_from_ground_truth(markers)
            self._publish_debug()
        except rospy.ROSException:
            rospy.loginfo("/object_markers unavailable; switching to camera+LiDAR")
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
            # Replace entire world for now (skeleton); later, merge incrementally
            self.world = wm
            self._publish_debug()
        except Exception as e:
            rospy.logerr(f"Fusion failed: {e}")

    def _publish_debug(self):
        try:
            payload = json.dumps(self.world.as_dict())
            self.pub_debug.publish(payload)
        except Exception as e:
            rospy.logwarn(f"Failed to publish debug world model: {e}")

    def spin(self):
        rospy.loginfo("world_model_node spinning")
        rospy.spin()


if __name__ == "__main__":
    try:
        node = WorldModelNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
