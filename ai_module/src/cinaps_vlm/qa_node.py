#!/usr/bin/env python3
"""
cinaps_vlm QA node: listens to /challenge_question and responds on
- /selected_object_marker (Marker) for "find" queries
- /numerical_response (Int32) for "how many" queries
- /way_point_with_heading (Pose2D) for simple navigation goals

Uses world_model_node's debug JSON on /world_model/debug as the source of objects.
No dependency on dummy_vlm.
"""

import json
import math
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry


def _norm_text(s: str) -> str:
    return s.strip().lower()


class QANode:
    def __init__(self):
        rospy.init_node("cinaps_qa", anonymous=True)

        # Publishers
        self.pub_marker = rospy.Publisher("/selected_object_marker", Marker, queue_size=1, latch=True)
        self.pub_waypoint = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=1)
        self.pub_number = rospy.Publisher("/numerical_response", Int32, queue_size=1)

        # Subscribers
        self.sub_question = rospy.Subscriber("/challenge_question", String, self._on_question, queue_size=1)
        self.sub_world = rospy.Subscriber("/world_model/debug", String, self._on_world, queue_size=5)
        self.sub_pose = rospy.Subscriber("/state_estimation", Odometry, self._on_pose, queue_size=5)

        # Params
        self._reach_dist = rospy.get_param("~waypointReachDis", 1.0)

        # State
        self._world: Dict[str, Dict] = {}
        self._vehicle_xy: Tuple[float, float] = (0.0, 0.0)
        self._lock = threading.Lock()
        self._nav_plan: List[Tuple[float, float, float]] = []  # (x,y,theta)
        self._nav_idx: int = 0
        self._nav_active: bool = False
        self._last_pub_time: float = 0.0

        # Synonyms and color words
        self._synonyms = self._build_synonyms()
        self._rev_syn = self._build_reverse_syn(self._synonyms)
        self._color_words = {"red", "blue", "green", "black", "white", "gray", "grey", "yellow", "brown", "pink", "orange", "purple"}

        # Timer for sequential navigation gating
        self._timer = rospy.Timer(rospy.Duration(0.25), self._nav_tick)

        rospy.loginfo("cinaps_qa node ready")

    def _on_world(self, msg: String):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                with self._lock:
                    self._world = data
        except Exception as e:
            rospy.logwarn(f"Failed to parse world model JSON: {e}")

    def _on_pose(self, msg: Odometry):
        self._vehicle_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_question(self, msg: String):
        q = _norm_text(msg.data)
        rospy.loginfo(f"Received question: {q}")

        # Decide intent
        if q.startswith("how many"):
            self._answer_how_many(q)
        elif q.startswith("find") or q.startswith("where is") or q.startswith("locate"):
            self._answer_find(q)
        else:
            # treat as navigation: go to the best-matching object
            self._answer_navigate(q)

    # --- Helpers ---
    def _current_objects(self) -> List[Dict]:
        with self._lock:
            # world dict is {id: {class, center, size, ...}}
            return list(self._world.values()) if isinstance(self._world, dict) else []

    def _build_synonyms(self) -> Dict[str, List[str]]:
        return {
            # furniture
            "sofa": ["sofa", "couch"],
            "pillow": ["pillow", "cushion"],
            "chair": ["chair"],
            "stool": ["stool"],
            "bench": ["bench"],
            "bed": ["bed"],
            "table": ["table", "coffee table", "dining table", "tea table", "small table"],
            "nightstand": ["nightstand", "bedside table"],
            "cabinet": ["cabinet", "file cabinet"],
            "tv cabinet": ["tv cabinet", "tv stand", "tv table"],
            "shelf": ["shelf", "bookcase"],
            # appliances / electronics
            "tv": ["tv", "television"],
            "microwave": ["microwave"],
            "refrigerator": ["refrigerator", "fridge"],
            "computer monitor": ["computer monitor", "monitor"],
            # decor / small objects
            "lamp": ["lamp", "lantern"],
            "vase": ["vase"],
            "bowl": ["bowl"],
            "cup": ["cup", "paper cup"],
            "picture": ["picture", "painting", "photo"],
            "clock": ["clock"],
            "guitar": ["guitar"],
            "flowers": ["flowers", "flower"],
            "beer bottle": ["beer bottle", "bottle"],
            "record": ["record", "framed record"],
            # structures
            "window": ["window"],
            "door": ["door", "door frame"],
            "whiteboard": ["whiteboard"],
            "fireplace": ["fireplace"],
            # containers / misc
            "trash can": ["trash can", "trashcan", "bin"],
            "box": ["box"],
            "folder": ["folder"],
            "phone": ["phone"],
            # plants
            "plant": ["plant", "potted plant"],
        }

    def _build_reverse_syn(self, syn: Dict[str, List[str]]) -> Dict[str, str]:
        rev = {}
        for canon, words in syn.items():
            for w in words:
                rev[_norm_text(w)] = canon
        return rev

    def _singularize(self, w: str) -> str:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    def _canonical_label(self, label: str) -> str:
        l = _norm_text(label)
        l = self._singularize(l)
        return self._rev_syn.get(l, l)

    def _labels_in_text(self, q: str) -> List[str]:
        ql = _norm_text(q)
        found = []
        phrases = sorted(self._rev_syn.keys(), key=len, reverse=True)
        for p in phrases:
            if p in ql:
                canon = self._rev_syn[p]
                if canon not in found:
                    found.append(canon)
        return found

    def _select_label(self, q: str, candidates: List[str]) -> Optional[str]:
        labels_in_q = self._labels_in_text(q)
        if labels_in_q:
            for l in labels_in_q:
                for c in candidates:
                    if l in _norm_text(c):
                        return l
            return labels_in_q[0]
        ql = _norm_text(q)
        matches = [c for c in candidates if c and _norm_text(c) in ql]
        if matches:
            return sorted(matches, key=len, reverse=True)[0]
        return None

    def _filter_by_label(self, objs: List[Dict], label: str) -> List[Dict]:
        lab = self._canonical_label(label)
        res = []
        for o in objs:
            c = self._canonical_label(o.get("class", ""))
            if lab in c:
                res.append(o)
        return res

    def _filter_by_color(self, objs: List[Dict], color: Optional[str]) -> List[Dict]:
        if not color:
            return objs
        col = _norm_text(color)
        res = []
        for o in objs:
            attrs = o.get("attributes", {}) or {}
            c = _norm_text(str(attrs.get("color", "")))
            if col in c:
                res.append(o)
        return res

    def _closest(self, objs: List[Dict]) -> Optional[Dict]:
        if not objs:
            return None
        vx, vy = self._vehicle_xy
        def dist2(o: Dict) -> float:
            cx, cy = 0.0, 0.0
            try:
                center = o.get("center", [0, 0, 0])
                cx, cy = float(center[0]), float(center[1])
            except Exception:
                pass
            dx, dy = cx - vx, cy - vy
            return dx * dx + dy * dy
        return min(objs, key=dist2)

    def _publish_marker(self, obj: Dict):
        try:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = obj.get("class", "object")
            m.id = int(rospy.Time.now().to_nsec() % 2147483647)
            m.action = Marker.ADD
            m.type = Marker.CUBE
            c = obj.get("center", [0, 0, 0])
            s = obj.get("size", [0.5, 0.5, 0.5])
            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = float(c[2])
            m.pose.orientation.w = 1.0
            m.scale.x = max(float(s[0]), 1e-3)
            m.scale.y = max(float(s[1]), 1e-3)
            m.scale.z = max(float(s[2]), 1e-3)
            m.color.a = 0.6
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.2
            self.pub_marker.publish(m)
        except Exception as e:
            rospy.logwarn(f"Failed to publish marker: {e}")

    def _publish_waypoint(self, obj: Dict, heading_deg: float = 0.0):
        try:
            p = Pose2D()
            c = obj.get("center", [0, 0, 0])
            p.x = float(c[0])
            p.y = float(c[1])
            p.theta = math.radians(heading_deg)
            self.pub_waypoint.publish(p)
        except Exception as e:
            rospy.logwarn(f"Failed to publish waypoint: {e}")

    def _publish_waypoint_xy(self, x: float, y: float, heading_deg: float = 0.0):
        try:
            p = Pose2D()
            p.x = float(x)
            p.y = float(y)
            p.theta = math.radians(heading_deg)
            self.pub_waypoint.publish(p)
        except Exception as e:
            rospy.logwarn(f"Failed to publish waypoint: {e}")

    # --- Relations helpers ---
    def _apply_spatial_filters(self, q: str, candidates: List[Dict], all_objs: List[Dict]) -> List[Dict]:
        """Apply on/above/below/near/between filters based on text."""
        try:
            from cinaps_vlm.reasoning import relations as R
        except Exception:
            return candidates

        filtered = candidates
        ql = _norm_text(q)

        def pick_ref_objs(label: str) -> List[Dict]:
            return self._filter_by_label(all_objs, label)

        # on X
        m = re.search(r"on (?:the |a )?([a-z \-]+)", ql)
        if m:
            ref_label = self._canonical_label(m.group(1).strip())
            refs = pick_ref_objs(ref_label)
            if refs:
                tmp = []
                for a in filtered:
                    for r in refs:
                        if R.is_on(a, r):
                            tmp.append(a)
                            break
                if tmp:
                    filtered = tmp

        # above/below X
        m = re.search(r"above (?:the |a )?([a-z \-]+)", ql)
        if m:
            refs = pick_ref_objs(m.group(1).strip())
            if refs:
                tmp = []
                for a in filtered:
                    if any(R.is_above(a, r) for r in refs):
                        tmp.append(a)
                if tmp:
                    filtered = tmp
        m = re.search(r"below (?:the |a )?([a-z \-]+)", ql)
        if m:
            refs = pick_ref_objs(m.group(1).strip())
            if refs:
                tmp = []
                for a in filtered:
                    if any(R.is_below(a, r) for r in refs):
                        tmp.append(a)
                if tmp:
                    filtered = tmp

        # near X
        m = re.search(r"near (?:the |a )?([a-z \-]+)", ql)
        if m:
            refs = pick_ref_objs(m.group(1).strip())
            if refs:
                tmp = []
                for a in filtered:
                    if any(R.is_near(a, r, th=1.5) for r in refs):
                        tmp.append(a)
                if tmp:
                    filtered = tmp

        # between A and B
        m = re.search(r"between (?:the |a )?([a-z \-]+) and (?:the |a )?([a-z \-]+)", ql)
        if m:
            a_label = self._canonical_label(m.group(1).strip())
            b_label = self._canonical_label(m.group(2).strip())
            as_ = pick_ref_objs(a_label)
            bs_ = pick_ref_objs(b_label)
            if as_ and bs_:
                tmp = []
                for cand in filtered:
                    if any(any(R.is_between(cand, a, b) for b in bs_) for a in as_):
                        tmp.append(cand)
                if tmp:
                    filtered = tmp
        return filtered

    def _rank_by_proximity_to_label(self, q: str, candidates: List[Dict], all_objs: List[Dict], mode: str) -> Optional[Dict]:
        try:
            from cinaps_vlm.reasoning import relations as R
        except Exception:
            R = None
        ql = _norm_text(q)
        m = re.search(r"(closest|nearest) to (?:the |a )?([a-z \-]+)", ql)
        if not m:
            m = re.search(r"farthest from (?:the |a )?([a-z \-]+)", ql)
            if m:
                mode = "farthest"
        if not m:
            return None
        label = self._canonical_label(m.groups()[-1].strip())
        refs = self._filter_by_label(all_objs, label)
        if not refs:
            return None
        ref = refs[0]
        if mode == "closest":
            idxs = [0]
            if R:
                idxs = R.closest(ref, candidates, k=1)
            return candidates[idxs[0]] if candidates else None
        else:
            idxs = [0]
            if R:
                idxs = R.farthest(ref, candidates, k=1)
            return candidates[idxs[0]] if candidates else None

    # --- Intents ---
    def _answer_how_many(self, q: str):
        objs = self._current_objects()
        labels = list({self._canonical_label(o.get("class", "object")) for o in objs})
        m = re.search(r"how many ([a-z \-]+?)\b", _norm_text(q))
        target = None
        if m:
            target = self._canonical_label(self._singularize(m.group(1).strip()))
        if not target:
            target = self._select_label(q, labels) or "object"

        # color
        color = None
        for cw in self._color_words:
            if cw in _norm_text(q):
                color = "grey" if cw == "grey" else cw
                break

        matched = self._filter_by_label(objs, target)
        matched = self._filter_by_color(matched, color)
        matched = self._apply_spatial_filters(q, matched, objs)

        count = len(matched)
        rospy.loginfo(f"how many '{target}': {count}")
        self.pub_number.publish(Int32(data=count))

    def _answer_find(self, q: str):
        objs = self._current_objects()
        labels = list({self._canonical_label(o.get("class", "object")) for o in objs})
        target = self._select_label(q, labels) or (labels[0] if labels else "object")
        # color
        color = None
        for cw in self._color_words:
            if cw in _norm_text(q):
                color = "grey" if cw == "grey" else cw
                break

        matched = self._filter_by_label(objs, target) or objs
        matched = self._filter_by_color(matched, color)
        matched = self._apply_spatial_filters(q, matched, objs)

        pick = self._rank_by_proximity_to_label(q, matched, objs, mode="closest")
        best = pick or self._closest(matched)
        if best is None:
            rospy.logwarn("No objects available to find.")
            return
        rospy.loginfo(f"find -> selecting '{best.get('class','object')}'")
        self._publish_marker(best)
        self._publish_waypoint(best, heading_deg=0.0)

    def _answer_navigate(self, q: str):
        # Build a sequence of waypoints from instruction clauses
        objs = self._current_objects()
        if not objs:
            rospy.logwarn("No objects; cannot navigate.")
            return

        clauses = self._split_navigation(q)
        plan: List[Tuple[float, float, float]] = []
        for cl in clauses:
            wp = self._waypoint_from_clause(cl, objs)
            if wp is not None:
                plan.append(wp)

        # avoid the path between X and Y -> insert detour waypoint before others
        detour = self._detour_from_avoid(q, objs)
        if detour is not None:
            plan.insert(0, detour)

        if not plan:
            # fallback single waypoint to closest mentioned object
            labels = list({self._canonical_label(o.get("class", "object")) for o in objs})
            chosen_label = self._select_label(q, labels)
            candidates = self._filter_by_label(objs, chosen_label) if chosen_label else objs
            best = self._closest(candidates)
            if best is None:
                rospy.logwarn("No navigation target found.")
                return
            c = best.get("center", [0, 0, 0])
            plan = [(float(c[0]), float(c[1]), 0.0)]

        # Start gated navigation
        with self._lock:
            self._nav_plan = plan
            self._nav_idx = 0
            self._nav_active = True
        self._publish_current_nav()

    def spin(self):
        rospy.loginfo("cinaps_qa spinning")
        rospy.spin()

    # --- Navigation utilities ---
    def _split_navigation(self, q: str) -> List[str]:
        ql = _norm_text(q)
        parts = re.split(r"(?:first,?\s*|then,?\s*|finally,?\s*|and then\s*| and )", ql)
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    def _find_label_object(self, label: str, objs: List[Dict]) -> Optional[Dict]:
        cands = self._filter_by_label(objs, label)
        return self._closest(cands)

    def _waypoint_from_clause(self, cl: str, objs: List[Dict]) -> Optional[Tuple[float, float, float]]:
        # between A and B -> midpoint
        m = re.search(r"between (?:the |a )?([a-z \-]+) and (?:the |a )?([a-z \-]+)", cl)
        if m:
            a = self._find_label_object(m.group(1).strip(), objs)
            b = self._find_label_object(m.group(2).strip(), objs)
            if a and b:
                ax, ay = float(a["center"][0]), float(a["center"][1])
                bx, by = float(b["center"][0]), float(b["center"][1])
                mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                return (mx, my, 0.0)

        # closest/farthest to Z
        m = re.search(r"(closest|nearest) to (?:the |a )?([a-z \-]+)", cl)
        mode = "closest"
        if not m:
            m = re.search(r"farthest from (?:the |a )?([a-z \-]+)", cl)
            if m:
                mode = "farthest"
        if m:
            labels_in = self._labels_in_text(cl)
            target = labels_in[0] if labels_in else None
            cands = self._filter_by_label(objs, target) if target else objs
            pick = self._rank_by_proximity_to_label(cl, cands, objs, mode=mode)
            if pick is None and cands:
                pick = self._closest(cands)
            if pick:
                cx, cy = float(pick["center"][0]), float(pick["center"][1])
                return (cx, cy, 0.0)

        # generic go to/near X
        m = re.search(r"(?:go|stop|to|near) (?:the |a )?([a-z \-]+)", cl)
        if m:
            obj = self._find_label_object(m.group(1).strip(), objs)
            if obj:
                cx, cy = float(obj["center"][0]), float(obj["center"][1])
                return (cx, cy, 0.0)

        labels_in = self._labels_in_text(cl)
        if labels_in:
            obj = self._find_label_object(labels_in[0], objs)
            if obj:
                cx, cy = float(obj["center"][0]), float(obj["center"][1])
                return (cx, cy, 0.0)
        return None

    def _detour_from_avoid(self, q: str, objs: List[Dict]) -> Optional[Tuple[float, float, float]]:
        m = re.search(r"avoid (?:the )?path between (?:the |a )?([a-z \-]+) and (?:the |a )?([a-z \-]+)", _norm_text(q))
        if not m:
            return None
        a = self._find_label_object(m.group(1).strip(), objs)
        b = self._find_label_object(m.group(2).strip(), objs)
        if not (a and b):
            return None
        ax, ay = float(a["center"][0]), float(a["center"][1])
        bx, by = float(b["center"][0]), float(b["center"][1])
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        vx, vy = bx - ax, by - ay
        px, py = -vy, vx
        norm = math.hypot(px, py) + 1e-6
        px, py = px / norm, py / norm
        rx, ry = self._vehicle_xy
        cand1 = (mx + px, my + py)
        cand2 = (mx - px, my - py)
        d1 = (cand1[0] - rx) ** 2 + (cand1[1] - ry) ** 2
        d2 = (cand2[0] - rx) ** 2 + (cand2[1] - ry) ** 2
        sel = cand1 if d1 > d2 else cand2
        return (sel[0], sel[1], 0.0)

    def _publish_current_nav(self):
        with self._lock:
            if not self._nav_active or self._nav_idx >= len(self._nav_plan):
                return
            x, y, th = self._nav_plan[self._nav_idx]
        now = time.time()
        if now - self._last_pub_time > 0.3:
            self._publish_waypoint_xy(x, y, heading_deg=math.degrees(th))
            self._last_pub_time = now

    def _nav_tick(self, _event):
        if not self._nav_active:
            return
        rx, ry = self._vehicle_xy
        with self._lock:
            if self._nav_idx >= len(self._nav_plan):
                self._nav_active = False
                return
            x, y, th = self._nav_plan[self._nav_idx]
        dist = math.hypot(x - rx, y - ry)
        if dist <= self._reach_dist:
            with self._lock:
                self._nav_idx += 1
                done = self._nav_idx >= len(self._nav_plan)
            if done:
                rospy.loginfo("Navigation plan completed")
                self._nav_active = False
                return
        self._publish_current_nav()


if __name__ == "__main__":
    try:
        node = QANode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
