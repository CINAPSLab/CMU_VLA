#!/usr/bin/env python3
"""
cinaps_vlm QA node: listens to /challenge_question and responds on
- /selected_object_marker (Marker) for "find" queries
- /numerical_response (Int32) for "how many" queries
- /way_point_with_heading (Pose2D) for simple navigation goals

Uses world_model_node's debug JSON on /world_model/debug as the source of objects.
Features enhanced synonym dictionary and improved parsing based on diagnostics.
"""

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry


def _norm_text(s: str) -> str:
    return s.strip().lower()


class QuestionType(Enum):
    UNKNOWN = "unknown"
    COUNT = "count"
    FIND = "find"
    NAVIGATE = "navigate"


@dataclass
class ParsedQuestion:
    qtype: QuestionType
    target: Optional[str] = None
    color: Optional[str] = None
    relation: Optional[str] = None
    reference: Optional[str] = None
    extreme: Optional[str] = None


class QANode:
    def __init__(self):
        rospy.init_node("cinaps_qa", anonymous=True)

        # Publishers
        self.pub_marker = rospy.Publisher("/selected_object_marker", Marker, queue_size=1, latch=True)
        self.pub_waypoint = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=1)
        self.pub_number = rospy.Publisher("/numerical_response", Int32, queue_size=1)
        self.pub_exploration_request = rospy.Publisher("/exploration/request", String, queue_size=1, latch=True)

        # Subscribers
        self.sub_question = rospy.Subscriber("/challenge_question", String, self._on_question, queue_size=1)
        self.sub_world = rospy.Subscriber("/world_model/debug", String, self._on_world, queue_size=5)
        self.sub_pose = rospy.Subscriber("/state_estimation", Odometry, self._on_pose, queue_size=5)
        self.sub_exploration_state = rospy.Subscriber("/exploration/state", String, self._on_exploration_state, queue_size=1)

        # State
        self._world: Dict[str, Dict] = {}
        self._vehicle_xy: Tuple[float, float] = (0.0, 0.0)
        self._lock = threading.Lock()
        self._exploration_state: str = "unknown"
        self._exploration_wait_timeout = rospy.get_param("~exploration_wait_timeout", 30.0)
        self._answer_sent = False

        # ENHANCED SYNONYM DICTIONARY based on ground truth analysis
        self._synonyms = self._build_enhanced_synonyms()
        self._rev_syn = self._build_reverse_syn(self._synonyms)
        self._color_words = {"red", "blue", "green", "black", "white", "gray", "grey", "yellow", "brown", "pink", "orange", "purple"}

        # Precompiled patterns for improved parsing
        self._count_type_patterns = [
            re.compile(r"\bhow many\b"),
            re.compile(r"\bcount (?:the )?(?:number|amount)\b"),
            re.compile(r"\bnumber of\b"),
            re.compile(r"\bwhat(?:'s| is) the count\b"),
        ]
        self._find_type_patterns = [
            re.compile(r"\bfind\b"),
            re.compile(r"\bwhere (?:is|are)\b"),
            re.compile(r"\blocate\b"),
            re.compile(r"\bshow (?:me|the)\b"),
            re.compile(r"\bpoint (?:to|at)\b"),
        ]
        self._navigate_type_patterns = [
            re.compile(r"\bgo to\b"),
            re.compile(r"\bnavigate to\b"),
            re.compile(r"\bmove to\b"),
            re.compile(r"\bdrive to\b"),
            re.compile(r"\bhead to\b"),
            re.compile(r"\btake me to\b"),
        ]

        self._navigate_target_patterns = [
            re.compile(r"(?:go|navigate|move|drive|head|take me) to (?:the |a )?([a-z ]+?)(?:\s|$|\?)"),
        ]

        self._count_target_patterns = [
            re.compile(r"how many ([a-z ]+?)(?:\s+are|\s+is|\?|$)"),
            re.compile(r"count (?:the )?(?:number of )?([a-z ]+?)(?:\s+(?:that|which|with)|\?|$)"),
            re.compile(r"number of ([a-z ]+?)(?:\s+(?:that|which|with)|\?|$)"),
        ]
        self._find_target_patterns = [
            re.compile(r"find (?:the |a )?([a-z ]+?)(?:\s+(?:that|which|closest|nearest|farthest|furthest)|\?|$)"),
            re.compile(r"where (?:is|are) (?:the |a )?([a-z ]+?)(?:\s|$|\?)"),
            re.compile(r"locate (?:the |a )?([a-z ]+?)(?:\s|$|\?)"),
            re.compile(r"show (?:me |the )?([a-z ]+?)(?:\s|$|\?)"),
            re.compile(r"point (?:me )?(?:to|at) (?:the |a )?([a-z ]+?)(?:\s|$|\?)"),
        ]

        self._relation_patterns = [
            ("on", re.compile(r"(?:on top of|on) (?:the |a )?([a-z ]+?)(?:\s|$|\?)")),
            ("above", re.compile(r"above (?:the |a )?([a-z ]+?)(?:\s|$|\?)")),
            ("below", re.compile(r"(?:below|under|beneath) (?:the |a )?([a-z ]+?)(?:\s|$|\?)")),
            ("near", re.compile(r"(?:near|next to|beside) (?:the |a )?([a-z ]+?)(?:\s|$|\?)")),
            ("in", re.compile(r"(?:in|inside|within) (?:the |a )?([a-z ]+?)(?:\s|$|\?)")),
        ]

        self._extreme_keywords = {
            "closest": "closest",
            "nearest": "closest",
            "most nearby": "closest",
            "farthest": "farthest",
            "furthest": "farthest",
        }

        rospy.loginfo("QA node ready with enhanced synonyms")

    def _build_enhanced_synonyms(self) -> Dict[str, List[str]]:
        """Enhanced synonym dictionary based on diagnostic findings"""
        return {
            # Furniture (existing + new)
            "sofa": ["sofa", "couch"],
            "pillow": ["pillow", "cushion"],
            "chair": ["chair"],
            "stool": ["stool"],
            "bench": ["bench", "bed bench"],
            "bed": ["bed"],
            "table": ["table", "coffee table", "dining table", "tea table", "small table", "side table", "round table"],
            "nightstand": ["nightstand", "bedside table", "night stand"],
            "cabinet": ["cabinet", "file cabinet", "tv cabinet", "kitchen cabinet", "bathroom cabinet", "sink cabinet"],
            "tv cabinet": ["tv cabinet", "tv stand", "tv table"],
            "shelf": ["shelf", "bookcase"],
            "dressing table": ["dressing table", "dresser"],
            "wardrobe": ["wardrobe", "wardrobe doors"],
            "ottoman": ["ottoman"],
            
            # Electronics & Appliances (enhanced)
            "tv": ["tv", "television"],
            "microwave": ["microwave"],
            "refrigerator": ["refrigerator", "fridge", "refridgerator"],
            "computer monitor": ["computer monitor", "monitor"],
            "projector screen": ["projector screen", "screen"],
            "speaker": ["speaker"],
            "air conditioner": ["air conditioner"],
            "oven": ["oven"],
            "stove": ["stove", "cooker"],
            "range hood": ["range hood"],
            
            # Lighting (new from diagnostics)
            "lamp": ["lamp", "lantern", "ceiling lamp", "bedroom light", "kitchen light", "circular light", "ceiling light"],
            "focus light": ["focus light"],
            
            # Decor & Objects (enhanced)
            "vase": ["vase"],
            "bowl": ["bowl"],
            "cup": ["cup", "paper cup"],
            "kettle": ["kettle"],
            "tray": ["tray"],
            "book": ["book", "books"],
            "picture": ["picture", "painting", "photo"],
            "clock": ["clock"],
            "mirror": ["mirror"],
            "curtain": ["curtain", "curtains"],
            
            # Plants (new)
            "plant": ["plant", "potted plant", "potted bamboo", "potted branch", "palm", "tree", "bamboo"],
            "flowers": ["flowers", "flower"],
            
            # Kitchen items (new)
            "knife": ["knife", "kitchen knife"],
            "fork": ["fork"],
            "spoon": ["spoon"],
            "plate": ["plate", "dish"],
            "cutting board": ["cutting board"],
            "knife rack": ["knife rack"],
            "bottle": ["bottle", "beer bottle", "oil bottle", "spice jar"],
            
            # Containers & Storage (new)
            "trash can": ["trash can", "trashcan", "bin"],
            "box": ["box", "cube"],
            "drawer": ["drawer"],
            "cupboard": ["cupboard"],
            
            # Floor & Wall items (new)  
            "carpet": ["carpet", "rug"],
            "window": ["window", "windows"],
            "door": ["door", "balcony door", "outside door", "kitchen door"],
            "door frame": ["door frame", "doorframe", "kitchen door frame"],
            "wall": ["wall", "exterior walls", "partition wall"],
            "ceiling": ["ceiling", "celling"],
            "floor": ["floor"],
            
            # Bathroom items (new)
            "toilet": ["toilet"],
            "sink": ["sink"],
            "shower": ["shower"],
            "shower head": ["shower head"],
            "towel": ["towel"],
            "toilet paper": ["toilet paper"],
            "toilet glass": ["toilet glass"],
            
            # Other (new from diagnostics)
            "mattress": ["mattress"],
            "quilt": ["quilt"],
            "blanket": ["blanket"],
            "slipper": ["slipper"],
            "shoes": ["shoes"],
            "umbrella": ["umbrella"],
            "sculpture": ["sculpture", "pottery"],
            "ashtray": ["ashtray"],
            "desk": ["desk"],
            
            # Decorative items (new)
            "figurine": ["figurine", "elephant figurine", "horse figurine"],
            "decoration": ["decoration", "tower decoration", "symbol decoraion", "circle decoration"],
            "folding screen": ["folding screen"],
            "partition": ["partition"],
            
            # Arabic room specific
            "hookah": ["hookah"],
            "arabic jar": ["arabic jar", "jar"],
            "coffee pot": ["coffee pot"],
            
            # Chinese room specific  
            "tea table": ["tea table"],
            "chopsticks": ["chopsticks"],
            "eye glasses": ["eye glasses"],
            "tv remote": ["tv remote"],
            
            # Home items
            "dumbbell": ["dumbbell"],
            "face cream": ["face cream"],
            "lotion": ["lotion"],
            "dvd player": ["dvd player"],
            "notecards": ["notecards"],
            "tablecloth": ["tablecloth"],
            "light switch": ["light switch"],
        }

    def _build_reverse_syn(self, syn: Dict[str, List[str]]) -> Dict[str, str]:
        rev = {}
        for canon, words in syn.items():
            for w in words:
                rev[_norm_text(w)] = canon
        return rev

    def _canonical_label(self, label: str) -> str:
        l = _norm_text(label)
        l = self._singularize(l)
        return self._rev_syn.get(l, l)

    def _singularize(self, w: str) -> str:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    def _parse_question(self, raw: str) -> ParsedQuestion:
        q = _norm_text(raw)
        qtype = QuestionType.UNKNOWN

        if any(pat.search(q) for pat in self._count_type_patterns):
            qtype = QuestionType.COUNT
        elif any(pat.search(q) for pat in self._find_type_patterns):
            qtype = QuestionType.FIND
        elif any(pat.search(q) for pat in self._navigate_type_patterns):
            qtype = QuestionType.NAVIGATE

        if qtype is QuestionType.UNKNOWN and q.startswith("how many"):
            qtype = QuestionType.COUNT

        target = None
        if qtype is QuestionType.COUNT:
            target = self._match_target(q, self._count_target_patterns)
        elif qtype is QuestionType.FIND:
            target = self._match_target(q, self._find_target_patterns)
        elif qtype is QuestionType.NAVIGATE:
            target = self._match_target(q, self._navigate_target_patterns) or self._match_target(q, self._find_target_patterns)

        color = self._extract_color(q)
        relation, reference = self._extract_relation(q)
        extreme = self._extract_extreme(q)

        if target:
            target = self._singularize(target)

        return ParsedQuestion(
            qtype=qtype,
            target=target,
            color=color,
            relation=relation,
            reference=reference,
            extreme=extreme,
        )

    def _match_target(self, q: str, patterns: List[re.Pattern]) -> Optional[str]:
        for pat in patterns:
            m = pat.search(q)
            if m:
                return m.group(1).strip()
        return None

    def _extract_color(self, q: str) -> Optional[str]:
        for color in sorted(self._color_words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(color)}\b", q):
                return color
        return None

    def _extract_extreme(self, q: str) -> Optional[str]:
        for key, value in self._extreme_keywords.items():
            if key in q:
                return value
        return None

    def _extract_relation(self, q: str) -> Tuple[Optional[str], Optional[str]]:
        for rel, pat in self._relation_patterns:
            m = pat.search(q)
            if m:
                ref = m.group(1).strip()
                return rel, self._canonical_label(ref)
        return None, None

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

    def _on_exploration_state(self, msg: String):
        try:
            self._exploration_state = msg.data.strip().lower()
        except Exception:
            self._exploration_state = "unknown"

    def _on_question(self, msg: String):
        raw_q = msg.data.strip()
        q_norm = _norm_text(raw_q)
        rospy.loginfo(f"Received question: {q_norm}")

        self._ensure_exploration_before_answer()

        parsed = self._parse_question(raw_q)

        if parsed.qtype is QuestionType.COUNT:
            self._answer_how_many(raw_q, parsed)
            return
        if parsed.qtype is QuestionType.FIND:
            self._answer_find(raw_q, parsed)
            return
        if parsed.qtype is QuestionType.NAVIGATE:
            self._answer_navigate(raw_q, parsed)
            return

        # Fallback to legacy heuristics if parsing failed
        if self._is_numerical_question(q_norm):
            legacy = ParsedQuestion(
                qtype=QuestionType.COUNT,
                target=self._extract_numerical_target(q_norm),
                color=self._extract_color(q_norm),
            )
            self._answer_how_many(raw_q, legacy)
        elif self._is_find_question(q_norm):
            legacy = ParsedQuestion(
                qtype=QuestionType.FIND,
                target=self._match_target(q_norm, self._find_target_patterns),
                color=self._extract_color(q_norm),
            )
            self._answer_find(raw_q, legacy)
        else:
            self._answer_navigate(raw_q, ParsedQuestion(qtype=QuestionType.NAVIGATE))

    def _ensure_exploration_before_answer(self):
        if self._exploration_state == "complete":
            return

        self.pub_exploration_request.publish(String(data="start"))
        rospy.loginfo("Exploration requested prior to answering question")

        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self._exploration_state == "complete":
                rospy.loginfo("Exploration complete; proceeding with question answer")
                return
            elapsed = rospy.Time.now().to_sec() - start_time
            if elapsed > self._exploration_wait_timeout:
                rospy.logwarn("Exploration timeout reached; answering with current world model")
                return
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def _is_numerical_question(self, q: str) -> bool:
        """Enhanced numerical question detection"""
        patterns = [
            r"how many",
            r"count the number",
            r"how much",
            r"what is the count"
        ]
        return any(re.search(pattern, q) for pattern in patterns)
    
    def _is_find_question(self, q: str) -> bool:
        """Enhanced find question detection"""
        patterns = [
            r"find",
            r"where is",
            r"locate",
            r"show me",
            r"identify"
        ]
        return any(re.search(pattern, q) for pattern in patterns)

    def _extract_numerical_target(self, q: str) -> Optional[str]:
        """Enhanced target extraction for numerical questions"""
        for pattern in self._count_target_patterns:
            match = pattern.search(q)
            if match:
                target = match.group(1).strip()
                # Handle complex targets like "chairs with pillows on them"
                if " with " in target:
                    # Extract main object before "with"
                    target = target.split(" with ")[0].strip()
                return self._singularize(target)
        return None

    def _current_objects(self) -> List[Dict]:
        with self._lock:
            return list(self._world.values()) if isinstance(self._world, dict) else []

    def _filter_by_label(self, objs: List[Dict], label: str) -> List[Dict]:
        if not label:
            return objs
        lab = self._canonical_label(label)
        res = []
        for o in objs:
            c = self._canonical_label(o.get("class", ""))
            if lab in c or c in lab:  # Bidirectional matching
                res.append(o)
        return res

    def _answer_how_many(self, original_q: str, parsed: ParsedQuestion):
        """Enhanced numerical answer with better parsing"""
        objs = self._current_objects()
        target = parsed.target or self._extract_numerical_target(_norm_text(original_q))

        if not target:
            rospy.logwarn(f"Could not extract target from numerical question: {original_q}")
            self.pub_number.publish(Int32(data=0))
            self._finish_query()
            return

        rospy.loginfo(f"Extracted target: '{target}' from question: '{original_q}'")

        # Apply label filter
        matched = self._filter_by_label(objs, target)

        # Apply color filter if present
        color = parsed.color
        if color:
            matched = self._filter_by_color(matched, color)

        # Apply spatial filters
        matched = self._apply_enhanced_spatial_filters(original_q, matched, objs, parsed)

        count = len(matched)
        rospy.loginfo(f"Final count for '{target}': {count}")
        self.pub_number.publish(Int32(data=count))
        self._finish_query()

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

    def _apply_enhanced_spatial_filters(self, q: str, candidates: List[Dict], all_objs: List[Dict], parsed: Optional[ParsedQuestion] = None) -> List[Dict]:
        """Enhanced spatial filtering with better pattern matching"""
        try:
            from cinaps_vlm.reasoning import relations as R
        except Exception:
            return candidates

        filtered = candidates

        # First use structured parse if available
        if parsed and parsed.relation and parsed.reference:
            filtered = self._filter_by_relation(filtered, all_objs, parsed.reference, parsed.relation, R)
            return filtered

        ql = _norm_text(q)
        for rel_name, pattern in self._relation_patterns:
            m = pattern.search(ql)
            if not m:
                continue
            ref_label = self._canonical_label(m.group(1).strip())
            tmp = self._filter_by_relation(filtered, all_objs, ref_label, rel_name, R)
            if tmp is not None:
                filtered = tmp
                break

        return filtered

    def _filter_by_relation(
        self,
        candidates: List[Dict],
        all_objs: List[Dict],
        reference_label: str,
        relation: str,
        R,
    ) -> List[Dict]:
        refs = self._filter_by_label(all_objs, reference_label)
        if not refs:
            return candidates

        relation = relation.lower()
        checker_map = {
            "on": R.is_on,
            "above": R.is_above,
            "below": R.is_below,
            "under": R.is_below,
            "near": R.is_near,
            "next to": R.is_near,
            "beside": R.is_near,
            "in": R.is_in,
            "inside": R.is_in,
            "within": R.is_in,
        }

        fn = checker_map.get(relation)
        if fn is None:
            return candidates

        result = []
        for cand in candidates:
            for ref in refs:
                try:
                    if fn(cand, ref):
                        result.append(cand)
                        break
                except Exception:
                    continue

        return result or candidates

    def _answer_find(self, original_q: str, parsed: ParsedQuestion):
        """Enhanced find with better object selection"""
        objs = self._current_objects()

        target = parsed.target or self._match_target(_norm_text(original_q), self._find_target_patterns)
        if not target:
            rospy.logwarn("Could not extract target from find question")
            self._finish_query()
            return
        
        matched = self._filter_by_label(objs, target) or objs

        if parsed.color:
            matched = self._filter_by_color(matched, parsed.color)
        
        # Apply spatial filters and ranking
        matched = self._apply_enhanced_spatial_filters(original_q, matched, objs, parsed)
        
        if parsed.extreme == "farthest":
            best = self._farthest(matched)
        else:
            best = self._closest(matched)
        if best is None:
            rospy.logwarn("No objects available to find.")
            self._finish_query()
            return
            
        rospy.loginfo(f"Selected: {best.get('class','object')}")
        self._publish_marker(best)
        self._publish_waypoint(best)

    def _closest(self, objs: List[Dict]) -> Optional[Dict]:
        if not objs:
            return None
        vx, vy = self._vehicle_xy

        def dist2(o: Dict) -> float:
            try:
                center = o.get("center", [0, 0, 0])
                cx, cy = float(center[0]), float(center[1])
                dx, dy = cx - vx, cy - vy
                return dx * dx + dy * dy
            except Exception:
                return float('inf')
        return min(objs, key=dist2)

    def _farthest(self, objs: List[Dict]) -> Optional[Dict]:
        if not objs:
            return None
        vx, vy = self._vehicle_xy

        def dist2(o: Dict) -> float:
            try:
                center = o.get("center", [0, 0, 0])
                cx, cy = float(center[0]), float(center[1])
                dx, dy = cx - vx, cy - vy
                return dx * dx + dy * dy
            except Exception:
                return float('-inf')

        return max(objs, key=dist2)

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
        finally:
            self._finish_query()

    def _answer_navigate(self, q: str, parsed: ParsedQuestion):
        """Answer navigation questions by selecting a waypoint near the requested object."""
        objs = self._current_objects()
        target = parsed.target or self._match_target(_norm_text(q), self._navigate_target_patterns)

        if not target and parsed.reference:
            target = parsed.reference

        if not target:
            rospy.logwarn(f"Navigation question lacks a recognizable target: {q}")
            self._finish_query()
            return

        candidates = self._filter_by_label(objs, target)
        if parsed.color:
            candidates = self._filter_by_color(candidates, parsed.color)

        candidates = self._apply_enhanced_spatial_filters(q, candidates, objs, parsed)

        if not candidates:
            rospy.logwarn(f"No navigation candidates match target '{target}'")
            self._finish_query()
            return

        if parsed.extreme == "farthest":
            goal_obj = self._farthest(candidates)
        else:
            goal_obj = self._closest(candidates)

        if goal_obj is None:
            rospy.logwarn("Navigation target selection failed")
            self._finish_query()
            return

        vx, vy = self._vehicle_xy
        center = goal_obj.get("center", [0.0, 0.0, 0.0])
        try:
            cx, cy = float(center[0]), float(center[1])
        except Exception:
            rospy.logwarn("Selected navigation object has invalid center")
            return

        heading = math.degrees(math.atan2(cy - vy, cx - vx))
        rospy.loginfo(f"Publishing navigation goal for '{goal_obj.get('class', target)}' at ({cx:.2f}, {cy:.2f})")
        self._publish_marker(goal_obj)
        self._publish_waypoint(goal_obj, heading)

    def _finish_query(self):
        if self._answer_sent:
            return
        self._answer_sent = True
        try:
            self.pub_exploration_request.publish(String(data="stop"))
        except Exception:
            pass
        rospy.loginfo("QA node answered question; shutting down.")
        rospy.signal_shutdown("question answered")

    def spin(self):
        rospy.loginfo("Improved QA node spinning")
        rospy.spin()


if __name__ == "__main__":
    try:
        node = QANode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
