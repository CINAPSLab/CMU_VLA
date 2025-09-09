#!/usr/bin/env python3
"""
Minimal geometric relations on axis-aligned 3D boxes.

Object format expectation (dict):
 - center: [x,y,z]
 - size: [sx,sy,sz]

These are intentionally simple; tune thresholds as needed.
"""

from typing import Dict, List, Tuple
import numpy as np


def _bbox_corners(center: List[float], size: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, dtype=float)
    s = np.asarray(size, dtype=float)
    pmin = c - s / 2.0
    pmax = c + s / 2.0
    return pmin, pmax


def is_above(a: Dict, b: Dict, z_gap: float = 0.01) -> bool:
    _, amax = _bbox_corners(a["center"], a["size"])
    bmin, _ = _bbox_corners(b["center"], b["size"])
    return amax[2] + z_gap <= bmin[2]


def is_below(a: Dict, b: Dict, z_gap: float = 0.01) -> bool:
    amin, _ = _bbox_corners(a["center"], a["size"])
    _, bmax = _bbox_corners(b["center"], b["size"])
    return amin[2] >= bmax[2] + z_gap


def is_near(a: Dict, b: Dict, th: float = 0.5) -> bool:
    ca = np.asarray(a["center"], dtype=float)
    cb = np.asarray(b["center"], dtype=float)
    return np.linalg.norm(ca - cb) <= th


def is_in(a: Dict, container: Dict) -> bool:
    amin, amax = _bbox_corners(a["center"], a["size"])
    cmin, cmax = _bbox_corners(container["center"], container["size"])
    return (amin >= cmin).all() and (amax <= cmax).all()


def is_on(a: Dict, support: Dict, z_eps: float = 0.02) -> bool:
    amin, _ = _bbox_corners(a["center"], a["size"])
    _, smax = _bbox_corners(support["center"], support["size"])
    # Contact on z, and XY overlap (very simplified)
    contact = abs(amin[2] - smax[2]) <= z_eps
    a_xy = a["center"][0], a["center"][1]
    s_xy = support["center"][0], support["center"][1]
    return contact and np.linalg.norm(np.array(a_xy) - np.array(s_xy)) < 0.5


def is_between(a: Dict, b: Dict, c: Dict, xy_eps: float = 0.3) -> bool:
    ca = np.asarray(a["center"])[:2]
    cb = np.asarray(b["center"])[:2]
    cc = np.asarray(c["center"])[:2]
    # check projection on line bc
    v = cc - cb
    if np.linalg.norm(v) < 1e-6:
        return False
    t = np.dot(ca - cb, v) / np.dot(v, v)
    proj = cb + t * v
    on_segment = 0.0 < t < 1.0
    dist_xy = np.linalg.norm(ca - proj)
    return on_segment and dist_xy < xy_eps


def closest(a: Dict, candidates: List[Dict], k: int = 1) -> List[int]:
    ca = np.asarray(a["center"], dtype=float)
    dists = [np.linalg.norm(ca - np.asarray(x["center"], dtype=float)) for x in candidates]
    order = np.argsort(dists)
    return [int(order[i]) for i in range(min(k, len(order)))]


def farthest(a: Dict, candidates: List[Dict], k: int = 1) -> List[int]:
    ca = np.asarray(a["center"], dtype=float)
    dists = [np.linalg.norm(ca - np.asarray(x["center"], dtype=float)) for x in candidates]
    order = np.argsort(dists)[::-1]
    return [int(order[i]) for i in range(min(k, len(order)))]

