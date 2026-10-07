"""ROS-free cube yaw estimation from the overhead camera (numpy + OpenCV only, unit-testable).

Image to world mapping used by detect_objects.py (camera looks straight down, see CAM_POS there):
    world x = -(v - cy) * d / fy      (image row v grows toward -x)
    world y = -(u - cx) * d / fx      (image column u grows toward -y)
so an edge with image direction (du, dv) points along world (dx, dy) = (-dv, -du).
"""
import math
import numpy as np
import cv2

MIN_POINTS = 8


def yaw_from_pixels(us, vs):
    """Yaw (rad, folded into [-pi/4, pi/4]) of the square-ish blob made of pixels (us, vs); 0.0 if too few pixels.
    A cube looks the same every 90 degrees, so only the offset from the nearest axis-aligned pose matters.
    Method: convex hull edges, each edge direction (in world coordinates) is multiplied by 4 so that all four sides of a
    square agree, then a length-weighted circular mean. More stable on a 15 px wide cube than minAreaRect."""
    us, vs = np.asarray(us, dtype=np.float32), np.asarray(vs, dtype=np.float32)
    if us.size < MIN_POINTS:
        return 0.0
    pts = np.column_stack([us, vs])
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) < 3:
        return 0.0
    nxt = np.roll(hull, -1, axis=0)
    du, dv = (nxt - hull)[:, 0], (nxt - hull)[:, 1]
    length = np.hypot(du, dv)
    if length.sum() < 1e-6:
        return 0.0
    theta = np.arctan2(-du, -dv)                     # world direction of each hull edge (see module docstring)
    z = np.sum(length * np.exp(4j * theta))
    if abs(z) < 1e-9:
        return 0.0
    return float(np.angle(z) / 4.0)                  # already in [-pi/4, pi/4]


def yaw_to_quat(yaw):
    """(x, y, z, w) of a rotation about world z."""
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def quat_to_yaw(z, w):
    """Inverse of yaw_to_quat for a pure z rotation."""
    return 2.0 * math.atan2(z, w)
