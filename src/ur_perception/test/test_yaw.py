"""Run: python3 src/ur_perception/test/test_yaw.py   (needs numpy + opencv, no ROS)"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ur_perception.yaw import yaw_from_pixels, yaw_to_quat, quat_to_yaw

CX, CY, F, D = 160.0, 120.0, 277.0, 0.95        # roughly the simulated camera (320x240, 1.05 rad fov, 0.95 m to the cube top)


def project(x, y):
    """world (x, y) on the cube top plane -> pixel (u, v), the inverse of detect_objects.py."""
    return CX - y * F / D, CY - x * F / D


def square_pixels(yaw, side=0.05, cx=0.6, cy=0.0):
    """Pixels covered by a cube top of the given world yaw (dense sampling)."""
    us, vs = [], []
    c, s = math.cos(yaw), math.sin(yaw)
    for a in np.linspace(-side / 2, side / 2, 25):
        for b in np.linspace(-side / 2, side / 2, 25):
            x, y = cx + a * c - b * s, cy + a * s + b * c
            u, v = project(x, y)
            us.append(u); vs.append(v)
    return np.round(us), np.round(vs)


def diff(a, b):
    """angle difference folded to [-45, 45) degrees (a cube repeats every 90)."""
    d = math.degrees(a - b)
    return (d + 45) % 90 - 45

for deg in (0, 10, 20, 30, 40, -10, -25, -40, 55, 100, -80):
    yaw = math.radians(deg)
    est = yaw_from_pixels(*square_pixels(yaw))
    assert -math.pi / 4 - 1e-9 <= est < math.pi / 4 + 1e-9, est
    assert abs(diff(est, yaw)) < 4.0, (deg, math.degrees(est))
    # off-centre cubes give the same yaw
    est2 = yaw_from_pixels(*square_pixels(yaw, cx=0.45, cy=0.2))
    assert abs(diff(est2, yaw)) < 4.0, (deg, math.degrees(est2))
assert yaw_from_pixels([1, 2], [1, 2]) == 0.0                        # too few pixels
q = yaw_to_quat(math.radians(30))
assert abs(q[2] - math.sin(math.radians(15))) < 1e-12 and abs(q[3] - math.cos(math.radians(15))) < 1e-12
assert abs(quat_to_yaw(q[2], q[3]) - math.radians(30)) < 1e-12
print("all yaw tests passed")
