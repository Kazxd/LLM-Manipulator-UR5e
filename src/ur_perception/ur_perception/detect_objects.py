import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from ur_interfaces.srv import DetectObjects
from ur_perception.yaw import yaw_from_pixels, yaw_to_quat

CAM_POS = np.array([0.6, 0.0, 1.2])   # must match the world file
MIN_AREA = 30
TOP_BAND = 0.012                      # depth pixels within this of the nearest ones (the top face) are used for the yaw
COLORS = ["red", "green", "blue"]

def color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    if color == "red":   return (r > 100) & (r > 2 * g) & (r > 2 * b)
    if color == "green": return (g > 100) & (g > 2 * r) & (g > 2 * b)
    return (b > 100) & (b > 2 * r) & (b > 2 * g)

class DetectService(Node):
    def __init__(self):
        super().__init__("detect_objects",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.br = CvBridge()
        self.rgb = self.depth = self.info = None
        self.create_subscription(CameraInfo, "/overhead/camera_info", lambda m: setattr(self, "info", m), 1)
        self.create_subscription(Image, "/overhead/image", lambda m: setattr(self, "rgb", m), 1)
        self.create_subscription(Image, "/overhead/depth_image", lambda m: setattr(self, "depth", m), 1)
        self.create_service(DetectObjects, "/detect_objects", self.on_request)

    def on_request(self, req, res):
        if self.rgb is None or self.depth is None or self.info is None:
            res.success, res.message = False, "no camera data yet"
            return res
        want = COLORS if req.color.lower() in ("", "all") else [req.color.lower()]
        if any(c not in COLORS for c in want):
            res.success, res.message = False, f"unknown color '{req.color}'"
            return res
        rgb = self.br.imgmsg_to_cv2(self.rgb, "rgb8")
        depth = self.br.imgmsg_to_cv2(self.depth, "32FC1")
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        for c in want:
            mask = color_mask(rgb, c).astype(np.uint8)
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
            k = 0
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                    continue
                comp = labels == i
                d = depth[comp]
                d = d[np.isfinite(d) & (d > 0)]
                if len(d) < MIN_AREA:
                    continue
                u, v = cents[i]
                dd = float(np.percentile(d, 20))
                y_left = -(u - cx) * dd / fx
                z_up = -(v - cy) * dd / fy
                p = np.array([z_up, y_left, -dd]) + CAM_POS
                # yaw of the cube from its top face only (the side faces seen in perspective would bias it)
                top = comp & np.isfinite(depth) & (depth > 0) & (depth <= dd + TOP_BAND)
                top_v, top_u = np.nonzero(top)
                yaw = yaw_from_pixels(top_u, top_v)
                _, _, qz, qw = yaw_to_quat(yaw)
                ps = PoseStamped()
                ps.header.frame_id = "base_link"
                ps.header.stamp = self.depth.header.stamp
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
                ps.pose.orientation.z, ps.pose.orientation.w = float(qz), float(qw)   # rotation about z, in (-45, 45] deg
                res.poses.append(ps)
                res.ids.append(f"{c}_{k}")
                k += 1
        res.success = len(res.poses) > 0
        res.message = f"found {len(res.poses)} object(s)" if res.success else "nothing found"
        return res

def main():
    rclpy.init()
    rclpy.spin(DetectService())
