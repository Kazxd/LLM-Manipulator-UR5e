"""Action server exposing /move_home, /pick_object, /place_object.

Uses a 'magic gripper': a held cube is teleported to follow the tool in Gazebo.
Only one skill runs at a time; a second goal is rejected while busy.
Services: /reset_state (clear held state + go home), /get_status (what is held).
Collision scene: the table, plus the cubes (re-detected after going home) except the held/picked one.
Disable with  --ros-args -p collision_scene:=false
"""
import math, time, threading, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from pymoveit2 import MoveIt2
from ur_interfaces.srv import DetectObjects
from std_srvs.srv import Trigger
from ur_interfaces.action import MoveHome, Pick, Place

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
DOWN = [1.0, 0.0, 0.0, 0.0]      # tool z pointing down (x, y, z, w)
TCP = 0.15                        # virtual fingertip distance below tool0
CUBE = 0.05
APPROACH = 0.12                   # hover height above grasp/place
TABLE_TOP = 0.20                  # must match the world file
SHOULDER_Z = 0.163                # UR5e shoulder height
REACH = 0.92                      # loose sanity gate only; MoveIt is the real judge
RELEASE_CLEARANCE = 0.005         # lower the cube with 5 mm to spare, never start interpenetrating
CUBE_MARGIN = 0.01                # collision boxes for cubes are inflated by this (total, all axes)
COLORS = ("red", "green", "blue")
IDENT_QUAT = (0.0, 0.0, 0.0, 1.0)


def reachable(x, y, z):
    """Rough UR5e reach check for a downward-pointing tool at (x, y, z)."""
    wrist_dz = z + 0.10 - SHOULDER_Z
    return math.sqrt(x * x + y * y + wrist_dz ** 2) <= REACH and math.hypot(x, y) > 0.25


class SkillServer(Node):
    def __init__(self):
        super().__init__("skill_server", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.cb = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(node=self, joint_names=JOINTS, base_link_name="base_link",
                               end_effector_name="tool0", group_name="ur_manipulator",
                               callback_group=self.cb)
        self.moveit2.max_velocity = 0.2
        self.moveit2.max_acceleration = 0.2
        for attr, val in (("allowed_planning_time", 5.0), ("num_planning_attempts", 10)):
            try:
                setattr(self.moveit2, attr, val)
            except Exception:
                pass
        self.detect_cli = self.create_client(DetectObjects, "/detect_objects", callback_group=self.cb)
        self.pose_cli = self.create_client(SetEntityPose, "/world/pick_place/set_pose", callback_group=self.cb)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.held = None
        self.held_lock = threading.Lock()
        self.pose_future = None
        self.active = False
        self.lock = threading.Lock()
        self.collision_scene = self.declare_parameter("collision_scene", True).value
        self.create_timer(0.03, self.follow_tool, callback_group=self.cb)

        for cls, name, fn in ((MoveHome, "move_home", self.do_home),
                              (Pick, "pick_object", self.do_pick),
                              (Place, "place_object", self.do_place)):
            ActionServer(self, cls, name,
                         execute_callback=lambda gh, c=cls, f=fn: self.run(gh, c, f),
                         goal_callback=self.on_goal,
                         cancel_callback=lambda gh: CancelResponse.ACCEPT,
                         callback_group=self.cb)
        self.create_service(Trigger, "reset_state", self.on_reset, callback_group=self.cb)
        self.create_service(Trigger, "get_status", self.on_status, callback_group=self.cb)
        self.get_logger().info(
            f"skill server ready: /move_home /pick_object /place_object /reset_state /get_status "
            f"(collision_scene={self.collision_scene})")

    # ---------- action plumbing ----------
    def on_goal(self, goal):
        with self.lock:
            if self.active:
                return GoalResponse.REJECT
            self.active = True
            return GoalResponse.ACCEPT

    def on_reset(self, req, res):
        """Forget any held object and go home (used by the benchmark / manual recovery)."""
        with self.lock:
            if self.active:
                res.success, res.message = False, "a skill is running"
                return res
            self.active = True
        try:
            with self.held_lock:
                self.held = None
            self.add_table()
            ok = self.home()
            res.success = ok
            res.message = "state cleared, at home" if ok else "state cleared but failed to move home"
        finally:
            self.active = False
        return res

    def on_status(self, req, res):
        held = self.held
        res.success = True
        res.message = f"holding {held.replace('_cube', '')}" if held else "holding nothing"
        return res

    def stage(self, gh, text):
        self.get_logger().info(text)
        msg = self._fb_cls()
        msg.stage = text
        gh.publish_feedback(msg)
        return not gh.is_cancel_requested

    def run(self, gh, cls, fn):
        self._fb_cls = cls.Feedback
        cancelled = False
        try:
            ok, msg = fn(gh)
            cancelled = (msg == "cancelled")
        except Exception as e:
            ok, msg = False, f"exception: {e}"
        finally:
            self.active = False
        res = cls.Result()
        res.success, res.message = ok, msg
        if cancelled:
            gh.canceled()
        elif ok:
            gh.succeed()
        else:
            gh.abort()
        self.get_logger().info(f"result: {ok} - {msg}")
        return res

    # ---------- magic gripper ----------
    def _send_pose(self, name, x, y, z):
        req = SetEntityPose.Request()
        req.entity.name, req.entity.type = name, Entity.MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, z
        req.pose.orientation.w = 1.0
        return self.pose_cli.call_async(req)

    def follow_tool(self):
        with self.held_lock:
            if not self.held:
                return
            if self.pose_future is not None and not self.pose_future.done():
                return                      # previous teleport still in flight: don't pile up
            try:
                t = self.tf_buf.lookup_transform("base_link", "tool0", rclpy.time.Time()).transform.translation
            except Exception:
                return
            self.pose_future = self._send_pose(self.held, t.x, t.y, t.z - TCP)

    def release_cube(self, x, y, z):
        """Stop following, let any in-flight teleport finish, then put the cube at its exact rest pose."""
        with self.held_lock:
            name, self.held = self.held, None
        if self.pose_future is not None:
            self.wait(self.pose_future, 2.0)
        if name:
            self.wait(self._send_pose(name, x, y, z), 2.0)

    # ---------- helpers ----------
    def wait(self, fut, timeout=5.0):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def locate_all(self):
        """{colour: (x, y, top_z)} for every visible cube (first detection per colour)."""
        req = DetectObjects.Request(); req.color = "all"
        res = self.wait(self.detect_cli.call_async(req))
        if res is None or not res.success:
            return {}
        out = {}
        for i, p in zip(res.ids, res.poses):
            out.setdefault(i.split("_")[0], (p.pose.position.x, p.pose.position.y, p.pose.position.z))
        return out

    def add_table(self):
        if not self.collision_scene:
            return
        try:
            self.moveit2.add_collision_box(id="table", size=(0.6, 0.8, 0.2), position=(0.6, 0.0, 0.1),
                                           quat_xyzw=IDENT_QUAT, frame_id="base_link")
        except Exception as e:
            self.get_logger().warn(f"could not add table to the collision scene: {e}")

    def sync_cubes(self, objs, exclude=()):
        """Cubes as obstacles, except the held one and the `exclude` colours. Stale boxes are removed first."""
        if not self.collision_scene:
            return
        held = (self.held or "").replace("_cube", "")
        try:
            for c in COLORS:
                self.moveit2.remove_collision_object(id=f"{c}_cube")
                if c in exclude or c == held or c not in objs:
                    continue
                x, y, top = objs[c]
                self.moveit2.add_collision_box(id=f"{c}_cube", size=(CUBE + CUBE_MARGIN,) * 3,
                                               position=(x, y, top - CUBE / 2),
                                               quat_xyzw=IDENT_QUAT, frame_id="base_link")
            time.sleep(0.3)            # scene updates are published asynchronously
        except Exception as e:
            self.get_logger().warn(f"could not update cubes in the collision scene: {e}")

    def z_hint(self, top):
        """Explain a suspicious detected height (usually an outdated world file)."""
        expected = TABLE_TOP + CUBE
        if abs(top - expected) > 0.05:
            return (f" [detected top z={top:.2f} m but expected about {expected:.2f} m: "
                    f"the world file in use probably has a different table height - rebuild and restart the sim]")
        return ""

    def go(self, xyz, cartesian=False):
        self.moveit2.move_to_pose(position=list(xyz), quat_xyzw=DOWN, cartesian=cartesian)
        return bool(self.moveit2.wait_until_executed())

    def home(self):
        self.moveit2.move_to_configuration(HOME)
        return bool(self.moveit2.wait_until_executed())

    # ---------- skills ----------
    def do_home(self, gh):
        self.add_table()
        self.stage(gh, "moving home")
        return (True, "at home") if self.home() else (False, "failed to move home")

    def do_pick(self, gh):
        color = gh.request.color.lower()
        if color not in COLORS:
            return False, f"unknown color '{color}', use one of {COLORS}"
        if self.held:
            return False, f"already holding {self.held}; place it first"
        self.add_table()
        if not self.stage(gh, "moving home"): return False, "cancelled"
        if not self.home(): return False, "failed to move home"
        if not self.stage(gh, f"detecting {color}"): return False, "cancelled"
        objs = self.locate_all()
        if color not in objs:
            return False, f"no {color} object detected"
        self.sync_cubes(objs, exclude=(color,))
        x, y, top = objs[color]
        self.get_logger().info(f"{color} detected at ({x:.3f}, {y:.3f}), top z={top:.3f}")
        hint = self.z_hint(top)
        grasp_z = (top - CUBE / 2) + TCP
        pre_z = grasp_z + APPROACH
        if not reachable(x, y, pre_z):
            return False, (f"{color} object at ({x:.2f},{y:.2f}) is out of the arm's reach "
                           f"(hover z={pre_z:.2f}){hint}")
        steps = (("moving above object", [x, y, pre_z], False),
                 ("descending", [x, y, grasp_z], True))
        for text, xyz, cart in steps:
            if not self.stage(gh, text): return False, "cancelled"
            if not self.go(xyz, cart): return False, f"motion failed while {text}"
        with self.held_lock:
            self.held = f"{color}_cube"
        time.sleep(0.5)
        if not self.stage(gh, "lifting"): return False, "cancelled"
        if not self.go([x, y, pre_z], True): return False, "motion failed while lifting"
        return True, f"picked {color} object"

    def do_place(self, gh):
        if not self.held:
            return False, "not holding anything"
        r = gh.request
        target = r.target.lower()
        if target and target not in COLORS:
            return False, f"unknown target '{target}'"
        hint = ""
        self.add_table()
        if not self.stage(gh, "moving home"): return False, "cancelled"
        if not self.home(): return False, "failed to move home"
        objs = {}
        if target or self.collision_scene:
            if not self.stage(gh, f"detecting {target or 'objects'}"): return False, "cancelled"
            objs = self.locate_all()
            self.sync_cubes(objs)
        if target:
            if target not in objs:
                return False, f"no {target} object detected to place on"
            x, y, surface = objs[target]
            self.get_logger().info(f"{target} detected at ({x:.3f}, {y:.3f}), top z={surface:.3f}")
            hint = self.z_hint(surface)
        else:
            x, y = r.x, r.y
            surface = r.surface_z if r.surface_z > 0 else TABLE_TOP
        place_z = (surface + CUBE / 2) + TCP + RELEASE_CLEARANCE
        pre_z = place_z + APPROACH
        if not reachable(x, y, pre_z):
            return False, (f"place position ({x:.2f},{y:.2f}) is out of the arm's reach "
                           f"(hover z={pre_z:.2f}){hint}")
        steps = (("moving above target", [x, y, pre_z], False),
                 ("descending", [x, y, place_z], True))
        for text, xyz, cart in steps:
            if not self.stage(gh, text): return False, "cancelled"
            if not self.go(xyz, cart): return False, f"motion failed while {text}"
        self.release_cube(x, y, surface + CUBE / 2 + 0.002)   # exact rest pose, no drop
        time.sleep(0.5)
        if not self.stage(gh, "retreating"): return False, "cancelled"
        if not self.go([x, y, pre_z], True): return False, "placed, but retreat failed"
        return True, "placed"


def main():
    rclpy.init()
    node = SkillServer()
    ex = MultiThreadedExecutor(8)
    ex.add_node(node)
    ex.spin()
