"""Action server exposing /move_home, /pick_object, /place_object.

Uses a 'magic gripper': a held object is teleported to follow the tool in Gazebo.
Only one skill runs at a time; a second goal is rejected while busy.
Services: /reset_state (clear held state, put the extra objects back, go home), /get_status (what is held).

Objects: the cubes red/green/blue (found by the colour detector) and, when `ros2 run ur_perception detect_open`
is running, the extra objects can/ball/block (OBJECTS below). Pick/Place accept any of these names
(Pick.color = "can", Place.target = "block", ...). Cubes are always available; the extra objects need /detect_open.
Collision scene: the table, plus every detected object except the held/picked one.
Disable with  --ros-args -p collision_scene:=false ; ignore /detect_open with  -p open_vocab:=false
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
from ur_interfaces.srv import DetectObjects, DetectOpen
from std_srvs.srv import Trigger
from ur_interfaces.action import MoveHome, Pick, Place

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
DOWN = [1.0, 0.0, 0.0, 0.0]      # tool z pointing down (x, y, z, w)
TCP = 0.15                        # virtual fingertip distance below tool0
CUBE = 0.05                       # height of EVERY object (cubes and extras are all 5 cm tall)
APPROACH = 0.12                   # hover height above grasp/place
TABLE_TOP = 0.20                  # must match the world file
SHOULDER_Z = 0.163                # UR5e shoulder height
REACH = 0.92                      # loose sanity gate only; MoveIt is the real judge
RELEASE_CLEARANCE = 0.005         # lower the object with 5 mm to spare, never start interpenetrating
CUBE_MARGIN = 0.01                # collision boxes are inflated by this (total, all axes)
COLORS = ("red", "green", "blue")
IDENT_QUAT = (0.0, 0.0, 0.0, 1.0)

# Extra objects: key -> (gazebo model, label sent to /detect_open, box size x,y,z for the collision scene).
# Labels describe how the object LOOKS from the overhead camera (a can is a circle). Keep in sync with the SDF
# and with detect_open_check.py.
OBJECTS = {
    "can":   ("yellow_can",  "yellow circle", (0.05, 0.05, 0.05)),
    "ball":  ("purple_ball", "purple ball",   (0.05, 0.05, 0.05)),
    "block": ("white_block", "white block",   (0.04, 0.07, 0.05)),
}
OBJECT_START = {"yellow_can": (0.72, -0.30, 0.225), "purple_ball": (0.38, 0.34, 0.225),
                "white_block": (0.80, 0.00, 0.225)}
NOT_STACKABLE = ("ball",)         # cannot be a place-on-top target
ALL_NAMES = COLORS + tuple(OBJECTS)
DUP_RADIUS = 0.04                 # an open-vocab box this close to a colour-detected cube is that cube, not an object


def canon(name):
    """'Red', 'can', 'yellow can', 'yellow_can' -> 'red' / 'can'; None if unknown."""
    n = (name or "").lower().strip().replace("_", " ")
    if n.endswith(" cube"):
        n = n[:-5]
    if n in COLORS:
        return n
    for key, (model, _, _) in OBJECTS.items():
        if n in (key, model.replace("_", " ")):
            return key
    return None


def model_of(name):
    return f"{name}_cube" if name in COLORS else OBJECTS[name][0]


def size_of(name):
    return (CUBE,) * 3 if name in COLORS else OBJECTS[name][2]


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
        self.open_cli = self.create_client(DetectOpen, "/detect_open", callback_group=self.cb)
        self.pose_cli = self.create_client(SetEntityPose, "/world/pick_place/set_pose", callback_group=self.cb)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.held = None                 # gazebo model name of the held object
        self.held_key = None             # its short name: 'red', 'can', ...
        self.held_lock = threading.Lock()
        self.pose_future = None
        self.active = False
        self.lock = threading.Lock()
        self.open_note = ""              # why the extra objects were not detected (for failure messages)
        self.scene_boxes = {}            # name -> (x, y, top_z) of the object boxes currently in the MoveIt scene
        self.collision_scene = self.declare_parameter("collision_scene", True).value
        self.open_vocab = self.declare_parameter("open_vocab", True).value
        # debugging aid: if a motion fails, remove all object boxes and retry once (hides real collisions!)
        self.retry_no_boxes = self.declare_parameter("retry_no_boxes", False).value
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
            f"(collision_scene={self.collision_scene}, open_vocab={self.open_vocab}, names={ALL_NAMES})")

    # ---------- action plumbing ----------
    def on_goal(self, goal):
        with self.lock:
            if self.active:
                return GoalResponse.REJECT
            self.active = True
            return GoalResponse.ACCEPT

    def on_reset(self, req, res):
        """Forget any held object, put the extra objects back, go home (benchmark / manual recovery)."""
        with self.lock:
            if self.active:
                res.success, res.message = False, "a skill is running"
                return res
            self.active = True
        try:
            with self.held_lock:
                self.held = self.held_key = None
            for model, (x, y, z) in OBJECT_START.items():
                self.wait(self._send_pose(model, x, y, z), 3.0)
            self.add_table()
            ok = self.home()
            res.success = ok
            res.message = "state cleared, at home" if ok else "state cleared but failed to move home"
        finally:
            self.active = False
        return res

    def on_status(self, req, res):
        res.success = True
        res.message = f"holding {self.held_key}" if self.held_key else "holding nothing"
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
        """Stop following, let any in-flight teleport finish, then put the object at its exact rest pose."""
        with self.held_lock:
            name, self.held, self.held_key = self.held, None, None
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

    def locate_all(self, need=()):
        """{name: (x, y, top_z)} for every visible cube (colour detector) and, if /detect_open is running,
        for the extra objects. `need` lists names the caller must have; it only changes the failure note."""
        req = DetectObjects.Request(); req.color = "all"
        res = self.wait(self.detect_cli.call_async(req))
        out = {}
        if res is not None and res.success:
            for i, p in zip(res.ids, res.poses):
                out.setdefault(i.split("_")[0], (p.pose.position.x, p.pose.position.y, p.pose.position.z))
        self.open_note = ""
        wants_extra = any(n in OBJECTS for n in need)
        if not self.open_vocab:
            if wants_extra:
                self.open_note = " (open_vocab is disabled: restart skill_server without -p open_vocab:=false)"
            return out
        if not (self.open_cli.service_is_ready() or self.open_cli.wait_for_service(timeout_sec=0.5)):
            if wants_extra:
                self.open_note = " (/detect_open is not running: ros2 run ur_perception detect_open)"
            return out
        # object labels + the cube labels, so that cubes are claimed by their own label instead of a weak object label
        q = DetectOpen.Request()
        q.labels = [v[1] for v in OBJECTS.values()] + [f"{c} cube" for c in COLORS]
        r = self.wait(self.open_cli.call_async(q), 120.0)     # the first call loads the model
        if r is None:
            self.open_note = " (/detect_open timed out)"
            return out
        by_label = {v[1]: k for k, v in OBJECTS.items()}
        for lab, p in zip(r.found_labels, r.poses):           # best score first
            key = by_label.get(lab)
            if key is None or key in out:
                continue
            x, y = p.pose.position.x, p.pose.position.y
            if any(math.hypot(x - cx, y - cy) < DUP_RADIUS for n, (cx, cy, _) in out.items() if n in COLORS):
                continue                                       # that is a cube, not an extra object
            top = p.pose.position.z
            if not (TABLE_TOP + 0.02 < top < TABLE_TOP + 3 * CUBE + 0.03):
                self.get_logger().warn(f"ignoring '{lab}' at ({x:.2f}, {y:.2f}): top z {top:.2f} m is not table level "
                                       f"(probably the arm)")
                continue
            out[key] = (x, y, top)
        if wants_extra and not r.success:
            self.open_note = f" (/detect_open: {r.message})"
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
        """All detected objects as obstacles, except the held one and the `exclude` names. Stale boxes are removed first."""
        if not self.collision_scene:
            return
        held = self.held_key
        self.scene_boxes = {}
        try:
            for n in ALL_NAMES:
                cid = model_of(n)
                self.moveit2.remove_collision_object(id=cid)
                if n in exclude or n == held or n not in objs:
                    continue
                x, y, top = objs[n]
                sx, sy, sz = size_of(n)
                self.moveit2.add_collision_box(id=cid, size=(sx + CUBE_MARGIN, sy + CUBE_MARGIN, sz + CUBE_MARGIN),
                                               position=(x, y, top - sz / 2),
                                               quat_xyzw=IDENT_QUAT, frame_id="base_link")
                self.scene_boxes[n] = (x, y, top)
            time.sleep(0.3)            # scene updates are published asynchronously
        except Exception as e:
            self.get_logger().warn(f"could not update objects in the collision scene: {e}")

    def z_hint(self, top):
        """Explain a suspicious detected height (usually an outdated world file)."""
        expected = TABLE_TOP + CUBE
        if abs(top - expected) > 0.05:
            return (f" [detected top z={top:.2f} m but expected about {expected:.2f} m: "
                    f"the world file in use probably has a different table height - rebuild and restart the sim]")
        return ""

    def fail_report(self, xyz, cartesian):
        """What to look at when a motion fails: the goal, every box in the scene and its xy distance from the goal."""
        boxes = ", ".join(f"{n} xy=({bx:.2f},{by:.2f}) top={t:.2f} d_xy={math.hypot(bx - xyz[0], by - xyz[1]):.2f}"
                          for n, (bx, by, t) in self.scene_boxes.items()) or "none"
        return (f"motion to ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) cartesian={cartesian} failed. "
                f"boxes in scene: {boxes}. holding: {self.held_key}. {self.state_report()}")

    def go(self, xyz, cartesian=False):
        for attempt in (1, 2):
            self.moveit2.move_to_pose(position=list(xyz), quat_xyzw=DOWN, cartesian=cartesian)
            if self.moveit2.wait_until_executed():
                if attempt == 2:
                    self.get_logger().warn("motion worked only WITHOUT object boxes: one of them blocks it (see the report above)")
                return True
            self.get_logger().warn(self.fail_report(xyz, cartesian))
            if attempt == 1 and self.retry_no_boxes and self.scene_boxes:
                self.get_logger().warn("retrying without object boxes (-p retry_no_boxes:=true)")
                self.clear_objects()
            else:
                break
        return False

    def clear_objects(self):
        """Remove every object box (not the table) from the MoveIt scene."""
        if not self.collision_scene:
            return
        try:
            for n in ALL_NAMES:
                self.moveit2.remove_collision_object(id=model_of(n))
            self.scene_boxes = {}
            time.sleep(0.3)
        except Exception as e:
            self.get_logger().warn(f"could not clear object boxes: {e}")

    def state_report(self):
        """One line about the arm's joint state, to tell stale/paused-sim problems from collision problems."""
        try:
            js = self.moveit2.joint_state
            age = (self.get_clock().now() - rclpy.time.Time.from_msg(js.header.stamp)).nanoseconds / 1e9
            pos = ", ".join(f"{n.replace('_joint', '')}={p:.2f}" for n, p in zip(js.name, js.position))
            return f"joint_state age {age:.2f} s (sim clock); {pos}"
        except Exception as e:
            return f"joint state unavailable ({e})"

    def home(self):
        for attempt in (1, 2):
            self.moveit2.move_to_configuration(HOME)
            if self.moveit2.wait_until_executed():
                return True
            self.get_logger().warn(f"homing failed (attempt {attempt}): {self.state_report()}")
            if attempt == 1:
                self.get_logger().warn("retrying home with all object boxes removed "
                                       "(if this works, an object box was the problem)")
                self.clear_objects()
        return False

    # ---------- skills ----------
    def do_home(self, gh):
        self.add_table()
        self.stage(gh, "moving home")
        return (True, "at home") if self.home() else (False, "failed to move home")

    def do_pick(self, gh):
        raw = gh.request.color
        name = canon(raw)
        if name is None:
            return False, f"unknown object '{raw}', use one of {ALL_NAMES}"
        if self.held_key:
            return False, f"already holding {self.held_key}; place it first"
        self.add_table()
        if not self.stage(gh, "moving home"): return False, "cancelled"
        if not self.home(): return False, "failed to move home"
        if not self.stage(gh, f"detecting {name}"): return False, "cancelled"
        objs = self.locate_all(need=(name,))
        if name not in objs:
            return False, f"no {name} object detected{self.open_note}"
        self.sync_cubes(objs, exclude=(name,))
        x, y, top = objs[name]
        self.get_logger().info(f"{name} detected at ({x:.3f}, {y:.3f}), top z={top:.3f}")
        hint = self.z_hint(top)
        grasp_z = (top - CUBE / 2) + TCP
        pre_z = grasp_z + APPROACH
        if not reachable(x, y, pre_z):
            return False, (f"{name} object at ({x:.2f},{y:.2f}) is out of the arm's reach "
                           f"(hover z={pre_z:.2f}){hint}")
        steps = (("moving above object", [x, y, pre_z], False),
                 ("descending", [x, y, grasp_z], True))
        for text, xyz, cart in steps:
            if not self.stage(gh, text): return False, "cancelled"
            if not self.go(xyz, cart): return False, f"motion failed while {text}"
        with self.held_lock:
            self.held, self.held_key = model_of(name), name
        time.sleep(0.5)
        if not self.stage(gh, "lifting"): return False, "cancelled"
        if not self.go([x, y, pre_z], True): return False, "motion failed while lifting"
        return True, f"picked {name} object"

    def do_place(self, gh):
        if not self.held_key:
            return False, "not holding anything"
        r = gh.request
        target = canon(r.target) if r.target.strip() else ""
        if target is None:
            return False, f"unknown target '{r.target}', use one of {ALL_NAMES} or leave it empty"
        if target in NOT_STACKABLE:
            return False, f"cannot place on top of the {target}: its top is round"
        if target == self.held_key:
            return False, "cannot place an object on itself"
        hint = ""
        self.add_table()
        if not self.stage(gh, "moving home"): return False, "cancelled"
        if not self.home(): return False, "failed to move home"
        objs = {}
        if target or self.collision_scene:
            if not self.stage(gh, f"detecting {target or 'objects'}"): return False, "cancelled"
            objs = self.locate_all(need=(target,) if target else ())
            self.sync_cubes(objs)
        if target:
            if target not in objs:
                return False, f"no {target} object detected to place on{self.open_note}"
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
