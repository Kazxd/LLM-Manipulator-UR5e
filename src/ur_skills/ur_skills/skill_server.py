"""Action server exposing /move_home, /pick_object, /place_object.

Gripper modes (-p gripper:=robotiq | magic, default robotiq; must match sim.launch.py gripper:=...):
  robotiq: Robotiq 2F-85. One joint (robotiq_85_left_knuckle_joint, 0 = open 85 mm, about 0.79 = closed).
           Closing is a ramp: the command creeps toward closed until the finger stalls on the object, then the command
           is set to the stalled position plus a small squeeze (-p squeeze:=0.05 rad). That gives a bounded, tunable
           grip force instead of a full-effort crush. Opening first relaxes the squeeze, then ramps open slowly, and
           (-p snap_release:=true) finally puts the object at its exact rest pose so it cannot drift away.
           The fingers open and close along world x (tool pointing down).
           At the start of every skill the gripper's self-contacts are allowed in MoveIt's collision matrix (acm_fix.py).
  magic  : a held object is teleported to follow the tool (no fingers).
Hold modes for robotiq (-p hold:=physics | attach, default physics):
  physics: the object is carried by finger friction only.
  attach : after a passed grasp check the object follows the tool by teleport (fallback while tuning physics).
-p tcp:=0.15  distance from tool0 to the fingertip centre (tune this if the pads grab too high or too low).
Only one skill runs at a time; a second goal is rejected while busy.
Services: /reset_state (clear held state, open gripper, put the extra objects back, go home), /get_status.

Objects: cubes red/green/blue (colour detector) and, when detect_open runs, can/ball/block (OBJECTS below).
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
from std_msgs.msg import Float64MultiArray
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from pymoveit2 import MoveIt2
from ur_interfaces.srv import DetectObjects, DetectOpen
from std_srvs.srv import Trigger
from ur_interfaces.action import MoveHome, Pick, Place
from ur_skills.acm_fix import allow_gripper_collisions

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
DOWN = [1.0, 0.0, 0.0, 0.0]      # tool z pointing down (x, y, z, w)
TCP_DEFAULT = 0.15                # tool0 to fingertip centre (parameter tcp overrides)
CUBE = 0.05                       # height of EVERY object
APPROACH = 0.12                   # hover height above grasp/place
TABLE_TOP = 0.20                  # must match the world file
SHOULDER_Z = 0.163                # UR5e shoulder height
REACH = 0.92                      # loose sanity gate only; MoveIt is the real judge
RELEASE_CLEARANCE = 0.005
CUBE_MARGIN = 0.01                # collision boxes are inflated by this (total, all axes)
COLORS = ("red", "green", "blue")
IDENT_QUAT = (0.0, 0.0, 0.0, 1.0)

# Robotiq knuckle joint (radians). Calibrate GRIP_MIN_Q and GRIP_MISS_Q from the logged "fingers closed: q=" values.
GRIP_OPEN = 0.0
GRIP_CLOSED = 0.79                # the closing ramp ends here if nothing is between the fingers
GRIP_MISS_Q = 0.70                # at or above this the fingers closed on nothing
GRIP_MIN_Q = 0.10                 # below this the fingers did not close (blocked or not commanded)
CLOSE_RATE = 0.4                  # rad/s command ramp (the joint velocity limit is 0.5)
OPEN_RATE = 0.3                   # rad/s, slow so the finger pads do not drag the object
STALL_LAG = 0.06                  # command ahead of the joint by more than this and not moving means stalled on an object
FINGER_JOINTS = ("robotiq_85_left_knuckle_joint",)
GRIPPER_TOPIC = "/gripper_controller/commands"

OBJECTS = {
    "can":   ("yellow_can",  "yellow circle", (0.05, 0.05, 0.05)),
    "ball":  ("purple_ball", "purple ball",   (0.05, 0.05, 0.05)),
    "block": ("white_block", "white block",   (0.04, 0.07, 0.05)),
}
OBJECT_START = {"yellow_can": (0.72, -0.30, 0.225), "purple_ball": (0.38, 0.34, 0.225),
                "white_block": (0.80, 0.00, 0.225)}
NOT_STACKABLE = ("ball",)
ALL_NAMES = COLORS + tuple(OBJECTS)
DUP_RADIUS = 0.04


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
        self.gripper = str(self.declare_parameter("gripper", "robotiq").value).lower()
        if self.gripper not in ("robotiq", "magic"):
            self.get_logger().warn(f"unknown gripper '{self.gripper}', using 'robotiq'")
            self.gripper = "robotiq"
        self.hold = str(self.declare_parameter("hold", "physics").value).lower()
        if self.hold not in ("attach", "physics"):
            self.get_logger().warn(f"unknown hold '{self.hold}', using 'physics'")
            self.hold = "physics"
        self.tcp = float(self.declare_parameter("tcp", TCP_DEFAULT).value)
        self.squeeze = float(self.declare_parameter("squeeze", 0.05).value)
        self.snap_release = bool(self.declare_parameter("snap_release", True).value)
        self.grasp_check = self.declare_parameter("grasp_check", True).value
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
        self.grip_pub = self.create_publisher(Float64MultiArray, GRIPPER_TOPIC, 10)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.held = None                 # gazebo model name of the held object
        self.held_key = None             # its short name: 'red', 'can', ...
        self.held_lock = threading.Lock()
        self.pose_future = None
        self.active = False
        self.lock = threading.Lock()
        self.open_note = ""
        self.scene_boxes = {}
        self.acm_ok = None               # last result of the gripper collision-matrix update (for logging only)
        self.collision_scene = self.declare_parameter("collision_scene", True).value
        self.open_vocab = self.declare_parameter("open_vocab", True).value
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
            f"(gripper={self.gripper}, hold={self.hold}, tcp={self.tcp}, squeeze={self.squeeze}, "
            f"snap_release={self.snap_release}, collision_scene={self.collision_scene}, "
            f"open_vocab={self.open_vocab}, names={ALL_NAMES})")

    def teleport_hold(self):
        """True when the held object is carried by teleporting it with the tool."""
        return self.gripper == "magic" or self.hold == "attach"

    def ensure_gripper_acm(self):
        """Allow the gripper's own contacts in MoveIt (cheap; repeated every skill so a move_group restart is covered)."""
        if self.gripper != "robotiq":
            return
        ok, msg = allow_gripper_collisions(self, self.wait, self.cb)
        if ok != self.acm_ok:
            (self.get_logger().info if ok else self.get_logger().warn)(f"collision matrix: {msg}")
        self.acm_ok = ok

    # ---------- action plumbing ----------
    def on_goal(self, goal):
        with self.lock:
            if self.active:
                return GoalResponse.REJECT
            self.active = True
            return GoalResponse.ACCEPT

    def on_reset(self, req, res):
        with self.lock:
            if self.active:
                res.success, res.message = False, "a skill is running"
                return res
            self.active = True
        try:
            with self.held_lock:
                self.held = self.held_key = None
            self.open_gripper()
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

    # ---------- gripper ----------
    def finger_q(self):
        """Knuckle joint position from joint states, or None."""
        try:
            js = self.moveit2.joint_state
            d = dict(zip(js.name, js.position))
            vals = [d[j] for j in FINGER_JOINTS if j in d]
            return sum(vals) / len(vals) if vals else None
        except Exception:
            return None

    def send_grip(self, q, repeat=1):
        msg = Float64MultiArray()
        msg.data = [float(q)]
        for _ in range(repeat):
            self.grip_pub.publish(msg)
            if repeat > 1:
                time.sleep(0.03)

    def ramp_gripper(self, target, rate, dt=0.1):
        """Move the command to target at `rate` rad/s starting from the current joint position, then settle."""
        if self.gripper != "robotiq":
            return
        q = self.finger_q()
        if q is None:
            self.send_grip(target, repeat=3)
            time.sleep(2.0)
            return
        cmd = q
        while abs(target - cmd) > 1e-3:
            step = rate * dt
            cmd = min(target, cmd + step) if target > cmd else max(target, cmd - step)
            self.send_grip(cmd)
            time.sleep(dt)
        self.send_grip(target, repeat=3)
        time.sleep(0.5)

    def open_gripper(self, wait=None):
        """Open slowly (wait is kept only for old call sites)."""
        self.ramp_gripper(GRIP_OPEN, OPEN_RATE)

    def close_gripper(self):
        """Ramp closed until the finger stalls on the object, then hold with a bounded squeeze. Returns (ok, message)."""
        if self.gripper != "robotiq":
            return True, ""
        q0 = self.finger_q()
        cmd = q0 if q0 is not None else 0.0
        dt, last, still, stalled = 0.1, None, 0, False
        while cmd < GRIP_CLOSED - 1e-3:
            cmd = min(GRIP_CLOSED, cmd + CLOSE_RATE * dt)
            self.send_grip(cmd)
            time.sleep(dt)
            q = self.finger_q()
            if q is None:
                continue
            if cmd - q > STALL_LAG:
                still = still + 1 if (last is not None and abs(q - last) < 0.002) else 0
                if still >= 3:
                    stalled = True
                    break
            last = q
        q = self.finger_q()
        if stalled and q is not None:
            hold = min(GRIP_CLOSED, q + self.squeeze)
            self.send_grip(hold, repeat=3)          # bounded squeeze: force is about gain times (hold - q)
            time.sleep(0.6)
            q = self.finger_q()
            self.get_logger().info(f"stalled on an object at q={q}, holding command {hold:.3f} (squeeze {self.squeeze})")
        else:
            self.send_grip(GRIP_CLOSED, repeat=3)
            time.sleep(0.5)
            q = self.finger_q()
        self.get_logger().info(f"fingers closed: q={q}  (miss >= {GRIP_MISS_Q}, too open < {GRIP_MIN_Q})")
        if not self.grasp_check:
            return True, ""
        if q is None:
            return False, "grasp check failed: no gripper joint state (is gripper_controller active?)"
        if q >= GRIP_MISS_Q:
            return False, f"grasp missed: fingers closed fully (q={q:.3f}), nothing between them"
        if q < GRIP_MIN_Q:
            return False, f"grasp failed: fingers did not close (q={q:.3f}), something blocks them"
        return True, ""

    # ---------- teleport carrying ----------
    def _send_pose(self, name, x, y, z):
        req = SetEntityPose.Request()
        req.entity.name, req.entity.type = name, Entity.MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, z
        req.pose.orientation.w = 1.0
        return self.pose_cli.call_async(req)

    def follow_tool(self):
        if not self.teleport_hold():
            return
        with self.held_lock:
            if not self.held:
                return
            if self.pose_future is not None and not self.pose_future.done():
                return
            try:
                t = self.tf_buf.lookup_transform("base_link", "tool0", rclpy.time.Time()).transform.translation
            except Exception:
                return
            self.pose_future = self._send_pose(self.held, t.x, t.y, t.z - self.tcp)

    def release_cube(self, x, y, z):
        """Teleport modes: put the object at its exact rest pose, then open. Physics: relax the squeeze, open slowly,
        then (snap_release) put the object at its exact rest pose so it cannot drift off."""
        with self.held_lock:
            name, self.held, self.held_key = self.held, None, None
        if self.teleport_hold():
            if self.pose_future is not None:
                self.wait(self.pose_future, 2.0)
            if name:
                self.wait(self._send_pose(name, x, y, z), 2.0)
            if self.gripper == "robotiq":
                self.open_gripper()
            return
        if self.gripper == "robotiq":
            q = self.finger_q()
            if q is not None:
                self.send_grip(q, repeat=3)         # command = actual position: grip force drops to about zero
                time.sleep(0.4)
            self.open_gripper()
        if name and self.snap_release:
            self.wait(self._send_pose(name, x, y, z), 2.0)

    # ---------- helpers ----------
    def wait(self, fut, timeout=5.0):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def locate_all(self, need=()):
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
        q = DetectOpen.Request()
        q.labels = [v[1] for v in OBJECTS.values()] + [f"{c} cube" for c in COLORS]
        r = self.wait(self.open_cli.call_async(q), 120.0)
        if r is None:
            self.open_note = " (/detect_open timed out)"
            return out
        by_label = {v[1]: k for k, v in OBJECTS.items()}
        for lab, p in zip(r.found_labels, r.poses):
            key = by_label.get(lab)
            if key is None or key in out:
                continue
            x, y = p.pose.position.x, p.pose.position.y
            if any(math.hypot(x - cx, y - cy) < DUP_RADIUS for n, (cx, cy, _) in out.items() if n in COLORS):
                continue
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
        """Runs at the start of every skill: also refreshes the gripper collision exceptions."""
        self.ensure_gripper_acm()
        if not self.collision_scene:
            return
        try:
            self.moveit2.add_collision_box(id="table", size=(0.6, 0.8, 0.2), position=(0.6, 0.0, 0.1),
                                           quat_xyzw=IDENT_QUAT, frame_id="base_link")
        except Exception as e:
            self.get_logger().warn(f"could not add table to the collision scene: {e}")

    def sync_cubes(self, objs, exclude=()):
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
            time.sleep(0.3)
        except Exception as e:
            self.get_logger().warn(f"could not update objects in the collision scene: {e}")

    def z_hint(self, top):
        expected = TABLE_TOP + CUBE
        if abs(top - expected) > 0.05:
            return (f" [detected top z={top:.2f} m but expected about {expected:.2f} m: "
                    f"the world file in use probably has a different table height - rebuild and restart the sim]")
        return ""

    def fail_report(self, xyz, cartesian):
        boxes = ", ".join(f"{n} xy=({bx:.2f},{by:.2f}) top={t:.2f} d_xy={math.hypot(bx - xyz[0], by - xyz[1]):.2f}"
                          for n, (bx, by, t) in self.scene_boxes.items()) or "none"
        return (f"motion to ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) cartesian={cartesian} failed. "
                f"boxes in scene: {boxes}. holding: {self.held_key}. {self.state_report()}")

    def go(self, xyz, cartesian=False):
        for attempt in (1, 2):
            self.moveit2.move_to_pose(position=list(xyz), quat_xyzw=DOWN, cartesian=cartesian)
            if self.moveit2.wait_until_executed():
                if attempt == 2:
                    self.get_logger().warn("motion worked only WITHOUT object boxes: one of them blocks it")
                return True
            self.get_logger().warn(self.fail_report(xyz, cartesian))
            if attempt == 1 and self.retry_no_boxes and self.scene_boxes:
                self.get_logger().warn("retrying without object boxes (-p retry_no_boxes:=true)")
                self.clear_objects()
            else:
                break
        return False

    def clear_objects(self):
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
                self.get_logger().warn("retrying home with all object boxes removed")
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
        self.open_gripper()
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
        grasp_z = (top - CUBE / 2) + self.tcp
        pre_z = grasp_z + APPROACH
        if not reachable(x, y, pre_z):
            return False, (f"{name} object at ({x:.2f},{y:.2f}) is out of the arm's reach "
                           f"(hover z={pre_z:.2f}){hint}")
        steps = (("moving above object", [x, y, pre_z], False),
                 ("descending", [x, y, grasp_z], True))
        for text, xyz, cart in steps:
            if not self.stage(gh, text): return False, "cancelled"
            if not self.go(xyz, cart): return False, f"motion failed while {text}"
        if self.gripper == "robotiq":
            if not self.stage(gh, "closing gripper"): return False, "cancelled"
            ok, why = self.close_gripper()
            if not ok:
                self.open_gripper()
                self.go([x, y, pre_z], True)
                return False, why
        with self.held_lock:
            self.held, self.held_key = model_of(name), name
        time.sleep(0.3)
        if not self.stage(gh, "lifting"): return False, "cancelled"
        if not self.go([x, y, pre_z], True): return False, "motion failed while lifting"
        if self.gripper == "robotiq" and self.grasp_check and not self.teleport_hold():
            q = self.finger_q()
            self.get_logger().info(f"after lift: q={q}")
            if q is not None and q >= GRIP_MISS_Q:
                with self.held_lock:
                    self.held = self.held_key = None
                self.open_gripper()
                return False, f"object slipped out of the gripper during the lift (q={q:.3f})"
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
        place_z = (surface + CUBE / 2) + self.tcp + RELEASE_CLEARANCE
        pre_z = place_z + APPROACH
        if not reachable(x, y, pre_z):
            return False, (f"place position ({x:.2f},{y:.2f}) is out of the arm's reach "
                           f"(hover z={pre_z:.2f}){hint}")
        steps = (("moving above target", [x, y, pre_z], False),
                 ("descending", [x, y, place_z], True))
        for text, xyz, cart in steps:
            if not self.stage(gh, text): return False, "cancelled"
            if not self.go(xyz, cart): return False, f"motion failed while {text}"
        if self.gripper == "robotiq" and not self.stage(gh, "releasing"): return False, "cancelled"
        self.release_cube(x, y, surface + CUBE / 2 + 0.002)
        time.sleep(0.3)
        if not self.stage(gh, "retreating"): return False, "cancelled"
        if not self.go([x, y, pre_z], True): return False, "placed, but retreat failed"
        return True, "placed"


def main():
    rclpy.init()
    node = SkillServer()
    ex = MultiThreadedExecutor(8)
    ex.add_node(node)
    ex.spin()
