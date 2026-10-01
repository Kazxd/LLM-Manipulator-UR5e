"""Chat with the robot: Ollama LLM -> tool calls -> ROS 2 skills.

  ros2 run ur_llm_bridge llm_bridge --ros-args -p model:=qwen3:4b
  (add -p think:=true to let the model reason; default appends /no_think)
Needs: sim, detect_objects, skill_server, and `ollama serve` with the model pulled.
"""
import threading, time, rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from ur_interfaces.srv import DetectObjects
from ur_interfaces.action import MoveHome, Pick, Place
from ur_llm_bridge.agent import (TOOLS, compute_relative, new_history, ollama_chat, run_agent,
                                 zone_contains, zone_slot)


class LLMBridge(Node):
    def __init__(self, name="llm_bridge"):
        super().__init__(name)
        self.model = self.declare_parameter("model", "qwen3:4b").value
        self.host = self.declare_parameter("host", "http://localhost:11434").value
        self.think = self.declare_parameter("think", False).value
        self.detect_cli = self.create_client(DetectObjects, "/detect_objects")
        self.status_cli = self.create_client(Trigger, "/get_status")
        self.pick_cli = ActionClient(self, Pick, "pick_object")
        self.place_cli = ActionClient(self, Place, "place_object")
        self.home_cli = ActionClient(self, MoveHome, "move_home")
        self.holding = None

    def _wait(self, fut, timeout):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.05)
        return True

    def call_action(self, client, goal, timeout=180.0):
        if not client.wait_for_server(timeout_sec=3.0):
            return False, "action server not available (is skill_server running?)"
        fut = client.send_goal_async(goal)
        if not self._wait(fut, 10.0):
            return False, "timed out sending goal"
        gh = fut.result()
        if not gh.accepted:
            return False, "goal rejected (robot busy?)"
        rf = gh.get_result_async()
        if not self._wait(rf, timeout):
            gh.cancel_goal_async()
            return False, "timed out waiting for the robot"
        r = rf.result().result
        return r.success, r.message

    def detect(self, color):
        if not self.detect_cli.wait_for_service(timeout_sec=3.0):
            return {"success": False, "message": "detect service not available", "objects": []}
        req = DetectObjects.Request(); req.color = color
        fut = self.detect_cli.call_async(req)
        if not self._wait(fut, 10.0):
            return {"success": False, "message": "detection timed out", "objects": []}
        res = fut.result()
        objs = [{"id": i, "x": round(p.pose.position.x, 3), "y": round(p.pose.position.y, 3)}
                for i, p in zip(res.ids, res.poses)]
        return {"success": res.success, "message": res.message, "objects": objs}

    def get_holding(self):
        """Ask skill_server what it really holds; fall back to the locally tracked value."""
        if self.status_cli.wait_for_service(timeout_sec=1.0):
            fut = self.status_cli.call_async(Trigger.Request())
            if self._wait(fut, 3.0):
                m = fut.result().message
                return None if m.endswith("nothing") else m.split()[-1]
        return self.holding

    def place_relative(self, a):
        """Compute a table spot next to the reference cube, then place the held cube there."""
        held = self.get_holding()
        if not held:
            return {"success": False, "message": "not holding anything; pick a cube first"}
        if a["reference"] == held:
            return {"success": False, "message": "cannot place a cube relative to itself"}
        ok, msg = self.call_action(self.home_cli, MoveHome.Goal())   # arm out of the camera's view
        if not ok:
            return {"success": False, "message": "could not clear the camera view: " + msg}
        d = self.detect("all")
        pos = {o["id"].split("_")[0]: (o["x"], o["y"]) for o in d["objects"]}
        if a["reference"] not in pos:
            return {"success": False, "message": f"{a['reference']} cube not visible"}
        others = {c: p for c, p in pos.items() if c not in (a["reference"], held)}
        xy, why = compute_relative(pos[a["reference"]], a["relation"], a["distance"], others)
        if xy is None:
            return {"success": False, "message": why}
        g = Place.Goal(); g.target = ""
        g.x, g.y, g.surface_z = xy[0], xy[1], 0.0
        ok, msg = self.call_action(self.place_cli, g)
        if ok:
            self.holding = None
            msg = f"{msg}: {why} of the {a['reference']} cube, at ({xy[0]:.2f}, {xy[1]:.2f})"
        return {"success": ok, "message": msg}

    def place_in_zone(self, a):
        """Place the held cube in a free slot of the zone (computed from a fresh detection)."""
        held = self.get_holding()
        if not held:
            return {"success": False, "message": "not holding anything; pick a cube first"}
        ok, msg = self.call_action(self.home_cli, MoveHome.Goal())   # arm out of the camera's view
        if not ok:
            return {"success": False, "message": "could not clear the camera view: " + msg}
        d = self.detect("all")
        pos = {o["id"].split("_")[0]: (o["x"], o["y"]) for o in d["objects"]}
        others = {c: p for c, p in pos.items() if c != held}
        xy, why = zone_slot(a["zone"], others)
        if xy is None:
            return {"success": False, "message": why}
        g = Place.Goal(); g.target = ""
        g.x, g.y, g.surface_z = xy[0], xy[1], 0.0
        ok, msg = self.call_action(self.place_cli, g)
        if ok:
            self.holding = None
            msg = f"{msg}: in the {a['zone']} zone at ({xy[0]:.2f}, {xy[1]:.2f})"
        return {"success": ok, "message": msg}

    def sort_cubes(self, a):
        """assignments {colour: zone}. Skips cubes already on their zone; stops at the first failure."""
        if self.get_holding():
            return {"success": False, "message": "already holding a cube; place it first", "sorted": 0}
        ok, msg = self.call_action(self.home_cli, MoveHome.Goal())
        if not ok:
            return {"success": False, "message": "could not clear the camera view: " + msg, "sorted": 0}
        d = self.detect("all")
        pos = {o["id"].split("_")[0]: (o["x"], o["y"]) for o in d["objects"]}
        done, skipped = [], []
        for color, zone in a["assignments"].items():
            if color not in pos:
                return {"success": False, "message": f"{color} cube not visible", "sorted": len(done)}
            if zone_contains(zone, *pos[color]):
                skipped.append(color)
                continue
            r = self.execute("pick_object", {"color": color})
            if not r["success"]:
                return {"success": False, "message": f"pick {color} failed: {r['message']}", "sorted": len(done)}
            r = self.execute("place_in_zone", {"zone": zone})
            if not r["success"]:
                return {"success": False, "message": f"place {color} in {zone} failed: {r['message']}",
                        "sorted": len(done)}
            done.append(color)
        note = f"; already in place: {', '.join(skipped)}" if skipped else ""
        return {"success": True, "message": "sorted: " + ", ".join(f"{c}->{a['assignments'][c]}" for c in done) + note,
                "sorted": len(done)}

    def build_tower(self, a):
        """order = colours bottom to top. The bottom cube never moves; each upper cube is picked and placed on the one below."""
        order = a["order"]
        if self.get_holding():
            return {"success": False, "message": "already holding a cube; place it first", "stacked": 1}
        stacked = 1
        for lower, upper in zip(order[:-1], order[1:]):
            r = self.execute("pick_object", {"color": upper})
            if not r["success"]:
                return {"success": False, "message": f"pick {upper} failed: {r['message']}", "stacked": stacked}
            r = self.execute("place_object", {"target": lower})
            if not r["success"]:
                return {"success": False, "message": f"place {upper} on {lower} failed: {r['message']}",
                        "stacked": stacked}
            stacked += 1
        return {"success": True, "message": "tower built, bottom to top: " + ", ".join(order), "stacked": stacked}

    def execute(self, name, a):
        if name == "detect_objects":
            return self.detect(a["color"])
        if name == "place_relative":
            return self.place_relative(a)
        if name == "place_in_zone":
            return self.place_in_zone(a)
        if name == "sort_cubes":
            return self.sort_cubes(a)
        if name == "build_tower":
            return self.build_tower(a)
        if name == "move_home":
            ok, msg = self.call_action(self.home_cli, MoveHome.Goal())
        elif name == "pick_object":
            g = Pick.Goal(); g.color = a["color"]
            ok, msg = self.call_action(self.pick_cli, g)
            if ok:
                self.holding = a["color"]
        else:  # place_object
            g = Place.Goal(); g.target = a["target"]
            if not a["target"]:
                g.x, g.y, g.surface_z = a["x"], a["y"], 0.0
            ok, msg = self.call_action(self.place_cli, g)
            if ok:
                self.holding = None
        return {"success": ok, "message": msg}

    def scene_summary(self):
        d = self.detect("all")
        objs = ", ".join(f"{o['id']} at ({o['x']}, {o['y']})" for o in d["objects"]) or d["message"]
        return f"objects: {objs}. Holding: {self.get_holding() or 'nothing'}."


def main():
    rclpy.init()
    node = LLMBridge()
    ex = MultiThreadedExecutor(4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    history = new_history()
    chat = lambda msgs: ollama_chat(node.host, node.model, msgs, TOOLS, no_think=not node.think)
    print(f"LLM bridge ready (model {node.model}). Type a command, or 'quit'.")
    while rclpy.ok():
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if text:
            print("robot> " + run_agent(history, text, node.execute, chat, node.scene_summary))
    rclpy.shutdown()
