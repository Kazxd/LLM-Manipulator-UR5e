"""Chat with the robot: Ollama LLM -> tool calls -> ROS 2 skills.

  ros2 run ur_llm_bridge llm_bridge --ros-args -p model:=qwen3:4b
Needs: sim, detect_objects, skill_server, and `ollama serve` with the model pulled.
"""
import threading, time, rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from ur_interfaces.srv import DetectObjects
from ur_interfaces.action import MoveHome, Pick, Place
from ur_llm_bridge.agent import TOOLS, new_history, ollama_chat, run_agent


class LLMBridge(Node):
    def __init__(self, name="llm_bridge"):
        super().__init__(name)
        self.model = self.declare_parameter("model", "qwen3:4b").value
        self.host = self.declare_parameter("host", "http://localhost:11434").value
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

    def execute(self, name, a):
        if name == "detect_objects":
            return self.detect(a["color"])
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
    chat = lambda msgs: ollama_chat(node.host, node.model, msgs, TOOLS)
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
