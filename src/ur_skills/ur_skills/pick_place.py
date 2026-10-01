"""Usage:
  ros2 run ur_skills pick_place red blue   # pick red, place it on blue
  ros2 run ur_skills pick_place red        # pick red only
(needs skill_server + detect_objects running)
"""
import sys, rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ur_interfaces.action import MoveHome, Pick, Place


def call(node, client, goal):
    if not client.wait_for_server(timeout_sec=5.0):
        return False, "action server not available"
    fut = client.send_goal_async(
        goal, feedback_callback=lambda m: node.get_logger().info(f"  [{m.feedback.stage}]"))
    rclpy.spin_until_future_complete(node, fut)
    gh = fut.result()
    if not gh.accepted:
        return False, "goal rejected (server busy?)"
    rf = gh.get_result_async()
    rclpy.spin_until_future_complete(node, rf)
    r = rf.result().result
    return r.success, r.message


def main():
    rclpy.init()
    node = Node("pick_place_client")
    args = [a for a in sys.argv[1:] if not a.startswith("__")]
    src = args[0] if args else "red"
    dst = args[1] if len(args) >= 2 else ("blue" if not args else None)
    pick = ActionClient(node, Pick, "pick_object")
    place = ActionClient(node, Place, "place_object")
    home = ActionClient(node, MoveHome, "move_home")

    g = Pick.Goal(); g.color = src
    ok, msg = call(node, pick, g)
    node.get_logger().info(f"pick: {ok} - {msg}")
    if ok and dst:
        g = Place.Goal(); g.target = dst
        ok, msg = call(node, place, g)
        node.get_logger().info(f"place: {ok} - {msg}")
    if not ok:
        node.get_logger().warn("failed; the robot may still be holding the object. "
                               "Run: ros2 action send_goal /place_object ur_interfaces/action/Place \"{target: '', x: 0.5, y: 0.0, surface_z: 0.0}\"")
    rclpy.shutdown()
