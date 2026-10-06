#!/usr/bin/env python3
"""Ask move_group whether the current robot state is valid and list the colliding link pairs.
Run with the sim and MoveIt up:  python3 scripts/check_state.py"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState


def main():
    rclpy.init()
    n = Node("check_state", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    js = []
    n.create_subscription(JointState, "/joint_states", lambda m: js.append(m), 1)
    cli = n.create_client(GetStateValidity, "/check_state_validity")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("/check_state_validity not available (is move_group running?)")
        return
    while not js:
        rclpy.spin_once(n, timeout_sec=0.5)
    print("joint_states:", ", ".join(f"{a}={b:.4f}" for a, b in zip(js[-1].name, js[-1].position)))
    req = GetStateValidity.Request()
    req.robot_state = RobotState()
    req.robot_state.joint_state = js[-1]
    req.group_name = "ur_manipulator"
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
    r = fut.result()
    if r is None:
        print("no answer")
        return
    print("valid:", r.valid)
    for c in r.contacts:
        print("  contact:", c.contact_body_1, "<->", c.contact_body_2)
    if not r.valid and not r.contacts:
        print("invalid but no contacts: probably a joint bound problem (a joint value just outside its limit).")
    rclpy.shutdown()


if __name__ == "__main__":
    main()