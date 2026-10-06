#!/usr/bin/env python3
"""Apply the Robotiq collision exceptions to the running move_group.  Run with sim + MoveIt up:
  python3 scripts/fix_acm.py && python3 scripts/check_state.py"""
import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
from ur_skills.acm_fix import allow_gripper_collisions

rclpy.init()
n = Node("fix_acm", parameter_overrides=[Parameter("use_sim_time", value=True)])


def wait(fut, timeout):
    rclpy.spin_until_future_complete(n, fut, timeout_sec=timeout)
    return fut.result() if fut.done() else None


print(allow_gripper_collisions(n, wait))
rclpy.shutdown()