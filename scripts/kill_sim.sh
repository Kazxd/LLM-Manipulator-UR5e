#!/usr/bin/env bash
# Kill a leftover simulator / MoveIt / bridges (a stale `gz sim` keeps the OLD world alive).
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f ruby 2>/dev/null
pkill -9 -f move_group 2>/dev/null; pkill -9 -f ros2_control_node 2>/dev/null
pkill -9 -f parameter_bridge 2>/dev/null; pkill -9 -f robot_state_publisher 2>/dev/null
pkill -9 -f skill_server 2>/dev/null; pkill -9 -f detect_objects 2>/dev/null
pkill -9 -f detect_open 2>/dev/null
ros2 daemon stop 2>/dev/null
echo "remaining gz processes:"; ps -eo args | grep -i "gz sim" | grep -v grep || echo "  none"
