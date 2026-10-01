#!/usr/bin/env bash
# Run from ur_llm_ws in a sourced shell. Catches the stale-workspace / stale-world problems.
ok=1
echo "== stray workspaces in AMENT_PREFIX_PATH (should be empty)"
bad=$(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | grep -i -E "trash|ros2_ws" || true)
[ -n "$bad" ] && { echo "$bad"; ok=0; } || echo "  none"
echo "== package prefix (should be inside this workspace)"
ros2 pkg prefix ur_sim_bringup || ok=0
echo "== installed world has the 0.2 m table? (should be > 0)"
n=$(grep -c "0.225" install/ur_sim_bringup/share/ur_sim_bringup/worlds/pick_place.sdf 2>/dev/null || echo 0)
echo "  $n"; [ "$n" -gt 0 ] || ok=0
echo "== running Gazebo world (should be under this workspace)"
ps -eo args | grep -i "gz sim" | grep -v grep | grep -o "[^ ]*pick_place.sdf" | sort -u || echo "  gz not running"
echo "== skills available"
ros2 pkg executables ur_skills ur_perception ur_llm_bridge
[ $ok = 1 ] && echo "ENV OK" || echo "ENV PROBLEM: see above"
