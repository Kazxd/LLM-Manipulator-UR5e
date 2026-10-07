#!/usr/bin/env bash
# Run the 8 extra-object benchmark cases live (can / ball / block). Run from ur_llm_ws in a sourced shell.
# Needs: sim, detect_objects, detect_open, skill_server (all running). Do NOT run llm_bridge at the same time.
set -e
MODEL="${1:-qwen3:4b-instruct}"
REPEAT="${2:-2}"
echo "== checks"
ros2 service list | grep -q "/detect_open"   || { echo "detect_open is not running: ros2 run ur_perception detect_open"; exit 1; }
ros2 service list | grep -q "/detect_objects" || { echo "detect_objects is not running"; exit 1; }
ros2 service list | grep -q "/reset_state"    || { echo "skill_server is not running (need the NEW one with /reset_state)"; exit 1; }
echo "ok. First case is slow: OWL-ViT loads on its first request."
ros2 run ur_llm_bridge llm_benchmark --model "$MODEL" --repeat "$REPEAT" \
  --only pick_can pick_ball pick_block xy_can xy_ball can_on_red ball_on_block zone_can_right
echo "== TELEPORT lines in your skill_server log must be empty for a valid run"
