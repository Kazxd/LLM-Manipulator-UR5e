# UR5e + MoveIt 2 + colour vision (ROS 2 Jazzy, Gazebo Harmonic)

Packages (src/): ur_interfaces (srv + actions), ur_perception (detect_objects service),
ur_skills (skill_server action server + pick_place client), ur_llm_bridge (Ollama chat -> skills),
ur_sim_bringup (world + launch).

## Setup
    ./setup.sh && source install/setup.bash     # use a clean shell (no other workspaces sourced)

## Run (3 terminals, all sourced)
    ros2 launch ur_sim_bringup sim.launch.py
    ros2 run ur_perception detect_objects
    ros2 run ur_skills skill_server

## Use
    ros2 run ur_skills pick_place red blue

    ros2 service call /detect_objects ur_interfaces/srv/DetectObjects "{color: 'all'}"
    ros2 action send_goal /move_home   ur_interfaces/action/MoveHome "{}" --feedback
    ros2 action send_goal /pick_object ur_interfaces/action/Pick "{color: red}" --feedback
    ros2 action send_goal /place_object ur_interfaces/action/Place "{target: blue}" --feedback
    ros2 action send_goal /place_object ur_interfaces/action/Place "{target: '', x: 0.5, y: -0.2, surface_z: 0.0}" --feedback

Table top is at z = 0.20 m (world file and TABLE_TOP in skill_server.py must agree).

## Talk to the robot with a local LLM
    ollama pull qwen3:4b && ollama serve          # (serve may already run as a service)
    ros2 run ur_llm_bridge llm_bridge --ros-args -p model:=qwen3:4b
    you> put the red cube on the blue one
    you> move the green cube to x 0.5 y -0.2

Offline test of the agent loop (no ROS/Ollama): python3 src/ur_llm_bridge/test/test_agent.py

## If you change the world file
Rebuild with `colcon build --symlink-install` (or rebuild + restart the sim) - otherwise the OLD
world in install/ is still loaded. Check: detected cube top z should be about 0.25 m.

## Benchmark (measure before changing things)
    ros2 run ur_llm_bridge llm_benchmark --list
    ros2 run ur_llm_bridge llm_benchmark --model qwen3:4b             # 20 cases, results in benchmark_results/
    ros2 run ur_llm_bridge llm_benchmark --only pick_red stack_red_blue --repeat 3
    ros2 run ur_llm_bridge reset_scene                                # put cubes back, clear held state, go home
Pass/fail is scored against Gazebo ground-truth cube poses, not the vision system.
Do not run llm_bridge at the same time (only one thing should command the robot).
Run `python3 src/ur_llm_bridge/test/test_bench_lib.py` for the offline checks.

## Helper scripts
    scripts/kill_sim.sh     # kill leftover gz/MoveIt/bridges (a stale gz keeps the OLD world)
    scripts/check_env.sh    # detect stray workspaces / stale world before launching
    scripts/git_init.sh     # first commit + tag
