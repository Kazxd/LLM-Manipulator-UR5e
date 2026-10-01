"""LLM benchmark: resets the scene, sends ~20 commands, scores the result against ground truth.

  ros2 run ur_llm_bridge llm_benchmark --model qwen3:4b            # all cases
  ros2 run ur_llm_bridge llm_benchmark --only pick_red stack_red_blue --repeat 3
  ros2 run ur_llm_bridge llm_benchmark --list
  ros2 run ur_llm_bridge reset_scene                                # just reset the cubes + robot
Needs: sim (with pose bridge), detect_objects, NEW skill_server (has /reset_state), ollama.
"""
import argparse, csv, json, os, sys, threading, time
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import remove_ros_args
from tf2_msgs.msg import TFMessage
from std_srvs.srv import Trigger
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from ur_llm_bridge import bench_lib as B
from ur_llm_bridge.agent import TOOLS, new_history, ollama_chat, run_agent
from ur_llm_bridge.llm_bridge import LLMBridge


class BenchNode(LLMBridge):
    def __init__(self):
        super().__init__("llm_benchmark")
        self.poses, self.pose_stamp = {}, 0.0
        self.create_subscription(TFMessage, B.POSE_TOPIC, self.on_poses, 10)
        self.reset_cli = self.create_client(Trigger, "/reset_state")
        self.setpose_cli = self.create_client(SetEntityPose, f"/world/{B.WORLD}/set_pose")

    def on_poses(self, msg):
        for t in msg.transforms:
            base = t.child_frame_id.split("::")[0].split("/")[-1]
            if base in B.CUBES:
                tr = t.transform.translation
                self.poses[base] = (tr.x, tr.y, tr.z)
        self.pose_stamp = time.time()

    def get_poses(self):
        if time.time() - self.pose_stamp < 2.0 and all(c in self.poses for c in B.CUBES):
            return dict(self.poses)
        return B.poses_from_gz()          # fallback: ask Gazebo directly

    def reset_scene(self):
        if not self.reset_cli.wait_for_service(timeout_sec=3.0):
            return False, "/reset_state not available (run the NEW skill_server)"
        fut = self.reset_cli.call_async(Trigger.Request())
        if not self._wait(fut, 120.0):
            return False, "/reset_state timed out"
        if not fut.result().success:
            return False, fut.result().message
        self.holding = None
        if not self.setpose_cli.wait_for_service(timeout_sec=3.0):
            return False, "set_pose service not available (is the bridge running?)"
        bad = list(B.INIT)
        for attempt in range(3):
            for name, (x, y, z) in B.INIT.items():
                req = SetEntityPose.Request()
                req.entity.name, req.entity.type = name, Entity.MODEL
                req.pose.position.x, req.pose.position.y = x, y
                req.pose.position.z = z        # cubes have gravity off: exact rest height
                req.pose.orientation.w = 1.0
                self._wait(self.setpose_cli.call_async(req), 5.0)
            time.sleep(2.0 + attempt)          # let the cubes settle
            poses = self.get_poses()
            if poses is None:
                return False, "no ground-truth poses (pose bridge missing and gz fallback failed)"
            bad = [n for n, p in B.INIT.items() if n not in poses or B._d3(poses[n], p) > 0.02]
            if not bad:
                return True, "ok"
        return False, f"cubes not back in place after 3 tries: {bad}"

    def run_case(self, case, model, host, max_steps):
        cid, cat, cmd, check = case
        rec = {"id": cid, "category": cat, "command": cmd, "passed": False, "detail": "", "reply": "",
               "tool_calls": 0, "tool_failures": 0, "rejected": 0, "llm_calls": 0,
               "llm_s": 0.0, "wall_s": 0.0, "trace": [], "error": ""}
        ok, msg = self.reset_scene()
        if not ok:
            rec["error"], rec["detail"] = "reset failed: " + msg, "reset failed"
            return rec
        init = self.get_poses()
        llm = {"n": 0, "t": 0.0}

        def chat(msgs):
            t = time.time()
            m = ollama_chat(host, model, msgs, TOOLS)
            llm["n"] += 1
            llm["t"] += time.time() - t
            return m

        t0 = time.time()
        rec["reply"] = run_agent(new_history(), cmd, self.execute, chat, self.scene_summary,
                                 max_steps=max_steps, log=rec["trace"].append)
        rec["wall_s"] = time.time() - t0
        rec["llm_calls"], rec["llm_s"] = llm["n"], llm["t"]
        tr = rec["trace"]
        rec["tool_calls"] = sum(1 for l in tr if l.startswith("  -> "))
        rec["rejected"] = sum(1 for l in tr if l.startswith("  !!"))
        rec["tool_failures"] = sum(1 for l in tr if l.startswith("  <- ") and "'success': False" in l)
        time.sleep(1.5)
        poses = self.get_poses()
        if poses is None:
            rec["detail"] = "no ground-truth poses at the end"
        else:
            rec["passed"], rec["detail"] = B.evaluate(check, poses, init, rec["reply"])
        if rec["reply"].startswith("LLM error"):
            rec["error"] = rec["reply"]
        return rec


def _spin(node):
    ex = MultiThreadedExecutor(4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--only", nargs="*", help="case ids to run")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--out", default="benchmark_results")
    ap.add_argument("--list", action="store_true")
    args, _ = ap.parse_known_args(remove_ros_args(sys.argv)[1:])

    cases = [c for c in B.CASES if not args.only or c[0] in args.only]
    if args.list:
        for c in cases:
            print(f"{c[0]:<18}{c[1]:<11}{c[2]}")
        return
    if not cases:
        print("no matching cases; use --list")
        return

    rclpy.init()
    node = BenchNode()
    _spin(node)
    t0 = time.time()
    while node.get_poses() is None and time.time() - t0 < 10:
        time.sleep(0.5)
    if node.get_poses() is None:
        print("No ground-truth cube poses from Gazebo. Is the sim running with the pose bridge?")
        rclpy.shutdown()
        return

    results, total = [], len(cases) * args.repeat
    print(f"model={args.model}  cases={len(cases)} x{args.repeat}")
    for rep in range(args.repeat):
        for i, case in enumerate(cases, 1):
            rec = node.run_case(case, args.model, args.host, args.max_steps)
            rec["repeat"] = rep
            results.append(rec)
            mark = "PASS" if rec["passed"] else "FAIL"
            print(f"[{len(results)}/{total}] {mark} {rec['id']:<18} {rec['wall_s']:5.1f}s "
                  f"tools={rec['tool_calls']} | {rec['detail']}" + (f" | {rec['error']}" if rec["error"] else ""))
    node.reset_scene()

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace(":", "-").replace("/", "_")
    base = os.path.join(args.out, f"bench_{tag}_{stamp}")
    with open(base + ".json", "w") as f:
        json.dump({"model": args.model, "time": stamp, "results": results}, f, indent=2)
    with open(base + ".csv", "w", newline="") as f:
        cols = ["id", "category", "repeat", "passed", "detail", "tool_calls", "tool_failures",
                "rejected", "llm_calls", "llm_s", "wall_s", "reply"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print("\n" + B.summarize(results))
    print(f"\nsaved {base}.json / .csv")
    rclpy.shutdown()


def reset_main():
    rclpy.init()
    node = BenchNode()
    _spin(node)
    ok, msg = node.reset_scene()
    print(("OK: " if ok else "FAILED: ") + msg)
    rclpy.shutdown()
