"""Find good grasp settings by measuring them (ground truth from Gazebo, no LLM involved).

  ros2 run ur_llm_bridge grasp_sweep                          # 5 heights x 3 squeezes x 3 trials (about 30 min)
  ros2 run ur_llm_bridge grasp_sweep --dz 0 0.005 --squeeze 0.08 --trials 4
  ros2 run ur_llm_bridge grasp_sweep --colors red --trials 5
  ros2 run ur_llm_bridge grasp_sweep --colors can ball block --dz -0.0075 --squeeze 0.08 --trials 3   # extra objects

For every setting it resets the scene, picks a cube (colours rotate), waits 2 s and checks that the cube is still
lifted (z > 0.30) according to Gazebo. grasp_retries is forced to 0 so every failed grasp counts.
Needs: sim, detect_objects, skill_server (the NEW one with live parameters). Do not run llm_bridge at the same time.
grasp_dz: + raises the pads relative to the cube centre, - lowers them.
"""
import argparse, csv, os, re, subprocess, sys, time
import rclpy
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from rclpy.utilities import remove_ros_args
from ur_interfaces.action import Pick
from ur_llm_bridge import bench_lib as B
from ur_llm_bridge.benchmark import BenchNode, _spin

LIFT_Z = 0.30
EXTRA_MODELS = {"can": "yellow_can", "ball": "purple_ball", "block": "white_block"}   # need detect_open running
DEFAULTS = {"grasp_dz": -0.0075, "squeeze": 0.08, "grasp_retries": 2}


def gz_poses(names, timeout=10):
    """Ground truth straight from Gazebo (`gz topic -e`): the ROS pose topic only reports objects that move."""
    try:
        txt = subprocess.run(["gz", "topic", "-e", "-t", B.POSE_TOPIC, "-n", "1"],
                             capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return {}
    out = {}
    for name in names:
        for m in re.finditer(r'name:\s*"%s"' % name, txt):
            seg = txt[m.end(): m.end() + 600]
            nxt = re.search(r'name:\s*"', seg)
            if nxt:
                seg = seg[:nxt.start()]
            pm = re.search(r'position\s*\{([^}]*)\}', seg)
            if not pm:
                continue
            v = {"x": 0.0, "y": 0.0, "z": 0.0}      # zero fields are omitted in this format
            for k, val in re.findall(r'([xyz]):\s*([-+0-9.eE]+)', pm.group(1)):
                v[k] = float(val)
            out[name] = (v["x"], v["y"], v["z"])
    return out


def set_params(node, cli, **kw):
    req = SetParameters.Request()
    for k, v in kw.items():
        pv = ParameterValue()
        if isinstance(v, int) and not isinstance(v, bool):
            pv.type, pv.integer_value = ParameterType.PARAMETER_INTEGER, int(v)
        else:
            pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, float(v)
        req.parameters.append(ParamMsg(name=k, value=pv))
    fut = cli.call_async(req)
    if not node._wait(fut, 5.0):
        return False
    return all(r.successful for r in fut.result().results)


def one_trial(node, color):
    ok, msg = node.reset_scene()
    if not ok:
        return False, "reset failed: " + msg, None
    g = Pick.Goal()
    g.color = color
    ok, msg = node.call_action(node.pick_cli, g)
    time.sleep(2.0)
    if color in EXTRA_MODELS:
        model = EXTRA_MODELS[color]
        p = gz_poses([model]).get(model)
    else:
        p = (node.get_poses() or {}).get(f"{color}_cube")
    z = p[2] if p else None
    held = bool(ok and z is not None and z > LIFT_Z)
    if ok and not held:
        msg += f" | skill said ok but object z={z}" + (" (no ground-truth pose from gz topic)" if z is None else "")
    return held, msg, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", nargs="*", type=float, default=[-0.015, -0.0075, 0.0, 0.0075, 0.015])
    ap.add_argument("--squeeze", nargs="*", type=float, default=[0.05, 0.08, 0.12])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--colors", nargs="*", default=["red", "green", "blue"])
    ap.add_argument("--out", default="benchmark_results")
    args, _ = ap.parse_known_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = BenchNode()
    _spin(node)
    cli = node.create_client(SetParameters, "/skill_server/set_parameters")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("/skill_server parameters not available: run the NEW skill_server")
        rclpy.shutdown()
        return
    t0 = time.time()
    while node.get_poses() is None and time.time() - t0 < 10:
        time.sleep(0.5)
    if node.get_poses() is None:
        print("No ground-truth cube poses from Gazebo (pose bridge?).")
        rclpy.shutdown()
        return

    set_params(node, cli, grasp_retries=0)
    rows, n = [], 0
    total = len(args.dz) * len(args.squeeze) * args.trials
    for dz in args.dz:
        for sq in args.squeeze:
            if not set_params(node, cli, grasp_dz=dz, squeeze=sq):
                print("could not set parameters")
                continue
            wins, fails = 0, []
            for t in range(args.trials):
                color = args.colors[(n) % len(args.colors)]
                held, msg, z = one_trial(node, color)
                n += 1
                wins += held
                if not held:
                    fails.append(msg[:70])
                print(f"[{n}/{total}] dz={dz:+.4f} squeeze={sq:.3f} {color:<5} "
                      f"{'HELD' if held else 'FAIL'}  z={z if z is None else round(z, 3)}  {'' if held else msg[:90]}")
            rows.append({"grasp_dz": dz, "squeeze": sq, "trials": args.trials, "held": wins,
                         "rate": wins / args.trials, "failures": " || ".join(fails)})

    node.reset_scene()
    set_params(node, cli, **DEFAULTS)
    rows.sort(key=lambda r: -r["rate"])
    print("\nRESULTS (best first)")
    print(f"{'grasp_dz':>9} {'squeeze':>8} {'held':>6}")
    for r in rows:
        print(f"{r['grasp_dz']:>+9.4f} {r['squeeze']:>8.3f} {r['held']:>3}/{r['trials']:<2}")
    if rows:
        best = rows[0]
        print(f"\nbest: ros2 run ur_skills skill_server --ros-args -p grasp_dz:={best['grasp_dz']} "
              f"-p squeeze:={best['squeeze']}")
        if best["rate"] < 1.0:
            print("no setting was perfect; send me this table and the failure lines above.")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"grasp_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["none"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")
    rclpy.shutdown()
