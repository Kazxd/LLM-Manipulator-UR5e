"""Score /detect_open against the known object positions in worlds/pick_place.sdf.

  ros2 run ur_perception detect_open_check                       # all 6 objects (the can is searched as 'yellow circle')
  ros2 run ur_perception detect_open_check --threshold 0.02
  ros2 run ur_perception detect_open_check "yellow can=yellow cylinder" "white block=white box"
      name=label tries another wording: the object `name` (must be a known object) is searched as `label`.
      A plain argument is searched under its own name.
Needs: sim running (arm at home), detect_open running. The first call is slow because the model loads.
Positions below must match the SDF.
"""
import math, sys, rclpy
from rclpy.node import Node
from ur_interfaces.srv import DetectOpen

# name -> (x, y) of the object centre at start; keep in sync with the world file
GROUND_TRUTH = {
    "red cube": (0.50, 0.10), "green cube": (0.70, 0.12), "blue cube": (0.60, -0.15),
    "yellow can": (0.72, -0.30), "purple ball": (0.38, 0.34), "white block": (0.80, 0.00),
}
# wording that works best from the overhead camera (found by trial, session 5): a can looks like a circle
DEFAULT_QUERY = {"yellow can": "yellow circle"}
MATCH_RADIUS = 0.06      # a detection within 6 cm of an object counts as that object
EXPECTED_TOP = 0.25      # all objects are 5 cm tall on a 0.20 m table


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("__")]
    thr = 0.0
    if "--threshold" in args:
        i = args.index("--threshold")
        thr = float(args[i + 1])
        del args[i:i + 2]
    query = {}                                   # object name -> label used in the query
    for a in (args or [f"{n}={DEFAULT_QUERY[n]}" if n in DEFAULT_QUERY else n for n in GROUND_TRUTH]):
        name, _, label = a.partition("=")
        name, label = name.strip(), (label.strip() or name.strip())
        query[name] = label
    labels = list(dict.fromkeys(query.values()))

    rclpy.init()
    node = Node("detect_open_check")
    cli = node.create_client(DetectOpen, "/detect_open")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("/detect_open not available (ros2 run ur_perception detect_open)")
        rclpy.shutdown()
        return
    req = DetectOpen.Request()
    req.labels, req.threshold = labels, float(thr)
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=180.0)
    res = fut.result()
    if res is None:
        print("timed out waiting for /detect_open")
        rclpy.shutdown()
        return
    print(f"service: {res.success} - {res.message}\n")
    dets = [(i, l, s, p.pose.position) for i, l, s, p in zip(res.ids, res.found_labels, res.scores, res.poses)]
    used, hits, right_label = set(), 0, 0
    wanted = [n for n in query if n in GROUND_TRUTH]
    for name in wanted:
        gx, gy = GROUND_TRUTH[name]
        best = None
        for j, (i, l, s, p) in enumerate(dets):
            d = math.hypot(p.x - gx, p.y - gy)
            if d <= MATCH_RADIUS and (best is None or d < best[0]):
                best = (d, j)
        if best is None:
            print(f"MISS   {name:<12} (searched as '{query[name]}') nothing within "
                  f"{MATCH_RADIUS * 100:.0f} cm of ({gx:.2f}, {gy:.2f})")
            continue
        used.add(best[1])
        i, l, s, p = dets[best[1]]
        hits += 1
        ok_label = l.lower() == query[name].lower()
        right_label += ok_label
        print(f"{'OK    ' if ok_label else 'WRONG '} {name:<12} err {best[0] * 100:4.1f} cm  "
              f"top z {p.z:.3f} (want ~{EXPECTED_TOP})  label '{l}'  score {s:.2f}")
    for j, (i, l, s, p) in enumerate(dets):
        if j not in used:
            print(f"EXTRA  {l:<12} at ({p.x:.2f}, {p.y:.2f}) top z {p.z:.3f} score {s:.2f}  (no known object here)")
    print(f"\nlocalised {hits}/{len(wanted)}, correct label {right_label}/{len(wanted)}, "
          f"extra detections {len(dets) - len(used)}")
    rclpy.shutdown()
