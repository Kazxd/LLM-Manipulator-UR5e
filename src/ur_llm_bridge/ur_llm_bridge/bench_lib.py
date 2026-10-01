"""ROS-free benchmark pieces: test cases, ground-truth checks, gz fallback parser, summary."""
import math, re, subprocess
from collections import defaultdict

WORLD = "pick_place"
POSE_TOPIC = f"/world/{WORLD}/pose/info"
CUBES = ("red_cube", "green_cube", "blue_cube")
TABLE_Z = 0.225                      # cube centre when resting on the table
CUBE = 0.05
INIT = {"red_cube": (0.5, 0.1, TABLE_Z),
        "green_cube": (0.7, 0.12, TABLE_Z),
        "blue_cube": (0.6, -0.15, TABLE_Z)}

# (id, category, command, check)   check specs: see evaluate()
CASES = [
    ("pick_red",        "pick",       "pick up the red cube",   ("held", "red")),
    ("pick_green",      "pick",       "pick up the green cube", ("held", "green")),
    ("pick_blue",       "pick",       "pick up the blue cube",  ("held", "blue")),
    ("stack_red_blue",  "stack",      "put the red cube on the blue one",        ("on", "red", "blue")),
    ("stack_green_red", "stack",      "put the green cube on the red cube",      ("on", "green", "red")),
    ("stack_blue_green", "stack",     "put the blue cube on top of the green one", ("on", "blue", "green")),
    ("stack_red_green", "stack",      "stack red on top of green",               ("on", "red", "green")),
    ("xy_red",          "place_xy",   "move the red cube to x 0.5 and y -0.25",  ("at", "red", 0.5, -0.25)),
    ("xy_green",        "place_xy",   "place the green cube on the table at x 0.45, y 0.25", ("at", "green", 0.45, 0.25)),
    ("xy_blue",         "place_xy",   "put the blue cube at x 0.4 y 0.0",        ("at", "blue", 0.4, 0.0)),
    ("multi_stack3",    "multistep",  "stack all three cubes: red on the bottom, green in the middle, blue on top",
        ("all", [("on", "green", "red"), ("on", "blue", "green")])),
    ("multi_there_back", "multistep", "put the red cube on the blue one, then put it back on the table at x 0.5 y 0.1",
        ("at", "red", 0.5, 0.1)),
    ("multi_xy_then_on", "multistep", "put the blue cube on the table at x 0.6 y 0.3, then put the red cube on top of it",
        ("all", [("at", "blue", 0.6, 0.3), ("on", "red", "blue")])),
    ("multi_two_moves", "multistep",  "move the green cube to x 0.45 y 0.25 and the blue cube to x 0.45 y -0.25",
        ("all", [("at", "green", 0.45, 0.25), ("at", "blue", 0.45, -0.25)])),
    ("rel_next_to",     "relational", "move the green cube next to the blue cube", ("near", "green", "blue")),
    ("rel_left",        "relational", "put the red cube to the left of the blue cube",
        ("rel", "red", "blue", "left")),
    ("rel_right",       "relational", "place the blue cube on the right side of the green cube",
        ("rel", "blue", "green", "right")),
    ("rel_front",       "relational", "put the red cube in front of the green cube",
        ("rel", "red", "green", "front")),
    ("info_objects",    "info",       "what objects are on the table?",
        ("all", [("unchanged",), ("reply_has", "red", "green", "blue")])),
    ("info_home",       "info",       "go to the home position",  ("unchanged",)),
    ("refuse_purple",   "refusal",    "pick up the purple cube",  ("unchanged",)),
    ("refuse_ambiguous", "refusal",   "move the cube to the left", ("unchanged",)),
    ("refuse_range",    "refusal",    "put the red cube at x 3 and y 3", ("unchanged",)),
]

# direction -> (axis index along which it points, sign); base frame, seen from the robot looking along +x
REL_AXES = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, -1.0), "behind": (0, 1.0)}


def _c(name):
    return name if name.endswith("_cube") else name + "_cube"


def _dxy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _d3(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def evaluate(spec, poses, init, reply=""):
    """Return (passed, detail) by comparing final cube poses (world frame) with the spec."""
    kind = spec[0]
    if kind == "all":
        res = [evaluate(s, poses, init, reply) for s in spec[1]]
        return all(r[0] for r in res), "; ".join(r[1] for r in res)
    if kind == "unchanged":
        moved = [n for n in INIT if n not in poses or _d3(poses[n], init[n]) > 0.02]
        return (not moved), ("nothing moved" if not moved else f"moved: {moved}")
    if kind == "reply_has":
        missing = [w for w in spec[1:] if w.lower() not in reply.lower()]
        return (not missing), ("reply ok" if not missing else f"reply missing {missing}")
    a = _c(spec[1])
    if a not in poses:
        return False, f"{a} pose unknown"
    pa = poses[a]
    if kind == "held":
        return pa[2] > 0.30, f"{a} z={pa[2]:.3f} (need >0.30)"
    if kind == "at":
        d = _dxy(pa, (spec[2], spec[3]))
        ok = d < 0.05 and abs(pa[2] - TABLE_Z) < 0.02
        return ok, f"{a} {d:.3f} m from target, z={pa[2]:.3f}"
    b = _c(spec[2])
    if b not in poses:
        return False, f"{b} pose unknown"
    pb = poses[b]
    if kind == "on":
        d, dz = _dxy(pa, pb), pa[2] - pb[2]
        ok = d < 0.035 and abs(dz - CUBE) < 0.02
        return ok, f"{a} on {b}? dxy={d:.3f} dz={dz:.3f}"
    if kind == "near":
        d = _dxy(pa, pb)
        ok = 0.05 <= d <= 0.15 and abs(pa[2] - TABLE_Z) < 0.02
        return ok, f"{a}-{b} distance {d:.3f} (want 0.05-0.15), z={pa[2]:.3f}"
    if kind == "rel":
        axis, sign = REL_AXES[spec[3]]
        diff = (pa[0] - pb[0], pa[1] - pb[1])
        along, across = diff[axis] * sign, abs(diff[1 - axis])
        ok = 0.06 <= along <= 0.16 and across < 0.04 and abs(pa[2] - TABLE_Z) < 0.02
        return ok, (f"{a} {spec[3]} of {b}? along={along:.3f} (want 0.06-0.16) "
                    f"across={across:.3f} (want <0.04) z={pa[2]:.3f}")
    return False, f"unknown check {kind}"


def parse_gz_poses(text):
    """Parse `gz topic -e` protobuf text for the cube models (zero fields are omitted there)."""
    out = {}
    for name in CUBES:
        for m in re.finditer(r'name:\s*"%s"' % name, text):
            seg = text[m.end(): m.end() + 600]
            nxt = re.search(r'name:\s*"', seg)
            if nxt:
                seg = seg[:nxt.start()]
            pm = re.search(r'position\s*\{([^}]*)\}', seg)
            if not pm:
                continue
            v = {"x": 0.0, "y": 0.0, "z": 0.0}
            for k, val in re.findall(r'([xyz]):\s*([-+0-9.eE]+)', pm.group(1)):
                v[k] = float(val)
            out[name] = (v["x"], v["y"], v["z"])
    return out


def poses_from_gz(timeout=10):
    """Fallback ground truth straight from Gazebo (no ROS bridge needed)."""
    try:
        txt = subprocess.run(["gz", "topic", "-e", "-t", POSE_TOPIC, "-n", "1"],
                             capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None
    p = parse_gz_poses(txt)
    return p if all(c in p for c in CUBES) else None


def summarize(results):
    by = defaultdict(list)
    for r in results:
        by[r["category"]].append(r)

    def line(name, rs):
        n = len(rs)
        ok = sum(r["passed"] for r in rs)
        first = sum(r["passed"] and r["tool_failures"] == 0 and r["rejected"] == 0 for r in rs)
        calls = sum(r["tool_calls"] for r in rs) / n
        t = sum(r["wall_s"] for r in rs) / n
        llm = sum(r["llm_s"] for r in rs) / n
        return f"{name:<11}{ok:>3}/{n:<3} pass {100 * ok / n:5.0f}%  clean {100 * first / n:4.0f}%  tools {calls:4.1f}  time {t:5.1f}s  llm {llm:5.1f}s"

    lines = [line(c, rs) for c, rs in by.items()] + ["-" * 72, line("OVERALL", results)]
    bad = [r for r in results if not r["passed"]]
    if bad:
        lines += ["", "FAILURES:"]
        for r in bad:
            lines.append(f"- {r['id']}: {r['detail']}" + (f"  [error: {r['error']}]" if r.get("error") else ""))
            lines.append(f"    reply: {r['reply'][:120]}")
    return "\n".join(lines)
