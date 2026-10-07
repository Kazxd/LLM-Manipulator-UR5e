"""Run: python3 src/ur_llm_bridge/test/test_bench_lib.py   (no ROS needed)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ur_llm_bridge import bench_lib as B

I = dict(B.INIT)
def P(**kw):
    p = dict(I); p.update({k + "_cube": v for k, v in kw.items()}); return p

# case table sanity
ids = [c[0] for c in B.CASES]
assert len(ids) == len(set(ids)) == 35, len(ids)
assert "multi_tower_brg" in ids and "refuse_orange" in ids and "refuse_purple" not in ids
assert sum(1 for c in B.CASES if c[1] == "objects") == 8

ev = B.evaluate
assert ev(("unchanged",), I, I)[0]
assert not ev(("unchanged",), P(red=(0.5, 0.2, 0.225)), I)[0]
assert ev(("held", "red"), P(red=(0.5, 0.1, 0.35)), I)[0]
assert not ev(("held", "red"), I, I)[0]
# red on blue: blue centre (0.6,-0.15,0.225), red centre 0.05 above
assert ev(("on", "red", "blue"), P(red=(0.6, -0.15, 0.275)), I)[0]
assert not ev(("on", "red", "blue"), P(red=(0.6, -0.15, 0.225)), I)[0]      # overlapping, not on top
assert not ev(("on", "red", "blue"), P(red=(0.7, -0.15, 0.275)), I)[0]      # offset 10 cm
assert ev(("at", "red", 0.5, -0.25), P(red=(0.51, -0.24, 0.226)), I)[0]
assert not ev(("at", "red", 0.5, -0.25), P(red=(0.5, -0.25, 0.30)), I)[0]   # still in the air
assert ev(("near", "green", "blue"), P(green=(0.6, -0.08, 0.225)), I)[0]
assert not ev(("near", "green", "blue"), P(green=(0.6, -0.14, 0.225)), I)[0]  # touching/colliding
assert ev(("reply_has", "red", "blue"), I, I, "I see Red and BLUE")[0]
assert ev(("all", [("unchanged",), ("reply_has", "red")]), I, I, "red")[0]

# tower blue / red / green (bottom to top) on blue's spot
spec = [c for c in B.CASES if c[0] == "multi_tower_brg"][0][3]
assert ev(spec, P(red=(0.6, -0.15, 0.275), green=(0.6, -0.15, 0.325)), I)[0]
assert not ev(spec, P(red=(0.6, -0.15, 0.275), green=(0.7, 0.12, 0.225)), I)[0]   # green never moved

# zone checks: left pad centre (0.55, 0.28), right (0.55, -0.28); cube must be fully on the pad
assert ev(("in_zone", "red", "left"), P(red=(0.51, 0.24, 0.225)), I)[0]
assert not ev(("in_zone", "red", "left"), P(red=(0.51, 0.24, 0.30)), I)[0]          # in the air
assert not ev(("in_zone", "red", "left"), P(red=(0.51, 0.10, 0.225)), I)[0]         # off the pad
assert not ev(("in_zone", "red", "left"), P(red=(0.51, -0.24, 0.225)), I)[0]        # wrong zone
assert ev(("in_zone", "red", "right"), P(red=(0.59, -0.32, 0.225)), I)[0]
zspec = [c for c in B.CASES if c[0] == "zone_sort3"][0][3]
assert ev(zspec, P(red=(0.51, 0.24, 0.225), green=(0.59, 0.24, 0.225), blue=(0.51, -0.24, 0.225)), I)[0]

# relational checks: reference blue at (0.6, -0.15); left = +y, right = -y, front = -x (toward robot), behind = +x
assert ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.05, 0.225)), I)[0]
assert not ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.25, 0.225)), I)[0]     # that is the right side
assert not ev(("rel", "red", "blue", "left"), P(red=(0.68, -0.05, 0.225)), I)[0]    # too far off-axis
assert not ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.12, 0.225)), I)[0]     # touching
assert ev(("rel", "red", "blue", "right"), P(red=(0.6, -0.25, 0.225)), I)[0]
assert ev(("rel", "red", "blue", "front"), P(red=(0.5, -0.15, 0.225)), I)[0]
assert ev(("rel", "red", "blue", "behind"), P(red=(0.7, -0.15, 0.225)), I)[0]
assert not ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.05, 0.30)), I)[0]      # in the air

# extra objects: names map to the Gazebo models, poses are keyed by model name
assert B._c("can") == "yellow_can" and B._c("ball") == "purple_ball" and B._c("block") == "white_block"
assert B._c("red") == "red_cube" and B._c("red_cube") == "red_cube"
assert set(B.EXTRA_MODELS.values()) <= set(B.CUBES) and len(B.CUBES) == 6
E = dict(I); E.update({"yellow_can": (0.72, -0.30, 0.225), "purple_ball": (0.38, 0.34, 0.225),
                       "white_block": (0.80, 0.0, 0.225)})
def Q(**kw):
    p = dict(E); p.update(kw); return p
assert ev(("held", "can"), Q(yellow_can=(0.72, -0.30, 0.343)), E)[0]
assert not ev(("held", "can"), E, E)[0]
assert not ev(("held", "ball"), E, E)[0] and ev(("held", "ball"), Q(purple_ball=(0.4, 0.3, 0.34)), E)[0]
assert ev(("held", "block"), Q(white_block=(0.8, 0.0, 0.34)), E)[0]
assert ev(("at", "can", 0.5, 0.0), Q(yellow_can=(0.51, 0.01, 0.226)), E)[0]
assert not ev(("at", "can", 0.5, 0.0), Q(yellow_can=(0.72, -0.30, 0.225)), E)[0]          # never moved
assert ev(("at", "ball", 0.5, 0.2), Q(purple_ball=(0.48, 0.21, 0.225)), E)[0]
# can on the red cube (red at (0.5, 0.1)), ball on the white block (0.8, 0.0)
assert ev(("on", "can", "red"), Q(yellow_can=(0.5, 0.1, 0.275)), E)[0]
assert not ev(("on", "can", "red"), Q(yellow_can=(0.5, 0.1, 0.225)), E)[0]                 # overlapping
assert ev(("on", "ball", "block"), Q(purple_ball=(0.8, 0.0, 0.275)), E)[0]
assert not ev(("on", "ball", "block"), Q(purple_ball=(0.8, 0.06, 0.275)), E)[0]            # rolled off
assert ev(("in_zone", "can", "right"), Q(yellow_can=(0.51, -0.32, 0.225)), E)[0]
assert not ev(("in_zone", "can", "right"), E, E)[0]
assert not ev(("held", "can"), I, I)[0] and "pose unknown" in ev(("held", "can"), I, I)[1]  # extras missing -> clear failure
assert ev(("unchanged",), Q(yellow_can=(0.1, 0.1, 0.5)), E)[0]                            # `unchanged` only watches the cubes

# gz protobuf-text fallback parser (zero fields are omitted in this format!)
txt = '''
pose { name: "ground" id: 1 }
pose {
  name: "red_cube"
  id: 7
  position {
    x: 0.5
    y: 0.1
    z: 0.225
  }
  orientation { w: 1 }
}
pose {
  name: "link"
  id: 8
  position { x: 9 }
}
pose {
  name: "green_cube"
  id: 9
  position {
    x: 0.7
    z: 0.225
  }
}
pose { name: "blue_cube" id: 10 position { x: 0.6 y: -0.15 z: 0.225 } }
pose { name: "yellow_can" id: 11 position { x: 0.72 y: -0.3 z: 0.225 } }
pose { name: "purple_ball" id: 12 position { x: 0.38 y: 0.34 z: 0.225 } }
pose { name: "white_block" id: 13 position { x: 0.8 z: 0.225 } }
'''
g = B.parse_gz_poses(txt)
assert g["red_cube"] == (0.5, 0.1, 0.225), g
assert g["green_cube"] == (0.7, 0.0, 0.225), g          # y omitted -> 0
assert g["blue_cube"] == (0.6, -0.15, 0.225), g
assert g["yellow_can"] == (0.72, -0.3, 0.225), g
assert g["purple_ball"] == (0.38, 0.34, 0.225), g
assert g["white_block"] == (0.8, 0.0, 0.225), g          # y omitted -> 0

fake = [dict(id="a", category="pick", passed=True, tool_failures=0, rejected=0, tool_calls=1, wall_s=10, llm_s=3, detail="", reply=""),
        dict(id="b", category="pick", passed=False, tool_failures=1, rejected=0, tool_calls=2, wall_s=20, llm_s=6, detail="bad", reply="x")]
out = B.summarize(fake); assert "OVERALL" in out and "FAILURES" in out
print("all bench_lib tests passed")
