"""Run: python3 test/test_bench_lib.py   (no ROS needed)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ur_llm_bridge import bench_lib as B

I = dict(B.INIT)
def P(**kw):
    p = dict(I); p.update({k + "_cube": v for k, v in kw.items()}); return p

# case table sanity
ids = [c[0] for c in B.CASES]
assert len(ids) == len(set(ids)) == 23, len(ids)

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

# relational checks: reference blue at (0.6, -0.15); left = +y, right = -y, front = -x (toward robot), behind = +x
assert ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.05, 0.225)), I)[0]
assert not ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.25, 0.225)), I)[0]     # that is the right side
assert not ev(("rel", "red", "blue", "left"), P(red=(0.68, -0.05, 0.225)), I)[0]    # too far off-axis
assert not ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.12, 0.225)), I)[0]     # touching
assert ev(("rel", "red", "blue", "right"), P(red=(0.6, -0.25, 0.225)), I)[0]
assert ev(("rel", "red", "blue", "front"), P(red=(0.5, -0.15, 0.225)), I)[0]
assert ev(("rel", "red", "blue", "behind"), P(red=(0.7, -0.15, 0.225)), I)[0]
assert not ev(("rel", "red", "blue", "left"), P(red=(0.6, -0.05, 0.30)), I)[0]      # in the air

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
'''
g = B.parse_gz_poses(txt)
assert g["red_cube"] == (0.5, 0.1, 0.225), g
assert g["green_cube"] == (0.7, 0.0, 0.225), g          # y omitted -> 0
assert g["blue_cube"] == (0.6, -0.15, 0.225), g

fake = [dict(id="a", category="pick", passed=True, tool_failures=0, rejected=0, tool_calls=1, wall_s=10, llm_s=3, detail="", reply=""),
        dict(id="b", category="pick", passed=False, tool_failures=1, rejected=0, tool_calls=2, wall_s=20, llm_s=6, detail="bad", reply="x")]
out = B.summarize(fake); assert "OVERALL" in out and "FAILURES" in out
print("all bench_lib tests passed")
