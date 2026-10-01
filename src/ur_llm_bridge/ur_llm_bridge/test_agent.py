"""Run: python3 test/test_agent.py   (no ROS or Ollama needed)"""
import json, sys, os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ur_llm_bridge.agent import *

SCRIPT = [
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "pick_object", "arguments": {"color": "red"}}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "place_object", "arguments": {"target": "blue"}}}]},
    {"role": "assistant", "content": "<think>ok</think>Put the red cube on the blue one."},
]
class H(BaseHTTPRequestHandler):
    i = 0
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"message": SCRIPT[H.i]}).encode(); H.i += 1
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
host = f"http://127.0.0.1:{srv.server_port}"
calls = []
def execute(name, args): calls.append((name, args)); return {"success": True, "message": "ok"}
h = new_history()
out = run_agent(h, "put red on blue", execute, lambda m: ollama_chat(host, "x", m, TOOLS),
                scene_fn=lambda: "objects: red, blue. Holding: nothing.", log=lambda *_: None)
assert calls == [("pick_object", {"color": "red"}), ("place_object", {"target": "blue"})], calls
assert out == "Put the red cube on the blue one.", out
assert h[1]["content"].startswith("[Current scene]")
assert [m["role"] for m in h] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]

# validation
assert validate("pick_object", {"color": "purple"})[1]
assert validate("place_object", {"x": 5, "y": 0})[1]
assert validate("place_object", {"x": 0.5, "y": 0.1})[0]["target"] == ""
assert validate("fly", {})[1]
# place_relative validation
v = validate("place_relative", {"reference": "Blue", "relation": "left"})[0]
assert v == {"reference": "blue", "relation": "left", "distance": 0.1}, v
assert validate("place_relative", {"reference": "blue", "relation": "next to", "distance_cm": 15})[0]["relation"] == "next_to"
assert validate("place_relative", {"reference": "blue", "relation": "up"})[1]
assert validate("place_relative", {"reference": "purple", "relation": "left"})[1]
assert validate("place_relative", {"reference": "blue", "relation": "left", "distance_cm": 2})[1]    # too close
assert validate("place_relative", {"reference": "blue", "relation": "left", "distance_cm": "far"})[1]
assert validate("place_relative", {"reference": "blue", "relation": "left", "distance_cm": None})[1]
assert any(t["function"]["name"] == "place_relative" for t in TOOLS)

# compute_relative: left=+y, right=-y, front=-x (toward robot), behind=+x
def near(a, b): return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
ref = (0.6, -0.15)
assert near(compute_relative(ref, "left", 0.10, {})[0], (0.6, -0.05))
assert near(compute_relative(ref, "right", 0.10, {})[0], (0.6, -0.25))
assert near(compute_relative(ref, "front", 0.10, {})[0], (0.5, -0.15))
assert near(compute_relative(ref, "behind", 0.10, {})[0], (0.7, -0.15))
assert compute_relative((0.35, 0.0), "front", 0.10, {})[0] is None                   # x would be 0.25
assert compute_relative((0.6, 0.35), "left", 0.10, {})[0] is None                    # y would be 0.45
assert compute_relative(ref, "left", 0.10, {"red": (0.6, -0.05)})[0] is None         # occupied
xy, side = compute_relative(ref, "next_to", 0.10, {"red": (0.6, -0.05)})             # next_to skips the blocked side
assert near(xy, (0.6, -0.25)) and side == "right", (xy, side)
assert compute_relative((0.35, 0.0), "next_to", 0.10, {})[1] == "left"               # first free side
msg = compute_relative(ref, "left", 0.10, {"red": (0.6, -0.05)})[1]
assert "red" in msg, msg

# a tool call written as text is nudged, and a stuck model gets an honest reply (not raw tool text)
class Stuck:
    n = 0
    def __call__(self, msgs):
        Stuck.n += 1
        return {"role": "assistant", "content": 'pick_object(color="red")'}
r = run_agent(new_history(), "pick red", execute, Stuck(), log=lambda *_: None)
assert Stuck.n == 3 and "pick_object" not in r, (Stuck.n, r)

# history trimming keeps whole turns
hh = new_history()
for k in range(30):
    hh += [{"role": "user", "content": str(k)}, {"role": "assistant", "content": "a"}]
trim(hh, 20); assert hh[0]["role"] == "system" and hh[1]["role"] == "user" and len(hh) <= 20
print("all agent tests passed")
