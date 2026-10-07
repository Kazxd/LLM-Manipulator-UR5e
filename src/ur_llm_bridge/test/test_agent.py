"""Run: python3 src/ur_llm_bridge/test/test_agent.py   (no ROS or Ollama needed)"""
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
    last = None
    stats = {}
    def do_POST(self):
        H.last = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"message": dict(SCRIPT[H.i]), **H.stats}).encode(); H.i += 1
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
assert "/no_think" not in h[1]["content"]          # appended on a copy only
assert [m["role"] for m in h] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]

# validation
assert validate("pick_object", {"color": "purple"})[1]
assert validate("place_object", {"x": 5, "y": 0})[1]
assert validate("place_object", {"x": 0.5, "y": 0.1})[0]["target"] == ""
assert validate("fly", {})[1]

# extra objects: can / ball / block (names, aliases, rules)
for raw, want in (("can", "can"), ("Yellow Can", "can"), ("yellow_can", "can"), ("ball", "ball"),
                  ("purple ball", "ball"), ("block", "block"), ("white block", "block"),
                  ("Red", "red"), ("red cube", "red"), ("object", None), ("purple", None), ("purple cube", None),
                  ("", None), (None, None)):
    assert canon_name(raw) == want, (raw, canon_name(raw))
assert validate("pick_object", {"color": "can"})[0] == {"color": "can"}
assert validate("pick_object", {"color": "Purple Ball"})[0] == {"color": "ball"}
assert validate("pick_object", {"object": "block"})[0] == {"color": "block"}          # tolerated alias for the argument
assert validate("pick_object", {"color": "purple cube"})[1]                             # there is no purple cube
assert validate("pick_object", {"color": "orange"})[1]
assert validate("place_object", {"target": "can"})[0] == {"target": "can"}
assert validate("place_object", {"target": "white block"})[0] == {"target": "block"}
assert validate("place_object", {"target": "ball"})[1] and "round" in validate("place_object", {"target": "ball"})[1]
assert validate("place_object", {"target": "purple"})[1]
assert validate("place_object", {"target": "", "x": 0.5, "y": 0.0})[0] == {"target": "", "x": 0.5, "y": 0.0}
assert validate("place_in_zone", {"zone": "right"})[0] == {"zone": "right"}           # works for a held extra object too
assert validate("sort_cubes", {"assignments": {"can": "left"}})[1]                     # cubes only
assert validate("build_tower", {"order": ["red", "can"]})[1]                           # cubes only
assert validate("place_relative", {"reference": "can", "relation": "left"})[1]         # reference must be a cube
tool = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in TOOLS}
assert tool["pick_object"]["color"]["enum"] == ["red", "green", "blue", "can", "ball", "block"]
assert tool["place_object"]["target"]["enum"] == ["red", "green", "blue", "can", "ball", "block", ""]
assert "can" in SYSTEM and "ball" in SYSTEM and "block" in SYSTEM and "no purple cube" in SYSTEM
# a scripted run: pick the can, put it on the red cube
SCRIPT[:] = [
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "pick_object", "arguments": {"color": "yellow can"}}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "place_object", "arguments": {"target": "red"}}}]},
    {"role": "assistant", "content": "Done."},
]
H.i = 0
calls.clear()
out = run_agent(new_history(), "put the can on the red cube", execute, lambda m: ollama_chat(host, "x", m, TOOLS),
                log=lambda *_: None)
assert calls == [("pick_object", {"color": "can"}), ("place_object", {"target": "red"})], calls
# a place on the ball is rejected before it reaches the robot
SCRIPT[:] = [
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "place_object", "arguments": {"target": "ball"}}}]},
    {"role": "assistant", "content": "I cannot stack on the ball."},
]
H.i = 0
calls.clear()
out = run_agent(new_history(), "put it on the ball", execute, lambda m: ollama_chat(host, "x", m, TOOLS),
                log=lambda *_: None)
assert calls == [] and out == "I cannot stack on the ball.", (calls, out)

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

# build_tower validation
assert validate("build_tower", {"order": ["Red", "green", "blue"]})[0] == {"order": ["red", "green", "blue"]}
assert validate("build_tower", {"order": "red, green"})[0] == {"order": ["red", "green"]}
assert validate("build_tower", {"order": ["red"]})[1]
assert validate("build_tower", {"order": ["red", "red"]})[1]
assert validate("build_tower", {"order": ["red", "purple"]})[1]
assert validate("build_tower", {})[1]
assert any(t["function"]["name"] == "build_tower" for t in TOOLS)

# zones
assert validate("place_in_zone", {"zone": "Left"})[0] == {"zone": "left"}
assert validate("place_in_zone", {"zone": "left zone"})[0] == {"zone": "left"}
assert validate("place_in_zone", {"zone": "middle"})[1]
v = validate("sort_cubes", {"assignments": {"Red": "left", "blue": "right"}})[0]
assert v == {"assignments": {"red": "left", "blue": "right"}}, v
assert validate("sort_cubes", {"assignments": '{"red": "left"}'})[0] == {"assignments": {"red": "left"}}
assert validate("sort_cubes", {"assignments": {}})[1]
assert validate("sort_cubes", {"assignments": {"purple": "left"}})[1]
assert validate("sort_cubes", {"assignments": {"red": "up"}})[1]
assert validate("sort_cubes", {})[1]
assert any(t["function"]["name"] == "sort_cubes" for t in TOOLS)
assert any(t["function"]["name"] == "place_in_zone" for t in TOOLS)
assert zone_contains("left", 0.51, 0.24) and not zone_contains("left", 0.51, 0.10)
assert not zone_contains("left", 0.55, 0.38)                                   # half a cube off the pad edge
xy, _ = zone_slot("left", {})
assert abs(xy[0] - 0.51) < 1e-9 and abs(xy[1] - 0.24) < 1e-9, xy
xy, _ = zone_slot("left", {"red": (0.51, 0.24)})                                # first slot taken -> next
assert abs(xy[0] - 0.59) < 1e-9 and abs(xy[1] - 0.24) < 1e-9, xy
full = {"a": (0.51, 0.24), "b": (0.59, 0.24), "c": (0.51, 0.32), "d": (0.59, 0.32)}
assert zone_slot("left", full)[0] is None and "full" in zone_slot("left", full)[1]
for zn in ZONES:                                                                # every slot is on its pad
    for dx, dy in SLOT_OFFSETS:
        assert zone_contains(zn, ZONE_CENTERS[zn][0] + dx, ZONE_CENTERS[zn][1] + dy)

# clean_text strips reasoning, including a bare </think>
assert clean_text("<think>x</think>Hi") == "Hi"
assert clean_text("reasoning...</think>\n\nDone.") == "Done."
assert clean_text(None) == ""

# salvage_calls: only when the WHOLE message is a call
s = salvage_calls('pick_object(color="red")')
assert s and s[0]["function"] == {"name": "pick_object", "arguments": {"color": "red"}}, s
s = salvage_calls('{"name": "move_home", "arguments": {}}')
assert s and s[0]["function"]["name"] == "move_home", s
s = salvage_calls('place_object(target="", x=0.5, y=-0.2)')
assert s and s[0]["function"]["arguments"] == {"target": "", "x": 0.5, "y": -0.2}, s
assert salvage_calls('I will now run pick_object(color="red") for you.') == []
assert salvage_calls("All done.") == []

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

# a text-only call (whole message) is salvaged and executed
class Salv:
    n = 0
    def __call__(self, msgs):
        Salv.n += 1
        if Salv.n == 1:
            return {"role": "assistant", "content": "</think>\nmove_home()"}
        return {"role": "assistant", "content": "Done."}
calls.clear()
r = run_agent(new_history(), "go home", execute, Salv(), log=lambda *_: None)
assert r == "Done." and calls == [("move_home", {})], (r, calls)

# a call buried in prose is nudged (with the request restated); a stuck model gets an honest reply
seen = []
class Stuck:
    n = 0
    def __call__(self, msgs):
        Stuck.n += 1
        seen.append(msgs[-1]["content"])
        return {"role": "assistant", "content": 'I will now run pick_object(color="red") for you.'}
r = run_agent(new_history(), "pick red", execute, Stuck(), log=lambda *_: None)
assert Stuck.n == 3 and "pick_object" not in r, (Stuck.n, r)
assert any("pick red" in s for s in seen[1:]), seen          # the nudge restates the request

# reasoning text alone must not trigger a nudge
class Reasoner:
    n = 0
    def __call__(self, msgs):
        Reasoner.n += 1
        return {"role": "assistant",
                "content": "Okay, I should call pick_object( ... later.\n</think>\n\nDone."}
r = run_agent(new_history(), "hi", execute, Reasoner(), log=lambda *_: None)
assert Reasoner.n == 1 and r == "Done.", (Reasoner.n, r)

# history trimming keeps whole turns
hh = new_history()
for k in range(30):
    hh += [{"role": "user", "content": str(k)}, {"role": "assistant", "content": "a"}]
trim(hh, 20); assert hh[0]["role"] == "system" and hh[1]["role"] == "user" and len(hh) <= 20

# ---- latency features ----
# request options: fixed context and a hard cap on generated tokens
H.i = 0
SCRIPT[:] = [{"role": "assistant", "content": "ok"}]
ollama_chat(host, "x", new_history(), TOOLS)
opt = H.last["options"]
assert opt["num_predict"] == NUM_PREDICT == 256 and opt["num_ctx"] == NUM_CTX == 4096 and opt["temperature"] == 0, opt
# timing stats from Ollama are attached to the message, logged by run_agent, and never stored in the history
H.i = 0
H.stats = {"load_duration": 2_000_000_000, "prompt_eval_count": 900, "prompt_eval_duration": 3_000_000_000,
           "eval_count": 25, "eval_duration": 1_500_000_000}
SCRIPT[:] = [{"role": "assistant", "content": "Hello."}]
logs = []
hh = new_history()
out = run_agent(hh, "hi", execute, lambda m: ollama_chat(host, "x", m, TOOLS), log=logs.append)
assert out == "Hello." and any("prompt 900 tok in 3.0s" in l and "output 25 tok in 1.5s" in l for l in logs), logs
assert all("_stats" not in m for m in hh)
H.stats = {}
# the prompt got shorter (latency): system prompt and tool schemas are compact
assert len(SYSTEM) < 2400, len(SYSTEM)
assert len(json.dumps(TOOLS)) < 3300, len(json.dumps(TOOLS))
# fast macros: a successful macro tool is reported without the closing LLM call
n_chat = []
def chat_macro(msgs):
    n_chat.append(1)
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "build_tower", "arguments": {"order": ["red", "green"]}}}]}
res_ok = {"success": True, "message": "tower built, bottom to top: red, green", "stacked": 2}
r = run_agent(new_history(), "stack green on red", lambda n, a: res_ok, chat_macro, fast_macros=True, log=lambda *_: None)
assert r == "Done: tower built, bottom to top: red, green." and len(n_chat) == 1, (r, n_chat)
# ... but not on failure, not for pick/place, and not when disabled
class Seq:
    def __init__(self, first_call):
        self.first, self.n = first_call, 0
    def __call__(self, msgs):
        self.n += 1
        if self.n == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{"function": self.first}]}
        return {"role": "assistant", "content": "Finished."}
res_bad = {"success": False, "message": "pick green failed"}
c = Seq({"name": "build_tower", "arguments": {"order": ["red", "green"]}})
assert run_agent(new_history(), "x", lambda n, a: res_bad, c, fast_macros=True, log=lambda *_: None) == "Finished." and c.n == 2
c = Seq({"name": "pick_object", "arguments": {"color": "red"}})
assert run_agent(new_history(), "x", lambda n, a: res_ok, c, fast_macros=True, log=lambda *_: None) == "Finished." and c.n == 2
c = Seq({"name": "build_tower", "arguments": {"order": ["red", "green"]}})
assert run_agent(new_history(), "x", lambda n, a: res_ok, c, fast_macros=False, log=lambda *_: None) == "Finished." and c.n == 2
print("all agent tests passed")
