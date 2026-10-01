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
# history trimming keeps whole turns
hh = new_history()
for k in range(30):
    hh += [{"role": "user", "content": str(k)}, {"role": "assistant", "content": "a"}]
trim(hh, 20); assert hh[0]["role"] == "system" and hh[1]["role"] == "user" and len(hh) <= 20
print("all agent tests passed")
