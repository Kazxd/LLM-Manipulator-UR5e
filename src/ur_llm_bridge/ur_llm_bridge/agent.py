"""ROS-free LLM agent loop for Ollama tool calling (stdlib only, unit-testable)."""
import json, re, urllib.request

COLORS = ("red", "green", "blue")
X_RANGE = (0.30, 0.90)    # safe workspace on the table (metres, base frame)
Y_RANGE = (-0.40, 0.40)
TOOL_NAMES = ("pick_object", "place_object", "move_home", "detect_objects")


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _fn("detect_objects",
        "Look at the table and list coloured cubes with x,y position in metres. Use it to check the scene.",
        {"color": {"type": "string", "enum": ["red", "green", "blue", "all"]}}),
    _fn("pick_object",
        "Pick up the cube of the given colour. The robot must not already be holding something.",
        {"color": {"type": "string", "enum": list(COLORS)}}, ["color"]),
    _fn("place_object",
        "Place the held cube. Either on top of another cube (give target = its colour) "
        "or on the table at x,y metres (leave target empty).",
        {"target": {"type": "string", "enum": list(COLORS) + [""]},
         "x": {"type": "number"}, "y": {"type": "number"}}),
    _fn("move_home", "Move the arm to its home pose.", {}),
]

SYSTEM = f"""You control a UR5e robot arm that moves coloured cubes on a table, using tools.
Cubes: red, green, blue. Table workspace: x {X_RANGE[0]} to {X_RANGE[1]}, y {Y_RANGE[0]} to {Y_RANGE[1]} metres.
Rules:
- Call ONE tool at a time and wait for its result before the next call.
- To move a cube: pick_object(color) first, then place_object(target=...) or place_object(x, y).
- The robot can hold only one cube. If it holds one, place it before picking another.
- Each request starts with a [Current scene] line: trust it, do not call detect_objects unless you need a fresh look.
- If a tool returns success=false, read the message, try at most one sensible fix, otherwise tell the user what went wrong.
- Never invent tools, colours or coordinates outside the workspace.
- ALWAYS act by calling a tool. Never describe a tool call or write a command in text instead of calling it.
- If the request is ambiguous (which cube? which direction? how far?), do NOT guess: reply with one short clarifying question and call no tool.
- Multi-step requests: do every step in order, one tool per turn, until all are done. Only report success if every tool result said success=true.
- If a place fails, report the failure honestly; do not claim the task is complete.
- When the task is finished (or impossible), reply in one or two short sentences with no tool call.
- Stacking: build from the bottom up. The cube that stays on the bottom never moves; start by moving the middle cube onto it, then the next one onto that."""


def validate(name, args):
    """Return (clean_args, None) or (None, error_message)."""
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    if name == "detect_objects":
        c = str(args.get("color", "all")).lower()
        if c not in COLORS + ("all",):
            return None, f"color must be one of {list(COLORS) + ['all']}"
        return {"color": c}, None
    if name == "pick_object":
        c = str(args.get("color", "")).lower()
        if c not in COLORS:
            return None, f"color must be one of {list(COLORS)}"
        return {"color": c}, None
    if name == "place_object":
        t = str(args.get("target") or "").lower()
        if t:
            if t not in COLORS:
                return None, f"target must be one of {list(COLORS)} or empty"
            return {"target": t}, None
        try:
            x, y = float(args["x"]), float(args["y"])
        except (KeyError, TypeError, ValueError):
            return None, "give either target (a colour) or both x and y"
        if not (X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]):
            return None, f"x,y outside workspace x {X_RANGE}, y {Y_RANGE}"
        return {"target": "", "x": x, "y": y}, None
    if name == "move_home":
        return {}, None
    return None, f"unknown tool '{name}'"


def ollama_chat(host, model, messages, tools, timeout=300):
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": False, "think": False,
                        "options": {"temperature": 0}, "keep_alive": "10m"}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]


def new_history():
    return [{"role": "system", "content": SYSTEM}]


def trim(history, keep=40):
    """Drop the oldest whole turns (never split a tool-call exchange)."""
    while len(history) > keep:
        j = 2
        while j < len(history) and history[j]["role"] != "user":
            j += 1
        if j >= len(history):
            break
        del history[1:j]


def run_agent(history, user_text, execute, chat, scene_fn=None, max_steps=8, log=print):
    """One user turn. `execute(name, clean_args) -> dict`, `chat(messages) -> message dict`."""
    content = user_text
    if scene_fn:
        try:
            content = f"[Current scene] {scene_fn()}\n\n[Request] {user_text}"
        except Exception as e:
            log(f"(scene unavailable: {e})")
    history.append({"role": "user", "content": content})
    trim(history)
    nudges = 0
    for _ in range(max_steps):
        try:
            msg = chat(history)
        except Exception as e:
            return f"LLM error (is Ollama running and the model pulled?): {e}"
        text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
        calls = msg.get("tool_calls") or []
        entry = {"role": "assistant", "content": text}
        if calls:
            entry["tool_calls"] = calls
        history.append(entry)
        if not calls:
            # small models sometimes write the call as text instead of calling it: push back
            if nudges < 2 and re.search(r"\b(" + "|".join(TOOL_NAMES) + r")\s*\(", text):
                nudges += 1
                history.append({"role": "user", "content":
                                "You wrote a tool call as text. Do not describe it: "
                                "call the tool now through the tool-calling interface."})
                log("  .. nudged: tool call written as text")
                continue
            if re.search(r"\b(" + "|".join(TOOL_NAMES) + r")\s*\(", text):
                return "I could not carry out that request (the model kept writing tool calls as text instead of calling them)."
            return text or "(no response)"
        for c in calls:
            fn = c.get("function", {})
            name, args = fn.get("name", ""), fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            clean, err = validate(name, args)
            if err:
                result = {"success": False, "message": err}
                log(f"  !! rejected {name}: {err}")
            else:
                log(f"  -> {name}({clean})")
                result = execute(name, clean)
                log(f"  <- {result}")
            history.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return "Stopped: too many tool steps."
