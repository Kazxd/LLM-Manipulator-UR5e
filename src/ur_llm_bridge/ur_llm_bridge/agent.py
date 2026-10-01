"""ROS-free LLM agent loop for Ollama tool calling (stdlib only, unit-testable)."""
import json, math, re, urllib.request

COLORS = ("red", "green", "blue")
X_RANGE = (0.30, 0.90)    # safe workspace on the table (metres, base frame)
Y_RANGE = (-0.40, 0.40)
TOOL_NAMES = ("pick_object", "place_object", "place_relative", "place_in_zone", "sort_cubes",
              "build_tower", "move_home", "detect_objects")
RELATIONS = ("left", "right", "front", "behind", "next_to")
# Directions as seen from the robot base looking along +x (ROS base frame: +y is left).
DIRS = {"left": (0.0, 1.0), "right": (0.0, -1.0), "front": (-1.0, 0.0), "behind": (1.0, 0.0)}
MIN_GAP = 0.07            # min centre distance to any other cube (cube is 0.05)

# Named table zones (flat pads in the world file). Keep in sync with worlds/pick_place.sdf.
ZONE_HALF = 0.10          # pad half-size (m)
ZONE_CENTERS = {"left": (0.55, 0.28), "right": (0.55, -0.28)}
ZONES = tuple(ZONE_CENTERS)
SLOT_OFFSETS = ((-0.04, -0.04), (0.04, -0.04), (-0.04, 0.04), (0.04, 0.04))   # 4 cubes per zone


def zone_contains(zone, x, y, margin=0.025):
    """True if a cube centred at (x, y) lies fully on the zone pad (margin = half a cube)."""
    cx, cy = ZONE_CENTERS[zone]
    return abs(x - cx) <= ZONE_HALF - margin and abs(y - cy) <= ZONE_HALF - margin


def zone_slot(zone, others):
    """First free slot in the zone, or (None, reason). others = {colour: (x, y)} of cubes that stay."""
    cx, cy = ZONE_CENTERS[zone]
    for dx, dy in SLOT_OFFSETS:
        x, y = cx + dx, cy + dy
        if all(math.hypot(x - ox, y - oy) >= MIN_GAP for ox, oy in others.values()):
            return (x, y), None
    return None, f"the {zone} zone is full"

_CALL_RE = re.compile(r"^\s*`{0,3}\w*\s*(" + "|".join(TOOL_NAMES) + r")\s*\((.*)\)\s*`{0,3}\s*$", re.S)
_ARG_RE = re.compile(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[-+0-9.eE]+)')


def clean_text(text):
    """Strip reasoning, including a bare </think> with no opening tag."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def salvage_calls(text):
    """Recover a tool call only if the WHOLE message is one (JSON or name(k=v)). Else return []."""
    t = (text or "").strip()
    if not t:
        return []
    try:
        d = json.loads(t)
        if isinstance(d, dict) and d.get("name") in TOOL_NAMES:
            return [{"function": {"name": d["name"], "arguments": d.get("arguments", {})}}]
    except ValueError:
        pass
    m = _CALL_RE.match(t)
    if not m:
        return []
    args = {}
    for k, v in _ARG_RE.findall(m.group(2)):
        v = v.strip("\"'")
        args[k] = float(v) if re.fullmatch(r"[-+0-9.eE]+", v) else v
    return [{"function": {"name": m.group(1), "arguments": args}}]


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
    _fn("place_relative",
        "Place the held cube on the table next to another cube. relation is seen from the robot base looking "
        "along +x: left = +y side, right = -y side, front = toward the robot, behind = away from the robot, "
        "next_to = any free side. distance_cm is the centre-to-centre distance (default 10).",
        {"reference": {"type": "string", "enum": list(COLORS)},
         "relation": {"type": "string", "enum": list(RELATIONS)},
         "distance_cm": {"type": "number"}}, ["reference", "relation"]),
    _fn("place_in_zone",
        f"Place the held cube in a named table zone ({', '.join(ZONES)}); the robot picks a free spot in it.",
        {"zone": {"type": "string", "enum": list(ZONES)}}, ["zone"]),
    _fn("sort_cubes",
        "Move several cubes into zones in one call. assignments maps a colour to a zone, e.g. "
        "{\"red\": \"left\", \"green\": \"left\", \"blue\": \"right\"}. Cubes already in their zone are skipped. "
        "The robot must hold nothing.",
        {"assignments": {"type": "object",
                         "properties": {c: {"type": "string", "enum": list(ZONES)} for c in COLORS}}},
        ["assignments"]),
    _fn("build_tower",
        "Stack cubes into a tower in one call. order = colours from BOTTOM to TOP, e.g. [\"red\",\"green\",\"blue\"] "
        "puts green on red and blue on green. Use this for any request to stack 2-3 cubes. "
        "The robot must hold nothing and the cubes must be on the table.",
        {"order": {"type": "array", "items": {"type": "string", "enum": list(COLORS)}}}, ["order"]),
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
- For "left of / right of / in front of / behind / next to <cube>": pick_object first, then place_relative. Do not compute coordinates yourself.
- Directions are from the robot base looking along +x: left = +y, right = -y, front = toward the robot, behind = away from it.
- Table zones: {", ".join(ZONES)} (flat pads on the table, "left" = +y side, "right" = -y side). For ONE cube: pick_object, then place_in_zone(zone). For SEVERAL cubes: call sort_cubes(assignments) ONCE.
- To stack 2 or 3 cubes, call build_tower(order) ONCE (order is bottom to top). Do not do the picks and places yourself.
- ALWAYS act by calling a tool. Never describe a tool call or write a command in text instead of calling it.
- If the request is ambiguous (which cube? which direction? how far?), do NOT guess: reply with one short clarifying question and call no tool.
- Multi-step requests: do every step in order, one tool per turn, until all are done. Only report success if every tool result said success=true.
- If a place fails, report the failure honestly; do not claim the task is complete.
- When the task is finished (or impossible), reply in one or two short sentences with no tool call."""


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
    if name == "place_relative":
        ref = str(args.get("reference", "")).lower()
        rel = str(args.get("relation", "")).lower().replace(" ", "_")
        if ref not in COLORS:
            return None, f"reference must be one of {list(COLORS)}"
        if rel not in RELATIONS:
            return None, f"relation must be one of {list(RELATIONS)}"
        try:
            cm = float(args.get("distance_cm", 10))
        except (TypeError, ValueError):
            return None, "distance_cm must be a number"
        if not 7 <= cm <= 30:
            return None, "distance_cm must be between 7 and 30 (centre to centre)"
        return {"reference": ref, "relation": rel, "distance": cm / 100.0}, None
    if name == "place_in_zone":
        z = str(args.get("zone", "")).lower().replace(" zone", "").strip()
        if z not in ZONES:
            return None, f"zone must be one of {list(ZONES)}"
        return {"zone": z}, None
    if name == "sort_cubes":
        a = args.get("assignments")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except ValueError:
                a = None
        if not isinstance(a, dict) or not a:
            return None, 'assignments must be an object like {"red": "left", "blue": "right"}'
        clean = {}
        for c, z in a.items():
            c, z = str(c).lower(), str(z).lower().replace(" zone", "").strip()
            if c not in COLORS or z not in ZONES:
                return None, f"each assignment needs a colour from {list(COLORS)} and a zone from {list(ZONES)}"
            clean[c] = z
        for z in set(clean.values()):
            if list(clean.values()).count(z) > len(SLOT_OFFSETS):
                return None, f"at most {len(SLOT_OFFSETS)} cubes fit in one zone"
        return {"assignments": clean}, None
    if name == "build_tower":
        order = args.get("order")
        if isinstance(order, str):
            order = [s for s in re.split(r"[\s,>/]+", order.lower()) if s]
        if not isinstance(order, list):
            return None, "order must be a list of colours, bottom to top"
        order = [str(c).lower() for c in order]
        if not 2 <= len(order) <= 3 or len(set(order)) != len(order) or any(c not in COLORS for c in order):
            return None, f"order needs 2-3 different colours from {list(COLORS)}, bottom to top"
        return {"order": order}, None
    if name == "move_home":
        return {}, None
    return None, f"unknown tool '{name}'"


def compute_relative(ref, relation, dist, others):
    """ref=(x,y); others={colour:(x,y)} of cubes that stay on the table.
    Return ((x,y), description) or (None, reason). next_to tries left, right, front, behind in turn."""
    order = [relation] if relation in DIRS else list(DIRS)
    reasons = []
    for r in order:
        dx, dy = DIRS[r]
        x, y = ref[0] + dx * dist, ref[1] + dy * dist
        if not (X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]):
            reasons.append(f"{r}: outside the workspace")
            continue
        clash = [c for c, (ox, oy) in others.items() if math.hypot(x - ox, y - oy) < MIN_GAP]
        if clash:
            reasons.append(f"{r}: too close to the {', '.join(clash)} cube")
            continue
        return (x, y), r
    return None, "no valid spot (" + "; ".join(reasons) + ")"


def ollama_chat(host, model, messages, tools, timeout=300, no_think=True):
    """no_think appends /no_think to the latest user message (on a copy; history is untouched)."""
    msgs = [dict(m) for m in messages]
    if no_think:
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = m["content"] + " /no_think"
                break
    body = json.dumps({"model": model, "messages": msgs, "tools": tools, "stream": False, "think": False,
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
    call_re = r"\b(" + "|".join(TOOL_NAMES) + r")\s*\("
    for _ in range(max_steps):
        try:
            msg = chat(history)
        except Exception as e:
            return f"LLM error (is Ollama running and the model pulled?): {e}"
        text = clean_text(msg.get("content"))
        calls = msg.get("tool_calls") or salvage_calls(text)
        if calls and not msg.get("tool_calls"):
            text = ""                       # the text WAS the call
            log("  .. salvaged tool call from text")
        entry = {"role": "assistant", "content": text}
        if calls:
            entry["tool_calls"] = calls
        history.append(entry)
        if not calls:
            looks_like_call = re.search(call_re, text)
            if looks_like_call and nudges < 2:
                nudges += 1
                history.append({"role": "user", "content":
                                f'You wrote a tool call as text instead of calling it. The request is: "{user_text}". '
                                "Continue it by calling the next tool through the tool-calling interface."})
                log("  .. nudged: tool call written as text")
                continue
            if looks_like_call:
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
