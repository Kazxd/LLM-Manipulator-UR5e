"""ROS-free LLM agent loop for Ollama tool calling (stdlib only, unit-testable)."""
import json, math, os, re, urllib.request

# Latency options (see the notes at ollama_chat / run_agent):
NUM_CTX = 4096            # fixed context size; changing it makes Ollama reload the model
NUM_PREDICT = 256         # hard cap on generated tokens (a rambling answer once took 5 minutes)
MACRO_TOOLS = ("build_tower", "sort_cubes", "move_home")   # one call does the whole job
FAST_MACROS = os.environ.get("UR_FAST_MACROS") == "1"      # skip the final LLM call after a successful macro tool
MAX_REPLY_CHARS = 500     # a longer reply without a tool call is reasoning, not an answer
MAX_RAMBLES = 2           # retries after a rambling reply before giving up

COLORS = ("red", "green", "blue")
EXTRA_OBJECTS = ("can", "ball", "block")          # yellow cylinder, purple sphere, white box (need detect_open)
OBJECT_NAMES = COLORS + EXTRA_OBJECTS
ALIASES = {"yellow can": "can", "purple ball": "ball", "white block": "block"}
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


def canon_name(value):
    """'Red', 'red cube', 'red box', 'yellow can', 'yellow_can', 'Ball' -> 'red' / 'can' / 'ball'; None if unknown."""
    n = str(value or "").lower().strip().replace("_", " ")
    if n.endswith(" cube"):
        n = n[:-5]
    elif n.endswith(" box") and n[:-4] in COLORS:      # "red box" is a cube; "white box" stays the block
        n = n[:-4]
    n = ALIASES.get(n, n)
    return n if n in OBJECT_NAMES else None


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
POSITIONAL = {"pick_object": ("color",), "detect_objects": ("color",), "place_object": ("target", "x", "y"),
              "place_in_zone": ("zone",), "place_relative": ("reference", "relation", "distance_cm")}
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
    raw = m.group(2).strip()
    if not args and raw and not re.search(r"[\[\]{}=]", raw):      # positional: pick_object(can), place_object(0.5, 0.1)
        toks = [t.strip().strip("\"'") for t in raw.split(",") if t.strip()]
        names = list(POSITIONAL.get(m.group(1), ()))
        if m.group(1) == "place_object" and toks and re.fullmatch(r"[-+0-9.eE]+", toks[0]):
            names = ["x", "y"]
        for k, v in zip(names, toks):
            args[k] = float(v) if re.fullmatch(r"[-+0-9.eE]+", v) else v
    return [{"function": {"name": m.group(1), "arguments": args}}]


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _fn("detect_objects", "List the coloured cubes with x,y in metres (cubes only).",
        {"color": {"type": "string", "enum": ["red", "green", "blue", "all"]}}),
    _fn("pick_object", "Pick up a cube (red, green, blue) or can, ball, block. The argument is called color for all.",
        {"color": {"type": "string", "enum": list(OBJECT_NAMES)}}, ["color"]),
    _fn("place_object", "Place the held object on top of target (an object name, never the ball) "
        "or on the table at x,y metres (target empty).",
        {"target": {"type": "string", "enum": list(OBJECT_NAMES) + [""]},
         "x": {"type": "number"}, "y": {"type": "number"}}),
    _fn("place_relative", "Place the held object on the table next to a cube. left = +y, right = -y, "
        "front = toward the robot, behind = away, next_to = any free side. distance_cm default 10.",
        {"reference": {"type": "string", "enum": list(COLORS)},
         "relation": {"type": "string", "enum": list(RELATIONS)},
         "distance_cm": {"type": "number"}}, ["reference", "relation"]),
    _fn("place_in_zone", f"Place the held object in a table zone ({', '.join(ZONES)}).",
        {"zone": {"type": "string", "enum": list(ZONES)}}, ["zone"]),
    _fn("sort_cubes", "Move several cubes into zones in one call, e.g. {\"red\": \"left\", \"blue\": \"right\"}. "
        "Hold nothing first. Cubes only.",
        {"assignments": {"type": "object",
                         "properties": {c: {"type": "string", "enum": list(ZONES)} for c in COLORS}}},
        ["assignments"]),
    _fn("build_tower", "Stack 2-3 cubes in one call. order = colours from BOTTOM to TOP. Hold nothing first. Cubes only.",
        {"order": {"type": "array", "items": {"type": "string", "enum": list(COLORS)}}}, ["order"]),
    _fn("move_home", "Move the arm to its home pose.", {}),
]

SYSTEM = f"""You control a UR5e arm that moves objects on a table by calling tools.
Objects: cubes red, green, blue; also can (yellow cylinder), ball (purple sphere), block (white box). There is no purple cube: the purple object is the ball. "red box" means the red cube. Workspace x {X_RANGE[0]} to {X_RANGE[1]}, y {Y_RANGE[0]} to {Y_RANGE[1]} m.
Rules:
- NEVER think aloud or explain. Your reply is either a tool call or one short sentence.
- Call ONE tool at a time and wait for its result. ALWAYS act through the tool interface, never write a call as text.
- Move an object: pick_object(color), then place_object(target) (on top of that object, never the ball) or place_object(x, y). Hold one object at a time.
- Each request starts with a [Current scene] line (cubes only; can, ball and block are always there). Trust it.
- Directions seen from the robot looking along +x: left = +y, right = -y, front = toward the robot, behind = away from it.
- "left of / right of / in front of / behind / next to <cube>": pick_object, then place_relative.
- Zones {", ".join(ZONES)} (flat pads, left = +y). One object: pick_object, then place_in_zone. Several cubes: sort_cubes once.
- Stack 2-3 cubes: build_tower(order, bottom to top) once. sort_cubes and build_tower take cubes only.
- If a tool fails, try at most one fix, else say what went wrong. Report success only if every result was success=true.
- If the request is ambiguous (which object? direction? distance?), ask one short question and call no tool.
- When done or impossible, reply in one or two short sentences without a tool call."""


EXAMPLES = """
Examples (make the calls through the tool interface; arguments are always named):
- "stack red on top of green": build_tower(order=["green","red"]). "put red on blue": pick_object(color="red"), then place_object(target="blue"). Never say a stack is out of reach: call the tools, they check reach.
- "pick up the can": pick_object(color="can"). "put the can on the red cube": pick_object(color="can"), then place_object(target="red"). The can, ball and block are ALWAYS on the table even though the scene line omits them. Never refuse or ask because they are missing from the scene line.
- "move the ball to x 0.5 y 0.2": pick_object(color="ball"), then place_object(target="", x=0.5, y=0.2).
- "move green next to blue": pick_object(color="green"), then place_relative(reference="blue", relation="next_to"). Do not ask which side.
- Ask "Which cube?" ONLY if the request names no object at all (e.g. "move the cube to the left"). A request naming red, green, blue, can, ball or block is never ambiguous.
- "white block" and "block" are the same object."""


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
        c = canon_name(args.get("color") or args.get("object"))
        if c is None:
            return None, f"color must be one of {list(OBJECT_NAMES)}"
        return {"color": c}, None
    if name == "place_object":
        raw = str(args.get("target") or "").strip()
        if raw:
            t = canon_name(raw)
            if t is None:
                return None, f"target must be one of {list(OBJECT_NAMES)} or empty"
            if t == "ball":
                return None, "cannot place on top of the ball: its top is round"
            return {"target": t}, None
        try:
            x, y = float(args["x"]), float(args["y"])
        except (KeyError, TypeError, ValueError):
            return None, "give either target (an object name) or both x and y"
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


def ollama_chat(host, model, messages, tools, timeout=300, no_think=True, num_ctx=NUM_CTX, num_predict=NUM_PREDICT):
    """no_think appends /no_think to the latest user message (on a copy; history is untouched)."""
    msgs = [dict(m) for m in messages]
    if no_think:
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = m["content"] + " /no_think"
                break
    body = json.dumps({"model": model, "messages": msgs, "tools": tools, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
                       "keep_alive": "10m"}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = data["message"]
    # timing from Ollama (nanoseconds), kept under a private key; run_agent logs it and never stores it in the history
    msg["_stats"] = {k: data[k] for k in ("load_duration", "prompt_eval_count", "prompt_eval_duration",
                                          "eval_count", "eval_duration", "done_reason") if k in data}
    return msg


def new_history():
    return [{"role": "system", "content": SYSTEM + EXAMPLES}]


def trim(history, keep=40):
    """Drop the oldest whole turns (never split a tool-call exchange)."""
    while len(history) > keep:
        j = 2
        while j < len(history) and history[j]["role"] != "user":
            j += 1
        if j >= len(history):
            break
        del history[1:j]


def compact_old_scenes(history, upto):
    """Earlier turns keep only the request: their [Current scene] lines are stale and cost ~300 tokens each."""
    for m in history[:upto]:
        if m["role"] == "user" and m["content"].startswith("[Current scene]"):
            m["content"] = m["content"].split("[Request] ", 1)[-1]


def is_rambling(msg, text):
    """A reply with no tool call that hit the token cap or is far too long: the model is reasoning out loud."""
    st = msg.get("_stats") or {}
    return st.get("done_reason") == "length" or st.get("eval_count", 0) >= NUM_PREDICT - 1 \
        or len(text) > MAX_REPLY_CHARS


def run_agent(history, user_text, execute, chat, scene_fn=None, max_steps=8, log=print, fast_macros=None):
    """One user turn. `execute(name, clean_args) -> dict`, `chat(messages) -> message dict`.
    fast_macros (default: env UR_FAST_MACROS=1): after a successful build_tower / sort_cubes / move_home the result is
    reported directly instead of asking the LLM for a closing sentence (saves one LLM call, 15-60 s).
    Rambling replies (token cap hit, no tool call) are never stored; the model is retried with a temporary reminder."""
    if fast_macros is None:
        fast_macros = FAST_MACROS
    content = user_text
    if scene_fn:
        try:
            content = (f"[Current scene] {scene_fn()} Also on the table (not listed above): can, ball, block.\n\n"
                       f"[Request] {user_text}")
        except Exception as e:
            log(f"(scene unavailable: {e})")
    compact_old_scenes(history, len(history))
    history.append({"role": "user", "content": content})
    trim(history)
    nudges = rambles = 0
    transient = None                       # reminder sent with the next call only, never stored
    call_re = r"\b(" + "|".join(TOOL_NAMES) + r")\s*\("
    for _ in range(max_steps):
        msgs = history + [{"role": "user", "content": transient}] if transient else history
        transient = None
        try:
            msg = chat(msgs)
        except Exception as e:
            return f"LLM error (is Ollama running and the model pulled?): {e}"
        st = msg.get("_stats")
        if st:
            log(f"  .. llm: prompt {st.get('prompt_eval_count', '?')} tok in {st.get('prompt_eval_duration', 0) / 1e9:.1f}s, "
                f"output {st.get('eval_count', '?')} tok in {st.get('eval_duration', 0) / 1e9:.1f}s, "
                f"load {st.get('load_duration', 0) / 1e9:.1f}s")
        text = clean_text(msg.get("content"))
        calls = msg.get("tool_calls") or salvage_calls(text)
        if not calls and is_rambling(msg, text):
            rambles += 1
            log(f"  .. rambling reply dropped ({len(text)} chars, retry {rambles}/{MAX_RAMBLES})")
            if rambles > MAX_RAMBLES:
                return "I could not work out how to do that. Please rephrase the request."
            transient = (f'Stop explaining. The request is: "{user_text}". '
                         "Reply with ONLY the next tool call through the tool interface, or one short sentence if done.")
            continue
        if calls and not msg.get("tool_calls"):
            text = ""                       # the text WAS the call
            log("  .. salvaged tool call from text")
        entry = {"role": "assistant", "content": text}
        if calls:
            entry["tool_calls"] = calls
        history.append(entry)
        if not calls:
            looks_like_call = re.search(call_re, text) or text.strip(" `") in TOOL_NAMES
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
        done = []
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
            done.append((name, result))
        if fast_macros and done and all(n in MACRO_TOOLS and r.get("success") for n, r in done):
            reply = "Done: " + "; ".join(str(r.get("message", "ok")) for _, r in done) + "."
            history.append({"role": "assistant", "content": reply})
            log("  .. fast reply (no closing LLM call)")
            return reply
    return "Stopped: too many tool steps."
