"""Protocol version stamp + the dynamic-body census both emitters share.

The scene JSON ships every body once; telemetry frames re-ship only the
bodies that can move. Both sides must agree on which bodies those are and
which belong to the robot, so the census lives here, in one place.

Bump PROTOCOL_VERSION whenever either artifact's shape changes. The website
repo vendors fixture copies stamped with this version, so a bump is a
deliberate two-repo event -- never a side effect of an unrelated edit.
"""

PROTOCOL_VERSION = "0.6.0"
# 0.6.0: the robot is SCORED, and the score is on the wire (issue #14).
#        Frames may carry a "ledger" block -- per robot, what it has earned:
#        {"pluggybot": {"balance": 34, "earned": 34, "spent": 0, "tasks": 3,
#        "pending": 0, "recent": [{"seq", "task", "points", "ok", "t"}]}}.
#        Sparse and keyframe-refreshed on exactly the same rule as
#        "activities", "boards" and "screens", and for the same reason: a
#        balance is not a pose, so this block is the only record of it in the
#        stream. The header gains "ledger": [robot names].
#        A fourth typed message joins the three board ones:
#        {"type": "earned", ...} -- one finished task's VERDICT, as it is
#        banked: task, tier, ok, points, quality, the evaluator's reason and
#        its (public) metrics, plus the balance afterwards. Unlike a stroke,
#        it needs no snapshot message to catch a late joiner up: `recent` in
#        the block does that job on the keyframe cadence.
#        A settled visitor rating re-emits the same entry with
#        "settled": true (the deferred-verdict slot; nothing produces one
#        until the inbound channel lands, issue #16).
#        Additive: a 0.5.0 consumer that ignores both renders what it did.
#        ⚠ What is NOT on the wire is as deliberate as what is: a hidden
#        ground-truth task publishes its verdict without its ANSWER (the
#        census's `truth` is redacted by hub/scoring.py), because the ledger
#        is streamed to the site AND shown to the LLM overseer as context --
#        and a task the robot is supposed to discover must not arrive
#        pre-solved in its own scoreboard.
# 0.5.0: the robot has a FACE, and a board can be caught up on (issue #13,
#        rooftop-media-2026 #28). Two additions, both about surfaces the
#        browser paints rather than geometry MuJoCo carries:
#        Frames may carry a "screens" block -- per display module, what it
#        is showing: {"module_lcd": {"mode": "face", "face": "curious",
#        "hint": "blink", "powered": true}}. Sparse and keyframe-refreshed
#        on exactly the same rule as "activities" and "boards", and for the
#        same reason: an LCD's content is not a pose, so this block is the
#        only record of it on the wire. Modes are "off" / "face" / "text" /
#        "count"; the vocabularies are FACE_STATES / SCREEN_HINTS below and
#        are a two-repo contract, like VISUAL_HINTS. The header gains
#        "screens": [names], and the SCENE gains a top-level "screens" --
#        which geom on which body carries each display, with the outward
#        normal, so a client can place a face without name archaeology.
#        A third typed message joins "draw" and "board_cleared":
#        {"type": "board_snapshot", ...} -- every stroke a board is
#        currently carrying, emitted when a stream OPENS. Ink is not a
#        pose, so nothing else in the stream can catch a late joiner up:
#        keyframes re-ship the "boards" counters but never the lines, and a
#        board that survived a producer restart (hub/boards.py persists the
#        polylines as of this version) has strokes no live "draw" event
#        will ever describe again. Additive: a consumer that ignores it
#        renders exactly what 0.4.0 rendered.
# 0.4.0: whiteboards are world STATE, and ink is an event (issue #12).
#        Frames may carry a "boards" block -- per drawing surface, which
#        stroke programs are on it, how full the pen's reach is, and when it
#        was last cleared. Sparse and keyframe-refreshed on exactly the same
#        rule as "activities", and for the same reason: ink is not a body, so
#        nothing about a drawing appears in the pose stream. The header gains
#        "boards": [names].
#        RECORDINGS now interleave two typed messages with the frames --
#        {"type": "draw", ...} (board id + the polyline the pen actually
#        traced, board-local metres) and {"type": "board_cleared", ...}.
#        That is the part that is NOT backward compatible in practice: a
#        0.3.0 replayer that assumed every line after the header is a frame
#        will trip over them. Dispatch on "type"; no "type" means frame --
#        the rule the live stream already had, now true of recordings too.
#        The SCENE gains an optional top-level "boards" (geom, world pose and
#        half-extents per drawing surface, keyed by the name the events use).
#        Without it a `draw` polyline cannot be placed at all: the event
#        names "whiteboard_a" while the geom it lives on is called "board_b".
# 0.3.0: frames may carry an "activities" block -- the task state machines'
#        discrete world state (issue #8), e.g.
#        {"garden_gate": {"state": "open", "pressed": false}}. Sparse like
#        body poses (only activities whose flags changed) and re-shipped in
#        full on every keyframe, so a mid-stream joiner is complete within
#        one keyframe interval exactly as it is for poses. The header gains
#        "activities": [names]. Additive -- a 0.2.0 consumer ignores the
#        block -- but a shape change to both artifacts, so the version moves
#        and the website re-vendors.
#        ⚠ An activity's visible EFFECT is usually invisible to the pose
#        stream by construction: a gate that opens by geom toggle sits on a
#        STATIC body, which ships once in the scene and never again. The
#        flag is not a convenience duplicating the poses -- for those
#        changes it is the only channel there is.
# 0.2.0: frames may carry "key": true (a keyframe -- every dynamic body),
#        and they now RECUR every header["keyframeS"] sim-seconds instead
#        of happening only at t=0 and on a live reconnect. Additive, but a
#        shape change to both artifacts, so the version moves and the
#        website re-vendors. Rationale: rooftop-media-2026 #22.

# Visual-hint vocabulary v1 (issue #6, co-designed with the website's
# parametric assets -- rooftop-media-2026 issue #18). The generator's
# sidecar may only emit these strings; the website renders a parametric
# component per hint and falls back to raw primitives for anything else,
# so ADDING a hint is additive (no version bump), while renaming one is a
# breaking change (bump).
VISUAL_HINTS = ("wall", "fence", "floor", "ground", "whiteboard", "rack",
                "plant")

# The LCD module's display (issue #13). Three vocabularies on the same terms
# as VISUAL_HINTS: the sim may only emit these strings, the website draws a
# parametric component per face, and ADDING one is additive (an unknown face
# falls back to `idle`) while renaming one is a breaking change in both
# repos. The face is drawn in the browser and never rendered in MuJoCo --
# layer 3 of the three-layer model, exactly like ink.
SCREEN_MODES = ("off", "face", "text", "count")
FACE_STATES = ("idle", "happy", "curious", "determined", "surprised",
               "sleepy", "worried")
# The animation the browser LOOPS under the face. A hint, not an event: the
# sim never ticks a blink, because a 20 Hz pose stream is the wrong channel
# for a 150 ms eyelid and the browser owns everything organic.
SCREEN_HINTS = ("none", "blink", "bounce", "shake")

# The robot's root body. The planned multi-robot refactor (mjSpec attach with
# a namespace prefix per robot) will generalize this to a prefix; until then
# there is exactly one robot and it is called this everywhere.
ROBOT_ROOT = "pluggybot"


def dynamic_flags(model) -> list[bool]:
  """Per-body: can this body move in the world frame?

  A body is dynamic iff any joint sits between it and the world -- its own
  or an ancestor's. Only these need per-frame poses; walls, outlets and the
  rack frame ship once in the scene description and never again.
  """
  dyn = [False] * model.nbody
  for b in range(1, model.nbody):
    # bool(): body_jntnum is a numpy array, and `or` hands back numpy.bool_,
    # which json.dumps refuses -- caught the first time the scene was written
    dyn[b] = bool(dyn[model.body_parentid[b]] or model.body_jntnum[b] > 0)
  return dyn


def robot_body_ids(model) -> set[int]:
  """Ids of every body in the robot's subtree (ROBOT_ROOT and below)."""
  root = model.body(ROBOT_ROOT).id
  ids: set[int] = set()
  for b in range(model.nbody):
    x = b
    while x not in (0, root):
      x = int(model.body_parentid[x])
    if x == root:
      ids.add(b)
  return ids


def body_census(model) -> tuple[list[str], list[str]]:
  """(robot, world) name lists of the DYNAMIC bodies, in model order.

  The split mirrors the frame shape: the robot's own bodies stream under
  "robots"/<name>; shared dynamic bodies (the rack, the modules) stream
  under "world", because they are nobody's limbs -- either robot may move
  them once the shared world lands.
  """
  dyn = dynamic_flags(model)
  rob = robot_body_ids(model)
  robot = [model.body(b).name for b in range(model.nbody)
           if dyn[b] and b in rob]
  world = [model.body(b).name for b in range(model.nbody)
           if dyn[b] and b not in rob]
  return robot, world
