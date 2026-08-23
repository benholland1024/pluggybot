"""Protocol version stamp + the dynamic-body census both emitters share.

The scene JSON ships every body once; telemetry frames re-ship only the
bodies that can move. Both sides must agree on which bodies those are and
which belong to the robot, so the census lives here, in one place.

Bump PROTOCOL_VERSION whenever either artifact's shape changes. The website
repo vendors fixture copies stamped with this version, so a bump is a
deliberate two-repo event -- never a side effect of an unrelated edit.
"""

PROTOCOL_VERSION = "0.8.0"
# 0.8.0: the robot says what it is FOR (rooftop-media-2026 #30). One new
#        upstream message, emitted when a stream OPENS -- exactly like
#        `board_snapshot`, and for exactly the same reason: goals are not a
#        pose and no keyframe re-ships them, so a viewer who joined late
#        would never learn them.
#          {"type": "goals", "t": 0.0, "robot": "pluggybot",
#           "text": "Keep the house in good order...", "steering": true}
#        `text` is hub/journal.py's `read_goals` verbatim -- the mounted
#        goals.md Ben edits, or the built-in defaults when there is no file.
#        The site displays it; it is READ-ONLY in every direction, and there
#        is no inbound message that can change it. The file beside the sim
#        stays the one copy (docs/pluggyworld.md is explicit that `pw_goals`
#        was deliberately never built): this is a mirror on the wire, like
#        the journal, not a second place goals live.
#        ⚠ `steering` is the `accepts` lesson again -- ask whether the thing
#        is actually happening, not whether the usual cause is present. The
#        goals file is read on EVERY run, but only an overseer decides
#        anything with it; without one the robot flies a scripted rotation
#        and these are a statement of purpose rather than the thing choosing
#        its next errand. A site that showed them identically either way
#        would be claiming a robot is following its goals when nothing is
#        reading them.
#        Additive: a 0.7.0 consumer ignores the type and renders what it did.
# 0.7.0: the socket becomes BIDIRECTIONAL, and the robot answers back
#        (pluggybot #16, rooftop-media-2026 #29). The first version where the
#        sim reads its socket at all -- everything before this streamed and
#        never listened.
#        DOWNSTREAM (server -> sim), the new direction. Three types, listed in
#        INBOUND_TYPES below; the sim drops anything else with a counter, so
#        adding one is additive and a website ahead of its sim is a no-op:
#          {"type": "suggestion", "id": "s_01", "from": "ada",
#           "text": "draw a tree on whiteboard_b"}
#          {"type": "question",   "id": "q_01", "from": "ada",
#           "text": "what are you working on?"}
#          {"type": "rating",     "id": "r_01", "seq": 3, "quality": 0.8}
#        `id` is the SERVER's and the sim only echoes it back -- it is what
#        lets an outcome land on the right database row. `rating` settles a
#        deferred visitor-tier verdict (the slot 0.6.0 reserved), and the
#        ledger re-emits that entry with "settled": true.
#        ⚠ Visitor text is DATA, never instructions. The sim caps it at 280
#        characters, strips control characters, collapses it to one line, and
#        frames it to the overseer as something a person WANTS. The robot's
#        freedom to decline is the defence; see hub/inbox.py.
#        UPSTREAM, two new typed messages:
#          {"type": "visitor_reply", "t": 412.5, "robot": "pluggybot",
#           "id": "s_01", "kind": "suggestion", "outcome": "accepted",
#           "reply": "good idea, doing it now", "action": "draw"}
#           -- outcome is accepted | declined | answered. This is what closes
#           the loop back to the row the website is holding.
#          {"type": "journal", "t": 300.2, "robot": "pluggybot",
#           "text": "whiteboard_a is nearly full", "why": "..."}
#           -- a note the overseer wrote to itself (issue #15). Previously
#           these reached the site only as narration `event` lines; the site's
#           journal feed wants them structured.
#        Additive in both directions: a 0.6.0 consumer that ignores the two
#        new upstream types renders exactly what it rendered before, and a
#        0.6.0 PRODUCER simply never reads its socket, which no server can
#        tell apart from a robot that declines everything.
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

# The visitor channel's vocabularies (issue #16). Two-repo contracts on the
# same terms as VISUAL_HINTS and FACE_STATES: the website may only SEND these
# inbound types, and may only receive these outcomes. Adding one is additive
# (the sim counts and drops an unknown inbound type; a consumer ignores an
# unknown outcome); renaming one breaks both repos.
#
# They live HERE rather than in hub/inbox.py, which is where the parsing is,
# so the wire spec and the parser cannot disagree -- and in this direction,
# because `hub` already imports `telemetry` and the reverse would invert the
# layering for a tuple of strings.
#
# `move` and `clear_board` (tic-tac-toe) are named in issue #16 as LATER and
# are deliberately absent: there is no board game yet, and a vocabulary entry
# with nothing behind it is a promise the robot cannot keep.
INBOUND_TYPES = ("suggestion", "question", "rating")

#: What the robot may say back about one visitor message.
VISITOR_OUTCOMES = ("accepted", "declined", "answered")

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
