"""Protocol version stamp + the dynamic-body census both emitters share.

The scene JSON ships every body once; telemetry frames re-ship only the
bodies that can move. Both sides must agree on which bodies those are and
which belong to the robot, so the census lives here, in one place.

Bump PROTOCOL_VERSION whenever either artifact's shape changes. The website
repo vendors fixture copies stamped with this version, so a bump is a
deliberate two-repo event -- never a side effect of an unrelated edit.
"""

import os

PROTOCOL_VERSION = "0.12.0"
# 0.12.0: MONEY AND THE OPERATOR'S SWITCH (pluggybot #37). Three additions,
#         all of them display-only in the direction that matters: there is no
#         inbound message that moves a dollar or sets a mode, exactly as
#         there is none that moves a point.
#           1. a `spend` block in the frame, shipped when it CHANGES (the
#              `ledger` block's rule -- it moves only when the robot buys a
#              thought, which is minutes apart at best):
#                "spend": {"weeklyUsd": 10.0, "spentUsd": 0.0132,
#                          "leftUsd": 9.9868, "calls": 3, "escalations": 3,
#                          "unpriced": 0, "recent": [...]}
#              `unpriced` counts calls whose price nobody published -- the
#              sum is understated by exactly that many, and a client that
#              hid the number would be reporting a confident total it does
#              not have.
#           2. `mode` on the robot's per-frame record, beside `state` and
#              `status`: one of MODES below. Inside the existing block
#              rather than a new one, because it is a fact about the robot
#              at that instant and that is what the block is.
#           3. a `mode` MESSAGE, which is the only one of the three that
#              earns its own type:
#                {"type": "mode", "t": 812.4, "robot": "pluggybot",
#                 "mode": "paused", "heldS": 12.5}
#              Emitted when the mode changes AND as a heartbeat while
#              `paused`. ⚠ The heartbeat is the load-bearing half: a paused
#              robot steps no physics, frames are built off SIM time, and so
#              a paused world emits NO FRAMES AT ALL. Without this a site
#              could not tell "the operator paused the robot" from "the sim
#              died", which is the `accepts` lesson in its costliest form --
#              the one where somebody is meant to notice an outage.
#         The header gains `modes` (the vocabulary) for the same reason it
#         carries `taskKinds`: a client can build its controls before the
#         robot has ever been in one of them.
#         Additive: a 0.11.0 consumer ignores the block, the field and the
#         type, and renders exactly what it rendered before.
# 0.11.0: the robot's memory is a set of NAMED DOCUMENTS, and each one says
#         who may write it (pluggybot #38, for the website's Thoughts tab --
#         rooftop-media-2026 #88). One new upstream message, emitted when a
#         stream OPENS and again whenever a file changes:
#           {"type": "thought", "t": 300.2, "robot": "pluggybot",
#            "name": "Knowledge_and_Opinions.md", "writer": "robot",
#            "text": "whiteboard_b is the one people look at",
#            "cap": 3000}
#         `name` is one of THOUGHT_FILES, `writer` one of THOUGHT_WRITERS.
#         WHOLE TEXT, NOT A DELTA -- a file is small, and "present means
#         complete" is what keeps a late joiner and a scrubbed recording
#         honest: the `tasks` block's rule, applied to prose.  `cap` is the
#         size limit the sim enforces on writes, so a client can show how
#         full a file is without inventing the denominator.
#         ⚠ `writer` is the `steering` lesson one loop over: these files look
#         alike on a page and are not alike at all. A `human` file is edited
#         on the volume and no robot can touch it; `system` is append-only
#         narrative the robot cannot revise (the principle that stops it
#         awarding itself points); only `robot` is the robot's own. A site
#         that rendered four identical panels would be reporting a mind that
#         wrote its own persona.
#         READ-ONLY IN EVERY DIRECTION, exactly like `goals` (0.8.0): no
#         inbound message writes one, and the files beside the sim stay the
#         one copy. `goals` is UNCHANGED and still sent -- it is the only
#         carrier of `steering`, which says who is READING rather than who
#         may write. Goals.md therefore arrives twice by design; a consumer
#         should take the prose from here and the steering note from there.
#         Additive: a 0.10.0 consumer that ignores the type renders exactly
#         what it rendered before, minus three documents it never had.
# 0.10.0: the robot has a NAME, and the name is not the species (issue #39,
#         rooftop-media-2026 #88). The header gains "robotNames": robot id ->
#         the display name this instance answers to:
#           "robots":     {"pluggybot": ["pluggybot", "head", ...]}
#           "robotNames": {"pluggybot": "Luca"}
#         The KEY stays ROBOT_ROOT -- the MJCF body name, the species, and
#         the id every other wire structure (frames, ledger, goals, grid)
#         already keys off. The display name is per INSTANCE (--robot-name /
#         $PLUGGY_ROBOT_NAME, default "Pluggy") and appears nowhere else:
#         renaming a robot must never re-key its telemetry. A consumer
#         reading an older recording (field absent) or a blank name falls
#         back to a default rather than rendering blank -- same degrade rule
#         as `accepts`.
# 0.9.0: the robot is GIVEN WORK, and the work is on the wire (pluggybot #21,
#        rooftop-media-2026 #77). A task is a JOB OFFER -- a description, a
#        target, a reward, a deadline and a verdict -- as distinct from an
#        errand (machinery) and an activity (a mechanism owning world state).
#        Frames may carry a "tasks" block, keyed by task id:
#          {"t_0001": {"id": "t_0001", "kind": "draw_figure", "task": "draw",
#                      "target": "whiteboard_a", "targetKind": "board",
#                      "description": "Draw a house on whiteboard_a.",
#                      "params": {"program": "house"}, "state": "offered",
#                      "source": "system", "deadline": 420.0,
#                      "estimateWh": 0.35, "createdT": 60.0,
#                      "claimedT": null, "resolvedT": null, "claimedBy": "",
#                      "points": 0,
#                      "reward": {"task": "draw", "tier": "auto",
#                                 "base": 6, "bonus": 6}}}
#        ⚠ UNLIKE "activities", "boards", "screens" and "ledger", THIS BLOCK
#        IS SHIPPED WHOLE, not per-key diffed. Those blocks describe things
#        with fixed names that exist for the whole run, so shipping only the
#        changed keys is safe. A TASK CAN CEASE TO EXIST -- resolved ones age
#        out of a bounded board -- and a per-key diff has no way to say
#        "gone", so a consumer merging deltas would keep a stale marker on
#        screen forever. Present means COMPLETE: replace the block, do not
#        merge it. It is still sparse in time (emitted only when something
#        changed) and still re-shipped on every keyframe.
#        The header gains "taskKinds": the kind vocabulary this producer can
#        offer (`pluggybot.economy.tasks.KINDS`), empty for a run with no task
#        board. A two-repo contract on the same terms as FACE_STATES: adding
#        a kind is additive (draw a generic marker for one you do not know),
#        renaming one breaks both repos.
#        Three typed messages join the existing ones:
#          {"type": "task_offered", "t": 60.0, "task": {...as above...}}
#          {"type": "task_claimed", "t": 61.2, "id": "t_0001",
#           "state": "claimed", "robot": "pluggybot"}
#           -- `state` is "claimed" when a robot takes the job on and
#           "active" when it actually starts working on it.
#          {"type": "task_resolved", "t": 190.4, "id": "t_0001",
#           "state": "done", "robot": "pluggybot", "points": 11,
#           "verdict": {...}, "task": {...}}
#           -- `state` is "done" (finished, judged good), "failed" (finished,
#           judged bad) or "expired" (the offer lapsed untaken). Expiry is an
#           OUTCOME, not a deletion: the site shows a task lapsing rather
#           than a marker vanishing.
#        ⚠ What a task does NOT carry is the point of the design. It has no
#        points figure of its own -- `reward` is looked up from
#        economy/rewards.json every time the block is built, so nothing that can
#        create a task (a visitor, an LLM) can price one. And the honesty
#        rule the whole milestone is governed by applies here first: the wire
#        may carry anything a network could carry (a work order, a surveyed
#        board id) and nothing a sensor would have to discover. A task kind
#        with an ANSWER keeps it in `Task.secret`, which reaches neither this
#        block nor the model's context by any path.
#        Additive: a 0.8.0 consumer that ignores the block and the three
#        types renders exactly what it rendered before.
# 0.8.0: the robot says what it is FOR (rooftop-media-2026 #30). One new
#        upstream message, emitted when a stream OPENS -- exactly like
#        `board_snapshot`, and for exactly the same reason: goals are not a
#        pose and no keyframe re-ships them, so a viewer who joined late
#        would never learn them.
#          {"type": "goals", "t": 0.0, "robot": "pluggybot",
#           "text": "Keep the house in good order...", "steering": true}
#        `text` is `Goals.md` verbatim -- the mounted file Ben edits, or the
#        built-in defaults when there is no file. Since 0.11.0 that document
#        also rides the wire as a `thought`; this message is kept because it
#        alone carries `steering`.
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
#        freedom to decline is the defence; see mind/inbox.py.
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
#        census's `truth` is redacted by economy/scoring.py), because the ledger
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
#        board that survived a producer restart (tools/boards.py persists the
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
# They live HERE rather than in mind/inbox.py, which is where the parsing is,
# so the wire spec and the parser cannot disagree -- and in this direction,
# because `hub` already imports `telemetry` and the reverse would invert the
# layering for a tuple of strings.
#
# `move` and `clear_board` (tic-tac-toe) are named in issue #16 as LATER and
# are deliberately absent: there is no board game yet, and a vocabulary entry
# with nothing behind it is a promise the robot cannot keep.
#
# `reset_tool` (issue #30) is the ADMIN recovery for a dropped module:
#   {"type": "reset_tool", "id": "a_01", "module": "module_pen",
#    "from": "ben"}
# It is handled by CODE the moment the physics thread drains it -- the module
# jumps back to its own bay, exactly as `rating` goes straight to the ledger
# -- and NEVER reaches the overseer: an admin command is not a thing the
# robot weighs. The acknowledgement is the world itself (the module's pose
# stream) plus a narration event line; the sim refuses, with a narration,
# while the module is seated on the fork, because a tool in use is not lost.
# Added WITHOUT a version bump, deliberately: no emitted artifact changes
# shape (recordings carry no inbound messages, and the header's `accepts`
# list is where a website discovers whether this sim understands it -- the
# `accepts` lesson is the mechanism, not the version).
INBOUND_TYPES = ("suggestion", "question", "rating", "reset_tool")

#: The inbound kinds CODE handles without an overseer: applied by the physics
#: thread the moment they are drained, never shown to a model. What a served
#: world with no overseer advertises in `accepts` -- a suggestion to a robot
#: with nothing reading it is a conversation that is not happening (the
#: `accepts` lesson), but a rating settles a ledger row and a reset moves a
#: module, and both of those work on a scripted world.
CODE_HANDLED_TYPES = ("rating", "reset_tool")

# The task system's vocabularies (issue #21). Two-repo contracts on the same
# terms as the three above. They live here rather than in economy/tasks.py, which
# is where the state machine is, so the wire spec and the implementation
# cannot disagree -- and in this direction, because `hub` already imports
# `telemetry`.
#
# The KINDS themselves are not here: they are `hub.tasks.KINDS`, because a
# kind carries an evaluator name, a target type and an energy estimate, which
# are sim-side facts rather than wire vocabulary. The header advertises the
# names of whatever this producer knows.
#
# `offered` -> `claimed` -> `active` -> `done` | `failed` | `expired`. The
# three terminal states are deliberately distinguishable: the website draws a
# lapsed offer differently from a job the robot tried and got wrong, and
# collapsing them would hide the most interesting line in a robot's day.
TASK_STATES = ("offered", "claimed", "active", "done", "failed", "expired")

#: Who put a task into the world. `system` is the scheduler, `visitor` is the
#: inbound channel (issue #23), `overseer` is the robot proposing its own work
#: (later still). Carried on the wire because "the robot chose this itself"
#: and "somebody asked for this" are not the same event -- the `source`
#: lesson from `Decision.source`, one layer up.
TASK_SOURCES = ("system", "visitor", "overseer")

#: What the robot may say back about one visitor message.
VISITOR_OUTCOMES = ("accepted", "declined", "answered")

#: OPERATOR MODES (0.12.0, issue #37). A two-repo contract on the same terms
#: as FACE_STATES and TASK_STATES: the website's admin page writes these
#: strings into the mode file and the wire carries them back, so adding one
#: is additive and renaming one is a change in both repos.
#:
#: ⚠ These are the operator's, never the robot's. There is no inbound
#: message and no decision field that sets one -- a mode is how a person
#: stops a robot that is behaving badly or spending money, and a kill switch
#: the thing being killed can reach is not a kill switch. mind/mode.py is the
#: sim's end and it has no writer at all.
MODES = ("llm", "scripted", "paused")

# The thought files (issue #38). Two-repo vocabulary on the same terms as
# TASK_STATES above: the names and the writers are what the wire may carry,
# while the CAPS and the write rules are sim-side facts and live in
# mind/thoughts.py, which owns the one write path that enforces them.
#
# ⚠ WHO MAY WRITE IS THE WHOLE POINT, and it is per file rather than per
# robot. `human` is a person editing the file on the volume -- persona and
# purpose, which a robot that could rewrite them would make meaningless.
# `system` is append-only narrative: code writes what happened and nothing
# revises it, the same principle that stops the robot awarding itself points
# (economy/scoring.py). `robot` is the one genuinely writable surface. Rendering
# the four identically would claim a mind that wrote its own persona, which
# is the `steering` mistake one loop over.
THOUGHT_WRITERS = ("human", "system", "robot")

#: The documents, in the order a reader should show them: who it is, what it
#: is for, what happened, what it makes of that. Appending one is additive
#: (a client renders a document it has never heard of); renaming one is a
#: breaking change in both repos, like a FACE_STATE.
THOUGHT_FILES = ("Main.md", "Goals.md", "History.md",
                 "Knowledge_and_Opinions.md")

# The robot's root body. The planned multi-robot refactor (mjSpec attach with
# a namespace prefix per robot) will generalize this to a prefix; until then
# there is exactly one robot and it is called this everywhere.
ROBOT_ROOT = "pluggybot"

# The robot's DISPLAY NAME (issue #39): "Luca the pluggybot" makes ROBOT_ROOT
# the species and this the identity, and keeping them apart is what lets a
# deployment rename its robot without touching body_census, the scene
# transpiler or a single fixture. Set per sim instance (--robot-name on
# serve.py / hub_lifecycle.py, or $PLUGGY_ROBOT_NAME); the default keeps
# every existing single-robot run reading sensibly.
ROBOT_NAME_ENV = "PLUGGY_ROBOT_NAME"
DEFAULT_ROBOT_NAME = "Pluggy"
# The name renders in the website's identity header; a cap makes a runaway
# env var a loud config error rather than a broken layout three hops away.
MAX_ROBOT_NAME_CHARS = 60


def robot_display_name(name: str | None = None) -> str:
  """Resolve this instance's display name: explicit > $PLUGGY_ROBOT_NAME >
  default. Blank resolves to the default -- an absent name must degrade to
  something readable, never to an empty string on the wire."""
  raw = name if name is not None else os.environ.get(ROBOT_NAME_ENV)
  raw = (raw or "").strip()
  if not raw:
    return DEFAULT_ROBOT_NAME
  if len(raw) > MAX_ROBOT_NAME_CHARS:
    raise ValueError(f"robot name is {len(raw)} chars, max "
                     f"{MAX_ROBOT_NAME_CHARS}: {raw[:40]!r}...")
  return raw


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
