"""Protocol version stamp + the dynamic-body census both emitters share.

The scene JSON ships every body once; telemetry frames re-ship only the
bodies that can move. Both sides must agree on which bodies those are and
which belong to the robot, so the census lives here, in one place.

Bump PROTOCOL_VERSION whenever either artifact's shape changes. The website
repo vendors fixture copies stamped with this version, so a bump is a
deliberate two-repo event -- never a side effect of an unrelated edit.
"""

import os

PROTOCOL_VERSION = "0.13.0"
#: What changed at each version -- every entry from 0.2.0 on, with the
#: worked JSON and the reasoning -- is `protocol/README.md`, which is the
#: canonical spec and the half the website repo reads. It is not summarised
#: here: two copies of a two-repo contract drift, and the copy a reader
#: trusts is whichever one they opened first.

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

# The visitor channel's vocabularies (issue #16, `reset_tool` from #30).
# Two-repo contracts on the same terms as VISUAL_HINTS and FACE_STATES: the
# website may only SEND these inbound types and may only receive these
# outcomes. Adding one is additive (the sim counts and drops an unknown
# inbound type; a consumer ignores an unknown outcome); renaming one breaks
# both repos.
#
# They live HERE rather than in mind/inbox.py, which is where the parsing is,
# so the wire spec and the parser cannot disagree -- and in this direction,
# because the sim already imports `telemetry` and the reverse would invert
# the layering for a tuple of strings.
#
# ⚠ A vocabulary entry with nothing behind it is a promise the robot cannot
# keep, so `move` / `clear_board` (tic-tac-toe, named in #16 as later work)
# are absent until there is a board game.
#: What each type means on the wire, and why `reset_tool` was added without a
#: version bump: protocol/README.md, "Downstream: server -> sim".
INBOUND_TYPES = ("suggestion", "question", "rating", "reset_tool")

#: The inbound kinds CODE handles without an overseer: applied by the physics
#: thread the moment they are drained, never shown to a model. What a served
#: world with no overseer advertises in `accepts` -- a suggestion to a robot
#: with nothing reading it is a conversation that is not happening (the
#: `accepts` lesson), but a rating settles a ledger row and a reset moves a
#: module, and both of those work on a scripted world.
CODE_HANDLED_TYPES = ("rating", "reset_tool")

# The task system's vocabularies (issue #21). Two-repo contracts on the same
# terms as the three above, and here rather than in economy/tasks.py -- where
# the state machine is -- so the wire spec and the implementation cannot
# disagree.
#
# The KINDS themselves are NOT here: a kind carries an evaluator name, a
# target type and an energy estimate, which are sim-side facts rather than
# wire vocabulary (`economy.tasks.KINDS`). The header advertises the names of
# whatever this producer knows.
#
# ⚠ `offered` -> `claimed` -> `active` -> `done` | `failed` | `expired`, and
# the three terminal states must stay distinguishable: a lapsed offer is
# drawn differently from a job the robot tried and got wrong.
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

#: THE ROBOT'S APPETITE (0.13.0, issue #36). Points are food: consumed at a
#: steady rate on sim time, capped rather than accumulated, and once there is
#: enough the robot is SATISFIED and spends its time on its goals instead.
#: A two-repo vocabulary on FACE_STATES' terms -- adding a state is additive
#: (a client falls back to rendering the number), renaming one breaks both.
#:
#: ⚠ `satisfied` and `fed` are BOTH "above the hungry line", and they are not
#: interchangeable: `fed` is climbing and not there yet, `satisfied` is the
#: latch that says stop working. The gap between them is hysteresis
#: (economy/metabolism.py), and a client that collapsed the two would draw a
#: gauge that flickered exactly where the robot is most stable.
#:
#: ⚠ `starving` IS NOT A DISABLED ROBOT. It is narrative -- a face and a line
#: in History -- and nothing in the sim gates on it: a robot at zero points
#: charges, navigates and stows exactly as it always did. A client that
#: rendered it as a fault would be reporting a state that does not exist.
HUNGER_STATES = ("starving", "hungry", "fed", "satisfied")

# The thought files (issue #38). Two-repo vocabulary on the same terms as
# TASK_STATES above: the names and the writers are what the wire may carry,
# while the CAPS and the write rules are sim-side and live in mind/thoughts.py,
# which owns the one write path that enforces them.
#
# ⚠ WHO MAY WRITE IS PER FILE, and rendering the four identically would claim
# a mind that wrote its own persona. `human` is a person editing the volume;
# `system` is append-only narrative; `robot` is the one writable surface.
#: Why each file has the writer it has: docs/Overseer.md section 7.
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
