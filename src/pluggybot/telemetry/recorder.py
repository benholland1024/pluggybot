"""JSONL telemetry recorder on the mission's step-hook seam.

Every physics step in the hub stack bottoms out in HubSwap._step_once,
which fires HubMission.step_hooks -- the same seam the battery drains
through. The recorder is one more callback there. Two rules shape it:

  - DECIMATE by sim time. The hook fires at 500 Hz (1000 during swaps,
    when the timestep drops); a frame every 1/FRAME_HZ sim-seconds is what
    the browser wants, so the fast path of the hook is a single float
    comparison and nothing else.
  - NO I/O IN THE PHYSICS STEP. The hook builds a plain dict of rounded
    floats and puts it on a queue; a writer THREAD drains the queue,
    serializes, and writes. Neither json.dumps nor the disk ever runs
    inside a step, and the physics never waits on either.

Frames are sparse: the first frame carries every dynamic body (a
keyframe); later frames carry only bodies that moved more than POS_EPS /
QUAT_EPS since they were LAST EMITTED. Comparing against the emitted pose
rather than the previous step matters -- per-step creep smaller than the
threshold still accumulates against the emitted pose and eventually
crosses it, instead of hiding under the threshold forever. A replayer
holds the last value it saw for any absent body.

Keyframes RECUR, every KEYFRAME_S of sim time, and say so: a keyframe
carries "key": true. Sparse frames are deltas against the whole history
before them, so a consumer that joins mid-stream is stuck until the next
keyframe -- and with a single keyframe at t=0 that never comes. Recurring
keyframes bound the wait (and, for the website's relay hub, bound what it
must cache to serve a late joiner: the last keyframe plus the frames
since). The marker is what keeps that consumer PROTOCOL-DUMB: "is this a
keyframe" is a field read, not a set comparison against the body census.

The frame-building lives in FrameBuilder so the live WebSocket publisher
(webserver v1, telemetry/publisher.py) emits byte-identical frames: the
recorder and the publisher are the same producer with different sinks.
"""

import gzip
import json
import queue
import threading
from typing import Callable

from pluggybot.telemetry.protocol import PROTOCOL_VERSION, ROBOT_ROOT, body_census

FRAME_HZ = 20.0     # the design point: 15-20 Hz reads smooth after
                    # client-side interpolation at 60 fps
KEYFRAME_S = 5.0    # sim seconds between full keyframes: the worst-case
                    # wait before a mid-stream joiner is complete, and the
                    # depth of the relay hub's join cache. At 20 Hz that is
                    # 1 frame in 100 -- a few percent of the byte budget.
POS_EPS = 0.0005    # m: half a millimetre of world-frame motion
QUAT_EPS = 0.0005   # per-component, sign-normalized (q and -q are the
                    # same rotation; xquat may hand us either)
NDIGITS = 4         # 0.1 mm -- below anything a viewer can see


class FrameBuilder:
  """Decimation + sparse-frame state for one robot's telemetry stream.

  build() returns the next due frame as a plain dict (or None between
  frames); header() is the stream's opening message. Not thread-safe: call
  both only from the physics thread. reset() clears the last-emitted poses
  so the NEXT frame is a full keyframe again -- on top of the every-
  keyframe_s cadence, a live consumer that reconnected (or missed dropped
  frames) needs one IMMEDIATELY: it has lost the state sparse frames build
  on, and holds stale poses until the stream re-keys.

  keyframe_s is sim seconds between periodic keyframes; 0 disables them
  (the stream then re-keys only on reset(), which is the pre-0.2.0
  behaviour and is what a from-the-top replay strictly needs).
  """

  def __init__(self, model, data, hz: float = FRAME_HZ,
               status_fn: Callable[[], dict] | None = None,
               model_name: str | None = None,
               keyframe_s: float = KEYFRAME_S,
               activities=None, boards=None, screens=None,
               ledger=None, accepts=(), goals: str = "",
               steering: bool = False) -> None:
    if keyframe_s < 0:
      # A negative interval keys EVERY frame and advertises a negative
      # cache depth (keyframeS x hz) to the hub. Fail at construction.
      raise ValueError(f"keyframe_s must be >= 0, got {keyframe_s}")
    self.model, self.data = model, data
    self.status_fn = status_fn
    self.activities = activities
    # Boards present the same duck type an ActivitySet does (`names` +
    # `snapshot()`), so they diff through the same code below -- and they are
    # here for the same reason activities are: a drawing has no body, so its
    # state reaches a viewer through this block or not at all.
    self.boards = boards
    # ...and so does a face (issue #13). Third duck of the same shape.
    self.screens = screens
    # ...and so does a balance (issue #14). Fourth, keyed by ROBOT rather
    # than by world feature -- which is the only difference, and the frame
    # builder does not care.
    self.ledger = ledger
    # Inbound message types this producer will act on (0.7.0, issue #16).
    # Advertised in the header so the server can tell "the robot got it" from
    # "the socket accepted it and nothing is listening".
    self.accepts = tuple(accepts)
    # What the robot is FOR, as prose, and whether anything is actually
    # reading it (0.8.0, rooftop-media-2026 #30). Not a header field and not
    # a frame block: it is up to MAX_GOALS_CHARS of text that never changes
    # during a run, so it goes out once per stream as its own message -- the
    # `board_snapshot` shape, for the `board_snapshot` reason.
    self.goals = goals
    self.steering = bool(steering)
    self.hz = hz
    self.model_name = model_name
    self.keyframe_s = keyframe_s
    self.frames = 0
    self.keyframes = 0
    self._interval = 1.0 / hz
    self._next_t: float | None = None
    self._next_key: float | None = None
    self._key_due = True             # frame 1 is always a keyframe
    robot, world = body_census(model)
    self.robot_names, self.world_names = robot, world
    self._robot = [(n, model.body(n).id) for n in robot]
    self._world = [(n, model.body(n).id) for n in world]
    self._last: dict[int, tuple[list[float], list[float]]] = {}
    # Sparse-emission memory for activity flags -- this builder's own, so
    # two sinks over one world (serve.py --record) never eat each other's
    # deltas. Same reason `_last` holds poses here rather than on the bodies.
    self._last_acts: dict[str, dict] = {}
    self._last_boards: dict[str, dict] = {}
    self._last_screens: dict[str, dict] = {}
    self._last_ledger: dict[str, dict] = {}

  def header(self) -> dict:
    return {
      "type": "header",
      "protocolVersion": PROTOCOL_VERSION,
      "model": self.model_name,
      "hz": self.hz,
      "keyframeS": self.keyframe_s,
      "robots": {ROBOT_ROOT: self.robot_names},
      "world": self.world_names,
      "activities": self.activities.names if self.activities else [],
      "boards": self.boards.names if self.boards else [],
      "screens": self.screens.names if self.screens else [],
      "ledger": self.ledger.names if self.ledger else [],
      # What this producer will ACT ON if the server sends it (0.7.0). Empty
      # is the normal answer and the important one: a sim with no overseer
      # reads nothing, so a website that marked a suggestion "delivered"
      # because the socket accepted it would be reporting a conversation that
      # is not happening. "Delivered" has to mean somebody who can hear you
      # got it, which is why this is advertised rather than assumed.
      "accepts": list(self.accepts),
    }

  def goals_message(self, t: float) -> dict | None:
    """The stream's opening statement of purpose, or None when there is none.

    Emitted once per stream rather than per frame, and NOT folded into the
    header, because the header describes the stream's shape while this is
    content -- and content a viewer joining an hour in still needs, which is
    why the live publisher re-sends it on every connect.

    `steering` is the honest half. The goals file is read on every run, but
    only an overseer decides anything with it; a scripted rotation is living
    by these in the sense that they were chosen with it in mind, and in no
    other sense. Reporting both identically would be the `accepts` mistake
    over again -- a website saying "following its goals" about a robot with
    nothing reading them.
    """
    if not self.goals:
      return None
    return {"type": "goals", "t": round(float(t), 3), "robot": ROBOT_ROOT,
            "text": self.goals, "steering": self.steering}

  def reset(self) -> None:
    """Make the next frame a keyframe (every dynamic body shipped)."""
    self._last.clear()
    self._last_acts.clear()
    self._last_boards.clear()
    self._last_screens.clear()
    self._last_ledger.clear()
    self._key_due = True

  def build(self) -> dict | None:
    """The next frame if one is due at this sim time, else None."""
    t = float(self.data.time)
    if self._next_t is not None and t < self._next_t:
      return None
    self._next_t = t + self._interval
    self.frames += 1
    frame: dict = {"t": round(t, 3)}
    if self.keyframe_s and (self._next_key is None or t >= self._next_key):
      self._key_due = True
    if self._key_due:
      # Clearing the emitted-pose memory is what MAKES the frame a keyframe:
      # every body then reads as moved, so every body ships.
      self._last.clear()
      # Activities re-ship on a keyframe for the same reason poses do: a
      # consumer joining mid-stream has never seen them. It matters MORE
      # here -- an activity's visible effect often lives on a STATIC body
      # (a gate moved by geom toggle ships once in the scene and never
      # again), so the flag is the only record of it anywhere in the stream.
      self._last_acts.clear()
      self._last_boards.clear()
      self._last_screens.clear()
      self._last_ledger.clear()
      self._key_due = False
      if self.keyframe_s:      # 0 would schedule the NEXT frame, keying all
        self._next_key = t + self.keyframe_s
      self.keyframes += 1
      frame["key"] = True
    robot_rec: dict = {}
    bodies = {}
    for name, bid in self._robot:
      pose = self._pose_if_moved(bid)
      if pose is not None:
        bodies[name] = pose
    if bodies:
      robot_rec["bodies"] = bodies
    if self.status_fn is not None:
      robot_rec.update(self.status_fn())
    frame["robots"] = {ROBOT_ROOT: robot_rec}
    world = {}
    for name, bid in self._world:
      pose = self._pose_if_moved(bid)
      if pose is not None:
        world[name] = pose
    if world:
      frame["world"] = world
    if self.activities is not None:
      acts = self._sparse(self.activities.snapshot(), self._last_acts)
      if acts:
        frame["activities"] = acts
    if self.boards is not None:
      boards = self._sparse(self.boards.snapshot(), self._last_boards)
      if boards:
        frame["boards"] = boards
    if self.screens is not None:
      screens = self._sparse(self.screens.snapshot(), self._last_screens)
      if screens:
        frame["screens"] = screens
    if self.ledger is not None:
      points = self._sparse(self.ledger.snapshot(), self._last_ledger)
      if points:
        frame["ledger"] = points
    return frame

  @staticmethod
  def _sparse(snapshot: dict, last: dict) -> dict:
    """Whatever changed since it was last emitted, remembering it.

    `last` lives on the BUILDER, never on the activity or the board:
    `serve.py --record` runs a publisher and a recorder over one world, each
    with its own builder, and a shared memory would have the two sinks eating
    each other's deltas -- each shipping a random half of the changes.
    """
    changed = {name: flags for name, flags in snapshot.items()
               if last.get(name) != flags}
    last.update({k: dict(v) for k, v in changed.items()})
    return changed

  def _pose_if_moved(self, bid: int) -> list[float] | None:
    """[x,y,z,qw,qx,qy,qz] world-frame, or None if within eps of the pose
    last emitted for this body."""
    xp = self.data.xpos[bid]
    xq = self.data.xquat[bid]
    last = self._last.get(bid)
    if last is not None:
      lp, lq = last
      if all(abs(float(xp[i]) - lp[i]) < POS_EPS for i in range(3)):
        s = 1.0 if sum(float(xq[i]) * lq[i] for i in range(4)) >= 0 else -1.0
        if all(abs(s * float(xq[i]) - lq[i]) < QUAT_EPS for i in range(4)):
          return None
    p = [float(xp[i]) for i in range(3)]
    q = [float(xq[i]) for i in range(4)]
    self._last[bid] = (p, q)
    return [round(v, NDIGITS) for v in p + q]


class TelemetryRecorder:
  """Decimating JSONL recorder for one robot's mission.

  Append `step_hook` to HubMission.step_hooks; call `close()` when the
  mission ends (it drains the queue and joins the writer). A path ending
  in .gz records through gzip transparently. `status_fn`, if given, is
  called once per emitted frame and its dict is merged into the robot's
  record -- the lifecycle supplies state / status line / battery there.
  """

  def __init__(self, model, data, path: str, hz: float = FRAME_HZ,
               status_fn: Callable[[], dict] | None = None,
               model_name: str | None = None,
               keyframe_s: float = KEYFRAME_S,
               activities=None, boards=None, screens=None,
               ledger=None, accepts=(), goals: str = "",
               steering: bool = False) -> None:
    self._builder = FrameBuilder(model, data, hz=hz, status_fn=status_fn,
                                 model_name=model_name, keyframe_s=keyframe_s,
                                 activities=activities, boards=boards,
                                 screens=screens, ledger=ledger,
                                 accepts=accepts, goals=goals,
                                 steering=steering)
    self._queue: queue.SimpleQueue = queue.SimpleQueue()
    self._closed = False
    self._queue.put(self._builder.header())
    # What the robot is for, before the first frame (0.8.0). Same slot the
    # board snapshots below use, and the same argument: no keyframe re-ships
    # it, so a reader that missed this line never learns it at all.
    goals_msg = self._builder.goals_message(float(data.time))
    if goals_msg is not None:
      self._queue.put(goals_msg)
    # Whatever is already on the walls, before the first frame (0.5.0). A
    # recording made against boards that survived a previous run opens with
    # a robot standing in front of a drawing it did not make -- and without
    # this, a replayer would show it standing in front of a blank wall while
    # the `boards` block insisted the wall was full.
    if boards is not None:
      for snapshot in boards.snapshots(t=float(data.time)):
        self._queue.put(snapshot)
    self._thread = threading.Thread(target=self._write, args=(path,),
                                    daemon=True)
    self._thread.start()

  @property
  def frames(self) -> int:
    return self._builder.frames

  # ---- the hook (runs inside every physics step) ---------------------------

  def step_hook(self) -> None:
    frame = self._builder.build()
    if frame is not None:
      self._queue.put(frame)

  # ---- out-of-band messages ------------------------------------------------

  def emit(self, message: dict) -> None:
    """Write a typed message into the stream between frames (0.4.0).

    The `draw` and `board_cleared` events go through here. They cannot ride
    in a frame: a stroke is not a per-tick quantity, and decimating one to
    20 Hz would mean either shipping the same polyline a hundred times or
    dropping it. So the recording becomes a mixed stream, and a reader
    dispatches on "type" -- no "type" means frame.

    Ordering with frames is by queue arrival, which is by sim time, because
    both come off the physics thread. A consumer replaying in `t` order gets
    the stroke at the moment the pen finished it.
    """
    self._queue.put(dict(message))

  # ---- the writer (its own thread; owns all I/O) ---------------------------

  def _write(self, path: str) -> None:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
      while True:
        item = self._queue.get()
        if item is None:
          return
        f.write(json.dumps(item, separators=(",", ":")) + "\n")

  def close(self) -> None:
    """Drain everything queued and finish the file. Idempotent."""
    if self._closed:
      return
    self._closed = True
    self._queue.put(None)
    self._thread.join(timeout=30.0)
