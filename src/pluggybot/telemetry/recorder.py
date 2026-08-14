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
"""

import gzip
import json
import queue
import threading
from typing import Callable

from pluggybot.telemetry.protocol import PROTOCOL_VERSION, ROBOT_ROOT, body_census

FRAME_HZ = 20.0     # the design point: 15-20 Hz reads smooth after
                    # client-side interpolation at 60 fps
POS_EPS = 0.0005    # m: half a millimetre of world-frame motion
QUAT_EPS = 0.0005   # per-component, sign-normalized (q and -q are the
                    # same rotation; xquat may hand us either)
NDIGITS = 4         # 0.1 mm -- below anything a viewer can see


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
               model_name: str | None = None) -> None:
    self.model, self.data = model, data
    self.status_fn = status_fn
    self.frames = 0
    self._interval = 1.0 / hz
    self._next_t: float | None = None
    robot, world = body_census(model)
    self._robot = [(n, model.body(n).id) for n in robot]
    self._world = [(n, model.body(n).id) for n in world]
    self._last: dict[int, tuple[list[float], list[float]]] = {}
    self._queue: queue.SimpleQueue = queue.SimpleQueue()
    self._closed = False
    self._queue.put({
      "type": "header",
      "protocolVersion": PROTOCOL_VERSION,
      "model": model_name,
      "hz": hz,
      "robots": {ROBOT_ROOT: robot},
      "world": world,
    })
    self._thread = threading.Thread(target=self._write, args=(path,),
                                    daemon=True)
    self._thread.start()

  # ---- the hook (runs inside every physics step) ---------------------------

  def step_hook(self) -> None:
    t = float(self.data.time)
    if self._next_t is not None and t < self._next_t:
      return
    self._next_t = t + self._interval
    self.frames += 1
    frame: dict = {"t": round(t, 3)}
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
    self._queue.put(frame)

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
