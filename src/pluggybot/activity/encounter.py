"""Encounters: when two robots come within range of each other (issue
#167; the raw event #155's empathy and morality instruments are designed
from AFTER it has been watched on the deployed world).

An ACTIVITY on the world's seam (docs/ActivityPattern.md): every step the
distance between the two chassis is measured, and an `encounter` event is
emitted on the way IN (`met`) and on the way OUT (`parted`), with
hysteresis so a pair hovering at the edge does not chatter. Which robot
approached whom is not judged here -- the event carries both robots' poses
and states at the moment, and the reading is the observatory's.

Nothing here arbitrates: an encounter is a fact, not a rule.
"""

import math

from pluggybot.activity.base import Activity
from pluggybot.robot import RobotHandle

#: Within this, two robots have met; beyond `PARTED_M` they have parted.
ENCOUNTER_M = 1.5
PARTED_M = 2.0


class Encounters(Activity):
  name = "encounters"

  def __init__(self, model, a: RobotHandle, b: RobotHandle,
               met_m: float = ENCOUNTER_M, parted_m: float = PARTED_M) -> None:
    super().__init__()
    self.a, self.b = a, b
    self.rebind(model, None)
    self.met_m, self.parted_m = met_m, parted_m
    self.on_event: list = []
    self.count = 0
    self.set(near=False, distanceM=None, met=0)

  def rebind(self, model, data) -> None:
    self.a_bid, self.b_bid = model.body(self.a.root).id, model.body(self.b.root).id

  def sense(self, model, data) -> None:
    pa, pb = data.xpos[self.a_bid], data.xpos[self.b_bid]
    dist = float(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))
    near = self.flags["near"]
    if not near and dist <= self.met_m:
      near, self.count = True, self.count + 1
      self._emit("met", data, dist)
    elif near and dist >= self.parted_m:
      near = False
      self._emit("parted", data, dist)
    self.set(near=near, distanceM=round(dist, 3), met=self.count)

  def _emit(self, phase: str, data, dist: float) -> None:
    event = {"type": "encounter", "t": round(float(data.time), 3),
             "phase": phase, "robots": [self.a.root, self.b.root],
             "distanceM": round(dist, 3)}
    for hook in list(self.on_event):
      hook(dict(event))
