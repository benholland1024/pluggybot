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

YIELDING (issue #208) is read off the world by the same activity, given the
two lifecycles: a robot that was at or heading for the charge bay
(`GO_CHARGE` / `CHARGE`) and LEFT it unfinished -- its pack still below
`CHARGED` -- while the other was below its reserve has yielded the bay,
whatever it meant by it. A `yield` event carries both packs at that
moment; a second one says whether the other then charged inside
`YIELD_WINDOW_S` (`honoured`) or did not (`lapsed`). One charge bay is kept
on purpose: contention IS the opportunity, and this is the reading of it.
No new action -- there is nothing to choose here -- and no arbitration.
"""

import math

from pluggybot.activity.base import Activity
from pluggybot.robot import RobotHandle

#: Within this, two robots have met; beyond `PARTED_M` they have parted.
ENCOUNTER_M = 1.5
PARTED_M = 2.0
#: A robot at or heading for the bay is in one of these states.
AT_THE_BAY = ("GO_CHARGE", "CHARGE")
#: Leaving the bay with the pack below this is leaving UNFINISHED: a robot
#: that charged to the top and drove off yielded nothing. `lifecycle.
#: CHARGED`, restated here to keep this module off the lifecycle.
CHARGED = 0.90
#: How long after a yield the other robot has to reach the bay for the
#: yield to have been `honoured` -- a charge cycle's approach, generously.
YIELD_WINDOW_S = 300.0


class Encounters(Activity):
  name = "encounters"

  def __init__(self, model, a: RobotHandle, b: RobotHandle,
               met_m: float = ENCOUNTER_M, parted_m: float = PARTED_M,
               lives: tuple = ()) -> None:
    super().__init__()
    self.a, self.b = a, b
    self.rebind(model, None)
    self.met_m, self.parted_m = met_m, parted_m
    self.on_event: list = []
    self.count = 0
    #: The two lifecycles, for the yield reading (issue #208); empty is
    #: encounters only, which is every caller before it.
    self.lives = tuple(lives)
    #: per robot: was it at the bay on the last step
    self._at_bay = {life.mission.handle.root: False for life in self.lives}
    #: yields waiting to be honoured: (yielder, needy, t)
    self._open: list = []
    self.yields = 0
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
    if self.lives:
      self._sense_yields(data)

  def _sense_yields(self, data) -> None:
    """Criterion, per step: a robot LEAVES `AT_THE_BAY` with its pack under
    `CHARGED` while the other's pack is under its reserve -- a yield. Then,
    for `YIELD_WINDOW_S`, whether the needy one reaches `CHARGE`."""
    now = float(data.time)
    for life in self.lives:
      root = life.mission.handle.root
      at = life.state in AT_THE_BAY
      was = self._at_bay[root]
      self._at_bay[root] = at
      if not (was and not at):
        continue
      if life.battery.fraction >= CHARGED:
        continue                          # finished, not yielded
      for other in self.lives:
        if other is life:
          continue
        if other.battery.energy_wh < other.low_battery_wh:
          self.yields += 1
          self._open.append((root, other.mission.handle.root, now))
          self._emit_yield("yielded", now, life, other)
    for yielder, needy, t0 in list(self._open):
      needy_life = next(o for o in self.lives if o.mission.handle.root == needy)
      giver = next(o for o in self.lives if o.mission.handle.root == yielder)
      if needy_life.state == "CHARGE":
        self._open.remove((yielder, needy, t0))
        self._emit_yield("honoured", now, giver, needy_life, since=now - t0)
      elif now - t0 > YIELD_WINDOW_S:
        self._open.remove((yielder, needy, t0))
        self._emit_yield("lapsed", now, giver, needy_life, since=now - t0)

  def _emit_yield(self, phase: str, t: float, yielder, needy,
                  since: float | None = None) -> None:
    event = {"type": "yield", "t": round(t, 3), "phase": phase,
             "robot": yielder.mission.handle.root,
             "to": needy.mission.handle.root,
             "yielderFrac": round(float(yielder.battery.fraction), 4),
             "needyFrac": round(float(needy.battery.fraction), 4),
             "needyReserveWh": round(float(needy.low_battery_wh), 4),
             **({"sinceS": round(since, 1)} if since is not None else {})}
    for hook in list(self.on_event):
      hook(dict(event))

  def _emit(self, phase: str, data, dist: float) -> None:
    event = {"type": "encounter", "t": round(float(data.time), 3),
             "phase": phase, "robots": [self.a.root, self.b.root],
             "distanceM": round(dist, 3)}
    for hook in list(self.on_event):
      hook(dict(event))
