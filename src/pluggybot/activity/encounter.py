"""Encounters: when two robots come within range of each other (issue
#167; the raw event #155's empathy and morality instruments are designed
from AFTER it has been watched on the deployed world).

An ACTIVITY on the world's seam (docs/ActivityPattern.md): every step the
distance between the two chassis is measured, and an `encounter` event is
emitted on the way IN (`met`) and on the way OUT (`parted`), with
hysteresis so a pair hovering at the edge does not chatter. Which robot
approached whom is not judged here -- the event carries both robots' poses
and states at the moment, and the reading is the observatory's.

TOUCHING is the third phase (issue #316), read off the contact array the
same step: the two robots are in contact, which until now left nothing
behind but `HubSwap.collision_steps`, a counter on neither robot's record
and on no wire. So a collision was a thing somebody had to be watching to
see, and nine `stuck` deaths in a week could not be attributed to it
either way. It latches like the plate's press -- `touched` on the way in,
`separated` on the way out past `CONTACT_HOLD_S` -- because a bump
bounces.

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

import numpy as np

from pluggybot.activity.base import Activity
from pluggybot.perception.lidar import robot_geoms
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
#: A contact is over this long after the last one, and not before: the same
#: 50 ms the bumper holds past its last contact (`HubSwap`, issue #94),
#: because a robot meeting something at cruise speed bounces off it and a
#: row per bounce is a row per frame of one collision.
CONTACT_HOLD_S = 0.05


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
    #: Contacts between the two bodies (issue #316), and when the current
    #: one may be called over.
    self.bumps = 0
    self._touching = False
    self._touch_until = 0.0
    self.set(near=False, distanceM=None, met=0, touching=False, bumps=0)

  def rebind(self, model, data) -> None:
    self.a_bid, self.b_bid = model.body(self.a.root).id, model.body(self.b.root).id
    # WHOSE GEOM IS WHOSE, as a lookup indexed by geom id: the contact
    # array is read with one fancy index per step, never a Python loop over
    # `data.contact[i]` (rooftop #296 measured four such loops at 49 % of
    # the physics thread). A geom belongs to one robot or to neither, so
    # 1 | 2 == 3 is a contact between the two and nothing else is.
    self._owner = np.zeros(model.ngeom, dtype=np.int8)
    for mark, handle in ((1, self.a), (2, self.b)):
      for g in robot_geoms(model, handle.root):
        self._owner[g] = mark

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
    self._sense_contact(data, dist)
    self.set(near=near, distanceM=round(dist, 3), met=self.count,
             touching=self._touching, bumps=self.bumps)
    if self.lives:
      self._sense_yields(data)

  def _sense_contact(self, data, dist: float) -> None:
    """Are the two bodies touching, off the contact array (issue #316)?

    Sensed, never assumed: a robot that drove into its pair is a fact
    about the world, and the only instrument that could have caught it
    before was somebody watching. The reflex that should stop it first is
    the front stop, which was blind to another robot until the scan learnt
    to answer its peer returns separately (`Lidar.scan_split`).
    """
    g = data.contact.geom[:data.ncon]
    hit = False
    if g.size:
      owners = self._owner[g]
      hit = bool(((owners[:, 0] | owners[:, 1]) == 3).any())
    now = float(data.time)
    # ⚠ The flag moves through `set` and not by hand: `Activity.set` is
    # what fires `on_change`, and a flag written into the dict first is a
    # flag the wire never hears about.
    if hit:
      self._touch_until = now + CONTACT_HOLD_S
      if not self._touching:
        self.bumps += 1
        self._touching = True
        self._emit("touched", data, dist)
    elif self._touching and now >= self._touch_until:
      self._touching = False
      self._emit("separated", data, dist)

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
