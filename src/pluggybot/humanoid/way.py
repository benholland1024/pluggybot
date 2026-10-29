"""Making way, on two legs (issue #559; the quadruped's is #415's,
`legs/way.py`): a humanoid RESTING across the other robot's way is asked to
step off it, and does. A mixin over `MakeWay`: `HumanoidMission` is the
rest of the body.

THE RULE is the quadruped's, and so is where it steps (`MakeWay.aside_spot`:
the nearest floor it has seen, clear of the asker's way). What differs is
what resting is: a quadruped lies down by reflex 8.6 s after its last motion
command (`legs.posture.T_REST_S`), and a humanoid rests STANDING -- it has
had no motion command for as long, no walk of its own waiting, no hand out
and its legs on the floor (`resting`). Without it two standing humanoids
waited on each other at a doorway for 500 s of a minded day, each walk of
each giving up 1.2 m from the other (SimNotes, "The humanoid pair served").

The step aside is code beneath the mind, as the quadruped's: `_aside_command`
takes the place of a STILL command until the body stands where it was sent,
and gives way to any command that moves. Its legs are handed to the walking
policy there and then where they stand on the standing one's -- at rest its
arms are home and its posture held, which is all `to_walk_routine` waits for.
"""

from __future__ import annotations

import math

from pluggybot.behavior.navigation import drive_toward
from pluggybot.legs.posture import T_REST_S
from pluggybot.legs.way import MakeWay
from pluggybot.navigator import ARRIVAL_SLOW_RADIUS, CLOSE_ENOUGH_M, WAYPOINT_REACHED_M

#: How long it has had no motion command before it rests, s: the
#: quadruped's reflex's (`T_REST_S`), so a pair rests and is asked aside on
#: one clock whatever its bodies.
REST_S = T_REST_S


class HumanoidWay(MakeWay):
  """Making way (the module docstring), on `HumanoidMission`."""

  #: No step aside under way, and when it last moved, sim s (`_init_way`).
  _aside = None
  last_motion_t = 0.0

  def _init_way(self) -> None:
    super()._init_way()
    #: When it was last commanded to move, sim s: what `resting` reads.
    self.last_motion_t = 0.0

  @property
  def resting(self) -> bool:
    """Resting STANDING: no motion command for `REST_S`, no walk of its own,
    no hand out and on its feet."""
    return (self.legs != "getup" and not self.driving and self._aside is None
            and all(p == "rest" for p in self.reach.phase.values())
            and float(self.data.time) - self.last_motion_t >= REST_S)

  def make_way(self, route, by: str) -> bool:
    """`Body.make_way`, and why not (`way_refused`)."""
    self.way_refused = ""
    if self._aside is not None:
      return True
    self.way_refused = ("getting up" if self.legs == "getup"
                        else "working" if any(p != "rest" for p in self.reach.phase.values())
                        else "walking" if self.driving
                        else "standing" if not self.resting else "")
    if self.way_refused:
      return False
    spot = self.aside_spot(route)
    if spot is None:
      self.way_refused = "no spot"
      return False
    x, y = self.pose_xy()
    self._aside = {"by": by, "from": (round(float(x), 3), round(float(y), 3)),
                   "to": spot, "t0": float(self.data.time), "waypoints": None}
    self.making_way = by
    return True

  def _aside_command(self, command, t: float):
    """One physics step of making way, in place of the routine's command:
    the routine's own once it moves -- its walk takes over -- else the
    drive's law along a plan to the spot, and STILL once it is there."""
    from pluggybot.humanoid.mission import STILL, command_for
    a = self._aside
    if any(command):
      return self._end_aside("its own walk", command)
    if self.legs == "getup":
      return self._end_aside("fell", command)
    if any(p != "rest" for p in self.reach.phase.values()):
      return self._end_aside("arm out", STILL)
    x, y, _ = self.pose
    tx, ty = a["to"]
    if a["waypoints"] is None:
      planned = self._plan_to(tx, ty)
      if planned is None:
        return self._end_aside("no route", STILL)
      a["waypoints"] = list(planned)
    wps = a["waypoints"]
    while wps and math.hypot(wps[0][0] - x, wps[0][1] - y) < WAYPOINT_REACHED_M:
      wps.pop(0)
    if not wps and math.hypot(tx - x, ty - y) < CLOSE_ENOUGH_M:
      return self._end_aside("aside", STILL)
    if t - a["t0"] > self.ASIDE_BUDGET_S:
      return self._end_aside("out of time", STILL)
    if t < self.backoff_until:
      # ⚠ THE FRONT STOP'S REFLEX, as a walk's (`Navigator.drive_to_routine`):
      # straight back, then a new plan. Ended on it, as the quadruped's is, a
      # humanoid resting in a doorway -- its stop re-armed by every scan --
      # never stepped aside at all, 3 doors of 6 (`make_way_spike.py`)
      a["waypoints"] = None
      return command_for(-self.BACKOFF_V, 0.0, 0.0)
    if len(wps) > 1:
      v, w = drive_toward(self.pose, wps[0])
    else:
      v, w = drive_toward(self.pose, wps[0] if wps else (tx, ty),
                          slow_radius=ARRIVAL_SLOW_RADIUS)
    return command_for(v, 0.0, w)

  def drive_to_routine(self, wx: float, wy: float, timeout: float = 90.0,
                       stop=None, beyond=()):
    """The Navigator's walk, counted while it runs (`driving`): a body
    waiting inside a walk of its own is never asked aside."""
    self.driving += 1
    try:
      return (yield from super().drive_to_routine(wx, wy, timeout, stop, beyond))
    finally:
      self.driving -= 1
