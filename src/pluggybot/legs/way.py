"""Making way, on legs (issue #415): a robot lying down to rest across
another's way is asked to step off it, and does. A mixin: `QuadMission` is
the rest of the body.

THE RULE (Ben, 2026-09-30, option 1 of #415): resting by reflex is not a
fall (#365), so another robot's drive waited on a resting body or gave up
at its disc, and the resting robot never moved -- it stands only when its
own next command moves it, and a finished robot never gets one. A drive
whose way such a body cuts asks it (`Navigator._ask_way`, over the pair's
channel `Body.ask_way`), and a body that is RESTING and free to -- not on
the dock, not mid-move or mid-swap, no walk of its own waiting -- stands
and steps aside. Like the rest reflex it is code beneath the mind, and
beneath whatever routine the body is holding: `_aside_command` takes the
place of a STILL command until the body stands where `aside_spot` put it,
and gives way to any command that moves.

WHERE (`aside_spot`): the nearest floor this robot has SEEN, by its own
walk over its own map with the asker kept clear of, that is `ASIDE_CLEAR_M`
from every point of the asker's way -- out of the disc the asker keeps
round it, so the way it sent is open whatever it plans next. Into the room
past a doorway and to one side, where the asker's way crosses it.

The rule and the walk are ONE for both bodies (#559): what only a body says
are its hooks -- why it will not (`_way_refusal`), what it does before it
walks (`_aside_ready`), what stops the walk (`_aside_blocked`) and the
command that walks it (`_aside_twist`). The humanoid's are `humanoid.way`'s.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.mapping import optimistic
from pluggybot.navigator import (ARRIVAL_SLOW_RADIUS, CLOSE_ENOUGH_M, WAYPOINT_REACHED_M,
                                 Navigator)

#: A command that moves nothing: every body's (`Body.STILL`).
STILL = (0.0, 0.0, 0.0)


class MakeWay:
  """Making way (the module docstring), on `QuadMission`, and under its
  own hooks on `HumanoidMission` (`humanoid.way`)."""

  #: The farthest it walks to step aside, m, and the longest it takes, s:
  #: MEASURED over six doorways of the house (`scripts/make_way_spike.py`),
  #: the stand-up (2.2 s) and a walk of 0.82-0.88 m took 6.7-8.9 s.
  ASIDE_WALK_M, ASIDE_BUDGET_S = 6.0, 30.0
  #: It rests LYING (the quadruped's reflex, #387), as its record of a step
  #: aside says (`last_aside`'s `lay`).
  RESTS_LYING = True

  def _init_way(self) -> None:
    #: The step aside under way: for whom, from where, to where, when it
    #: began and its waypoints (None until it stands); None between.
    self._aside: dict | None = None
    self.making_way: str | None = None
    #: Why the last ask was said no to (`Body.way_refused`), or "".
    self.way_refused = ""
    self.asides = 0
    self.last_aside: dict | None = None
    #: Walks of its own under way (`drive_to_routine`): a body waiting in
    #: one is never asked aside -- its own walk decides where it goes.
    self.driving = 0

  @property
  def aside_clear_m(self) -> float:
    """How far off the asker's way it stands, m: the disc the asker keeps
    round a standing robot (`OTHER_ROBOT_CELLS`, 0.70 m) and three cells
    -- the asker's plan is laid on a 10 cm lattice whose cells pass only
    when every map cell in them does."""
    return (self.OTHER_ROBOT_CELLS + 3) * self.grid.resolution

  def make_way(self, route, by: str) -> bool:
    """`Body.make_way`, and why not (`way_refused`)."""
    self.way_refused = ""
    if self._aside is not None:
      return True
    self.way_refused = self._way_refusal()
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

  def aside_spot(self, route) -> tuple[float, float] | None:
    """Where to step off `route` (the module docstring): the floor it has
    seen, reachable round the other robots within `ASIDE_WALK_M`, at least
    `aside_clear_m` from every point of the route, the shortest walk from
    here; None where there is none."""
    from scipy.ndimage import distance_transform_edt
    pts = np.asarray(route, dtype=float).reshape(-1, 2)
    if not len(pts):
      return None
    g = self.grid
    side = g.resolution * optimistic.BLOCK
    lat = self.seen_floor_lattice(mask_others=True)
    rows, cols = lat.shape
    walk = self.walk_field(lat, self.pose_xy(), limit=self.ASIDE_WALK_M,
                           over=self.seen_floor_lattice())
    # the route on the lattice, every half cell of it, and each cell's
    # distance from it
    dense = [pts[:1]]
    for a, c in zip(pts[:-1], pts[1:]):
      n = max(1, int(math.ceil(math.hypot(*(c - a)) / (0.5 * side))))
      dense.append(a + (c - a) * (np.arange(1, n + 1)[:, None] / n))
    dense = np.concatenate(dense)
    ix = np.floor((dense[:, 0] - g.x_min) / side).astype(np.int64)
    iy = np.floor((dense[:, 1] - g.y_min) / side).astype(np.int64)
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    on = np.zeros(lat.shape, dtype=bool)
    on[iy[ok], ix[ok]] = True
    off = (distance_transform_edt(~on) * side if on.any()
           else np.full(lat.shape, np.inf))
    good = np.isfinite(walk) & (off >= self.aside_clear_m)
    if not good.any():
      return None
    k = int(np.argmin(np.where(good, walk, np.inf)))
    ky, kx = divmod(k, cols)
    return (round(g.x_min + (kx + 0.5) * side, 3), round(g.y_min + (ky + 0.5) * side, 3))

  def _way_refusal(self) -> str:
    """Why it will not step aside now, or "": on the dock, mid-move with its
    arm, inside a walk of its own, mid-move of its legs, or standing -- in
    nobody's way by its own rule, as it rests lying."""
    from pluggybot.legs.body import LYING
    return ("docked" if self.docked else "working" if self.working
            else "walking" if self.driving
            else "moving" if self._move is not None
            else "standing" if self.posture != LYING else "")

  def _aside_command(self, command, t: float):
    """One physics step of making way, in place of the routine's command
    (the module docstring): what the body does first (`_aside_ready`: the
    routine's own walk takes over, a fall ends it, a body lying stands),
    then the drive's law along a plan to the spot, and STILL once it is
    there."""
    ready = self._aside_ready(command)
    if ready is not None:
      return ready
    a = self._aside
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
    blocked = self._aside_blocked(t)
    if blocked is not None:
      return blocked
    if len(wps) > 1:
      v, w = drive_toward(self.pose, wps[0])
    else:
      v, w = drive_toward(self.pose, wps[0] if wps else (tx, ty),
                          slow_radius=ARRIVAL_SLOW_RADIUS)
    return self._aside_twist(v, w)

  def _aside_ready(self, command):
    """The command this step before the walk, or None to walk: the
    routine's own once it moves, the end of it on a fall, a motion command
    while it lies (the posture machine stands it first) and STILL while it
    stands up."""
    from pluggybot.legs.body import GETTING_UP, LYING, STANDING, W_MIN, is_motion
    if is_motion(command):
      return self._end_aside("its own walk", command)
    if self.posture == GETTING_UP:
      return self._end_aside("fell", command)
    if self.posture == LYING:
      return (0.0, 0.0, W_MIN)                    # stands it, by the reflex's own rule
    if self.posture != STANDING:
      return STILL
    return None

  def _aside_blocked(self, t: float):
    """The command that ends the walk, or None to walk on: its front stop
    or its bumper, or its arm out of its stow."""
    if t < self.backoff_until or (self.pressing and not (
        self.pressed_at is not None and self.robot_at(*self.pressed_at))):
      # ...never where another robot's reported pose puts its body (#455):
      # the pair collides, and a robot stepping off a way can start against
      # the asker -- which no touch can tell from a wall
      return self._end_aside("met something", STILL)
    if not self.arm_driving():
      return self._end_aside("arm out", STILL)
    return None

  def _aside_twist(self, v: float, w: float):
    """The command that walks (v, w): its turn held to `W_CARRY` with a
    tool aboard."""
    from pluggybot.legs.body import W_CARRY, command_for
    cmd = command_for(v, 0.0, w)
    if self.carrying is not None and abs(cmd[2]) > W_CARRY:
      cmd = (cmd[0], cmd[1], math.copysign(W_CARRY, cmd[2]))
    return cmd

  def _end_aside(self, why: str, command):
    """The step aside is over: its record, and the command to apply."""
    a, self._aside = self._aside, None
    self.making_way = None
    self.asides += 1
    x, y = self.pose_xy()
    self.last_aside = {"by": a["by"], "from": a["from"], "to": a["to"],
                       "at": (round(float(x), 3), round(float(y), 3)),
                       "seconds": round(float(self.data.time) - a["t0"], 1),
                       "why": why, "lay": self.RESTS_LYING}
    return command

  def drive_to_routine(self, wx: float, wy: float, timeout: float = 90.0,
                       stop=None, beyond=()):
    """`Navigator.drive_to_routine`, counted while it runs (`driving`): a
    body waiting inside a walk of its own is never asked aside (#415)."""
    self.driving += 1
    try:
      return (yield from Navigator.drive_to_routine(self, wx, wy, timeout, stop, beyond))
    finally:
      self.driving -= 1

  def clear_way_routine(self, way):
    """`Navigator.clear_way_routine`, counted as a walk of its own too
    (`driving`, #455): a press waiting for its way in lies down by the
    rest reflex 8.6 s into the wait, and asked aside then, it would walk
    off its standoff before its walk in."""
    self.driving += 1
    try:
      return (yield from Navigator.clear_way_routine(self, way))
    finally:
      self.driving -= 1
