"""Making way (issue #415; on two legs, #559): a humanoid RESTING across
the other robot's way is asked to step off it, and does. A mixin:
`HumanoidMission` is the rest of the body.

THE RULE (Ben, 2026-09-30, option 1 of #415): resting is not a fall (#365),
so another robot's drive waited on a resting body or gave up at its disc,
and the resting robot never moved -- it moves only when its own next
command moves it, and a finished robot never gets one. A drive whose way
such a body cuts asks it (`Navigator._ask_way`, over the pair's channel
`Body.ask_way`), and a body that is RESTING and free to -- not plugged in,
not getting up, no hand out, no walk of its own waiting -- steps aside.
It is code beneath the mind, and beneath whatever routine the body is
holding: `_aside_command` takes the place of a STILL command until the
body stands where `aside_spot` put it, and gives way to any command that
moves.

WHERE (`aside_spot`): the nearest floor this robot has SEEN, by its own
walk over its own map with the asker kept clear of, that is
`aside_clear_m` from every point of the asker's way -- out of the disc the
asker keeps round it, so the way it sent is open whatever it plans next.
Into the room past a doorway and to one side, where the asker's way
crosses it.

RESTING is STANDING (`resting`): no motion command for `REST_S`, no walk
of its own waiting, no hand out and its legs on the floor. Without it two
standing humanoids waited on each other at a doorway for 500 s of a minded
day, each walk of each giving up 1.2 m from the other (SimNotes, "The
humanoid pair served").

Its legs go to the walking policy there and then where they stand on the
standing one's (`HumanoidMission._before_step`), once its posture is the
stand's, as `to_walk_routine` hands them over; a routine that takes the
legs for the standing policy ends the step aside (`to_stand_routine`).
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.mapping import optimistic
from pluggybot.navigator import (ARRIVAL_SLOW_RADIUS, CLOSE_ENOUGH_M, WAYPOINT_REACHED_M,
                                 Navigator)

#: A command that moves nothing: the body's (`Body.STILL`).
STILL = (0.0, 0.0, 0.0)
#: How long it has had no motion command before it rests, s: the clock the
#: quadrupeds rested on by reflex (#387), kept, so the pair is asked aside
#: as it always was.
REST_S = 8.6


class HumanoidWay:
  """Making way (the module docstring), on `HumanoidMission`."""

  #: The farthest it walks to step aside, m, and the longest it takes, s:
  #: MEASURED over six doorways of the house (`scripts/make_way_spike.py`),
  #: the stand-up (2.2 s) and a walk of 0.82-0.88 m took 6.7-8.9 s.
  ASIDE_WALK_M, ASIDE_BUDGET_S = 6.0, 30.0
  #: It rests standing (`resting`), as its record of a step aside says
  #: (`last_aside`'s `lay`).
  RESTS_LYING = False
  #: No step aside under way, and when it last moved, sim s (`_init_way`).
  _aside = None
  last_motion_t = 0.0

  def _init_way(self) -> None:
    #: The step aside under way: for whom, from where, to where, when it
    #: began and its waypoints (None until it walks); None between.
    self._aside: dict | None = None
    self.making_way: str | None = None
    #: Why the last ask was said no to (`Body.way_refused`), or "".
    self.way_refused = ""
    self.asides = 0
    self.last_aside: dict | None = None
    #: Walks of its own under way (`drive_to_routine`): a body waiting in
    #: one is never asked aside -- its own walk decides where it goes.
    self.driving = 0
    #: When it was last commanded to move, sim s: what `resting` reads.
    self.last_motion_t = 0.0
    #: The station's catches its port is held by (`activity.station`).
    root = self.model.body(self.handle.root).name
    self.port_eqs = [e for e in range(self.model.neq)
                     if self.model.equality(e).name.endswith(f"_port_{root}")]

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

  def plugged_in(self) -> bool:
    """A port catch holds its plug: its pack takes the charger's current,
    which the BMS reads (`Body.charging`). ⚠ Asked aside then, it walked
    off with the plug in its port and pulled it out (#559's stage 4, its
    pair's recording) -- read off the plug, never the lifecycle's
    `Body.docked`, which a restart forgets with the robot still plugged in."""
    return any(self.data.eq_active[e] for e in self.port_eqs)

  def _hand_out(self) -> bool:
    return any(p != "rest" for p in self.reach.phase.values())

  @property
  def resting(self) -> bool:
    """Resting STANDING: no motion command for `REST_S`, no walk of its own,
    no hand out and on its feet."""
    return (self.legs != "getup" and not self.driving and self._aside is None
            and not self._hand_out()
            and float(self.data.time) - self.last_motion_t >= REST_S)

  def _way_refusal(self) -> str:
    return ("charging" if self.plugged_in()
            else "getting up" if self.legs == "getup"
            else "working" if self._hand_out()
            else "walking" if self.driving
            else "standing" if not self.resting else "")

  def _aside_command(self, command, t: float):
    """One physics step of making way, in place of the routine's command
    (the module docstring): what the body does first (`_aside_ready`: the
    routine's own walk takes over, a fall or a hand out ends it), then the
    drive's law along a plan to the spot, and STILL once it is there."""
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
    """The routine's own command once it moves, the end of the step aside
    on a fall or a hand out, else None: it walks from where it stands."""
    if any(command):
      return self._end_aside("its own walk", command)
    if self.legs == "getup":
      return self._end_aside("fell", command)
    if self._hand_out():
      return self._end_aside("arm out", STILL)
    return None

  def _aside_blocked(self, t: float):
    """⚠ THE FRONT STOP'S REFLEX, as a walk's (`Navigator.drive_to_routine`):
    straight back, then a new plan. Ended on it, a humanoid resting in a
    doorway -- its stop re-armed by every scan -- never stepped aside at
    all, 3 doors of 6 (`make_way_spike.py`)."""
    if t < self.backoff_until:
      from pluggybot.humanoid.mission import command_for
      self._aside["waypoints"] = None
      return command_for(-self.BACKOFF_V, 0.0, 0.0)
    return None

  def _aside_twist(self, v: float, w: float):
    from pluggybot.humanoid.mission import command_for
    return command_for(v, 0.0, w)

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
