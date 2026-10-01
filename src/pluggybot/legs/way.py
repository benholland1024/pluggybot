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
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.mapping import optimistic
from pluggybot.mapping.astar import nearest_traversable
from pluggybot.mapping.frontier import FREE_THRESH
from pluggybot.navigator import ARRIVAL_SLOW_RADIUS, CLOSE_ENOUGH_M, WAYPOINT_REACHED_M


class MakeWay:
  """Making way (the module docstring), on `QuadMission`."""

  #: The farthest it walks to step aside, m, and the longest it takes, s:
  #: MEASURED over six doorways of the house (`scripts/make_way_spike.py`),
  #: the stand-up (2.2 s) and a walk of 0.82-0.88 m took 6.7-8.9 s.
  ASIDE_WALK_M, ASIDE_BUDGET_S = 6.0, 30.0

  def _init_way(self) -> None:
    #: The step aside under way: for whom, from where, to where, when it
    #: began and its waypoints (None until it stands); None between.
    self._aside: dict | None = None
    self.making_way: str | None = None
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
    """`Body.make_way`."""
    if self._aside is not None:
      return True
    from pluggybot.legs.body import LYING
    if (self.posture != LYING or self.docked or self.working or self.driving
        or self._move is not None):
      return False
    spot = self.aside_spot(route)
    if spot is None:
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
    b = optimistic.BLOCK
    side = g.resolution * b
    planning = self._planning_grid()
    cost = optimistic.map_costs(planning, self.INFLATION_CELLS, self.UNKNOWN_COST)
    cost[planning >= FREE_THRESH] = np.inf        # floor it has SEEN, only
    floor = np.isfinite(cost)
    self._mask_others(floor)
    cost[~floor] = np.inf
    lat = optimistic.coarsen(cost, b)
    rows, cols = lat.shape
    x, y = self.pose_xy()
    cx, cy = g.world_to_cell(x, y)
    src = nearest_traversable(np.isfinite(lat), (cx // b, cy // b),
                              radius=optimistic.ESCAPE_CELLS)
    if src is None:
      return None
    graph = optimistic.lattice_for(lat.shape, side)
    graph.weigh(lat)
    walk, _ = graph.search(src[1] * cols + src[0], limit=self.ASIDE_WALK_M)
    walk = walk.reshape(lat.shape)
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

  def _aside_command(self, command, t: float):
    """One physics step of making way, in place of the routine's command
    (the module docstring): the routine's own once it moves -- its walk
    takes over -- else a motion command while it lies (the posture machine
    stands it first), STILL while it stands up, then the drive's law along
    a plan to the spot, and STILL once it is there."""
    from pluggybot.legs.body import (GETTING_UP, LYING, STANDING, STILL, W_CARRY, W_MIN,
                                     command_for, is_motion)
    a = self._aside
    if is_motion(command):
      return self._end_aside("its own walk", command)
    if self.posture == GETTING_UP:
      return self._end_aside("fell", command)
    if self.posture == LYING:
      return (0.0, 0.0, W_MIN)                    # stands it, by the reflex's own rule
    if self.posture != STANDING:
      return STILL
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
    if t < self.backoff_until or self.pressing:
      return self._end_aside("met something", STILL)
    if not self.arm_driving():
      return self._end_aside("arm out", STILL)
    if len(wps) > 1:
      v, w = drive_toward(self.pose, wps[0])
    else:
      v, w = drive_toward(self.pose, wps[0] if wps else (tx, ty),
                          slow_radius=ARRIVAL_SLOW_RADIUS)
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
                       "why": why}
    return command
