"""Hiding and seeking, on legs (issue #404): the two roles of hide and
seek as walks over the robot's own map, the game's referee
(`activity/hideseek.py`) watching. A mixin: `QuadMission` is the rest of
the body.

THE HIDER (`hide_routine`) picks its own spot, from its own map -- no
surveyed spot and no coordinate handed over (#419): floor it has SEEN, a
walk it can make before the seeking starts (`reach_m`), `HIDE_CLEAR_M`
from anything in its way (never a doorway, where a robot resting cuts the
other's way, #415), `KEEP_CLEAR_M` from the dock and the tool rack (the
other robot's, too: both are commissioned, and the robot knows where),
farther from where the seeker counts than a find, and out of the seeker's
sight from there wherever the map allows; of those, the one the seeker
would have the LONGEST WALK to over the same map, and of those, the
hider's own shortest. Where the seeker counts is what it says on the
network (`HubLifecycle.reported_xy`), as a real seeker would tell a real
hider.

THE SEEKER (`seek_routine`) is never told where the hider is: it searches
its own map outward from where it counted -- first the floor out of its
own sight from there, where a hider hides, then the rest -- ring by ring
of `SWEEP_RING_M` by its own walk, to the nearest viewpoint it has not
seen round yet (a lattice `SWEEP_STEP_M` apart over floor it has seen),
and each walk ends the moment that viewpoint has been within `cover_m` of
it in plain sight. The referee, not the seeker, decides a find.

SIGHT, both ways, is the map's (`_seen_from`): a straight line from a
point meeting nothing the planner plans round -- the LIDAR's walls and what
the depth camera laid under its plane. A couch hides a robot lying behind
it from an eye 0.51 m up, and the referee's rays agree.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.mapping import optimistic
from pluggybot.mapping.frontier import OCC_THRESH
from pluggybot.navigator import DRIVE_STOPPED

#: A hiding spot is this far from anything in the way, m: a 1 m door
#: leaves 0.5 m either side of its middle, and a robot resting there cuts
#: the seeker's way (#415) -- the depth camera's furniture counts.
HIDE_CLEAR_M = 0.6
#: ...and this far from the dock's seat and its standoff, and from the
#: tool rack, m: a robot resting there is in the way of the other's charge
#: or tool, and stood up to make way (#415) it is given away.
KEEP_CLEAR_M = 1.5
#: How far the seeker is reckoned to see when the hider picks a spot out of
#: its sight, m: past the LIDAR's 8 m, nothing in the house is far enough.
SIGHT_M = 20.0
#: The straight lines a sight is made of: 0.5 deg apart, 9 cm at 10 m,
#: under the lattice's 10 cm.
SIGHT_RAYS = 720
#: The seeker's viewpoints: a lattice this far apart over the floor it has
#: seen, m -- its sight covers 1.2 m either side of its walk, so 2 m leaves
#: no strip unseen, and MEASURED at 1 m it stopped and turned every 4 s
#: (28 viewpoints in 120 s) where at 2 m it walks on -- taken in rings this
#: wide by its walk from where it counted, the nearest first; and a walk to
#: one gives up after this long, s.
SWEEP_STEP_M = 2.0
SWEEP_RING_M = 2.0
SWEEP_LEG_S = 45.0


class GameWalk:
  """Hiding and seeking (the module docstring), on `QuadMission`."""

  def _init_game(self) -> None:
    #: The last hide's and the last search's records.
    self.last_hide: dict | None = None
    self.last_seek: dict | None = None

  def _sight_walls(self) -> np.ndarray:
    """What a line of sight is blocked by, on the map's cells: everything
    the planner plans round (the module docstring)."""
    return self._planning_grid() > OCC_THRESH

  def _seen_from(self, frm, radius: float, walls: np.ndarray) -> np.ndarray:
    """The planner lattice's cells `frm` sees within `radius`: `SIGHT_RAYS`
    straight lines over `walls`, each stopped by the first wall it meets
    (the map's edge is one)."""
    g, b = self.grid, optimistic.BLOCK
    rows, cols = walls.shape
    seen = np.zeros((-(-rows // b), -(-cols // b)), dtype=bool)
    a = np.linspace(0.0, 2.0 * math.pi, SIGHT_RAYS, endpoint=False)
    r = np.arange(0.0, float(radius), 0.5 * g.resolution)
    ix = np.floor((frm[0] + np.cos(a)[:, None] * r - g.x_min) / g.resolution).astype(np.int64)
    iy = np.floor((frm[1] + np.sin(a)[:, None] * r - g.y_min) / g.resolution).astype(np.int64)
    inside = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    hit = ~inside
    hit[inside] = walls[iy[inside], ix[inside]]
    clear = np.cumsum(hit, axis=1) == 0
    seen[iy[clear] // b, ix[clear] // b] = True
    return seen

  def _facilities(self) -> list[tuple[float, float]]:
    """Where the commissioned facilities are, the other robot's as much as
    this one's: the dock's seat and its standoff, and the tool rack."""
    out = []
    if self.dock_prior is not None:
      out += [tuple(self.dock_prior[:2]), tuple(self.charge_standoff()[:2])]
    if getattr(self, "tool_rack_prior", None) is not None:
      out.append(tuple(self.tool_rack_prior[:2]))
    return [(float(x), float(y)) for x, y in out]

  def _lattice_xy(self, shape) -> tuple[np.ndarray, np.ndarray]:
    """The planner lattice's cell centres in the map, (x, y) arrays."""
    g = self.grid
    side = g.resolution * optimistic.BLOCK
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return g.x_min + (xs + 0.5) * side, g.y_min + (ys + 0.5) * side

  # ---- hiding -------------------------------------------------------------------

  def hiding_spot(self, away_from, reach_m: float, clear_of_m: float) -> dict | None:
    """Where to hide from a robot counting at `away_from` (the module
    docstring): `at`, and whether it is out of that robot's sight
    (`hidden`), its walk and the seeker's (m); None where it can walk
    nowhere that will do."""
    from scipy.ndimage import distance_transform_edt
    g = self.grid
    mine = self.walk_field(self.seen_floor_lattice(mask_others=True), self.pose_xy(),
                           limit=reach_m)
    theirs = self.walk_field(self.seen_floor_lattice(), away_from)
    walls = self._sight_walls()
    clear = distance_transform_edt(~walls) * g.resolution
    # each lattice cell's nearest obstacle: the least of its map cells'
    clear = -optimistic.coarsen(-clear, optimistic.BLOCK)
    cx, cy = self._lattice_xy(mine.shape)
    far = np.hypot(cx - away_from[0], cy - away_from[1]) > clear_of_m
    for fx, fy in self._facilities():
      far &= np.hypot(cx - fx, cy - fy) > KEEP_CLEAR_M
    ok = np.isfinite(mine) & np.isfinite(theirs) & (clear >= HIDE_CLEAR_M) & far
    if not ok.any():
      return None
    hidden = ok & ~self._seen_from(away_from, SIGHT_M, walls)
    pool = hidden if hidden.any() else ok
    # ...as long a walk is within a lattice cell of it: the lattice's own
    # steps part two walks the same length by a cell's worth
    best = theirs[pool].max()
    near = pool & (theirs >= best - g.resolution * optimistic.BLOCK)
    k = int(np.argmin(np.where(near, mine, np.inf)))
    ky, kx = divmod(k, mine.shape[1])
    return {"at": (round(float(cx[ky, kx]), 3), round(float(cy[ky, kx]), 3)),
            "hidden": bool(hidden.any()),
            "walkM": round(float(mine[ky, kx]), 2),
            "seekerWalkM": round(float(theirs[ky, kx]), 2)}

  def hide_routine(self, away_from, reach_m: float, clear_of_m: float,
                   patience: float, stop=None):
    """`Body.hide_routine`: a spot (`hiding_spot`), and the walk there.
    Its record: `hid` (arrived), `why` ("hid", "nowhere" to go, "stopped"
    by `stop`, or why the walk gave up), `seconds`, and the spot's own."""
    t0 = float(self.data.time)
    rec: dict = {"from": [round(float(away_from[0]), 2), round(float(away_from[1]), 2)],
                 "reachM": round(float(reach_m), 2)}
    self.last_hide = rec

    def done(why: str, hid: bool = False) -> dict:
      rec.update(hid=hid, why=why, seconds=round(float(self.data.time) - t0, 1))
      return rec

    yield from self.stand_routine()
    spot = self.hiding_spot(away_from, reach_m, clear_of_m)
    if spot is None:
      return done("nowhere")
    rec.update(spot)
    tx, ty = spot["at"]
    left = max(0.0, float(patience) - (float(self.data.time) - t0))
    arrived = yield from self.drive_to_routine(tx, ty, timeout=left,
                                               **({"stop": stop} if stop is not None else {}))
    why = (self.last_drive or {}).get("why", "")
    if why == DRIVE_STOPPED:
      return done("stopped")
    return done("hid" if arrived else why or "not there", hid=bool(arrived))

  # ---- seeking ------------------------------------------------------------------

  def seek_routine(self, base, cover_m: float, patience: float, stop=None):
    """`Body.seek_routine` (the module docstring). Its record: `why`
    ("stopped" by `stop`, "out of time", or "searched" -- every viewpoint
    it could reach seen round), `seconds`, how many viewpoints it set out
    for (`targets`) and gave up on (`gaveUp`)."""
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec: dict = {"base": [round(float(base[0]), 2), round(float(base[1]), 2)],
                 "coverM": round(float(cover_m), 2), "targets": 0, "gaveUp": 0}
    self.last_seek = rec

    def done(why: str) -> dict:
      rec.update(why=why, seconds=round(float(self.data.time) - t0, 1))
      return rec

    def halted() -> bool:
      return stop is not None and bool(stop())

    yield from self.stand_routine()
    covered = self._seen_from(self.pose_xy(), cover_m, self._sight_walls())
    # ...where a hider hides: out of its own sight from where it counted
    in_view = self._seen_from(base, SIGHT_M, self._sight_walls())

    def cover() -> None:
      covered[...] |= self._seen_from(self.pose_xy(), cover_m, self._sight_walls())

    step = max(1, int(round(SWEEP_STEP_M / (self.grid.resolution * optimistic.BLOCK))))
    while True:
      if halted():
        return done("stopped")
      if float(self.data.time) >= until:
        return done("out of time")
      floor = self.seen_floor_lattice()
      walk = self.walk_field(floor, base)
      views = np.zeros(floor.shape, dtype=bool)
      views[::step, ::step] = True
      todo = views & np.isfinite(walk) & ~covered
      if not todo.any():
        return done("searched")
      if (todo & ~in_view).any():
        todo &= ~in_view
      # the nearest ring of the search, by its walk from where it counted,
      # and in it the viewpoint nearest where it stands
      ring = todo & (walk < (np.floor(walk[todo].min() / SWEEP_RING_M) + 1) * SWEEP_RING_M)
      cx, cy = self._lattice_xy(floor.shape)
      x, y = self.pose_xy()
      k = int(np.argmin(np.where(ring, np.hypot(cx - x, cy - y), np.inf)))
      ky, kx = divmod(k, floor.shape[1])
      tx, ty = float(cx[ky, kx]), float(cy[ky, kx])
      rec["targets"] += 1

      def leg_stop() -> bool:
        cover()
        return halted() or bool(covered[ky, kx])

      left = max(0.0, until - float(self.data.time))
      yield from self.drive_to_routine(tx, ty, timeout=min(left, SWEEP_LEG_S),
                                       stop=leg_stop)
      cover()
      if not covered[ky, kx]:
        covered[ky, kx] = True              # given up on: on to the next
        rec["gaveUp"] += 1
