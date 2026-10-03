"""Surveying an area and counting what stands in it, on legs (issue #407;
the census, #13): the walk round an area the robot found by its tag, the
depth camera's low layer filling in as it goes. A mixin: `QuadMission` is
the rest of the body; the counting is `economy/census.py`'s.

THE AREA is the floor the LIDAR's walls and fence enclose round the tag's
front, a doorway closed (`census.area_from_walls`), re-found after every
look: it grows as the map does. The count is the low layer's objects inside
it (`census.count_low`), the coverage the share of its floor the depth
camera has seen (`census.seen_share`) -- the camera's layer reaches 1.8 m
(`legs.body.LOW_RANGE_M`), so the survey walks: to the vantage nearest it
on a lattice over the area whose reach still holds floor it has not seen,
and looks round there, until `SURVEY_COVERED` of the floor is seen, no
vantage has any left, or its patience runs out.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.tick import Routine

#: The survey stops once this share of the area's floor has been seen.
SURVEY_COVERED = 0.90
#: Where it starts: this far out of the area's tag's face, m.
SEED_OUT_M = 1.0
#: The vantages: a lattice this far apart over the area, m, each at least
#: `VANTAGE_CLEAR_M` off its walls; a vantage still has floor to see where
#: an unseen cell is within `VANTAGE_SEES_M` (the low layer's reach, less
#: what the camera's view misses at the feet).
VANTAGE_STEP_M = 1.5
VANTAGE_CLEAR_M = 0.5
VANTAGE_SEES_M = 1.6
#: A walk to a vantage gives up after this long, s.
VANTAGE_PATIENCE_S = 60.0
#: An object this many cells from a LIDAR return is the LIDAR's: what its
#: plane hit there stands taller than a plant (`survey_count`).
LIDAR_NEAR_CELLS = 2
#: ...a return this near, m. Further out the plane is no height: level is
#: 1.5 deg (`QuadMission.LEVEL_TILT`), and MEASURED, a walking torso's pitch
#: put returns 7.4-8.0 m off onto two of the garden's four 0.30 m plants,
#: which the count then dropped. At 3 m even 3 deg leaves the plane 0.35 m up.
LIDAR_TALL_M = 3.0


class CensusLayer:
  """What one survey's depth frames saw, by cell of the map: how many found
  something standing in a cell (`legs.body.LOW_MIN_Z`..`LOW_MAX_Z` up, off
  the LIDAR's walls), and whether any saw the cell at all -- never taken
  back, as the planner's layer takes a stalk's cell back from the floor
  seen round it (`economy.census`). Fed by `QuadMission._fold_low` while
  `QuadMission.census` is one; a survey's alone, never kept."""

  def __init__(self, shape) -> None:
    self.hits = np.zeros(shape, dtype=np.int32)
    self.seen = np.zeros(shape, dtype=bool)
    #: ...and where the LIDAR's returns landed: what its plane, 0.51 m up,
    #: hit is taller than that, never a plant (`QuadMission._on_scan`)
    self.lidar = np.zeros(shape, dtype=bool)

  def fold(self, hit_cells, floor_cells) -> None:
    self.hits.flat[hit_cells] += 1
    self.seen.flat[hit_cells] = True
    self.seen.flat[floor_cells] = True


class AreaSurvey:
  """The census's walk (the module docstring), on `QuadMission`."""

  def _init_survey(self) -> None:
    #: The last survey's record.
    self.last_survey: dict | None = None

  def survey_area(self, seed: tuple[float, float]) -> np.ndarray:
    """The area round `seed` (the map, m) as the LIDAR's walls enclose it
    now: a mask over its grid."""
    from pluggybot.economy.census import area_from_walls
    from pluggybot.mapping.frontier import FREE_THRESH, OCC_THRESH
    g = self.grid
    return area_from_walls(g.grid > OCC_THRESH, g.grid < FREE_THRESH,
                           g.world_to_cell(*seed), g.resolution)

  def survey_count(self, area: np.ndarray, layer: CensusLayer) -> tuple[list, float]:
    """What this survey saw standing inside the area, under the LIDAR's
    plane -- each object's middle in the map -- and the share of its floor
    seen. ⚠ AN OBJECT THE LIDAR HIT IS NOT COUNTED: it is taller than the
    plane. The garden light's 40 mm pole, which rays slipping past it clear
    from the fused map, was counted as a fifth plant."""
    from scipy import ndimage
    from pluggybot.economy.census import HITS, count_low, seen_share
    g = self.grid
    tall = ndimage.binary_dilation(layer.lidar, iterations=LIDAR_NEAR_CELLS)
    objs = [g.cell_to_world(ix, iy) + (n,)
            for ix, iy, n in count_low((layer.hits >= HITS) & ~tall, area, g.resolution)]
    return objs, seen_share(layer.seen, area)

  def _on_scan(self, angles, ranges) -> None:
    """Where a level scan's returns land, into a survey's layer while one is
    under way (`Navigator._on_scan`)."""
    if self.census is None:
      return
    hit = ranges < min(LIDAR_TALL_M, self.lidar.max_range - 1e-6)
    if not hit.any():
      return
    x, y, th = self.pose
    ox, oy = self.LIDAR_ORIGIN
    c, s = np.cos(th), np.sin(th)
    sx, sy = x + c * ox - s * oy, y + s * ox + c * oy
    a = th + angles[hit]
    g = self.grid
    ix = np.floor((sx + ranges[hit] * np.cos(a) - g.x_min) / g.resolution).astype(np.int64)
    iy = np.floor((sy + ranges[hit] * np.sin(a) - g.y_min) / g.resolution).astype(np.int64)
    rows, cols = self.census.lidar.shape
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    self.census.lidar[iy[ok], ix[ok]] = True

  def _next_vantage(self, area: np.ndarray, visited: list,
                    layer: CensusLayer) -> tuple[float, float] | None:
    """The lattice point over the area nearest the robot, clear of its
    walls and of the vantages already stood at, whose reach still holds
    floor the camera has not seen -- or None."""
    from scipy import ndimage
    g = self.grid
    res = g.resolution
    clear = ndimage.distance_transform_edt(area) * res
    step = max(1, int(round(VANTAGE_STEP_M / res)))
    ys, xs = np.nonzero(clear >= VANTAGE_CLEAR_M)
    keep = (ys % step == 0) & (xs % step == 0)
    unseen = area & ~layer.seen
    if not unseen.any():
      return None
    near = ndimage.distance_transform_edt(~unseen) * res
    px, py, _ = self.pose
    best = None
    for iy, ix in zip(ys[keep], xs[keep]):
      if near[iy, ix] > VANTAGE_SEES_M:
        continue
      wx, wy = g.cell_to_world(int(ix), int(iy))
      if any(math.hypot(wx - vx, wy - vy) < VANTAGE_STEP_M * 0.75 for vx, vy in visited):
        continue
      d = math.hypot(wx - px, wy - py)
      if best is None or d < best[0]:
        best = (d, (wx, wy))
    return None if best is None else best[1]

  def survey_routine(self, tag: int, patience: float, stop=None, on_count=None) -> Routine:
    """Survey the area tag `tag` marks and count what stands in it (the
    module docstring): `on_count(n)` hears the count after every look. Its
    record: `surveyed`, `why` ("surveyed", "seen all it could", "not
    found", "out of time", "interrupted"), `count`, `coverage`, the
    objects' places, the vantages stood at."""
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec: dict = {"tag": int(tag), "vantages": 0}
    self.last_survey = rec

    def done(why: str) -> dict:
      rec.update(surveyed=why in ("surveyed", "seen all it could"), why=why,
                 seconds=round(float(self.data.time) - t0, 1))
      return rec

    p = self.places.get(int(tag))
    if p is None:
      return done("not found")
    facing, _ = self.places.facing(int(tag))
    seed = (p.x + SEED_OUT_M * math.cos(facing), p.y + SEED_OUT_M * math.sin(facing))
    yield from self.stand_routine()
    layer = self.census = CensusLayer(self.low.shape)
    try:
      return (yield from self._survey_walk_routine(seed, layer, rec, until, stop,
                                                   on_count, done))
    finally:
      self.census = None

  def _survey_walk_routine(self, seed, layer, rec, until, stop, on_count, done) -> Routine:
    visited: list = []
    goal = seed
    while True:
      left = until - float(self.data.time)
      if stop is not None and stop():
        return done("interrupted")
      if left <= 0.0:
        return done("out of time")
      kw = {"stop": stop} if stop is not None else {}
      yield from self.drive_to_routine(*goal, timeout=min(VANTAGE_PATIENCE_S, left), **kw)
      yield from self._spin_routine()
      visited.append(goal)
      rec["vantages"] = len(visited)
      area = self.survey_area(seed)
      objs, cover = self.survey_count(area, layer)
      rec.update(count=len(objs), coverage=round(cover, 3),
                 objects=[[round(x, 2), round(y, 2)] for x, y, _ in objs],
                 areaM2=round(float(area.sum()) * self.grid.resolution ** 2, 1))
      if on_count is not None:
        on_count(len(objs))
      if cover >= SURVEY_COVERED:
        return done("surveyed")
      goal = self._next_vantage(area, visited, layer)
      if goal is None:
        return done("seen all it could")
