"""What limits how big the world may get (issue #67): guardrails that change
no behaviour, so a growing world fails loudly instead of quietly.

A MuJoCo plane's `size` is used for RENDERING ONLY -- collision is with the
infinite half-space, so nothing can walk off the world. What the first guard
protects is what the WEBSITE DRAWS: `scene_dict` ships the plane's size and
the browser renders a floor of exactly that; a body outside it is drawn
floating over nothing.

The three budgets, and they fail in different directions:

  1. GEOMETRY vs the drawn floor. Cosmetic.
  2. THE GRID vs the world. The map must cover what the robot must map.
  3. THE GRID vs itself. A cell ceiling, so an accidentally 10x world is
     loud. Denominated in the half of the mapping stack that actually scales.
"""

import functools
import math

import mujoco
import numpy as np
import pytest

from pluggybot.home import world as home
from pluggybot.legs import world as lw
from pluggybot.lifecycle import QUAD_HOME, world_config
from pluggybot.mapping.frontier import traversable_mask
from pluggybot.mapping.occupancy_grid import MAX_CELLS, OccupancyGrid
from pluggybot.telemetry.protocol import robot_body_ids

#: Tightest margin between any non-robot geom and the DRAWN floor's edge,
#: m, the served world's (the house, its dock, its rack; measured at the
#: loop's east fence). A RATCHET: the assertion is that nothing gets worse.
KNOWN_MARGIN_M = 0.980

#: Slack for float noise only. Anything bigger is a real move.
RATCHET_TOL_M = 0.002

_GEOM = mujoco.mjtGeom


@functools.cache
def _model():
  """The served world compiled: the house with the quadruped, its dock and
  its rack (`legs.world.home_spec`)."""
  return lw.home_spec().compile()


def _local_half(model, g: int) -> np.ndarray | None:
  """A geom's half-extents in its OWN frame, as a box that contains it.

  Per type, because `geom_size` means something different for each and
  `geom_rbound` (the bounding SPHERE) is uselessly loose for the long thin
  slabs this world is made of -- it reported home overhanging by 2.3 m, which
  is the radius of a sphere around an 8 m wall and not a fact about the floor.
  """
  t, s = int(model.geom_type[g]), model.geom_size[g]
  if t in (_GEOM.mjGEOM_BOX, _GEOM.mjGEOM_ELLIPSOID):
    return np.array(s, dtype=float)
  if t == _GEOM.mjGEOM_SPHERE:
    return np.array([s[0]] * 3, dtype=float)
  if t == _GEOM.mjGEOM_CYLINDER:
    return np.array([s[0], s[0], s[1]], dtype=float)
  if t == _GEOM.mjGEOM_CAPSULE:
    return np.array([s[0], s[0], s[1] + s[0]], dtype=float)
  return None                       # planes, meshes, heightfields: not bounded


def _floor_margin() -> tuple[float, str, str]:
  """Tightest (margin, geom, body) against the drawn floor's half-extents.

  WORLD-FRAME, which is the whole point: a rotated slab's axis-aligned box is
  bigger than its own dimensions, so checking `geom_size` would pass a geom
  that genuinely overhangs. `|R| @ half` is the standard oriented-box -> AABB
  extent. The plane's size is read off the COMPILED model, never a literal:
  the generator grows it with the layout.
  """
  model = _model()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)                 # world poses at qpos0
  robot = robot_body_ids(model)

  planes = [g for g in range(model.ngeom)
            if int(model.geom_type[g]) == _GEOM.mjGEOM_PLANE]
  assert len(planes) == 1, f"{len(planes)} planes, expected one"
  # The plane's WORLD CENTRE, not the origin: the property is not centred
  # on (0, 0).
  pc = data.geom_xpos[planes[0]]
  psize = model.geom_size[planes[0]]
  hx, hy = float(psize[0]), float(psize[1])
  fx0, fx1 = float(pc[0]) - hx, float(pc[0]) + hx
  fy0, fy1 = float(pc[1]) - hy, float(pc[1]) + hy

  worst: tuple[float, str, str] | None = None
  for g in range(model.ngeom):
    if int(model.geom_bodyid[g]) in robot or g in planes:
      continue
    half = _local_half(model, g)
    if half is None:
      continue
    rot = data.geom_xmat[g].reshape(3, 3)
    ext = np.abs(rot) @ half
    lo, hi = (data.geom_xpos[g] - ext)[:2], (data.geom_xpos[g] + ext)[:2]
    margin = min(lo[0] - fx0, fx1 - hi[0], lo[1] - fy0, fy1 - hi[1])
    if worst is None or margin < worst[0]:
      worst = (float(margin), model.geom(g).name or f"geom{g}",
               model.body(int(model.geom_bodyid[g])).name)
  assert worst is not None, "no boundable geometry"
  return worst


# ---- 1. geometry vs the floor the browser draws ------------------------------


def test_nothing_hangs_off_the_ground_plane():
  """Every non-robot geom's world-frame AABB is inside the drawn floor, the
  floor being GENERATED from the layout (#68) so it grows with the plot.

  Shown to fail by shrinking the plane to `size="4 4 0.1"`: the failure
  names the offending body and the measured margin.
  """
  margin, geom, body = _floor_margin()
  assert margin >= 0.0, (
    f"{geom} (body {body}) hangs {-margin * 1000:.0f} mm past the "
    f"floor the scene draws. The robot cannot fall -- a plane's size is "
    f"rendering only -- but the website draws this body over nothing.")


def test_the_floor_margin_does_not_get_worse():
  """A ratchet over the measured margin: growing the world grows the plane
  to match, and this is that not happening."""
  margin, geom, body = _floor_margin()
  assert margin >= KNOWN_MARGIN_M - RATCHET_TOL_M, (
    f"the tightest floor margin moved from {KNOWN_MARGIN_M:+.3f} m to "
    f"{margin:+.3f} m at {geom} (body {body}).")


# ---- 2. the grid vs the world ------------------------------------------------


def test_the_grid_covers_everything_the_robot_must_map():
  """The map has to reach whatever the robot is asked to drive to.

  Checked against the ZONES rather than the geometry, deliberately: a wall's
  outer face may sit outside the grid with no consequence, but a zone is
  somewhere the robot is sent.
  """
  gx0, gy0, gx1, gy1 = home.GRID_BOUNDS
  for zone in home.ZONES:
    (zx0, zy0), (zx1, zy1) = zone["min"], zone["max"]
    assert gx0 <= zx0 and zx1 <= gx1 and gy0 <= zy0 and zy1 <= gy1, \
        f"zone {zone['name']} {zone['min']}..{zone['max']} is outside " \
        f"GRID_BOUNDS {home.GRID_BOUNDS}"


def test_the_grid_covers_every_bit_of_floor_the_visitor_can_see():
  """The map reaches at least as far as the ground the browser draws:
  nothing a visitor watches the robot stand on is off the edge of its map.
  The safety invariant that makes any mismatch harmless is asserted below.
  """
  model = _model()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  plane = next(g for g in range(model.ngeom)
               if int(model.geom_type[g]) == _GEOM.mjGEOM_PLANE)
  centre, size = data.geom_xpos[plane], model.geom_size[plane]
  gx0, gy0, gx1, gy1 = home.GRID_BOUNDS
  assert gx0 <= centre[0] - size[0] and centre[0] + size[0] <= gx1, \
      f"the drawn floor reaches outside GRID_BOUNDS in x: {home.GRID_BOUNDS}"
  assert gy0 <= centre[1] - size[1] and centre[1] + size[1] <= gy1, \
      f"the drawn floor reaches outside GRID_BOUNDS in y: {home.GRID_BOUNDS}"


def test_unknown_space_is_never_driveable():
  """The invariant that makes a grid/floor mismatch harmless in EITHER
  direction, and the reason #67 could leave one standing.

  A cell over nothing can never become known-free (there is no surface to
  reflect a ray), unknown space is never traversable, and a frontier is a
  known-FREE cell bordering unknown -- so the planner cannot route into such a
  strip and the explorer cannot target it.

  Shown to fail by making `traversable_mask` return `~inflated` instead of
  `free & ~inflated`: an all-unknown grid becomes wall-to-wall driveable.
  """
  unknown = np.zeros((40, 40))          # log-odds 0.0 == "no opinion"
  assert not traversable_mask(unknown).any(), \
      "unknown space became driveable: a grid that reaches past the floor is " \
      "no longer safe, and home.GRID_BOUNDS' comment is now wrong"


# ---- 3. the grid vs itself ---------------------------------------------------


def test_the_grid_stays_inside_its_cell_budget():
  """A tripwire, not a cliff. See `occupancy_grid.MAX_CELLS` for the measured
  table this is denominated in -- `binary_dilation` over every cell is the
  half of the mapping stack that scales, and `update()` is nearly flat."""
  x0, y0, x1, y1 = world_config(QUAD_HOME)["grid_bounds"]
  grid = OccupancyGrid(x_min=x0, y_min=y0, x_max=x1, y_max=y1, resolution=0.05)
  cells = int(grid.grid.size)
  assert cells <= MAX_CELLS, (
    f"{cells:,} cells against a {MAX_CELLS:,} ceiling. Re-measure "
    f"before raising it -- the numbers are at the constant.")


# ---- 4. the reserve's worst case ---------------------------------------------

#: The LIDAR's plane, the quadruped's (`legs/body.py`): a static box whose
#: z-extent crosses it is a wall the robot maps and plans around.
BEAM_Z = 0.51
#: The routing raster's cell, and the one-cell halo that stands in for the
#: robot's half-width at this resolution. Coarse on purpose: this is a
#: question about which POINT is farthest, not about a route's millimetres.
ROUTE_CELL_M = 0.25
#: How far a zone's corner is pulled inward before it is asked about, the
#: way `HOME_WORST_RETURN` sits 0.4 m inside its corner: a robot cannot
#: stand in a fence.
CORNER_INSET_M = 0.4


def _route_lengths(model, goal: tuple[float, float]) -> tuple:
  """Driving distance from `goal` to every cell of a coarse raster of the
  COMPILED world (Dijkstra, 8-connected), plus a `(x, y) -> metres` reader.

  Walls are every static geom whose z-extent crosses the beam -- read off
  the model the way `test_dressing.py` reads decor, never off the zone
  list, because a doorway is a fact about the walls and not about the
  rectangles either side of it. Dynamic bodies (the tools on the rack,
  the blocks, the masses) are skipped: they are small, and they move.
  """
  import heapq
  from scipy.ndimage import binary_dilation
  from pluggybot.telemetry.protocol import dynamic_flags

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  x0, y0, x1, y1 = home.GRID_BOUNDS
  cols, rows = int(round((x1 - x0) / ROUTE_CELL_M)), int(round((y1 - y0) / ROUTE_CELL_M))
  blocked = np.zeros((rows, cols), dtype=bool)
  robots = robot_body_ids(model)
  dyn = dynamic_flags(model)
  for g in range(model.ngeom):
    b = int(model.geom_bodyid[g])
    if b in robots or dyn[b] or int(model.geom_type[g]) == _GEOM.mjGEOM_PLANE:
      continue
    half = _local_half(model, g)
    if half is None:
      continue
    ext = np.abs(data.geom_xmat[g].reshape(3, 3)) @ half
    lo, hi = data.geom_xpos[g] - ext, data.geom_xpos[g] + ext
    if hi[2] < BEAM_Z or lo[2] > BEAM_Z:
      continue
    ix0, ix1 = int((lo[0] - x0) / ROUTE_CELL_M), int((hi[0] - x0) / ROUTE_CELL_M)
    iy0, iy1 = int((lo[1] - y0) / ROUTE_CELL_M), int((hi[1] - y0) / ROUTE_CELL_M)
    blocked[max(iy0, 0):min(iy1, rows - 1) + 1, max(ix0, 0):min(ix1, cols - 1) + 1] = True
  blocked = binary_dilation(blocked, iterations=1)

  def cell(x, y):
    return int((x - x0) / ROUTE_CELL_M), int((y - y0) / ROUTE_CELL_M)

  dist = np.full((rows, cols), np.inf)
  gx, gy = cell(*goal)
  assert not blocked[gy, gx], "the goal is inside a wall"
  dist[gy, gx] = 0.0
  heap = [(0.0, gx, gy)]
  steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
           (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (-1, -1, 2 ** 0.5)]
  while heap:
    d, ix, iy = heapq.heappop(heap)
    if d > dist[iy, ix]:
      continue
    for dx, dy, w in steps:
      nx, ny = ix + dx, iy + dy
      if 0 <= nx < cols and 0 <= ny < rows and not blocked[ny, nx]:
        nd = d + w * ROUTE_CELL_M
        if nd < dist[ny, nx]:
          dist[ny, nx] = nd
          heapq.heappush(heap, (nd, nx, ny))

  def metres(x, y):
    # The nearest free cell within a metre of the point: a point 0.4 m inside
    # a corner sits in the halo (the robot's width, not a wall), and a zone's
    # centre can fall inside a table -- the robot stands beside it.
    ix, iy = cell(x, y)
    best = np.inf
    for dy in range(-4, 5):
      for dx in range(-4, 5):
        if 0 <= ix + dx < cols and 0 <= iy + dy < rows:
          best = min(best, float(dist[iy + dy, ix + dx]))
    return best

  return dist, metres


def _places_the_robot_can_be_sent():
  """Every zone's centre (an overseer's `explore(zone)`) and its four
  corners pulled `CORNER_INSET_M` inward (where an explore can end up)."""
  for zone in home.ZONES:
    (zx0, zy0), (zx1, zy1) = zone["min"], zone["max"]
    yield zone["name"] + " centre", ((zx0 + zx1) / 2.0, (zy0 + zy1) / 2.0)
    for cx, cy in ((zx0, zy0), (zx0, zy1), (zx1, zy0), (zx1, zy1)):
      sx = 1 if cx == zx0 else -1
      sy = 1 if cy == zy0 else -1
      yield zone["name"] + " corner", (cx + sx * CORNER_INSET_M, cy + sy * CORNER_INSET_M)


def test_the_documented_worst_return_point_is_still_the_worst():
  """`HOME_WORST_RETURN` names the point `legs.world.RESERVE_WH` is
  measured from; a comment saying "the worst case is X" rots the moment
  somebody moves a wall, so the claim is checked against the compiled
  world's own routes to the living room's south wall (`RETURN_END`).

  BY ROUTE, not by straight line (issue #215): the loop's east legs are the
  farthest as the crow flies and among the nearest as the robot walks,
  because the living room is reached only through the middle street's gate.

  ⚠ This asserts WHERE the worst case is and that the documented path is
  that route's length, not that the reserve covers it -- that is
  `scripts/energy_spike.py --reserve`'s measurement, at the constant.
  """
  end = home.RETURN_END
  path = home.HOME_WORST_RETURN_PATH
  assert path[0] == home.HOME_WORST_RETURN and path[-1] == end, \
      "HOME_WORST_RETURN_PATH must run from the worst point to RETURN_END"
  routed = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
  assert routed == pytest.approx(home.HOME_WORST_RETURN_M, abs=0.05), \
      f"the routed distance from HOME_WORST_RETURN is now {routed:.2f} m, " \
      f"not the documented {home.HOME_WORST_RETURN_M} m"

  # A robot stands in the living room, off the wall's face.
  goal = (end[0], end[1] + 0.7)
  _, metres = _route_lengths(_model(), goal)
  ranked = sorted(((metres(x, y), where, (x, y))
                   for where, (x, y) in _places_the_robot_can_be_sent()
                   if math.isfinite(metres(x, y))), reverse=True)
  farthest_m, where, point = ranked[0]
  documented_m = metres(*home.HOME_WORST_RETURN)
  assert math.isfinite(documented_m), "HOME_WORST_RETURN is not reachable"
  assert documented_m >= farthest_m - 0.05 * farthest_m, (
    f"{where} at {point[0]:.2f},{point[1]:.2f} routes {farthest_m:.1f} m from "
    f"RETURN_END against HOME_WORST_RETURN's {documented_m:.1f} m. The "
    f"documented worst case is no longer the worst; re-measure the reserve.")
  # ...and the documented waypoints are that route, not a scenic one: the
  # raster's 8-connected path is within a tenth of the waypoint path.
  assert abs(routed - documented_m) <= 0.10 * documented_m, (
    f"HOME_WORST_RETURN_PATH is {routed:.1f} m where the world routes it in "
    f"{documented_m:.1f} m")


def test_every_zone_can_be_reached_from_the_living_room():
  """A zone the router cannot reach is a room the robot can be sent to and
  cannot get home from -- and the reserve's worst case would be silently
  measured over the rest. Every named region routes."""
  end = home.RETURN_END
  _, metres = _route_lengths(_model(), (end[0], end[1] + 0.7))
  unreachable = [where for where, (x, y) in _places_the_robot_can_be_sent()
                 if where.endswith("centre") and not math.isfinite(metres(x, y))]
  assert not unreachable, f"no route from the living room to: {unreachable}"
