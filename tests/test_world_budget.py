"""What limits how big the world may get (issue #67).

M13 grows the house, and the things that bound it were written down nowhere
and asserted by nothing. These are guardrails: they change no behaviour, and
they exist so the expansion's failures are loud instead of quiet.

⚠ THE GROUND PLANE IS NOT THE EDGE OF THE WORLD, and the issue's premise here
is wrong in a way worth knowing before reading the rest. A MuJoCo plane's
`size` is used for RENDERING ONLY -- collision is with the infinite
half-space. Verified rather than assumed: two boxes dropped onto a 1x1 plane,
one at x=0 and one at x=15 (far outside it), both come to rest at z=0.0999.
Nothing can drive off the world and nothing ever could.

So what this file's first guard actually protects is what the WEBSITE DRAWS.
`scene_dict` ships the plane as `size: [20, 20]` and the browser renders a
floor of exactly that; a body outside it is drawn floating over nothing. That
is a real bug on a milestone about dressing the world -- just a cosmetic one,
not a robot falling into the void.

The three budgets, and they fail in different directions:

  1. GEOMETRY vs the drawn floor. Cosmetic, and home already violates it by
     50 mm -- see `test_nothing_hangs_off_the_ground_plane`, which is a
     STRICT xfail for that world so the day #68 grows the plane, the suite
     goes red and somebody deletes the marker.
  2. THE GRID vs the world. The map must cover what the robot must map.
     Exceeding the drawn floor is deliberate; the invariant that makes it
     safe is pinned here because it is the one that could silently stop
     being true.
  3. THE GRID vs itself. A cell ceiling, so an accidentally 10x world is
     loud. Denominated in the half of the mapping stack that actually scales.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.home import world as home
from pluggybot.mapping.frontier import traversable_mask
from pluggybot.mapping.occupancy_grid import MAX_CELLS, OccupancyGrid
from pluggybot.telemetry.protocol import robot_body_ids
from pluggybot.lifecycle import world_config

#: The worlds that share `models/world_fork.xml`, and so share its floor.
#: `world_config`'s keys, not the model stems -- "home" builds home_world.xml.
WORLDS = ("home", "room_hub")

#: Tightest margin between any non-robot geom and the DRAWN floor's edge,
#: measured 2026-09-01. home is NEGATIVE: `garden_lamp_bulb` is a 5 cm sphere
#: centred at exactly x=10.0 on the garden gate post (the plate activity's
#: indicator lamp), so its far half hangs 50 mm past the slab the browser
#: draws. Issue #68 owns growing the plane -- its floor plan runs x -12..14.5,
#: so the plane MUST grow there regardless of this.
#:
#: These are a RATCHET, not a tolerance: the assertion is that nothing gets
#: worse, and the companion xfail below is what stops the exemption outliving
#: its cause.
KNOWN_MARGIN_M = {"home": -0.050, "room_hub": 3.990}

#: Slack for float noise only. Anything bigger is a real move.
RATCHET_TOL_M = 0.002

_GEOM = mujoco.mjtGeom


def _model(world: str):
  return mujoco.MjModel.from_xml_path(world_config(world)["model"])


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


def _floor_margin(world: str) -> tuple[float, str, str]:
  """Tightest (margin, geom, body) against the drawn floor's half-extents.

  WORLD-FRAME, which is the whole point: a rotated slab's axis-aligned box is
  bigger than its own dimensions, so checking `geom_size` would pass a geom
  that genuinely overhangs. `|R| @ half` is the standard oriented-box -> AABB
  extent. The plane's size is read off the COMPILED model, never a literal --
  `world_fork.xml` is included by both worlds and #68 may grow it.
  """
  model = _model(world)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)                 # world poses at qpos0
  robot = robot_body_ids(model)

  planes = [g for g in range(model.ngeom)
            if int(model.geom_type[g]) == _GEOM.mjGEOM_PLANE]
  assert len(planes) == 1, f"{world} has {len(planes)} planes, expected one"
  hx, hy = float(model.geom_size[planes[0]][0]), float(model.geom_size[planes[0]][1])

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
    margin = min(hx - abs(lo[0]), hx - abs(hi[0]),
                 hy - abs(lo[1]), hy - abs(hi[1]))
    if worst is None or margin < worst[0]:
      worst = (float(margin), model.geom(g).name or f"geom{g}",
               model.body(int(model.geom_bodyid[g])).name)
  assert worst is not None, f"{world} has no boundable geometry"
  return worst


# ---- 1. geometry vs the floor the browser draws ------------------------------


@pytest.mark.parametrize("world", [
  "room_hub",
  pytest.param("home", marks=pytest.mark.xfail(
    strict=True,
    reason="known 50 mm overhang: garden_lamp_bulb sits at exactly x=10.0 on "
           "the garden gate post, so half of it hangs past the drawn floor. "
           "Issue #68 grows the plane (its plan runs x -12..14.5); this is "
           "STRICT, so the day that lands this xpasses, the suite goes red, "
           "and whoever fixed it deletes this marker.")),
])
def test_nothing_hangs_off_the_ground_plane(world):
  """Every non-robot geom's world-frame AABB is inside the drawn floor.

  Shown to fail by shrinking `world_fork.xml`'s plane to `size="4 4 0.1"`:
  both worlds report the offending body and the measured margin.
  """
  margin, geom, body = _floor_margin(world)
  assert margin >= 0.0, (
    f"{world}: {geom} (body {body}) hangs {-margin * 1000:.0f} mm past the "
    f"floor the scene draws. The robot cannot fall -- a plane's size is "
    f"rendering only -- but the website draws this body over nothing.")


@pytest.mark.parametrize("world", WORLDS)
def test_the_floor_margin_does_not_get_worse(world):
  """A ratchet over the measured margin, so #68 cannot quietly make it worse.

  This is the half that keeps working while `home_world` is exempt above: the
  xfail says "known bad", and this says "and no worse than this". Without it,
  a new wall at x=20 would be swallowed by the same xfail.
  """
  margin, geom, body = _floor_margin(world)
  floor = KNOWN_MARGIN_M[world] - RATCHET_TOL_M
  assert margin >= floor, (
    f"{world}: the tightest floor margin moved from "
    f"{KNOWN_MARGIN_M[world]:+.3f} m to {margin:+.3f} m at {geom} "
    f"(body {body}). Growing the world past the drawn floor is issue #68's "
    f"job and it grows the plane to match; this is that not happening.")


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


def test_the_grid_may_reach_past_the_floor_because_unknown_is_never_driveable():
  """THE DELIBERATE DISCREPANCY, and the invariant that makes it safe.

  `GRID_BOUNDS` reaches x=11.0 and the drawn floor stops at 10.0, so the grid
  covers a strip with no floor under it. That is fine -- and it is fine by
  DESIGN, not by luck, which is why the design is what gets asserted:

  A cell out there can never become known-free (nothing reflects a ray), and
  unknown space is never traversable, so the planner cannot route into it and
  a frontier -- which is a known-FREE cell bordering unknown -- can never BE
  one. This test pins the second half, because it is the one a future change
  to `traversable_mask` could silently take away.

  Shown to fail by making `traversable_mask` return `~inflated` instead of
  `free & ~inflated`: an all-unknown grid becomes wall-to-wall driveable.
  """
  unknown = np.zeros((40, 40))          # log-odds 0.0 == "no opinion"
  assert not traversable_mask(unknown).any(), \
      "unknown space became driveable: the grid may no longer reach past " \
      "the floor, and home.GRID_BOUNDS' comment is now wrong"

  # ...and the strip really is outside the floor, or this test is about
  # nothing. Read off the compiled model, like the margin check.
  model = _model("home")
  plane = next(g for g in range(model.ngeom)
               if int(model.geom_type[g]) == _GEOM.mjGEOM_PLANE)
  assert home.GRID_BOUNDS[2] > float(model.geom_size[plane][0])


# ---- 3. the grid vs itself ---------------------------------------------------


@pytest.mark.parametrize("world", WORLDS)
def test_the_grid_stays_inside_its_cell_budget(world):
  """A tripwire, not a cliff. See `occupancy_grid.MAX_CELLS` for the measured
  table this is denominated in -- `binary_dilation` over every cell is the
  half of the mapping stack that scales, and `update()` is nearly flat."""
  x0, y0, x1, y1 = world_config(world)["grid_bounds"]
  grid = OccupancyGrid(x_min=x0, y_min=y0, x_max=x1, y_max=y1, resolution=0.05)
  cells = int(grid.grid.size)
  assert cells <= MAX_CELLS, (
    f"{world}: {cells:,} cells against a {MAX_CELLS:,} ceiling. Re-measure "
    f"before raising it -- the numbers are at the constant.")


# ---- 4. the reserve's worst case ---------------------------------------------


def test_the_documented_worst_return_point_is_still_the_worst():
  """`HOME_WORST_RETURN` names the point `HOME_LOW_BATTERY_WH` should be
  measured from. A comment saying "the worst case is X" rots the moment
  somebody moves a wall, and #68 moves several -- so the claim is checked.

  ⚠ This asserts WHERE the worst case is, not that the reserve covers it. It
  does not: 0.55 Wh was sized on a 2.89 m living-room crossing and this point
  routes 11.96 m. That gap is real, pre-existing and issue #70's to re-price;
  what this stops is the gap growing while nobody is looking.
  """
  rack = home.HOME_RACK_POS
  door = (home.GARDEN_X[0], sum(home.DOOR_GARDEN_Y) / 2.0)
  routed = math.dist(home.HOME_WORST_RETURN, door) + math.dist(door, rack)
  assert routed == pytest.approx(home.HOME_WORST_RETURN_M, abs=0.05), \
      f"the routed distance from HOME_WORST_RETURN is now {routed:.2f} m, " \
      f"not the documented {home.HOME_WORST_RETURN_M} m"

  # ...and it really is the farthest place the robot can be sent. Compared
  # LIKE WITH LIKE: every zone corner inset by the same margin, ranked by
  # straight-line distance from the rack. (The first cut of this compared a
  # bare corner's straight line against an inset point's ROUTED distance and
  # failed on the very zone the constant was taken from, which is a good
  # reminder that "farther" needs one metric, not two.)
  inset = math.dist(home.HOME_WORST_RETURN, (home.GARDEN_X[1], home.HOUSE_Y[1]))
  farthest, where = 0.0, None
  for zone in home.ZONES:
    (zx0, zy0), (zx1, zy1) = zone["min"], zone["max"]
    cx, cy = (zx0 + zx1) / 2.0, (zy0 + zy1) / 2.0
    for corner in ((zx0, zy0), (zx0, zy1), (zx1, zy0), (zx1, zy1)):
      # ...pulled `inset` metres toward the zone's middle, the way
      # HOME_WORST_RETURN is: a robot cannot stand in a fence.
      vx, vy = cx - corner[0], cy - corner[1]
      norm = math.hypot(vx, vy) or 1.0
      point = (corner[0] + vx / norm * inset, corner[1] + vy / norm * inset)
      d = math.dist(point, rack)
      if d > farthest:
        farthest, where = d, (zone["name"], point)
  documented = math.dist(home.HOME_WORST_RETURN, rack)
  assert farthest <= documented + 0.05, (
    f"{where[0]} reaches {where[1][0]:.2f},{where[1][1]:.2f} -- {farthest:.2f} m "
    f"from the rack against HOME_WORST_RETURN's {documented:.2f} m. The "
    f"documented worst case is no longer the worst; re-measure the reserve.")
