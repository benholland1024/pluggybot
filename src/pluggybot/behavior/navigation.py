"""Frontier navigation: shared by exploration and the recharge lifecycle.

Extracted from scripts/explore.py so that both explore.py (the milestone-4
demo, kept as the smallest repro of mapping behavior) and lifecycle.py can
drive the same planner without duplicating it. Pure functions over the
occupancy grid + pose; nothing here touches MuJoCo directly.
"""

import math

import numpy as np

from pluggybot.control import wrap_angle
from pluggybot.mapping.astar import astar, nearest_traversable
from pluggybot.mapping.frontier import find_frontiers, traversable_mask
from pluggybot.mapping.occupancy_grid import OccupancyGrid

Pose = tuple[float, float, float]     # (x, y, theta): axle midpoint + heading
Cell = tuple[int, int]                # (ix, iy) grid cell, per repo convention


SCAN_EVERY = 20          # physics steps between scans (500 Hz sim -> 25 Hz scanning)
REPLAN_PERIOD = 2.0      # sim seconds between replans (the map changes under us)
MAP_SAVE_PERIOD = 0.5    # sim seconds between map.png saves
V_MAX = 0.4              # m/s cruise speed
W_MAX = 1.5              # rad/s turn-rate clamp
K_HEADING = 2.5          # P gain: heading error -> turn rate
WAYPOINT_RADIUS = 0.08   # m: close enough to advance (small: big radii cut corners
                         # through the inflation ring and clip obstacles)
FRONT_STOP_RANGE = 0.25  # m: reflex threshold — camera-measured range dead ahead
BACKOFF_TIME = 0.8       # s of straight reverse after the reflex trips
MAX_PLAN_ATTEMPTS = 20   # frontiers tried per replan before giving up this round
STRIKES_TO_FINISH = 3    # post-spin pathless replans before declaring done
MIN_FRONTIER_CELLS = 6   # ignore frontiers closer than this (0.3 m): a forward camera
                         # can't observe the cells beside its own wheels -- chasing
                         # them deadlocks; they dissolve while driving to real goals
W_SPIN = 1.0             # rad/s during a look-around spin
TERMINAL_CONE = math.radians(25)   # a terminal approach translates only while
                         # the destination is within this cone of dead ahead;
                         # outside it, pivot in place. See drive_toward.


def plan(grid: OccupancyGrid, pose: Pose,
         blacklist: set[Cell]) -> tuple[list[Cell] | None, str]:
  """Pick the nearest reachable frontier and plan a path to it.

  Returns (path, status): path is a cell list or None; status is "ok",
  "no-frontiers" (map fully explored) or "no-reachable" (frontiers exist
  but none could be pathed to this round).
  """
  trav = traversable_mask(grid.grid)
  frontiers = find_frontiers(grid.grid, traversable=trav)
  if len(frontiers) == 0:
    return None, "no-frontiers"

  # ⚠ ESCAPE THE INFLATION HALO FIRST (issue #92). A robot parked against a
  # couch sits inside the couch's own inflation ring; `astar`'s start-cell
  # exemption is one cell deep, so every plan from there fails on the first
  # step, every failure permanently blacklists a frontier, and after a few
  # strikes explore declares a two-thirds-unmapped house finished. The twin
  # planner in `HubMission._plan_to` always had this escape -- this one is
  # what exploration uses, and it did not.
  start = nearest_traversable(trav, grid.world_to_cell(pose[0], pose[1]))
  if start is None:
    return None, "no-reachable"          # off the map, not merely sealed in
  rix, riy = start

  dist = np.hypot(frontiers[:, 0] - rix, frontiers[:, 1] - riy)
  eligible = dist >= MIN_FRONTIER_CELLS
  if not eligible.any():
    return None, "only-near"             # a look-around spin will resolve these

  order = np.argsort(np.where(eligible, dist, np.inf))
  attempts = 0
  for idx in order:
    if not eligible[idx]:
      break                              # only inf-distance (ineligible) cells remain
    cell = (int(frontiers[idx, 0]), int(frontiers[idx, 1]))
    if cell in blacklist:
      continue
    path = astar(trav, (rix, riy), cell)
    if path is not None:
      return path, "ok"
    blacklist.add(cell)                  # unreachable this round; skip it in future
    attempts += 1
    if attempts >= MAX_PLAN_ATTEMPTS:
      break
  return None, "no-reachable"


def path_to_waypoints(grid: OccupancyGrid,
                      path: list[Cell]) -> list[tuple[float, float]]:
  """Cell path -> sparse world waypoints (every 3rd cell keeps driving smooth)."""
  cells = path[3::3]
  if not cells or cells[-1] != path[-1]:
    cells.append(path[-1])
  return [grid.cell_to_world(ix, iy) for ix, iy in cells]


def drive_toward(pose: Pose, waypoint: tuple[float, float],
                 slow_radius: float | None = None) -> tuple[float, float]:
  """Proportional controller: (v, w) command toward a world waypoint.

  Default (`slow_radius=None`) is the pure-pursuit law every path-follower
  uses: turn proportionally, and translate at a speed scaled by how nearly
  the waypoint is ahead. That is the right shape for SWEEPING THROUGH a
  chain of waypoints, which is what exploration and A* path-following do.

  ⚠ It is the wrong shape for the last metre, and the failure is a limit
  cycle rather than a wobble. Chasing a nearby destination, the robot
  overshoots, the target ends up BESIDE it, and the law then settles into a
  stable orbit: `w` saturates at W_MAX while `cos(heading_err)` keeps a
  little forward speed alive, so the machine drives a small circle AROUND
  the point it is trying to reach. Measured on a 0.20 m hop with the seed
  dispenser: heading error pinned at 84-87 deg for ten solid seconds, three
  full revolutions, escaping only when drift happened to drop it inside the
  15 mm arrival radius. Roughly 900 deg of turning to cover 200 mm.

  It bites SHORT HOPS specifically, which is why it stayed hidden: the
  claw stages 0.45 m back and the plotter drives ~1 m to its board, and both
  measure ~185 deg of turning for their approaches -- near the geometric
  minimum. Nothing in the repo drove a 20 cm errand until the dispenser did.

  Passing `slow_radius` switches to a TERMINAL approach, which is the fix
  and which cannot orbit by construction:

    * outside a +/-TERMINAL_CONE cone, `v` is exactly ZERO -- pivot in
      place. The orbit lives on the sliver of speed that `cos(84 deg)` still
      allows; a hard cone removes it. Turning on the spot also costs no
      ground, so a badly-aimed approach is corrected rather than flown
      around.
    * inside the cone, speed tapers linearly to zero over the last
      `slow_radius` metres, so the robot arrives instead of overshooting and
      re-entering the chase.

  Terminal speed stays clear of the wheels' stiction deadband at any sane
  arrival radius (0.4 m/s x 15 mm / 0.25 m is 0.53 rad/s of wheel against a
  0.1 rad/s floor), so this does not need `turn_command`'s breakaway
  treatment. Path-followers are untouched: the parameter defaults to off.
  """
  px, py, theta = pose
  dx, dy = waypoint[0] - px, waypoint[1] - py
  heading_err = wrap_angle(math.atan2(dy, dx) - theta)
  w = max(-W_MAX, min(W_MAX, K_HEADING * heading_err))
  if slow_radius is None:
    v = V_MAX * max(0.0, math.cos(heading_err))  # pivot first, drive when aligned
    return v, w
  if abs(heading_err) > TERMINAL_CONE:
    return 0.0, w                                # pivot in place, cover no ground
  taper = min(1.0, math.hypot(dx, dy) / max(slow_radius, 1e-6))
  return V_MAX * taper * math.cos(heading_err), w


PATH_COLOR = (60, 90, 220)          # blue: the planned route
ROBOT_COLOR = (220, 50, 50)         # red: where the robot believes it is
OUTLET_CONFIRMED_COLOR = (40, 205, 90)    # green: a landmark trusted enough to drive to
OUTLET_TENTATIVE_COLOR = (130, 150, 55)   # olive: seen, but not yet confirmed


def _stamp(img, grid, wx, wy, color, half=0):
  """Paint a (2*half+1)-square block at a world point, clipped to the map.

  Landmarks come from projected detections, which can land outside the grid
  when a sighting is noisy -- unclipped, a negative index silently wraps and
  paints a marker on the opposite edge of the map.
  """
  rows, cols = img.shape[:2]
  ix, iy = grid.world_to_cell(wx, wy)
  if not (0 <= ix < cols and 0 <= iy < rows):
    return
  img[max(0, iy - half):iy + half + 1, max(0, ix - half):ix + half + 1] = color


def render_map(grid: OccupancyGrid, pose: Pose,
               waypoints: list[tuple[float, float]],
               landmarks=(), min_sightings: int = 3) -> np.ndarray:
  """Map image with overlays: blue planned path, red robot, green outlets.

  Outlets are drawn bright green once confirmed (>= min_sightings, matching
  LandmarkStore.confirmed) and olive while still tentative, so the map shows
  at a glance which memories the robot would actually act on.
  """
  img = np.stack([grid.to_image()] * 3, axis=-1)
  for wx, wy in waypoints:
    _stamp(img, grid, wx, wy, PATH_COLOR)
  for lm in landmarks:
    color = (OUTLET_CONFIRMED_COLOR if lm.n_sightings >= min_sightings
             else OUTLET_TENTATIVE_COLOR)
    _stamp(img, grid, lm.x, lm.y, color, half=1)
  _stamp(img, grid, pose[0], pose[1], ROBOT_COLOR, half=1)
  return np.flipud(img)