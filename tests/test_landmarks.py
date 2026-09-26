import math

import numpy as np

from pluggybot.mapping.landmarks import Landmark, wall_normal_conf
from pluggybot.mapping.occupancy_grid import OccupancyGrid

SEEN = (0.0, 0.0)   # a default observer position for tests that don't care


def _grid_with_wall_below_y0():
  """A map whose y > 0 half is known-free and y <= 0 half is solid wall."""
  grid = OccupancyGrid(x_min=-2, y_min=-2, x_max=2, y_max=2, resolution=0.05)
  ys = np.arange(grid.grid.shape[0]) * grid.resolution + grid.y_min
  grid.grid[ys > 0.0, :] = -1.0     # confidently free
  grid.grid[ys <= 0.0, :] = 1.0     # confidently occupied
  return grid


def test_merged_position_is_the_mean_of_sightings():
  """The running average must converge to the mean, so one bad first
  sighting (typically the farthest, noisiest view) gets corrected."""
  sightings = [(1.00, 2.00, 0.20), (1.20, 2.10, 0.26), (1.10, 1.90, 0.23)]
  lm = Landmark(*sightings[0], SEEN)
  for s in sightings[1:]:
    lm.merge(*s, SEEN)
  assert math.isclose(lm.x, sum(s[0] for s in sightings) / 3, abs_tol=1e-9)
  assert math.isclose(lm.y, sum(s[1] for s in sightings) / 3, abs_tol=1e-9)
  assert math.isclose(lm.z, sum(s[2] for s in sightings) / 3, abs_tol=1e-9)


def test_seen_from_averages_across_sightings():
  lm = Landmark(0.0, 0.0, 0.24, (1.0, 0.4))
  lm.merge(0.0, 0.0, 0.24, (1.0, -0.4))
  assert math.isclose(lm.seen_from_x, 1.0)
  assert math.isclose(lm.seen_from_y, 0.0, abs_tol=1e-9)


def test_wall_normal_points_out_of_the_wall():
  """The map knows which way a wall faces; the mean seen-from point only
  knows where the robot drove. Measured end to end, seen-from put the
  standoff heading 31 deg off the true normal and the map put it at 0.0."""
  nx, ny, _ = wall_normal_conf(_grid_with_wall_below_y0(), 0.0, 0.0)
  assert math.isclose(ny, 1.0, abs_tol=0.05)   # straight out of the wall
  assert abs(nx) < 0.05                        # no sideways bias


def test_wall_normal_flips_to_the_side_it_was_seen_from():
  """A free-standing partition is open on both sides, so the free-cell sums
  nearly cancel and the surviving direction may name the wrong face. The
  robot demonstrably saw this landmark, so the side it was seen from
  decides."""
  grid = _grid_with_wall_below_y0()
  # Claim it was observed from BELOW the wall (the solid side here), which
  # is the disagreement the sign check exists to catch.
  _, ny, _ = wall_normal_conf(grid, 0.0, 0.0, fallback=(0.0, -1.0))
  assert ny < 0.0, "normal must stay in the hemisphere it was seen from"


def test_wall_normal_falls_back_when_nothing_is_mapped():
  """A landmark seen from far off may sit in unmapped space: no free cells
  nearby means no evidence, so the caller's fallback must win rather than
  some arbitrary direction."""
  blank = OccupancyGrid(x_min=-2, y_min=-2, x_max=2, y_max=2, resolution=0.05)
  assert wall_normal_conf(blank, 0.0, 0.0, fallback=(0.6, -0.8)) == \
    (0.6, -0.8, 0.0)
