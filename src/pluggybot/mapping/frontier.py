"""Frontier detection: known-free cells that border unknown space.

A frontier cell is a legal destination (it's known-free) whose neighborhood
is not fully explored — standing there reveals what lies beyond. Exploration
is complete when no reachable frontiers remain.
"""

import numpy as np
from scipy.ndimage import distance_transform_cdt

FREE_THRESH = -0.5   # log-odds below this = confidently free
OCC_THRESH = 0.5     # log-odds above this = confidently occupied


def traversable_mask(logodds: np.ndarray, robot_radius_cells: int = 7) -> np.ndarray:
  """Cells the robot's CENTER may occupy: known-free and clear of obstacles.

  Occupied cells are inflated by the robot's radius so the planner can treat
  the robot as a point. The default (7 cells = 0.35 m) covers the ARMED
  robot's swing radius: the alignment-feeler tips sweep 0.27 m from the axle
  (sqrt(0.24^2 + 0.12^2)), plus margin for control and drift error. The old
  5-cell value was calibrated to the bare 0.15 m chassis half-diagonal, and
  the day the arm was added a feeler measurably clipped an obstacle while
  cornering. Inflation must track the robot's OUTERMOST geometry, not its
  body. Unknown space is never traversable — we don't plan through territory
  we haven't seen.
  """
  occupied = logodds > OCC_THRESH
  # `binary_dilation(occupied, iterations=r)` with scipy's default cross
  # structure, which this was: r iterated 4-neighbour dilations, i.e. every
  # cell within TAXICAB distance r of an occupied one. The chamfer distance
  # transform answers the same question in two passes over the grid instead
  # of r full passes -- MEASURED (rooftop-media-2026 #296) at ~300 ms a call
  # on the 49 x 21 m home grid, on every 2 s replan, 10 % of the served
  # pair's physics thread. `tests/test_frontier.py` pins the two identical.
  if occupied.any():
    inflated = distance_transform_cdt(~occupied, metric="taxicab") <= robot_radius_cells
  else:
    inflated = occupied
  free = logodds < FREE_THRESH
  return free & ~inflated


def find_frontiers(logodds: np.ndarray, traversable: np.ndarray | None = None) -> np.ndarray:
  """Return frontier cells as an (N, 2) array of (ix, iy) pairs.

  A frontier is a free cell with at least one unknown 4-neighbor. Unknown
  space sealed behind walls never yields frontiers, because no *free* cell
  ever borders it. If a traversable mask is given, only frontiers the robot
  could actually stand on are returned.
  """
  free = logodds < FREE_THRESH
  unknown = (logodds >= FREE_THRESH) & (logodds <= OCC_THRESH)

  # Poor man's convolution: OR together the unknown mask shifted one cell
  # in each cardinal direction -> "this cell touches unknown".
  touches_unknown = np.zeros_like(unknown)
  touches_unknown[1:, :] |= unknown[:-1, :]
  touches_unknown[:-1, :] |= unknown[1:, :]
  touches_unknown[:, 1:] |= unknown[:, :-1]
  touches_unknown[:, :-1] |= unknown[:, 1:]

  frontier = free & touches_unknown
  if traversable is not None:
    frontier &= traversable

  ys, xs = np.nonzero(frontier)          # array is indexed [iy, ix]
  return np.column_stack([xs, ys])       # ...but the API speaks (ix, iy)
