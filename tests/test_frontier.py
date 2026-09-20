import numpy as np

from pluggybot.mapping.frontier import find_frontiers, traversable_mask


def test_frontier_is_the_border_of_the_known():
  logodds = np.zeros((10, 10))           # everything unknown...
  logodds[2:5, 2:5] = -1.0               # ...except a 3x3 free block (iy 2-4, ix 2-4)
  cells = set(map(tuple, find_frontiers(logodds)))
  assert (2, 2) in cells                 # block corner: borders unknown
  assert (4, 3) in cells                 # block edge: borders unknown
  assert (3, 3) not in cells             # block interior: surrounded by known free


def test_fully_explored_map_has_no_frontiers():
  logodds = np.full((10, 10), -1.0)      # free room...
  logodds[0, :] = logodds[-1, :] = 1.0   # ...enclosed by occupied walls
  logodds[:, 0] = logodds[:, -1] = 1.0
  assert len(find_frontiers(logodds)) == 0


def test_unknown_behind_a_wall_is_not_a_frontier():
  # free | wall | unknown: the free cells touch the WALL, never the unknown,
  # so the sealed-off unknown region correctly produces zero frontiers.
  logodds = np.zeros((5, 7))
  logodds[:, :3] = -1.0                  # ix 0-2 free
  logodds[:, 3] = 1.0                    # ix 3 occupied wall
  assert len(find_frontiers(logodds)) == 0


def test_traversable_mask_inflates_obstacles():
  logodds = np.full((20, 20), -1.0)      # all free
  logodds[10, 10] = 1.0                  # one obstacle cell
  trav = traversable_mask(logodds, robot_radius_cells=2)
  assert not trav[10, 10]                # the obstacle itself
  assert not trav[10, 12]                # within 2 cells (Manhattan)
  assert not trav[8, 10]
  assert not trav[9, 11]
  assert trav[10, 13]                    # just beyond the inflation ring
  assert trav[13, 13]


def test_traversable_excludes_unknown():
  logodds = np.zeros((10, 10))           # all unknown
  logodds[5, 5] = -1.0                   # one free cell
  trav = traversable_mask(logodds)
  assert trav[5, 5]
  assert trav.sum() == 1                 # unknown space is never traversable


def test_inflation_is_the_iterated_cross_dilation_it_replaced():
  """rooftop-media-2026 #296: the chamfer transform must give the SAME
  mask as `binary_dilation(occupied, iterations=r)` with scipy's default
  cross structure -- a taxicab ball -- on any grid, edges and all. Random
  grids, three radii; a single differing cell is a changed planner."""
  from scipy.ndimage import binary_dilation
  from pluggybot.mapping.frontier import FREE_THRESH, OCC_THRESH
  rng = np.random.default_rng(296)
  for trial in range(12):
    h, w = rng.integers(3, 60, size=2)
    logodds = rng.choice([FREE_THRESH - 1.0, 0.0, OCC_THRESH + 1.0],
                         size=(h, w), p=[0.6, 0.3, 0.1])
    for r in (1, 4, 7):
      occupied = logodds > OCC_THRESH
      old = (logodds < FREE_THRESH) & ~binary_dilation(occupied, iterations=r)
      new = traversable_mask(logodds, robot_radius_cells=r)
      assert np.array_equal(old, new), (trial, r)
  # ...and a grid with nothing occupied inflates nothing.
  clear = np.full((20, 30), FREE_THRESH - 1.0)
  assert traversable_mask(clear).all()
