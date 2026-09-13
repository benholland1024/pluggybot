"""Guards for the robot-centric height map (perception/heightmap.py, issue
#34). Pure numpy: no sim, no frame -- points are handed in."""

import math

import numpy as np

from pluggybot.perception import heightmap as hmod
from pluggybot.perception.heightmap import HeightMap


def _cube(x, y, side, z_top, step=0.005):
  """Points on the top face of a cube in the ROBOT frame."""
  xs = np.arange(x - side / 2, x + side / 2, step)
  ys = np.arange(y - side / 2, y + side / 2, step)
  xx, yy = np.meshgrid(xs, ys)
  return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_top)], axis=1)


def _floor(x0, x1, y0, y1, step=0.01):
  xx, yy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
  return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def test_the_cell_count_is_the_budget():
  """Issue #34's budget: a robot-centric window, fewer cells than the 2D
  grid (56 000 at 5 cm), and a voxel map of the same volume is 25x that
  (`scripts/nearfield_spike.py --cost`)."""
  hm = HeightMap()
  assert hm.cells == 40_000
  assert hm.cells < 56_000
  assert hm.height.nbytes + hm.count.nbytes < 1_000_000


def test_the_highest_point_wins_and_a_remeasured_cell_can_fall():
  """A cell is the top of what the LAST frame saw there: max within a
  frame, replaced by the next. A thing carried away reads as floor again;
  a running max would remember it for ever."""
  hm = HeightMap()
  pose = (0.0, 0.0, 0.0)
  hm.update(pose, np.array([[1.0, 0.0, 0.01], [1.005, 0.003, 0.10]]))
  ix, iy = hm._index(1.0, 0.0)
  cell = (iy - hm.origin[1], ix - hm.origin[0])
  assert hm.height[cell] == 0.10
  hm.update(pose, np.array([[1.0, 0.0, 0.0]]))
  assert hm.height[cell] == 0.0
  assert hm.count[cell] == 2


def test_recentring_keeps_what_stays_in_the_window_and_forgets_what_leaves():
  """Content is world-anchored: a thing seen from one pose is at the same
  world place from the next, and what falls out of the +-2 m window is
  gone rather than wrapped round."""
  hm = HeightMap()
  hm.update((0.0, 0.0, 0.0), _cube(1.5, 0.0, 0.10, 0.08))
  before = hm.things()
  assert len(before) == 1 and abs(before[0]["x"] - 1.5) < 0.02
  hm.update((1.0, 0.3, 0.0), np.zeros((0, 3)))
  after = hm.things()
  assert len(after) == 1
  assert abs(after[0]["x"] - before[0]["x"]) < 1e-9
  assert abs(after[0]["y"] - before[0]["y"]) < 1e-9
  hm.update((4.0, 0.0, 0.0), np.zeros((0, 3)))
  assert hm.things() == []
  assert hm.seen_fraction() == 0.0


def test_things_come_out_in_world_coordinates_through_the_pose():
  """The robot at (2, 3) facing +y sees a 10 cm thing 1 m ahead: the world
  says (2, 4). The yaw sign is the whole test."""
  hm = HeightMap()
  hm.update((2.0, 3.0, math.pi / 2), _cube(1.0, 0.0, 0.10, 0.10))
  (t,) = hm.things()
  assert abs(t["x"] - 2.0) < 0.02 and abs(t["y"] - 4.0) < 0.02, t
  assert abs(t["height"] - 0.10) < 1e-9
  assert 20 <= t["cells"] <= 36   # a 10 cm square is 5x5 cells, +-1 on the edges


def test_a_point_above_the_ceiling_is_not_a_floor_thing():
  """A 2.5D map cannot say what is under an overhang, so it does not try:
  the rack shelf, a table top and the module on the fork are above `Z_MAX`
  and dropped, never painted down as a wall on the floor."""
  hm = HeightMap()
  pts = np.vstack([_floor(0.5, 1.5, -0.5, 0.5),
                   _cube(1.0, 0.0, 0.10, hmod.Z_MAX + 0.05),
                   _cube(1.0, 0.3, 0.10, hmod.Z_MAX - 0.05)])
  hm.update((0.0, 0.0, 0.0), pts)
  things = hm.things()
  assert len(things) == 1, things
  assert abs(things[0]["y"] - 0.3) < 0.02


def test_a_thing_under_raised_m_is_floor():
  """Floor noise is not a thing: below `RAISED_M` nothing is reported."""
  hm = HeightMap()
  hm.update((0.0, 0.0, 0.0), _cube(1.0, 0.0, 0.10, hmod.RAISED_M - 0.005))
  assert hm.things() == []
  assert hm.seen_fraction() > 0.0
