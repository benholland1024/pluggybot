import pytest 
import numpy as np
import math

from pluggybot.mapping.occupancy_grid import OccupancyGrid

@pytest.fixture
def grid():
  return OccupancyGrid(x_min=-2, y_min=-2, x_max=6, y_max=6, resolution=0.05)

# "Log odds" means logarithmic odds (the odds a given square is occupied)
def logodds_at(grid, x, y):
  """Log-odds value at a world coord. Hides the [row, col] index flip"""
  ix, iy = grid.world_to_cell(x,y)
  return grid.grid[iy, ix]     # note: [iy, ix], not [ix, iy]


def test_grid_coordinate_mapping(grid):
  """Ensure that coordinates in the world map correctly to the occupancy map, and vice versa"""

  # Basic test
  test_coords_1 = grid.cell_to_world(*grid.world_to_cell(1.02, 0.99))
  assert abs(test_coords_1[0] - 1.02) < (grid.resolution/2)
  assert abs(test_coords_1[1] - 0.99) < (grid.resolution/2)

  # Test corner pin
  corner_pin_coords = grid.world_to_cell(-2,-2)
  assert corner_pin_coords[0] == 0 and corner_pin_coords[1] == 0

  # Asymmetric transpose
  test_coord_3 = grid.world_to_cell(5.0, -1.5)
  assert test_coord_3[0] == 140 and test_coord_3[1] == 10

def test_empty_grid(grid):
  """A fresh grid holds no evidence and renders all all-unknown"""
  assert not grid.grid.any()   # .any() is True if ANY element is nonzero

  img = grid.to_image()
  assert img.shape == grid.grid.shape
  assert img.dtype == np.uint8
  assert (img == 127).all()    # all pixels are "unknown" gray

  
def test_one_ray(grid):
  """One straight ray: free along the beam, occupied at the tip, silent elsewhere
  
  pose(0,0,0) puts the camera (scan origin) at (0.03, 0.03), so a range-1.0
  ray at angle 0 travels along y=0.3 and ends at world (1.03, 0.03)
  """
  grid.update(pose=(0.0, 0.0, 0.0), angles=[0.0], ranges=[1.0], max_range=5.0)

  assert logodds_at(grid, 0.5, 0.03) < 0     # mid-beam: evidence of free 
  assert logodds_at(grid, 1.03, 0.03) > 0    # the tip: evidence of a wall
  assert logodds_at(grid, 0.5, 1.0) == 0     # bystander cell: untouched
  assert logodds_at(grid, -0.5, 0.03) == 0   # behind the camera (also untouched)

def test_max_range_ray_no_flag(grid):
  """If a ray hits the max range, it shouldn't mark it as occupied"""
  grid.update(pose=(0.0, 0.0, 0.0), angles=[0.0], ranges=[5.0], max_range=5.0)
  assert logodds_at(grid, 2.5, 0.03) < 0    # beam interior: marked free
  assert logodds_at(grid, 5.03, 0.03) <= 0  # the far tip: free or untouched
  assert (grid.grid <= 0).all()             # Redundant, more thorough:
                                            #  A miss leaves no positive evidence anywhere

def test_frames_compose(grid):
  """A ray at pose=(0, 0, π/2) must have end endpoint with +y axis (not x)
  
  at θ = π/2, the camera offset rotates too: 
    origin = (0 + 0.03*cos90° - 0.03*sin90°, 0 + 0.03*sin90° + 0.03*cos90°)
           = (-0.03, +0.03) 
   The ray now points along +y, ending at (−0.03, 1.03)
  """
  grid.update(pose=(0.0, 0.0, math.pi/2), angles=[0.0], ranges=[1.0], max_range=5.0)
  assert logodds_at(grid, -0.03, 0.5) < 0     # beaam runs up the +y axis
  assert logodds_at(grid, -0.03, 1.03) > 0    # tip is north of the robot
  assert logodds_at(grid, 1.03, 0.03) == 0    # wheree the UNrotated ray would have ended  (silent)

def test_evidence_saturation(grid):
  """The same scan 100 times. Cells hit the maximum limit (+/- 5.0)"""
  for _ in range(100):
    grid.update(pose=(0.0, 0.0, 0.0), angles=[0.0], ranges=[1.0], max_range=5.0)
  
  assert logodds_at(grid, 0.5, 0.03) == -5.0     # mid-beam: strong evidence of free 
  assert logodds_at(grid, 1.03, 0.03) == 5.0    # the tip: strong evidence of a wall
  assert grid.grid.min() >= -5.0
  assert grid.grid.max() <= 5.0
