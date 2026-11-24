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


def _thin_wall_scans(belief_dy: float):
  """#712's alley, as a scan sees it: the research center's south wall (4 cm,
  y -6.02..-5.98, x 16..28) met at 3-35 deg by a robot walking south past
  the alley's mouth at x 15.55, each scan's ranges the truth's and laid at a
  belief `belief_dy` north of it (the sidewalk's weak direction)."""
  for y in np.arange(-6.2, -9.2, -0.04):
    ox, oy = 15.55, float(y)
    bearings = np.radians(np.arange(3.0, 36.0, 1.0))
    ranges = (-6.02 - oy) / np.sin(bearings)
    keep = (ranges < 8.0) & (ox + ranges * np.cos(bearings) <= 28.0)
    yield (ox, oy + belief_dy, 0.0), bearings[keep], ranges[keep]


def _alley_grid():
  """The wall laid from inside first: head-on scans from the lobby, its
  north face's cells sure (+5) from x 19 to 23.5."""
  g = OccupancyGrid(x_min=10, y_min=-12, x_max=30, y_max=-2, resolution=0.05)
  for x in np.arange(19.5, 23.5, 0.5):
    bearings = np.radians(np.linspace(-115.0, -65.0, 51))
    ranges = (-5.98 - (-3.0)) / np.sin(bearings)
    for _ in range(8):
      g.update((float(x), -3.0, 0.0), bearings, ranges, 8.0, origin=(0.0, 0.0))
  return g


def _held_columns(g):
  """Columns of the wall's two rows (x 19-23.5) either of which reads wall."""
  r0, r1 = g.world_to_cell(19.0, -6.02)[1], g.world_to_cell(19.0, -5.98)[1]
  c0, c1 = g.world_to_cell(19.0, -6.0)[0], g.world_to_cell(23.5, -6.0)[0]
  return int((g.grid[r0:r1 + 1, c0:c1] > 0.5).any(axis=0).sum()), c1 - c0


def test_a_thin_wall_met_at_a_graze_from_a_belief_a_few_cm_off_is_kept():
  # #712: the base's rule erased this wall in 8.5 s of the live robots'
  # walk, every clear a grazing ray's run along the wall's own rows on its
  # way to a face laid 9 cm beyond them (at the true pose, none of them).
  # Pinned on the rule's direct call: the wall laid from inside, then the
  # alley's grazing scans laid 9 cm off.
  from pluggybot.mapping.occupancy_grid import GRAZE_M
  before = _alley_grid()
  assert _held_columns(before) == (90, 90)
  for graze in (None, GRAZE_M):
    g = _alley_grid()
    for pose, bearings, ranges in _thin_wall_scans(0.09):
      g.update(pose, bearings, ranges, 8.0, origin=(0.0, 0.0), graze_m=graze)
    held, cols = _held_columns(g)
    if graze is None:
      assert held <= 20, f"the base's rule kept {held} of {cols}: the premise is gone"
    else:
      assert held == cols, f"{held} of {cols} columns kept"


def test_a_wall_that_moved_is_still_cleared_by_rays_that_pass_it_by():
  # What the rule costs (#712): a cell it no longer frees is one a ray runs
  # within GRAZE_M of the surface it ends on. A door laid shut in a doorway
  # and then gone is still cleared by the rays through the doorway to the
  # room beyond, each passing it far from its own end.
  from pluggybot.mapping.occupancy_grid import GRAZE_M
  g = OccupancyGrid(x_min=-1, y_min=-3, x_max=6, y_max=3, resolution=0.05)
  bearings = np.radians(np.linspace(-20.0, 20.0, 41))
  door = 2.0 / np.cos(bearings)                  # a door across x = 2, seen head on
  for _ in range(10):
    g.update((0.0, 0.0, 0.0), bearings, door, 8.0, origin=(0.0, 0.0), graze_m=GRAZE_M)
  assert logodds_at(g, 2.0, 0.0) > 0.5
  room = 5.0 / np.cos(bearings)                  # the door gone: the wall 5 m off
  for _ in range(20):
    g.update((0.0, 0.0, 0.0), bearings, room, 8.0, origin=(0.0, 0.0), graze_m=GRAZE_M)
  assert logodds_at(g, 2.0, 0.0) < -0.5
  assert logodds_at(g, 5.0, 0.0) > 0.5


def test_a_ray_beside_an_edge_reads_the_surface_it_ends_on_not_the_jump():
  # #712's revision (its review's finding 1): a ray's slant off the line to
  # its neighbour's end read the depth jump at an occluding edge as a
  # graze, and kept up to 2 m of walled cells along every ray beside an
  # edge. Read where both neighbours agree -- the larger sine -- a wall
  # that went (a cell laid sure 0.5 m short of the edge) is cleared by the
  # rays either side of it, as it is with no rule at all.
  from pluggybot.mapping.occupancy_grid import GRAZE_M
  g = OccupancyGrid(x_min=-1, y_min=-3, x_max=6, y_max=3, resolution=0.05)
  ix, iy = g.world_to_cell(2.5, 0.02)
  g.grid[iy, ix] = 5.0
  bearings = np.radians(np.arange(-20.0, 21.0, 1.0))
  ranges = np.where(bearings <= 0, 3.0, 3.5) / np.cos(bearings)  # a near wall's edge on a far one
  for _ in range(20):
    g.update((0.0, 0.0, 0.0), bearings, ranges, 8.0, origin=(0.0, 0.0), graze_m=GRAZE_M)
  assert logodds_at(g, 2.5, 0.02) < -0.5


def test_a_near_wall_seen_square_on_through_the_range_noise_is_no_graze():
  # #712's revision (its review's finding 2): off the next bearing's end, a
  # chord 1.7 cm across a metre out, a 2 cm sigma of range noise read a wall
  # seen square on as met at a graze (a sine of 0.16). Off a neighbour's end
  # SLANT_CHORD_M or more away, it reads square on. A 2 m room round the
  # robot, every ray within 20 deg of its wall's normal.
  from pluggybot.mapping.occupancy_grid import _slant
  rng = np.random.default_rng(0)
  bearings = np.radians(np.arange(0.0, 360.0, 1.0))
  cos_b, sin_b = np.cos(bearings), np.sin(bearings)
  truth = np.minimum(1.0 / np.maximum(np.abs(cos_b), 1e-9), 1.0 / np.maximum(np.abs(sin_b), 1e-9))
  square = np.abs(np.remainder(np.degrees(bearings) + 45.0, 90.0) - 45.0) <= 20.0
  least = 1.0
  for _ in range(20):
    ranges = truth + rng.normal(0.0, 0.02, bearings.size)
    sine = _slant(bearings, ranges * cos_b, ranges * sin_b, np.ones(bearings.size, dtype=bool),
                  cos_b, sin_b)
    least = min(least, float(sine[square].min()))
  assert least > 0.8, f"a wall seen square on read at a sine of {least:.2f}"


def test_a_copy_of_a_wall_a_few_cm_in_front_of_it_wears_away_under_rays_that_run_alongside():
  # #712's revision (its review's finding 5): a veto kept every wall cell a
  # ray ran alongside within GRAZE_M of its end -- the alley's wall, and as
  # surely a copy of a corridor's wall laid 6 cm off it by a belief that
  # was, which held the research center's belief to it. Weighted by the
  # slant instead (a clear at 30-48 deg weighs 0.06-0.30 of one), rays laid
  # where they truly were wear a sure copy away in three walks along it.
  from pluggybot.mapping.occupancy_grid import GRAZE_M
  g = OccupancyGrid(x_min=-1, y_min=-3, x_max=8, y_max=3, resolution=0.05)
  row = g.world_to_cell(0.0, 0.94)[1]
  c0, c1 = g.world_to_cell(2.0, 0.94)[0], g.world_to_cell(4.0, 0.94)[0]
  g.grid[row, c0:c1] = 5.0                       # the copy, 6 cm in front of the wall at y = 1
  bearings = np.radians(np.arange(30.0, 49.0, 1.0))
  for _ in range(3):
    for x in np.arange(0.0, 4.0, 0.05):          # walked along it, the wall's truth each scan
      g.update((float(x), 0.0, 0.0), bearings, 1.0 / np.sin(bearings), 8.0, origin=(0.0, 0.0),
               graze_m=GRAZE_M)
  kept = int((g.grid[row, c0:c1] > 0.5).sum())
  assert kept == 0, f"{kept} of {c1 - c0} cells of the copy kept"


def test_a_walled_cell_nearer_than_free_from_is_never_cleared_by_a_slanted_ray():
  # #712's revision's review: the served scan carries both rules, and the
  # nearer one wins. A ray meeting a wall at 45 deg 0.7 m out clears a walled
  # cell in its last 0.15 m at sin(45 deg)^4, a quarter of a clear -- but
  # nearer than its `free_from` (1 m here) a ray frees no walled cell at all.
  from pluggybot.mapping.occupancy_grid import GRAZE_M
  g = OccupancyGrid(x_min=-1, y_min=-3, x_max=6, y_max=3, resolution=0.05)
  ix, iy = g.world_to_cell(0.6, 0.0)
  g.grid[iy, ix] = 1.0
  bearings = np.radians(np.arange(-20.0, 21.0, 1.0))
  ranges = 0.7 / (np.cos(bearings) + np.sin(bearings))  # the wall x + y = 0.7, met at 45 deg
  for _ in range(20):
    g.update((0.0, 0.0, 0.0), bearings, ranges, 8.0, origin=(0.0, 0.0),
             free_from=np.full(bearings.size, 1.0), graze_m=GRAZE_M)
  assert logodds_at(g, 0.6, 0.0) == 1.0


def test_a_slant_read_a_hair_over_square_on_is_square_on():
  # #712's revision's review: a sine of 1 plus a rounding error took the
  # square root of a negative number -- NaN, and numpy's warning on every
  # such scan. Read as square on, it holds nothing.
  import warnings
  from pluggybot.mapping import occupancy_grid as og
  g = OccupancyGrid(x_min=-1, y_min=-3, x_max=6, y_max=3, resolution=0.05)
  bearings = np.radians(np.arange(-20.0, 21.0, 1.0))
  ranges = 2.0 / np.cos(bearings)
  real = og._slant
  og._slant = lambda a, *rest: np.full(a.size, 1.0 + 2.0 ** -52)
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      g.update((0.0, 0.0, 0.0), bearings, ranges, 8.0, origin=(0.0, 0.0), graze_m=og.GRAZE_M)
  finally:
    og._slant = real
  assert logodds_at(g, 1.0, 0.0) < 0.0


def test_what_a_scan_looks_across_and_round_it_is_forgotten_and_nothing_else():
  # #741 (`ScanMatcher.relay`): the cells a scan from the pose looks across
  # -- out to each bearing's end, to max range where it met nothing -- and
  # within round_m of them go back to unknown; a wall's far side past that,
  # and what lies off every bearing the scan carries, keep their evidence.
  grid = OccupancyGrid(x_min=-2, y_min=-2, x_max=6, y_max=6, resolution=0.05)
  grid.grid[...] = 3.0                      # all of it held as a wall...
  ix, iy = grid.world_to_cell(1.0, 0.0)
  grid.grid[iy, ix - 10:ix + 10] = -3.0     # ...but a stretch of floor before the wall east
  angles = np.radians([0.0, 90.0])
  ranges = np.array([2.0, 8.0])             # a wall 2 m east; nothing north
  bare = grid.grid.copy()
  held = grid.forget_seen((0.0, 0.0, 0.0), angles, ranges, 3.0, origin=(0.0, 0.0),
                          round_m=0.2)
  assert logodds_at(grid, 1.0, 0.0) == 0.0, "floor it looked across, held"
  assert logodds_at(grid, 2.0, 0.0) == 0.0, "the wall it ended on, held"
  assert logodds_at(grid, 2.15, 0.0) == 0.0, "within round_m past its end"
  assert logodds_at(grid, 2.4, 0.0) == 3.0, "a wall's far side past round_m"
  assert logodds_at(grid, 0.0, 2.9) == 0.0, "to max range where it met nothing"
  assert logodds_at(grid, 0.0, 3.4) == 3.0, "past max range and round_m"
  assert logodds_at(grid, 1.0, 1.0) == 3.0, "off both bearings"
  assert logodds_at(grid, -1.0, 0.0) == 3.0, "behind the sensor"
  assert held == int(np.count_nonzero(grid.grid == 0.0))
  # ...and with no reach round it, the cell each ray ends in and not past it
  grid.grid[...] = bare
  grid.forget_seen((0.0, 0.0, 0.0), angles, ranges, 3.0, origin=(0.0, 0.0))
  assert logodds_at(grid, 2.02, 0.0) == 0.0 and logodds_at(grid, 2.07, 0.0) == 3.0
