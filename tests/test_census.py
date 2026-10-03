"""The census on legs (issues #13, #407): the area the walls enclose, what a
survey's depth frames saw standing in it, and the share of its floor seen.

The flown claim -- the garden's four plants counted at 91 % coverage from
the dock, the garden light's pole left out -- is a measurement
(`scripts/energy_spike.py --actions census`, reported on #407); every rule
it stands on is pinned here without flying:

  1. THE AREA IS THE WALLS': a doorway closes it, a wider gap does not.
  2. AN OBJECT IS PLANT-SIZED and inside the area; a diagonal pair is one.
  3. COVERAGE is the share of the area's floor seen, never of the map.
  4. THE TRUTH comes out of the world, over both garden rectangles.
  5. A SURVEY NEVER TAKES A STALK BACK: floor seen round it later does not
     erase it, and one frame is a stray point, not a plant.
  6. AN OBJECT THE LIDAR HIT IS NOT A PLANT -- but only a hit near enough
     that the scan plane is still over a plant's height.
  7. The survey's walk: it stops once it has seen enough, says why it
     stopped otherwise, and never leaves its layer on.
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot.economy.census import (DOOR_M, HITS, MAX_SPAN, Zone, area_from_walls,
                                      count_low, score, seen_share, true_count)
from pluggybot.home import world as home
from pluggybot.legs import survey as sv
from pluggybot.lifecycle import QUAD_HOME, world_config
from pluggybot.mapping.occupancy_grid import OccupancyGrid

RES = 0.05


def _run(gen):
  """Drive a routine that steps no physics to its result."""
  try:
    while True:
      next(gen)
  except StopIteration as stop:
    return stop.value


def _rooms(gap_m: float):
  """Two 3.5 m rooms side by side, all known floor, the wall between them
  open `gap_m` wide: (occupied, free) as a map holds them, [iy, ix]."""
  occ = np.zeros((80, 141), dtype=bool)
  occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
  occ[:, 70] = True
  gap = int(round(gap_m / RES))
  occ[40 - gap // 2:40 - gap // 2 + gap, 70] = False
  return occ, ~occ


# ---- 1. the area -----------------------------------------------------------------


def test_a_doorway_closes_the_area_and_a_wider_gap_does_not():
  """The garden's floor runs on through the living room's doorway: a fill
  of the free floor would survey the house. Every doorway in the house is
  1.0 m; `DOOR_M` is the widest gap that still closes an area."""
  occ, free = _rooms(1.0)
  area = area_from_walls(occ, free, (30, 40), RES)
  assert area[40, 30] and area[40, 65], "the area is its own room's floor"
  assert not area[40, 110], "the area ran on through a doorway"
  assert 1.0 < DOOR_M < 1.6
  occ, free = _rooms(1.6)
  assert area_from_walls(occ, free, (30, 40), RES)[40, 110], "an opening closed the area"


def test_an_area_round_no_floor_is_empty():
  occ, free = _rooms(1.0)
  assert not area_from_walls(occ, free, (70, 10), RES).any(), "seeded in a wall"
  assert not area_from_walls(occ, free, (500, 10), RES).any(), "seeded off the map"


# ---- 2. what counts --------------------------------------------------------------


def test_an_object_is_plant_sized_inside_the_area_and_a_diagonal_pair_is_one():
  area = np.ones((60, 60), dtype=bool)
  low = np.zeros_like(area)
  low[10:12, 10:12] = True                 # an 80 mm stalk, two cells across
  low[20, 20] = low[21, 21] = True         # one stalk seen from two bearings
  low[30:45, 30:45] = True                 # 0.75 m across: a couch, not a plant
  assert MAX_SPAN < 0.75
  found = count_low(low, area, RES)
  assert sorted((round(x), round(y)) for x, y, _ in found) == [(10, 10), (20, 20)]
  area[:, :15] = False                     # ...and outside the area, nothing
  assert [(round(x), round(y)) for x, y, _ in count_low(low, area, RES)] == [(20, 20)]


# ---- 3. coverage -----------------------------------------------------------------


def test_coverage_is_the_share_of_the_area_seen():
  """A right answer off a third of the garden is a lucky guess, and the
  score has to be able to tell: the share of the AREA's floor, so floor
  seen elsewhere in the map buys none of it."""
  area = np.zeros((10, 10), dtype=bool)
  area[:, :8] = True
  seen = np.zeros_like(area)
  seen[:, :4] = True
  seen[:, 9] = True                        # outside the area
  assert seen_share(seen, area) == pytest.approx(0.5)
  assert seen_share(seen, np.zeros_like(area)) == 0.0


def test_the_score_is_code_and_says_when_it_is_wrong():
  assert score(4, 4, 1.0)["correct"] is True
  wrong = score(3, 4, 0.5)
  assert wrong["correct"] is False and wrong["error"] == -1


# ---- 4. the truth ----------------------------------------------------------------


def test_the_truth_comes_out_of_the_world_over_both_garden_rectangles():
  """Hidden ground truth, hidden from the ROBOT: read off the model, as a
  query rather than the number 4, so a fifth plant re-scores the task. The
  area the robot finds runs round the house's south side, so the truth is
  over both rectangles the garden is (`census_zones`)."""
  model = mujoco.MjModel.from_xml_path(world_config(QUAD_HOME)["model"])
  zones = [Zone.from_meta(z) for z in world_config(QUAD_HOME)["census_zones"]]
  assert {z.name for z in zones} == {"garden", "garden_south"}
  assert true_count(model, zones) == len(home.PLANTS) == 4
  living = Zone.from_meta(next(z for z in home.ZONES if z["name"] == "living"))
  assert true_count(model, [living]) == 0


# ---- 5. the survey's own layer -----------------------------------------------------


def test_a_survey_never_takes_a_stalk_back_off_the_floor_seen_round_it():
  """The planner's layer takes a cell back for every floor point seen in
  it, and a stalk seen from one side was erased from another: a survey
  counted 2 of 4 plants at 91 % coverage. A survey's layer only adds."""
  layer = sv.CensusLayer((10, 10))
  cell = np.array([np.ravel_multi_index((5, 5), (10, 10))])
  none = np.array([], dtype=np.int64)
  layer.fold(cell, none)
  layer.fold(none, cell)                   # the same cell seen as floor, later
  assert layer.hits[5, 5] == 1 and layer.seen[5, 5]


class _Surveyor(sv.AreaSurvey):
  """The survey's arithmetic on a 6 m map with no body: a 5 m room, walls
  round it, all of it known floor."""

  LIDAR_ORIGIN = (0.0, 0.0)

  def __init__(self):
    self._init_survey()
    self.grid = OccupancyGrid(0.0, 0.0, 6.0, 6.0, resolution=RES)
    self.grid.grid[...] = -2.0
    self.grid.grid[[10, 110], 10:111] = 3.0
    self.grid.grid[10:111, [10, 110]] = 3.0
    self.pose = (1.0, 1.0, 0.0)
    self.lidar = SimpleNamespace(max_range=8.0)
    self.census = sv.CensusLayer(self.grid.grid.shape)
    self.data = SimpleNamespace(time=0.0)

  def cell(self, x, y):
    ix, iy = self.grid.world_to_cell(x, y)
    return iy, ix

  def stalk(self, x, y, frames):
    """`frames` depth frames each finding something standing at (x, y)."""
    iy, ix = self.cell(x, y)
    flat = np.array([np.ravel_multi_index((iy, ix), self.grid.grid.shape)])
    for _ in range(frames):
      self.census.fold(flat, np.array([], dtype=np.int64))


def test_one_frame_is_a_stray_point_and_hits_frames_are_a_plant():
  s = _Surveyor()
  area = s.survey_area((3.0, 3.0))
  s.stalk(2.0, 2.0, HITS - 1)
  s.stalk(4.0, 4.0, HITS)
  objs, _ = s.survey_count(area, s.census)
  assert [(round(x, 1), round(y, 1)) for x, y, _ in objs] == [(4.0, 4.0)]


# ---- 6. what the LIDAR hit ---------------------------------------------------------


def test_what_a_near_lidar_return_hits_is_not_a_plant_and_a_far_one_proves_nothing():
  """The garden light's 40 mm pole: rays slipping past clear it from the
  fused map, so the depth camera's layer saw it standing under the scan
  plane and the survey counted five plants. What the LIDAR hits is taller
  than its plane -- but MEASURED, a walking torso's pitch put returns 7.4-
  8.0 m off onto two 0.30 m plants, and the count fell to two. So a return
  marks what it hit only within `LIDAR_TALL_M`."""
  s = _Surveyor()
  s.pose = (1.0, 3.0, 0.0)
  near, far = 1.0 + 2.0, 1.0 + sv.LIDAR_TALL_M + 0.5
  for x in (near, far):
    s.stalk(x, 3.0, HITS)
  s._on_scan(np.array([0.0, 0.0]), np.array([near - 1.0, far - 1.0]))
  objs, _ = s.survey_count(s.survey_area((3.0, 2.0)), s.census)
  assert [round(x, 1) for x, _, _ in objs] == [round(far, 1)]
  # ...and nothing at all is marked outside a survey
  s.census = None
  s._on_scan(np.array([0.0]), np.array([1.0]))


# ---- 7. the walk -------------------------------------------------------------------


class _Walker(_Surveyor):
  """`_Surveyor` with a body: a tag it has found, a walk that arrives at
  once, and a look round -- many depth frames -- that sees everything
  within `reach` and finds the stalks there in `HITS` of them."""

  def __init__(self, stalks=(), reach=sv.VANTAGE_SEES_M, found=True):
    super().__init__()
    self.census = None
    self.low = np.zeros(self.grid.grid.shape)
    self.stalks, self.reach, self.walks = list(stalks), reach, []
    tag = SimpleNamespace(x=0.6, y=3.0)
    self.places = SimpleNamespace(get=lambda t: tag if found else None,
                                  facing=lambda t: (0.0, "fixture"))

  def stand_routine(self):
    return
    yield

  def drive_to_routine(self, x, y, timeout=90.0, stop=None):
    self.walks.append((round(x, 2), round(y, 2)))
    self.pose = (x, y, 0.0)
    self.data.time += 10.0
    return True
    yield

  def _look_round_routine(self):
    ys, xs = np.mgrid[0:self.grid.grid.shape[0], 0:self.grid.grid.shape[1]]
    wx = self.grid.x_min + (xs + 0.5) * RES
    wy = self.grid.y_min + (ys + 0.5) * RES
    self.census.seen |= np.hypot(wx - self.pose[0], wy - self.pose[1]) <= self.reach
    for x, y in self.stalks:
      if math.hypot(x - self.pose[0], y - self.pose[1]) <= self.reach:
        self.stalk(x, y, HITS)
    self.data.time += 5.0
    return
    yield


def test_a_survey_walks_until_it_has_seen_enough_and_counts_what_it_saw():
  w = _Walker(stalks=[(2.0, 2.0), (3.5, 3.5), (4.0, 2.0)])
  counts = []
  rec = _run(w.survey_routine(46, 900.0, on_count=counts.append))
  assert rec["surveyed"] and rec["why"] == "surveyed", rec
  assert rec["coverage"] >= sv.SURVEY_COVERED and rec["count"] == 3
  assert rec["vantages"] == len(w.walks) > 1 and counts[-1] == 3
  assert w.walks[0] == (0.6 + sv.SEED_OUT_M, 3.0), "it starts out of the tag's face"
  assert w.census is None, "the layer outlived the survey"


def test_a_survey_says_why_it_stopped_short():
  assert _run(_Walker(found=False).survey_routine(46, 900.0))["why"] == "not found"
  rec = _run(_Walker().survey_routine(46, 0.0))
  assert rec["why"] == "out of time" and not rec["surveyed"]
  blind = _Walker(reach=0.2)                 # its looks see next to nothing
  rec = _run(blind.survey_routine(46, 100.0))
  assert rec["why"] == "out of time" and rec["coverage"] < sv.SURVEY_COVERED


def test_the_survey_layer_goes_off_when_a_walk_throws():
  w = _Walker()

  def broken(*a, **kw):
    raise RuntimeError("the walk fell over")
    yield
  w.drive_to_routine = broken
  with pytest.raises(RuntimeError):
    _run(w.survey_routine(46, 900.0))
  assert w.census is None


# ---- review of #407 ---------------------------------------------------------


def test_the_area_grows_back_over_floor_never_through_a_wall():
  """By distance alone, the area grown back from its middle took in floor
  behind a wall a cell thick. Grown a cell a step over the free floor, it
  stops at the wall."""
  occ = np.zeros((60, 60), dtype=bool)
  occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
  occ[1:59, 40] = True                     # a thin wall, the floor behind it free
  free = ~occ
  area = area_from_walls(occ, free, (20, 30), RES)
  assert area[30, 20] and area[30, 39]
  assert not area[:, 41:].any(), "the area ran on behind a wall"


def test_a_wall_the_survey_saw_closes_the_area_where_the_map_lost_it():
  """A robot walking the sidewalk past the garden cleared its fence from
  the map, and the area ran round the street loop -- 354 m2 for a garden
  of 87. Every return of the survey's own scans is a wall to its area."""
  s = _Surveyor()
  s.grid.grid[10:111, 110] = -2.0          # the east wall gone from the map...
  s.grid.grid[:, 111:] = -2.0              # ...and the floor beyond it known
  open_area = s.survey_area((3.0, 3.0))
  assert open_area[60, 115], "premise: the area runs out through the lost wall"
  s.census.lidar[10:111, 110] = True       # what this survey's near returns hit
  assert not s.survey_area((3.0, 3.0))[:, 111:].any()


def test_the_look_round_turns_a_full_circle_at_the_carrying_rate():
  """Carrying the LCD a body turns at `W_CARRY` (0.45 rad/s); the rover's
  spin, timed for 1 rad/s, turned it 145 deg. The survey turns until its
  own heading has gone round, inside `LOOK_ROUND_S`."""
  from pluggybot.behavior.navigation import W_SPIN
  w = _Walker()
  dt = 0.002

  def turn(v, rate):
    th = w.pose[2] + min(rate, 0.45) * dt
    w.pose = (w.pose[0], w.pose[1], math.atan2(math.sin(th), math.cos(th)))
    w.data.time += dt
    return
    yield
  w._nav_routine = turn
  _run(sv.AreaSurvey._look_round_routine(w))
  assert w.data.time * 0.45 >= 2 * math.pi - 1e-6
  assert w.data.time < sv.LOOK_ROUND_S
  assert 2 * math.pi / W_SPIN < w.data.time, "timed at the bare rate, it would stop short"


def test_an_area_the_walls_do_not_close_is_no_survey():
  """Seeded on no floor -- a tag whose facing put the seed in its fence --
  the area is empty, and an empty area once read as surveyed, nothing in
  it, nothing seen."""
  w = _Walker()
  w.places = SimpleNamespace(get=lambda t: SimpleNamespace(x=5.6, y=3.0),
                             facing=lambda t: (0.0, "fixture"))
  rec = _run(w.survey_routine(46, 900.0))
  assert rec["why"] == "no area" and not rec["surveyed"]


def test_a_census_that_never_found_its_garden_says_so():
  """A census whose `find` failed told History only "no count came back
  from garden"; it leads with what failed (#350)."""
  from test_body import stub_life
  from pluggybot.lifecycle import census_errand
  life = stub_life(QUAD_HOME)
  try:
    missed = life.run_errand(census_errand(QUAD_HOME, "garden"))
    assert not missed["verdict"]["ok"]
    assert missed["verdict"]["reason"].startswith("never found garden: did not find tag 46")
  finally:
    life.body.close()


def test_only_a_scan_laid_into_the_map_reaches_the_survey():
  """The survey's layer is laid where the map is: after the match, at the
  pose the scan was laid from, and never from a scan the matcher refused."""
  from pluggybot.navigator import Navigator
  seen = []
  fake = SimpleNamespace(
    data=SimpleNamespace(time=1.0), _next_scan=0.0, backoff_until=9e9,
    lidar=SimpleNamespace(max_range=8.0, scan_split=lambda d: (np.zeros(2), np.ones(2),
                                                              np.zeros(0), np.zeros(0))),
    level=lambda: True, _match=lambda a, r: "fit", LIDAR_ORIGIN=(0.0, 0.0), pose=(0, 0, 0),
    grid=SimpleNamespace(update=lambda *a, **kw: seen.append("map")),
    _on_scan=lambda a, r: seen.append("survey"), _front_blocked=lambda a, r: False)
  fake.matcher = SimpleNamespace(fuses=lambda m, t: False, fused=lambda p, t: None)
  Navigator._scan_step(fake)
  assert seen == [], "a refused scan reached the survey"
  fake._next_scan, fake.matcher.fuses = 0.0, (lambda m, t: True)
  Navigator._scan_step(fake)
  assert seen == ["map", "survey"]
