"""Guards for the rack pose a second tool fetch is computed from.

Found by issue #22's acceptance mission, which was the first thing in this
repo to fetch the SAME tool twice with a charge cycle in between. It failed:
`SWAP_PICK FAILED`, the module still on its bracket, ~100 s and 42 % of the
pack spent grinding into the rack.

The chain, measured end to end:

  1. The trip to the charge bay drives the robot BEHIND the rack, so the
     mapped free space around it stops being "a wall with open space on one
     side" and becomes the free-standing partition `wall_normal` has always
     warned about. The two sides nearly cancel and the direction that
     survives is leftover noise -- ~20 deg of it.
  2. `RackFinder.estimate` adopted that as the rack's FACING. At the 0.63 m
     standoff radius, 20 deg moves the hand-off pose ~0.2 m and swings the
     approach heading into the rack's flank. Bay C's standoff went from
     (+0.173, -1.441) -- correct -- to (+0.026, -1.597).
  3. From there the bay tag ranged 1.09 m, so `_terminal_travel` asked for a
     0.93 m creep: 4x nominal, and just under the 1.0 m a 20 s creep can
     cover at APPROACH_V, so it did not even stop early. The robot drove
     into the rack and wheel slip counted as odometry progress the whole way.

Two fixes, because there are two defects, and the second one is what turns a
bad belief into a robot shoving furniture:

  A FACING IS KEPT ONCE IT IS KNOWN. More sightings improve a landmark's
  POSITION; they do not improve a facing whose input is the shape of the
  mapped free space. A rack does not turn round.

  A TERMINAL CREEP REFUSES AN IMPLAUSIBLE RANGE. It is a 0.22 m move ending
  in a +/-11 mm capture window; asked for four times that, it is not a creep.
"""

import math

import pytest

from pluggybot.rack.localize import MIN_FACING_CONF, RackFinder, RackPose
from pluggybot.mission.mission import (
  NOMINAL_TRAVEL, TRAVEL_SLACK, bay_standoff, plausible_travel,
)
from pluggybot.mapping.landmarks import wall_normal, wall_normal_conf
from pluggybot.mapping.occupancy_grid import OccupancyGrid

RACK_X, RACK_Y = 0.5, -1.98


def grid_with(free) -> OccupancyGrid:
  """A grid whose only free cells are the ones `free(dx, dy)` accepts,
  sampled over a metre around the rack."""
  g = OccupancyGrid(-3, -3, 7, 7, resolution=0.05)
  g.grid[:] = 0.0                     # unknown-ish; FREE_THRESH is below 0
  n = int(1.0 / 0.05)
  for iy in range(-n, n + 1):
    for ix in range(-n, n + 1):
      dx, dy = ix * 0.05, iy * 0.05
      if free(dx, dy):
        cx, cy = g.world_to_cell(RACK_X + dx, RACK_Y + dy)
        g.grid[cy, cx] = -5.0         # confidently free
  return g


# ---- the measurement: free space on both sides is not a measurement ---------


def test_a_wall_is_well_conditioned_and_a_free_standing_rack_is_not():
  """The number the fix hangs on, and the two cases it has to separate.

  A wall blocks about half the circle, and the sum of a half-circle of unit
  vectors is 2/pi ~= 0.64 of its count -- so a rack against a wall reads
  around that, and one with the robot's own tracks behind it reads near zero.
  """
  against_wall = grid_with(lambda dx, dy: dy > 0.10)
  _, _, wall_conf = wall_normal_conf(against_wall, RACK_X, RACK_Y,
                                     fallback=(0.0, 1.0))
  both_sides = grid_with(lambda dx, dy: abs(dy) > 0.10)
  nx, ny, open_conf = wall_normal_conf(both_sides, RACK_X, RACK_Y,
                                       fallback=(0.0, 1.0))
  assert wall_conf > MIN_FACING_CONF, \
    f"a rack against a wall must be believable: {wall_conf:.3f}"
  assert open_conf < MIN_FACING_CONF, \
    f"free space on both sides must not be: {open_conf:.3f}"
  assert wall_conf > open_conf * 3, "the two cases are not actually separated"
  # ...and the direction that survives the cancelling case is junk, which is
  # the whole point: it is not merely less precise, it is not an answer.
  assert abs(math.degrees(math.atan2(ny, nx)) - 90.0) > 5.0


def test_wall_normal_still_answers_exactly_what_it_used_to():
  """The confidence is ADDITIVE. Every existing caller -- the outlet
  landmarks of milestone 5 -- must get the identical direction back."""
  g = grid_with(lambda dx, dy: dy > 0.10)
  assert wall_normal(g, RACK_X, RACK_Y, fallback=(0.0, 1.0)) == \
    wall_normal_conf(g, RACK_X, RACK_Y, fallback=(0.0, 1.0))[:2]


# ---- a facing is kept once it is known ---------------------------------------


class FakeLandmark:
  n_sightings = 9

  def __init__(self, x, y, seen_from):
    self.x, self.y = x, y
    self.seen_from_x, self.seen_from_y = seen_from


def finder_seeing(x, y, seen_from=(0.5, -1.0)) -> RackFinder:
  """A RackFinder with a confirmed landmark and no camera anywhere near it."""
  finder = RackFinder.__new__(RackFinder)      # no MuJoCo model needed
  finder.landmarks = type("S", (), {"landmarks": [FakeLandmark(x, y, seen_from)]})()
  finder.facing = None
  return finder


def test_driving_behind_the_rack_does_not_turn_it_round():
  """THE regression test. A good look, then a bad one, and the good answer
  survives -- because the rack did not move and the second look is noise.

  Shown to fail without the fix: drop the `conf >= MIN_FACING_CONF` guard in
  `RackFinder.estimate` and the believed yaw swings with the free space,
  which is what put bay C's standoff 0.2 m off the approach axis.
  """
  finder = finder_seeing(RACK_X, RACK_Y)
  good = finder.estimate(grid_with(lambda dx, dy: dy > 0.10))
  assert good is not None
  # 8 deg, not 1: this grid is a 50 mm-cell cartoon of a wall and the
  # sampled ring quantises. What is under test is that the SECOND look does
  # not move it, and a synthetic grid precise enough to argue about degrees
  # would be testing the fixture.
  assert abs(good.yaw - math.pi / 2) < math.radians(8), \
    f"the well-conditioned look was already wrong: {math.degrees(good.yaw):.1f}"
  after_charging = finder.estimate(grid_with(lambda dx, dy: abs(dy) > 0.10))
  assert abs(after_charging.yaw - good.yaw) < 1e-9, (
    f"a drive behind the rack rotated it by "
    f"{math.degrees(abs(after_charging.yaw - good.yaw)):.1f} deg")


def test_the_first_facing_is_adopted_however_it_reads():
  """A robot that has only ever seen the rack badly still has to start
  somewhere -- `None` is not a pose, and refusing every look would leave the
  boot-time prior in place with nothing able to correct it."""
  finder = finder_seeing(RACK_X, RACK_Y)
  first = finder.estimate(grid_with(lambda dx, dy: abs(dy) > 0.10))
  assert first is not None and finder.facing is not None


def test_a_twenty_degree_error_is_what_it_costs():
  """Why a facing is worth guarding: the number that made this a bug rather
  than a rounding error. Bay C sits 0.375 m off the rack's axis, so its
  standoff is ~0.63 m out, and a rotation there is a translation here."""
  true = RackPose(RACK_X, RACK_Y, math.pi / 2)
  rotated = RackPose(RACK_X, RACK_Y, math.pi / 2 + math.radians(20))
  sx, sy, _ = bay_standoff(0.375, true)
  rx, ry, _ = bay_standoff(0.375, rotated)
  assert math.hypot(rx - sx, ry - sy) > 0.15, \
    "20 deg of yaw must move the hand-off pose enough to matter"


# ---- a terminal creep refuses an implausible range ---------------------------


@pytest.mark.parametrize("travel,ok", [
  (NOMINAL_TRAVEL, True),
  (NOMINAL_TRAVEL + TRAVEL_SLACK * 0.9, True),     # arrived 80 mm short: fine
  (NOMINAL_TRAVEL - TRAVEL_SLACK * 0.9, True),     # ...or 80 mm long
  (0.9307, False),                                 # the measured failure
  (-0.05, False),                                  # already past the plane
  (1.0, False),
])
def test_only_a_hand_off_pose_worth_of_travel_is_believed(travel, ok):
  assert plausible_travel(travel) is ok


def test_the_creep_that_drove_into_the_rack_is_refused():
  """The measured failure, as a number: a bay tag read at 1.09 m.

  0.93 m of creep at APPROACH_V is 18.6 s, against `_drive_until`'s 20 s
  timeout -- so the drive did not even end early. It ground into the rack for
  the full 20 s, twice, with wheel slip counting as odometry progress.

  Shown to fail without the fix: delete the `plausible_travel` checks in
  `_terminal_travel` and this range is used verbatim.
  """
  measured_travel = 0.9307    # m, from a bay tag ranged at 1.09 m
  assert not plausible_travel(measured_travel)
  assert measured_travel / NOMINAL_TRAVEL > 4.0, "4x nominal, as measured"
  # ...and what replaces it is the blind travel every swap used before there
  # were tags -- short of the peg from a wrong pose, and harmless.
  assert plausible_travel(NOMINAL_TRAVEL)
  assert NOMINAL_TRAVEL * 1.0 < 0.25


class FakeTags:
  """`bay_range` / `rack_range` on demand, with no renderer anywhere."""

  def __init__(self, bay=None, rack=None):
    self.bay, self.rack = bay, rack

  def bay_range(self, data, tag_id):
    return self.bay

  def rack_range(self, data):
    return self.rack


def mission_at(pose, bay=None, rack=None, vertex_ahead=0.165):
  """Enough HubMission for `_terminal_travel` and nothing else.

  Hand-built rather than a real mission: what is under test is ten lines of
  source selection, and standing a physics world up to ask them would make a
  0.2 s test a 30 s one.
  """
  from pluggybot.mission.mission import HubMission

  # `pose` is a read-only property on the real class, so the stand-in is a
  # subclass that overrides it rather than an instance with an attribute set.
  class Standing(HubMission):
    pose = property(lambda self: pose)

    def _vertex_ahead_of_camera(self):
      return vertex_ahead

  fake = Standing.__new__(Standing)
  fake.tags = FakeTags(bay, rack)
  fake.data = None
  fake.rack = RackPose(RACK_X, RACK_Y, math.pi / 2)
  fake.travel_source = ""
  return fake


def test_a_bay_range_from_a_hand_off_pose_is_used():
  """The fix must not cost the thing it is guarding: the bay tag is still
  the best source, and a normal reading still wins."""
  # A hand-off pose is STANDOFF from the station, so the tag reads about
  # that less the hang plane -- somewhere in the plausible band.
  m = mission_at((RACK_X - 0.375, RACK_Y + 0.45, -math.pi / 2), bay=0.38)
  travel = m._terminal_travel(0.375)
  assert plausible_travel(travel) and m.travel_source == "bay"


def test_an_implausible_bay_range_falls_through_instead_of_being_driven():
  """THE regression test for the creep that drove into the rack.

  The measured failure, reproduced at the seam: the bay tag decodes -- from a
  metre away, obliquely, because the believed rack was rotated -- and says
  1.09 m. Believed, that is a 0.93 m creep into a rack 0.38 m ahead.

  Shown to fail without the fix: drop the `plausible_travel` guards in
  `_terminal_travel` and this returns 0.93 m, which `swap_at_bay` hands
  straight to `HubSwap.pick` as its approach distance.
  """
  m = mission_at((RACK_X - 0.375, RACK_Y + 0.45, -math.pi / 2), bay=1.0916)
  travel = m._terminal_travel(0.375)
  assert plausible_travel(travel), f"a {travel:.3f} m creep was accepted"
  assert travel == pytest.approx(NOMINAL_TRAVEL, abs=0.15)


def test_with_no_plausible_source_at_all_it_creeps_the_nominal_distance():
  """A pose nothing can range from: no tag decodes and the odometry answer
  is a metre out too. The honest move is the blind travel every swap used
  before there were tags -- short of the peg, and harmless."""
  m = mission_at((RACK_X - 0.375, RACK_Y + 1.60, -math.pi / 2))
  assert m._terminal_travel(0.375) == NOMINAL_TRAVEL
  assert m.travel_source == "nominal"


def test_odometry_still_answers_when_no_tag_decodes():
  """The third source is unchanged, and still used: a bay whose marker is
  occluded falls back to the believed rack plus dead reckoning."""
  m = mission_at((RACK_X - 0.375, RACK_Y + 0.45, -math.pi / 2))
  travel = m._terminal_travel(0.375)
  assert plausible_travel(travel) and m.travel_source == "odometry"


# ---- the press is not travel --------------------------------------------------


def test_a_pinned_robot_does_not_reckon_its_way_across_the_room(world_model):
  """THE root cause, and the cheapest possible statement of it.

  Hold the wheels against something immovable and dead reckoning integrates
  every slipping revolution as progress. The charge press does exactly that
  for minutes at a time -- measured on the home world, 828 mm of travel that
  never happened -- and every pose computed afterwards is in the wrong frame.

  Here the robot is simply commanded forward with `pinned` set, which is the
  same arithmetic without needing a rack to shove: the wheels turn, the
  encoders advance, and the believed position must not.

  Shown to fail without the fix: drop the `held` save/restore in
  `HubSwap._step_once` and the reckoner walks.
  """
  import mujoco

  from pluggybot.rack.swap import HubSwap
  data = mujoco.MjData(world_model)
  swap = HubSwap(world_model, data)
  swap.reckoner.x, swap.reckoner.y = 1.0, 2.0
  left0 = float(data.qpos[swap.left_adr])

  swap.pinned = True
  swap._run(3.0, 0.12)                      # 3 s of commanded creep

  assert abs(float(data.qpos[swap.left_adr]) - left0) > 0.5, \
    "the wheels did not actually turn, so this test proves nothing"
  assert (swap.reckoner.x, swap.reckoner.y) == (1.0, 2.0), \
    (f"a pinned robot reckoned to "
     f"({swap.reckoner.x:.3f}, {swap.reckoner.y:.3f})")


def test_and_still_reckons_normally_once_it_is_free(world_model):
  """The flag is a hold, not an off switch: real travel is still counted, or
  the robot would come away from every charge with no idea where it is."""
  import mujoco

  from pluggybot.rack.swap import HubSwap
  data = mujoco.MjData(world_model)
  swap = HubSwap(world_model, data)
  swap.reckoner.x, swap.reckoner.y = 0.0, 0.0
  swap._run(3.0, 0.12)
  moved = math.hypot(swap.reckoner.x, swap.reckoner.y)
  assert moved > 0.05, f"free-running travel was not counted: {moved:.3f} m"
