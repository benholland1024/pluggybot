"""A long `drive_to` goes the way the house goes (issue #353).

Four robot-written `drive_to` legs never arrived on the deployed world,
each 6.6-9.1 m short: Rowan's `drive_to(22.0, 3.0)` from the house (three
times) and `drive_to(-9.7, -4.75)` from the hall. The planner plans over
the map only and the LIDAR reaches 8 m; `pick` and the cage programs
already walked the house's route (`zone_route`, `lab_route`), and the
robots' own procedures did not.

The rules, each pinned without a mission (docs/Testing.md):

  1. A goal one drive cannot plan to -- past the LIDAR's reach or off the
     map (`HubMission.in_sight`) -- is reached along `lifecycle.route_to`'s
     legs, in order, then the goal; a goal in sight is one drive, as before.
  2. A leg is a WAYPOINT (`steps.legs_routine`, `pick`'s travel too):
     reached on arrival or within `LEG_DONE_M`, passed by when another
     robot stands on it; one it cannot get near ends the verb, and the
     reason names the leg and why (#350's clause), asked of the leg.
  3. `route_to` joins the ways out at the house, drops the legs behind the
     robot and a last leg the goal stands in -- so the house's own legs,
     driven one `drive_to` at a time (the cage programs, `solutions.WEIGH`'s
     way home), compose to nothing and those programs are unchanged.
"""

import math
from types import SimpleNamespace

from pluggybot import tick
from pluggybot.activity.cage import ACTS
from pluggybot.lifecycle import WORKSHOP_ROUTE, cage_program, lab_route, route_to
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.mission.mission import HubMission
from pluggybot.procedure import steps as st

REACH = 8.0            # the LIDAR's, `perception.lidar.MAX_RANGE`


def _life(start, reach=REACH, fails_at=None):
  """A robot on an empty map: its planner answers only within `reach` of
  where it stands, so a drive inside arrives and one beyond gives up where
  it is. `fails_at` is a goal that gives up however near it is."""
  drives, asked = [], []
  mission = SimpleNamespace(pose=(*start, 0.0))

  def drive_to_routine(x, y, timeout):
    drives.append((x, y))
    px, py, _ = mission.pose
    if (x, y) == fails_at or math.hypot(x - px, y - py) > reach:
      return tick.result(False)
    mission.pose = (x, y, 0.0)
    return tick.result(True)
  mission.drive_to_routine = drive_to_routine
  mission.pose_xy = lambda: mission.pose[:2]
  mission.peer_on_the_goal = lambda x, y: None
  mission.in_sight = lambda x, y: math.hypot(x - mission.pose[0], y - mission.pose[1]) <= reach
  life = SimpleNamespace(mission=mission, world="home",
                         drive_why=lambda x, y: (asked.append((x, y)),
                                                 "the drive gave up (why)")[1])
  return life, drives, asked


def _drive_to(life, x, y):
  return tick.run(SimpleNamespace(_step_once=lambda *a: None),
                  st._drive_to(life, {"x": x, "y": y}))


# ---- 1. out of sight, the house's route; in sight, one drive ----------------


def test_rowans_drive_to_the_lab_goes_by_the_houses_route():
  """Rowan's `mass_check` step 2, from the hall: before, one drive that
  stopped short. Now the lab route's doorways in order, then the goal."""
  life, drives, _ = _life((-3.5, 1.0))
  r = _drive_to(life, 22.0, 3.0)
  assert drives == [*lab_route("home")[:4], (22.0, 3.0)], drives
  assert r["ok"] and r["route"] == [list(leg) for leg in lab_route("home")[:4]]


def test_rowans_drive_to_the_tower_goes_through_the_workshop_door():
  """`build_tower`'s `drive_to(-9.7, -4.75)` from the hall (8.4 m): the
  workshop's legs, less the one it stands on."""
  life, drives, _ = _life((-3.5, 1.0))
  assert _drive_to(life, -9.7, -4.75)["ok"]
  assert drives == [*WORKSHOP_ROUTE[1:], (-9.7, -4.75)]


def test_a_goal_in_sight_is_one_drive_as_before():
  """In sight is one drive, even where the house has a route: the rack to
  the garden is 7 m past the garden door's leg."""
  assert route_to("home", (0.5, -1.6), (7.0, 1.0)) == [lab_route("home")[0]]
  life, drives, _ = _life((0.5, -1.6))
  r = _drive_to(life, 7.0, 1.0)
  assert drives == [(7.0, 1.0)] and r["ok"] and "route" not in r


def test_in_sight_is_the_lidars_reach_and_the_map():
  """Past 8 m, or on a cell never seen, one plan cannot reach it; a seen
  cell -- free or a wall -- inside the reach, it can."""
  grid = OccupancyGrid(-10.0, -10.0, 10.0, 10.0)
  me = SimpleNamespace(pose=(0.0, 0.0, 0.0), grid=grid,
                       lidar=SimpleNamespace(max_range=REACH))
  cx, cy = grid.world_to_cell(3.0, 0.0)
  grid.grid[cy, cx] = -2.0                         # seen free
  wx, wy = grid.world_to_cell(0.0, 3.0)
  grid.grid[wy, wx] = 2.0                          # seen occupied
  far = grid.world_to_cell(8.5, 0.0)
  grid.grid[far[1], far[0]] = -2.0                 # seen, but past the reach
  assert HubMission.in_sight(me, 3.0, 0.0)
  assert HubMission.in_sight(me, 0.0, 3.0)
  assert not HubMission.in_sight(me, -3.0, 0.0)    # never seen
  assert not HubMission.in_sight(me, 8.5, 0.0)
  assert not HubMission.in_sight(me, 50.0, 0.0)    # off the grid


# ---- 2. a leg is a waypoint; one it cannot get near ends the verb ---------


def test_a_leg_that_gives_up_says_which_and_why():
  life, drives, asked = _life((-3.5, 1.0), fails_at=(16.0, 3.1))
  r = _drive_to(life, 22.0, 3.0)
  assert drives == lab_route("home")[:3]           # stopped at the failed leg
  assert not r["ok"] and asked == [(16.0, 3.1)]
  assert r["reason"] == ("did not arrive at (22, 3): on the house's route there, "
                         "the leg to (16, 3.1) -- the drive gave up (why), at (10.0, 3.1)")
  assert r["why"] == "the drive gave up (why)" and r["shortM"] == 12.0


def test_a_leg_is_a_waypoint_near_is_reached_and_a_robot_on_it_is_passed_by():
  """MEASURED on the pair (the ladder-A flights of this issue): the bench's
  route ended at the garden_2 gate, the drive's stand-in 0.2 m short of the
  leg, and the tower's in the hall, where Rowan stands by ON the workshop
  route's first leg -- arrival there is impossible by arithmetic
  (`peer_on_the_goal`, #313) and the drive held 60 s to learn it."""
  from pluggybot.lifecycle import LEG_DONE_M
  gate = lab_route("home")[2]

  def stops_short_of_the_gate(by):
    life, drives, _ = _life((-3.5, 1.0))
    inner = life.mission.drive_to_routine

    def drive(x, y, timeout):
      if (x, y) != gate:
        return inner(x, y, timeout)
      drives.append((x, y))
      life.mission.pose = (x - by, y, 0.0)         # the stand-in, and "no route"
      return tick.result(False)
    life.mission.drive_to_routine = drive
    return _drive_to(life, 22.0, 3.0), drives
  r, drives = stops_short_of_the_gate(0.2)
  assert r["ok"] and drives == [*lab_route("home")[:4], (22.0, 3.0)]
  # ...a leg ended farther out than LEG_DONE_M is not one it reached
  r, drives = stops_short_of_the_gate(LEG_DONE_M + 0.1)
  assert not r["ok"] and drives[-1] == gate
  # a leg the other robot stands on is not driven to: the next one is
  life, drives, _ = _life((0.5, -1.6))
  life.mission.peer_on_the_goal = lambda x, y: 0.0 if (x, y) == WORKSHOP_ROUTE[0] else None
  assert _drive_to(life, -9.7, -4.75)["ok"]
  assert drives == [*WORKSHOP_ROUTE[1:], (-9.7, -4.75)]


# ---- 3. the route itself ----------------------------------------------------


def test_the_ways_out_join_at_the_house():
  lab, shop = lab_route("home"), list(WORKSHOP_ROUTE)
  # the bench to the tower: the lab's way back, then the workshop's way out
  assert route_to("home", (26.9, 1.5), (-10.2, -4.75)) == [*lab[::-1], *shop]
  assert route_to("home", (-10.2, -4.75), (26.9, 1.5)) == [*shop[::-1], *lab]
  # along one way only the stretch between: the street to the lab
  assert route_to("home", (12.0, 3.0), (26.9, 1.5)) == lab[2:]
  # just inside the workshop door, the leg deeper in is behind the robot
  assert route_to("home", (-6.0, 0.0), (1.5, 0.5)) == [shop[0]]
  # in the house, or a world with no route written: nothing
  assert route_to("home", (1.5, 0.5), (3.0, 4.0)) == []
  assert route_to("room_hub", (0.0, 0.0), (5.0, 5.0)) == []


def test_the_houses_own_legs_compose_to_nothing():
  """Driven one `drive_to` at a time, the house's legs add no leg of their
  own: every step of a cage program, and `solutions.WEIGH`'s way home,
  route to nothing -- so ladder A's programs drive as they did."""
  lab = lab_route("home")
  for a, b in [*zip(lab, lab[1:]), *zip(lab[::-1], lab[::-1][1:])]:
    assert route_to("home", a, b) == [], (a, b)
  weigh_home = [(26.88, 1.5), (22.0, 3.0), (19.0, 3.1), (16.0, 3.1), (10.0, 3.1),
                (7.0, 1.2), (4.4, 0.7)]
  for a, b in zip(weigh_home, weigh_home[1:]):
    assert route_to("home", a, b) == [], (a, b)
  for act in ACTS:
    goals = [(s.args["x"], s.args["y"]) for s in cage_program("home", act).steps()
             if s.verb == "drive_to"]
    for a, b in zip([(0.5, -1.6), *goals], goals):
      assert route_to("home", a, b) == [], (act, a, b)
