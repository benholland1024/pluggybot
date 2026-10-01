"""A robot resting across the other's way makes way (issue #415; Ben's
decision, 2026-09-30: the resting robot moves).

Resting by reflex is not a fall (#365), so another robot's drive waited on
a resting body or gave up at its disc -- in a doorway, in 0 s -- and the
resting robot never moved: nothing told it it was in the way. Now the drive
asks (`Navigator._ask_way`, the pair's `Body.ask_way`) and waits, and the
resting body stands and steps aside beneath whatever it holds
(`legs/way.py`). Each rule is pinned here as cheaply as it fails: the
drive on a stub that walks, the spot and the step on a quadruped's mission
with nothing stepped, the lifecycle's half on the pair as built. The flown
scenes are `scripts/make_way_spike.py` (SimNotes, "A robot resting across
the other's way").
"""

import math
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import navigator as nav
from pluggybot import tick
from pluggybot.body import KeepClear
from pluggybot.legs import body as qb
from pluggybot.legs import world as lw
from pluggybot.legs.body import QuadMission
from pluggybot.lifecycle import QUAD_HOME, world_config
from pluggybot.navigator import MAKE_WAY_WAIT_S, OTHER_WAIT_S, STAGNATION_S, gave_up

STEPPER = SimpleNamespace(step=lambda *a: None)
BOUNDS = world_config(QUAD_HOME)["grid_bounds"]
GOAL = (5.0, 0.0)


# ---- 1. the drive asks, and waits while the robot makes way -------------------------


class _Drive:
  """`drive_to_routine`'s own `self` with the planner and the legs stubbed:
  every command is 0.1 s; the way to `GOAL` is cut until `opens_at` (sim s,
  None never), and then the robot walks straight there."""
  drive_to_routine = nav.Navigator.drive_to_routine
  _drove = nav.Navigator._drove
  _at_stand_in = nav.Navigator._at_stand_in
  _other_in_the_way = nav.Navigator._other_in_the_way
  _ask_way = nav.Navigator._ask_way
  _bodies = nav.Navigator._bodies
  peer_on_the_goal = nav.Navigator.peer_on_the_goal
  _cells = nav.Navigator._cells
  _left = nav.Navigator._left
  PROGRESS_ALONG_ROUTE = PROGRESS_MAP_GROWTH = False
  NEW_ROUTE_M = nav.Navigator.NEW_ROUTE_M
  BACKOFF_V, PEER_CLEARANCE_M = QuadMission.BACKOFF_V, QuadMission.PEER_CLEARANCE_M
  OTHER_ROBOT_CELLS, DOWN_ROBOT_CELLS = QuadMission.OTHER_ROBOT_CELLS, QuadMission.DOWN_ROBOT_CELLS
  pressing = False

  def __init__(self, others, says=True, opens_at=None, plans=None):
    self.data = SimpleNamespace(time=0.0)
    self.grid = SimpleNamespace(resolution=0.05)
    self.pose = (0.0, 0.0, 0.0)
    self.others = list(others)
    self.backoff_until, self.peer_holds = 0.0, 0
    self.last_drive, self._stand_in = None, None
    self.opens_at, self.plans = opens_at, plans
    self.asked: list = []
    self.ask_way = (lambda root, route: self.asked.append((root, list(route))) or says)

  def _open(self):
    return self.opens_at is not None and self.data.time >= self.opens_at

  def _plan_to(self, wx, wy):
    if self.plans is not None:
      return list(self.plans)
    return [(wx, wy)] if self._open() else None

  def _route_cut_by_others(self, wx, wy):
    return not self._open()

  def _route_past(self, root, wx, wy):
    return [(wx, wy)]

  def peer_sighting(self):
    return None

  def _known_cells(self):
    return 0

  def _nav_routine(self, v, w):
    self.data.time += 0.1
    if self.plans is None:
      self.pose = (GOAL[0], GOAL[1], 0.0)
    yield v, w

  def _drive_routine(self, seconds, v, w):
    self.data.time += seconds
    yield v, w


def _resting(x=2.0, y=0.0, **kw):
  return lambda: KeepClear(x, y, root="r2_pluggybot", **({"resting": True} | kw))


def test_a_drive_whose_way_a_resting_robot_cuts_asks_it_aside_and_waits():
  """The issue's failure: the way cut by a robot lying down to rest, and
  the drive gave up at once ("peer", 0 s). It asks the robot off its way --
  from where it stands, along the plan it would walk without it -- waits
  while the robot moves, and walks on once the way is open."""
  drive = _Drive([_resting()], opens_at=7.0)
  arrived = tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0))
  assert arrived and drive.data.time >= 7.0, drive.last_drive
  root, route = drive.asked[0]
  assert root == "r2_pluggybot" and route == [(0.0, 0.0), GOAL]
  assert drive.last_drive["askedWay"] == ["r2_pluggybot"]


def test_the_wait_for_a_robot_making_way_is_bounded():
  """A robot that says yes and never clears the way is waited for
  `MAKE_WAY_WAIT_S` from its first yes, and then given up on as before --
  never for the whole of the walk's patience."""
  drive = _Drive([_resting()])
  arrived = tick.run(STEPPER, drive.drive_to_routine(*GOAL, 300.0))
  rec = drive.last_drive
  assert not arrived and rec["why"] == "peer"
  assert MAKE_WAY_WAIT_S <= rec["seconds"] <= MAKE_WAY_WAIT_S + 2 * OTHER_WAIT_S, rec


@pytest.mark.parametrize("who", ["fallen", "standing", "nameless", "refuses"])
def test_a_robot_that_is_not_resting_or_will_not_move_is_given_up_on_at_once(who):
  """Asked only: a robot lying down TO REST, which the pair names. Fallen,
  it gets up by itself or not at all (#365); standing, it moves on its own;
  a robot that says no is a way still cut. Each ends as before, at once."""
  others = {"fallen": [_resting(down=True)],
            "standing": [_resting(resting=False)],
            "nameless": [lambda: KeepClear(2.0, 0.0, resting=True)],
            "refuses": [_resting()]}[who]
  drive = _Drive(others, says=False)
  arrived = tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0))
  assert not arrived and drive.last_drive["why"] == "peer"
  assert drive.last_drive["seconds"] == 0.0
  assert bool(drive.asked) is (who == "refuses")


def test_a_stagnated_drive_asks_a_resting_robot_near_it_and_none_far_off():
  """The drive's other wait: no progress for `STAGNATION_S` with a robot
  near it or its goal, which it waited on and which, resting, would never
  move. It asks that one first; one across the room is none of its
  business."""
  near = _Drive([_resting(0.8, 0.0)], plans=[GOAL])
  tick.run(STEPPER, near.drive_to_routine(*GOAL, STAGNATION_S + 5.0))
  assert [root for root, _ in near.asked] == ["r2_pluggybot"]
  # ...waiting on a robot standing beside it, with one resting far off
  standing = lambda: KeepClear(0.8, 0.0, root="pluggybot_3")  # noqa: E731
  far = _Drive([standing, _resting(2.5, 3.0)], plans=[GOAL])
  tick.run(STEPPER, far.drive_to_routine(*GOAL, STAGNATION_S + 5.0))
  assert far.data.time >= STAGNATION_S + 5.0, "the premise: it waited on the near one"
  assert far.asked == []


def test_a_resting_robot_is_said_to_rest_where_it_is_in_the_way():
  """The words of the issue's failure: a robot resting in the doorway
  3.8 m off read "standing 3.5 m from where it was going" -- nearer the goal
  than the robot, and nowhere on it. On the goal is inside its disc."""
  drive = _Drive([_resting(2.0, 0.0)], says=False)
  drive.pose = (-1.6, 0.0, 0.0)
  tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0))
  rec = drive.last_drive
  assert (rec["peerAt"], rec["peerM"], rec.get("peerRests")) == ("here", 3.6, True)
  assert gave_up(rec, "Rowan").endswith("(Rowan lying down to rest in the way, 3.6 m off)")
  on = _Drive([_resting(GOAL[0] + 0.3, 0.0)], says=False)
  tick.run(STEPPER, on.drive_to_routine(*GOAL, 60.0))
  assert (on.last_drive["peerAt"], on.last_drive["peerM"]) == ("goal", 0.3)


# ---- 2. the resting body: who makes way, where to, and the step ------------------


@pytest.fixture(scope="module")
def house():
  return lw.home_spec().compile()


#: A wall at x 2 with a 1 m door (y 0.25..1.25) in the open floor of a map
#: laid free everywhere else; the resting robot lies in the door, the asker
#: walks from the west of it to the east.
DOOR = (2.0, 0.75)
ASKER = (0.5, 0.75)
EAST = (4.0, 0.75)


def _door_map(m):
  g = m.grid
  g.grid[:] = -5.0
  wall = g.world_to_cell(DOOR[0], 0.0)[0]
  lo, hi = g.world_to_cell(0.0, 0.25)[1], g.world_to_cell(0.0, 1.25)[1]
  g.grid[:, wall:wall + 2] = 5.0
  g.grid[lo:hi, wall:wall + 2] = -5.0


def _mission(house, at=DOOR, posture=qb.LYING):
  m = QuadMission(house, mujoco.MjData(house), realtime=False, grid_bounds=BOUNDS)
  m.start_at(*at, 0.0)
  _door_map(m)
  m.others = [lambda: KeepClear(*ASKER, root="pluggybot")]
  m.posture = posture
  m._plan_memo = None
  return m


def _route():
  return [ASKER, DOOR, EAST]


def _off(route, x, y) -> float:
  """How far (x, y) is from a polyline."""
  best = math.inf
  for (ax, ay), (bx, by) in zip(route[:-1], route[1:]):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    best = min(best, math.hypot(x - ax - t * dx, y - ay - t * dy))
  return best


def test_the_spot_is_off_the_way_on_floor_it_has_seen_and_near(house):
  """Out of the disc the asker keeps round it, by `aside_clear_m` of every
  point of the way, so the way it sent is open whatever it plans next; on
  floor it has seen; the nearest by its own walk -- here, a step through
  the door and to one side."""
  m = _mission(house)
  try:
    x, y = m.aside_spot(_route())
    clear = (m.OTHER_ROBOT_CELLS + 3) * m.grid.resolution
    assert m.aside_clear_m == pytest.approx(clear)
    assert _off(_route(), x, y) >= clear - 1e-6
    assert math.dist((x, y), DOOR) < 1.5, "the nearest, not anywhere off the way"
    gx, gy = m.grid.world_to_cell(x, y)
    assert m.grid.grid[gy, gx] < 0
  finally:
    m.close()


def _corridor(m, rooms=()):
  """A corridor 1.2 m wide, x -1..6, and rooms off it to the north, each
  `(x0, x1, logodds)`: seen (-5) or not (0); walls everywhere else."""
  g = m.grid
  g.grid[:] = 5.0
  row = lambda y: g.world_to_cell(0.0, y)[1]   # noqa: E731
  col = lambda x: g.world_to_cell(x, 0.0)[0]   # noqa: E731
  g.grid[row(0.0):row(1.2), col(-1.0):col(6.0)] = -5.0
  for x0, x1, seen in rooms:
    g.grid[row(1.2):row(3.2), col(x0):col(x1)] = seen


@pytest.mark.parametrize("room", ["unseen", "past the asker"])
def test_the_spot_is_never_on_floor_it_has_not_seen_nor_past_the_asker(house, room):
  """The robot lies in a corridor the way runs along; the only floor off
  the way is a room it has not SEEN beside it -- an aside into the unknown
  is a walk into a wall -- or one it has, behind the asker, whose disc it
  may not walk through. Neither is an aside: there is none."""
  m = _mission(house, at=(3.0, 0.6))
  try:
    _corridor(m, [(3.5, 4.5, 0.0)] if room == "unseen" else [(0.0, 1.0, -5.0)])
    m.others = [lambda: KeepClear(1.5, 0.6, root="pluggybot")]
    assert m.aside_spot([(1.5, 0.6), (5.5, 0.6)]) is None
  finally:
    m.close()


def test_there_is_no_spot_in_a_corridor_the_way_runs_the_length_of(house):
  """A corridor 1.2 m wide that the way runs along: nowhere in it is far
  enough off the way, so there is no aside, the robot says no, and the
  asker gives up as it did."""
  m = _mission(house, at=(2.0, 0.6))
  try:
    m.grid.grid[:] = 5.0
    lo, hi = m.grid.world_to_cell(0.0, 0.0)[1], m.grid.world_to_cell(0.0, 1.2)[1]
    x0, x1 = m.grid.world_to_cell(-1.0, 0.0)[0], m.grid.world_to_cell(9.0, 0.0)[0]
    m.grid.grid[lo:hi, x0:x1] = -5.0
    m.others = [lambda: KeepClear(0.5, 0.6, root="pluggybot")]
    way = [(0.5, 0.6), (8.0, 0.6)]
    assert m.aside_spot(way) is None
    assert m.make_way(way, "pluggybot") is False and m.making_way is None
  finally:
    m.close()


@pytest.mark.parametrize("busy", ["standing", "docked", "working", "driving", "moving"])
def test_only_a_resting_body_free_to_makes_way(house, busy):
  """Lying down to rest and nothing else: not standing (it moves on its
  own), not on the dock, not mid arm-move or swap, not waiting inside a
  walk of its own (#395's head-on hold is that walk's to settle), not
  mid lie-down or stand-up."""
  m = _mission(house)
  try:
    if busy == "standing":
      m.posture = qb.STANDING
    elif busy == "moving":
      m._move = iter(())
    else:
      setattr(m, busy, 1 if busy == "driving" else True)
    assert m.make_way(_route(), "pluggybot") is False and m._aside is None
  finally:
    m.close()


def test_a_resting_body_asked_steps_aside_beneath_a_still_command(house):
  """The rule beneath whatever the body holds: asked, it says yes once (an
  ask while it moves is the same yes); under a STILL command it is stood up
  first -- the reflex's own motion rule -- and once standing walks the drive's
  law along a plan to the spot; there, STILL, and the record says so."""
  m = _mission(house)
  try:
    assert m.make_way(_route(), "pluggybot") and m.making_way == "pluggybot"
    aside = m._aside
    assert m.make_way(_route(), "pluggybot") and m._aside is aside
    m._before_step(qb.STILL)
    assert m.posture == qb.STANDING_UP, "a STILL command left it lying"
    m._move, m.posture = None, qb.STANDING
    cmd = m._aside_command(qb.STILL, float(m.data.time))
    assert qb.is_motion(cmd) and aside["waypoints"], cmd
    tx, ty = aside["to"]
    m._set_pose(tx, ty, 0.0)
    aside["waypoints"] = []
    assert m._aside_command(qb.STILL, float(m.data.time)) == qb.STILL
    assert m.making_way is None and m.asides == 1
    assert m.last_aside["why"] == "aside" and m.last_aside["by"] == "pluggybot"
    assert m.last_aside["from"] == DOOR and m.last_aside["at"] == (tx, ty)
  finally:
    m.close()


def test_its_own_walk_and_a_fall_end_the_step_aside(house):
  """A command that moves is the body's own business, and takes over at
  once, unchanged; a fall ends it too -- the get-up is the posture's."""
  m = _mission(house)
  try:
    walk = (0.4, 0.0, 0.0)
    m.make_way(_route(), "pluggybot")
    assert m._aside_command(walk, float(m.data.time)) == walk
    assert m.last_aside["why"] == "its own walk" and m._aside is None
    m.make_way(_route(), "pluggybot")
    m.posture = qb.GETTING_UP
    assert m._aside_command(qb.STILL, float(m.data.time)) == qb.STILL
    assert m.last_aside["why"] == "fell" and m.asides == 2
  finally:
    m.close()


# ---- 3. the pair's channel, a death, and what is said ---------------------------


def test_the_pair_asks_through_the_other_lifecycle_and_says_what_it_did(tmp_path):
  """Wired by the pair (one line): the others say who they are and whether
  they rest, a drive's ask reaches the other robot's lifecycle, which
  refuses when it is dead and otherwise hands it to its body, says so as
  it stands, remembers the step once it is over -- the body moved and the
  mind did not decide it -- and holds a restart's save off meanwhile."""
  from pluggybot.continuation import Keeper
  from pluggybot.pair import build_pair
  me, peer = build_pair(QUAD_HOME, errands=("none", "none"),
                        thoughts_root=str(tmp_path / "thoughts"))
  try:
    peer.body.start_at(*DOOR, 0.0)
    _door_map(peer.body.mission)
    me.body.start_at(*ASKER, 0.0)
    keep = me.body.others[0]()
    assert (keep.root, keep.resting) == (peer.root, False)
    peer.body.mission.posture = qb.LYING
    assert me.body.others[0]().resting
    peer.dead = {"cause": "flat"}
    assert me.body.ask_way(peer.root, _route()) is False
    peer.dead = None
    assert me.body.ask_way(peer.root, _route()) is True
    assert peer.body.making_way == me.root
    assert peer.status.startswith(f"MAKE WAY: lying across {me.robot_name}'s way")
    assert Keeper([me, peer], path=str(tmp_path / "w.npz")).busy()
    peer.body.mission._end_aside("aside", qb.STILL)
    peer._aside_step()
    peer._aside_step()
    lines = [ln for ln in peer.thoughts.read("History.md").splitlines()
             if "aside" in ln]
    assert len(lines) == 1 and f"aside for {me.robot_name}, whose way I was lying across" in lines[0]
  finally:
    for life in (me, peer):
      life.body.close()
