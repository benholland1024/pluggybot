"""Walking into the unknown (issue #381's walking stage): each rule it rests
on, pinned as cheaply as it can be and still fail for the right reason --
the planner on synthetic grids, the drive's two new endings (the map still
growing is progress; a stop is an abort, not a give-up), the verb's
patience and its cap, and the interrupt a walk asks as it goes, recorded
by both runners. The walk flown whole is behind --endurance."""

import heapq
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pluggybot import navigator as nav
from pluggybot import tick
from pluggybot.body import KeepClear
from pluggybot.legs.body import QuadMission
from pluggybot.lifecycle import QUAD_HOME, procedure_outcome, world_facts
from pluggybot.mapping import optimistic as op
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.mission.mission import HubMission
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st

FREE, WALL, UNSEEN = -5.0, 5.0, 0.0
STEPPER = SimpleNamespace(step=lambda *a: None)
LEGS = world_facts(QUAD_HOME)


class _Nav(nav.Navigator):
  """The planner's own `self` over a synthetic map at 5 cm: no world, no
  body, nobody else -- `_plan_to` exactly as a body calls it."""

  def __init__(self, logodds, at, optimistic=True):
    rows, cols = logodds.shape
    self.grid = OccupancyGrid(0.0, 0.0, cols * 0.05, rows * 0.05, resolution=0.05)
    self.grid.grid[:] = logodds
    self._at = at
    self.others = []
    self.OPTIMISTIC = optimistic

  @property
  def pose(self):
    return (*self._at, 0.0)


def _room_with_a_door():
  """A known room, x 0..2 m, its east wall at x 2 m with a 1 m door low down
  (y 0.25..1.25), and nothing seen beyond it: 4 m x 6 m."""
  g = np.full((80, 120), UNSEEN)
  g[:, :40] = FREE
  g[:, 40:42] = WALL
  g[5:25, 40:42] = FREE
  return g


# ---- the planner -------------------------------------------------------------------


def test_a_goal_past_the_map_is_walked_to_by_the_door_it_has_seen():
  """THE FAILURE (the issue's "before"): the goal beyond the room's east
  wall, level with no door. Over mapped floor only, the plan is a stand-in
  pressed against the wall nearest the goal in a straight line -- the robot
  stood there, 11 of 16 zones from a fresh start. Through the unknown, it
  leaves by the door and runs on to the goal itself."""
  goal = (3.5, 3.5)
  old = _Nav(_room_with_a_door(), (1.0, 1.5), optimistic=False)
  stood = old._plan_to(*goal)
  assert old._stand_in is not None and stood[-1][0] < 2.0, "the premise: a stand-in at the wall"
  new = _Nav(_room_with_a_door(), (1.0, 1.5))
  path = new._plan_to(*goal)
  assert new._stand_in is None and math.dist(path[-1], goal) < 0.08
  through = [y for x, y in path if 1.95 <= x <= 2.15]
  assert through and all(0.25 <= y <= 1.25 for y in through), "out by the door"


def test_a_wall_it_has_seen_is_never_crossed_and_its_inflation_reaches_the_unknown():
  """Unknown is floor at a price; a known wall, grown by the body, is not
  floor on either side of it."""
  g = np.full((100, 130), UNSEEN)
  g[:60, 60:62] = WALL                                  # a wall, y 0..3 m
  cost = op.map_costs(g, 7)
  assert np.isinf(cost[30, 60 + 5]) and np.isinf(cost[30, 60 - 5]), "grown into the unknown"
  assert cost[30, 60 + 12] == op.UNKNOWN_COST and cost[80, 60] == op.UNKNOWN_COST
  path = _Nav(g, (1.0, 1.0))._plan_to(5.5, 1.0)
  crossing = [y for x, y in path if 2.95 <= x <= 3.15]
  assert crossing and min(crossing) >= 3.0 + 7 * 0.05, "round the wall's end"


def test_the_unknown_costs_more_than_the_floor_it_has_mapped():
  """A known corridor round three sides against the straight line through
  nothing seen: 9.5 m of floor against 5.5 m of unknown. At the price the
  robot walks at, the corridor; at a price under 9.5 / 5.5, the unknown."""
  g = np.full((100, 130), UNSEEN)
  g[45:95, 0:10] = FREE
  g[85:95, 0:130] = FREE
  g[45:95, 110:120] = FREE

  def top(unknown_cost):                              # (0.25, 2.5) to (5.75, 2.5)
    cost = op.coarsen(op.map_costs(g, 0, unknown_cost))
    cells, _ = op.route(op.lattice_for(cost.shape, 0.1), cost,
                        (2, 25), (57, 25))
    return max(cy for _, cy in cells) * 0.1
  assert op.UNKNOWN_COST > 9.5 / 5.5
  assert top(op.UNKNOWN_COST) > 4.0, "the corridor, at the robot's price"
  assert top(1.5) < 3.0, "the straight line, where the unknown is cheap enough"


def test_a_goal_in_a_wall_is_a_stand_in_and_a_goal_walled_off_is_no_route():
  g = _room_with_a_door()
  g[30:40, 10:20] = WALL                                # a cabinet in the room
  n = _Nav(g, (1.0, 0.5))
  path = n._plan_to(0.75, 1.75)                         # inside it
  assert n._stand_in is not None and path[-1] == n._stand_in
  # half the cabinet, the inflation's reach, and a lattice cell
  assert math.dist(path[-1], (0.75, 1.75)) < 0.25 + 0.35 + 0.1, "the nearest reachable cell"
  shut = _room_with_a_door()
  shut[5:25, 40:42] = WALL                              # the door is a wall now
  assert _Nav(shut, (1.0, 1.5))._plan_to(5.5, 2.5) is None, "the walls it saw enclose it"


@pytest.mark.parametrize("width,passes", [(3, True), (1, False)])
def test_the_lattice_keeps_every_gap_three_map_cells_wide(width, passes):
  """`BLOCK`'s claim: a lattice cell passes only if every map cell in it
  does, so a gap keeps a lattice path at any alignment while it is three
  map cells wide -- a 1 m door, less the inflation, is six -- and never at
  one. Checked at every offset."""
  for offset in range(4):
    cost = np.full((40, 40), np.inf)
    cost[:, :18] = 1.0
    cost[:, 22:] = 1.0
    cost[offset + 10:offset + 10 + width, 18:22] = 1.0   # the gap, 4 cells deep
    lat = op.coarsen(cost)
    cells, _ = op.route(op.lattice_for(lat.shape, 0.1), lat, (2, 10), (17, 10))
    assert (cells is not None) is passes, offset


def _brute(cost, source):
  """Dijkstra in plain Python on the lattice's own rules: eight steps, the
  mean of two cells' costs over the step, no corner cut."""
  rows, cols = cost.shape
  dist = np.full(cost.size, np.inf)
  dist[source] = 0.0
  heap = [(0.0, source)]
  while heap:
    d, i = heapq.heappop(heap)
    if d > dist[i]:
      continue
    y, x = divmod(i, cols)
    for dx, dy in op.STEPS:
      nx, ny = x + dx, y + dy
      if not (0 <= nx < cols and 0 <= ny < rows):
        continue
      if dx and dy and not (np.isfinite(cost[y, nx]) and np.isfinite(cost[ny, x])):
        continue
      w = (cost[y, x] + cost[ny, nx]) * 0.5 * 0.1 * math.hypot(dx, dy)
      if d + w < dist[ny * cols + nx]:
        dist[ny * cols + nx] = d + w
        heapq.heappush(heap, (d + w, ny * cols + nx))
  return dist


def test_the_lattice_search_is_dijkstra_over_its_own_edges():
  """The CSR the lattice builds once and a plan rewrites, against Dijkstra
  in plain Python -- twice on one lattice, so a plan's weights cannot
  outlive it."""
  rng = np.random.default_rng(381)
  lat = op.Lattice(9, 13, 0.1)
  for _ in range(3):
    cost = rng.choice([1.0, 2.0, np.inf], size=(9, 13), p=[0.6, 0.25, 0.15])
    cost[4, 6] = 1.0
    lat.weigh(cost)
    dist, _ = lat.search(4 * 13 + 6)
    assert np.allclose(dist, _brute(cost, 4 * 13 + 6)), "a diagonal, an inf, or the order"


# ---- the navigator ---------------------------------------------------------------


def test_the_quadruped_walks_into_the_unknown_and_the_rover_does_not():
  """A new behaviour for one body is a class attribute, the rover's day as
  it was (CLAUDE.md, the body bullet)."""
  assert QuadMission.OPTIMISTIC and QuadMission.PROGRESS_MAP_GROWTH
  assert not HubMission.OPTIMISTIC and not HubMission.PROGRESS_MAP_GROWTH


def test_the_cut_route_check_answers_as_the_plan_would_without_the_other_robot():
  """`_route_cut_by_others` on the lattice: held to a real plan with the
  other robot left out, with it standing in the room's one door, the walls
  round the room seen, and the goal beyond, in a wall, and this side."""
  g = _room_with_a_door()
  g[:, 41] = WALL
  g[5:25, 40:42] = FREE
  g[:, 42:] = FREE                                      # the far side seen, too
  n = _Nav(g, (1.0, 1.5))
  for gap in (True, False):
    if not gap:
      n.grid.grid[5:25, 40:42] = WALL
    for goal in [(4.0, 0.75), (2.05, 2.5), (0.5, 2.5)]:
      n.others = [lambda: KeepClear(2.05, 0.75)]
      masked = n._plan_to(*goal)
      cut = n._route_cut_by_others(*goal)
      n.others = []
      alone = n._plan_to(*goal)
      assert cut == (alone is not None), (gap, goal)
      if goal == (4.0, 0.75):
        assert masked is None and cut is gap, "the door is the other robot's"


def test_a_gap_only_the_map_passes_is_no_route_and_not_the_other_robots():
  """The cut-route check reads the planner's own LATTICE: a door the 5 cm
  map passes by two cells at an odd alignment has no lattice path, so the
  plan is None -- and asked whether the other robot cut it, the answer is
  no, wherever it stands. Read off the map's own floor, it said yes."""
  g = _room_with_a_door()
  g[:, 40:42] = WALL
  g[10:26, 40:42] = FREE                                # passable: y cells 17, 18
  n = _Nav(g, (1.0, 1.5))
  n.others = [lambda: KeepClear(0.5, 3.5)]              # nowhere near the door
  assert n._plan_to(4.0, 0.9) is None
  assert n._floor[17:19, 40].all(), "the premise: the map's own floor passes"
  assert n._route_cut_by_others(4.0, 0.9) is False


# ---- the drive's two new endings --------------------------------------------------


class _Walk:
  """`drive_to_routine`'s own `self`, the planner and the legs stubbed:
  every command is 0.1 s, the robot moves `step` m along x, and the map is
  `grows` cells bigger each time it is counted."""
  drive_to_routine = nav.Navigator.drive_to_routine
  _drove = nav.Navigator._drove
  _at_stand_in = nav.Navigator._at_stand_in
  _other_in_the_way = nav.Navigator._other_in_the_way
  _bodies = nav.Navigator._bodies
  peer_on_the_goal = nav.Navigator.peer_on_the_goal
  _cells = nav.Navigator._cells
  _left = nav.Navigator._left
  PROGRESS_ALONG_ROUTE = True
  BACKOFF_V, PEER_CLEARANCE_M = QuadMission.BACKOFF_V, QuadMission.PEER_CLEARANCE_M
  OTHER_ROBOT_CELLS = QuadMission.OTHER_ROBOT_CELLS
  DOWN_ROBOT_CELLS = QuadMission.DOWN_ROBOT_CELLS
  NEW_ROUTE_M = nav.Navigator.NEW_ROUTE_M
  pressing = False

  def __init__(self, grows=0, growth=True, step=0.0):
    self.data = SimpleNamespace(time=0.0)
    self.grid = SimpleNamespace(resolution=0.05)
    self.pose = (0.0, 0.0, 0.0)
    self.others, self.backoff_until, self.peer_holds = [], 0.0, 0
    self.last_drive, self._stand_in = None, None
    self.PROGRESS_MAP_GROWTH = growth
    self._grows, self._known, self._step = grows, 0, step

  def _known_cells(self):
    self._known += self._grows
    return self._known

  def _plan_to(self, wx, wy):
    return [(wx, wy)]

  def _route_cut_by_others(self, wx, wy):
    return False

  def peer_sighting(self):
    return None

  def _nav_routine(self, v, w):
    self.data.time += 0.1
    self.pose = (self.pose[0] + self._step, self.pose[1], 0.0)
    yield v, w

  def _drive_routine(self, seconds, v, w):
    self.data.time += seconds
    yield v, w


def test_the_map_still_growing_is_progress_and_a_map_standing_still_is_not():
  """Nothing gets nearer the goal, and the map grows by more than
  `MAP_GROWTH_CELLS` a look: that is a walk still finding its way, and only
  its patience ends it. The same walk with the map standing still stalls."""
  grows = nav.MAP_GROWTH_CELLS + 1
  finding = _Walk(grows=grows)
  assert tick.run(STEPPER, finding.drive_to_routine(5.0, 0.0, 40.0)) is False
  assert finding.last_drive["why"] == "timeout" and finding.last_drive["seconds"] >= 40.0
  lost = _Walk(grows=nav.MAP_GROWTH_CELLS // 20)      # half of it in a window
  tick.run(STEPPER, lost.drive_to_routine(5.0, 0.0, 40.0))
  assert lost.last_drive["why"] == "stalled" and lost.last_drive["seconds"] < 12.0
  rover = _Walk(grows=grows, growth=False)
  tick.run(STEPPER, rover.drive_to_routine(5.0, 0.0, 40.0))
  assert rover.last_drive["why"] == "stalled", "the rover reads no growth"


def test_a_stop_ends_the_walk_as_an_abort_and_its_answer_costs_no_patience():
  """The drive asks `stop` every `STOP_EVERY_S`; a True ends it where it
  stands as `DRIVE_STOPPED` -- never one of the four ways a drive gives up
  -- and the time a False took (a question to the mind, standing still)
  counts against neither its patience nor its stagnation clock."""
  walk = _Walk(step=0.004)                              # 4 cm/s: slow progress
  asked = []

  def stop():
    asked.append(walk.data.time)
    if len(asked) == 3:
      walk.data.time += 50.0                            # the mind thought for 50 s
    return len(asked) == 6
  assert tick.run(STEPPER, walk.drive_to_routine(5.0, 0.0, 20.0, stop=stop)) is False
  rec = walk.last_drive
  assert rec["why"] == nav.DRIVE_STOPPED and nav.DRIVE_STOPPED not in nav.DRIVE_GAVE_UP
  assert len(asked) == 6 and rec["seconds"] < 20.0, "the 50 s ask came off its patience"
  assert "stopped by the robot's own interrupt" in nav.gave_up(rec)


# ---- the verb: patience, and the interrupt ---------------------------------------


def test_drive_to_takes_a_patience_the_code_caps():
  verb = st.VERBS["drive_to"]
  assert st.check_step(verb, {"x": 1.0, "y": 2.0}, LEGS) == []
  assert st.check_step(verb, {"x": 1.0, "y": 2.0, "patience": 300.0}, LEGS) == []
  assert st.check_step(verb, {"x": 1.0, "y": 2.0, "patience": 601.0}, LEGS) == [
    f"patience=601.0 is above {st.MAX_PATIENCE_S}"]
  assert lang.compile_procedure("def go():\n    drive_to(1, 2, 300)\n", LEGS)
  with pytest.raises(st.Refused, match="line 2: patience=700.0 is above"):
    lang.compile_procedure("def go():\n    drive_to(1, 2, patience=700)\n", LEGS)
  with pytest.raises(st.Refused, match="drive_to needs y"):
    lang.compile_procedure("def go():\n    drive_to(1)\n", LEGS)


def _no_arm(name):
  """`QuadBody.actuator`: a body with no arm names no actuator (#387)."""
  raise KeyError(f"no actuator {name!r}")


def test_a_body_with_no_arm_has_no_carrying_pose_to_take():
  """Every verb that moves takes the carrying pose first (`run_verb`), and
  asking a quadruped for its arm raised: no procedure on legs walked a step
  (#387's deploy). Found on the way to this stage."""
  life = SimpleNamespace(body=SimpleNamespace(actuator=_no_arm))
  assert st.travel_pose(life, None) == []


def _life(stops_at=None):
  """A life for one `drive_to`: a body that records what each walk was
  given and, asked `stop`, is ended by it (`last_drive`, as the drive
  records it) -- and an interrupt that answers "stop" once `stops_at`
  sim s have passed, latching `aborting` the way the lifecycle does."""
  walks = []
  life = SimpleNamespace(data=SimpleNamespace(time=0.0, ctrl=[0.0]), world=QUAD_HOME,
                         aborting=False, step_now=None, step_until=None,
                         _say=lambda *a, **k: None,
                         drive_why=lambda x, y: "the walk gave up (why)")

  def interrupted():
    if stops_at is not None and life.data.time >= stops_at:
      life.aborting = True
    return life.aborting

  def go(x, y, timeout=90.0, stop=None):
    walks.append({"timeout": timeout, "stop": stop is not None})
    life.data.time += 10.0
    if stop is not None and stop():
      life.body.last_drive = {"why": nav.DRIVE_STOPPED, "goal": (x, y)}
      return False
    life.body.last_drive = {"why": "", "goal": (x, y)}
    life.body.pose = (x, y, 0.0)
    return True
    yield

  def hold(seconds):
    life.data.time += seconds
    return
    yield
  life.interrupted = interrupted
  life.body = SimpleNamespace(pose=(0.0, 0.0, 0.0), last_drive=None,
                              actuator=_no_arm,
                              pose_xy=lambda: life.body.pose[:2],
                              in_sight=lambda x, y: True,
                              peer_on_the_goal=lambda x, y: None,
                              module_state=lambda tool: {"on_fork": False},
                              go_to_routine=go, hold_routine=hold)
  life.walks = walks
  return life


def test_a_verb_the_interrupt_ended_is_stopped_whatever_the_verb():
  """`run_verb` reads the latch: a verb that failed because the robot's
  own interrupt latched during it -- a `pick` whose way to the cube was
  stopped says only that it did not get there -- is `stopped:
  interrupted`; a verb that failed with no interrupt is a failure."""
  def run(life, args):
    life.aborting = life.stops
    return {"ok": False, "reason": "did not get there"}
    yield
  verb = st.Verb("walks", {}, run, "a verb that walks and fails")
  for stops in (True, False):
    life = SimpleNamespace(data=SimpleNamespace(time=0.0), aborting=False, stops=stops)
    verdict = tick.run(STEPPER, st.run_verb(life, verb, {}))
    assert (verdict.get("stopped") == "interrupted") is stops, verdict


def test_patience_is_the_walks_budget_and_never_past_the_programs():
  life = _life()
  tick.run(STEPPER, st._drive_to(life, {"x": 1.0, "y": 0.0}))
  tick.run(STEPPER, st._drive_to(life, {"x": 2.0, "y": 0.0, "patience": 300.0}))
  life.step_until = life.data.time + 25.0
  tick.run(STEPPER, st._drive_to(life, {"x": 3.0, "y": 0.0, "patience": 300.0}))
  assert [w["timeout"] for w in life.walks] == [st.DRIVE_TIMEOUT_S, 300.0, 25.0]
  assert all(w["stop"] for w in life.walks), "every walk asks the interrupt"


def test_a_walk_its_interrupt_stopped_stops_either_runner_as_interrupted():
  """The row fires mid-walk; the walk asks it, stops, and BOTH runners
  record the program `stopped: interrupted` -- no `failedAt`, and History
  says it was interrupted rather than that a step failed. Premise: the
  same walk with no interrupt runs both steps."""
  program = st.Program.single("out", [st.Step("drive_to", {"x": 3.0, "y": 0.0}),
                                      st.Step("wait", {"seconds": 1.0})])
  source = "def out():\n    drive_to(3, 0)\n    wait(1)\n"
  for run in (lambda life: st.run_program_routine(life, program, LEGS),
              lambda life: lang.run_procedure_routine(
                life, lang.compile_procedure(source, LEGS), LEGS)):
    calm = tick.run(STEPPER, run(_life()))
    assert calm["ok"] and calm["completed"] == 2, "the premise"
    r = tick.run(STEPPER, run(_life(stops_at=5.0)))
    assert r["stopped"] == "interrupted" and "failedAt" not in r, r
    assert r["completed"] == 0 and r["steps"][0]["stopped"] == "interrupted"
    said = procedure_outcome("out", r)[0]
    assert "it was interrupted" in said and "it stopped at" not in said, said


def test_a_decided_walk_to_a_zone_has_the_decided_patience():
  """`explore(zone)` walks to the zone with `ZONE_PATIENCE_S` (a decided
  action's default), not the 60 s that ended every walk to the lab."""
  from pluggybot import lifecycle as lc
  from pluggybot.body import StubBody
  from pluggybot.mind.overseer import Decision
  from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
  body = StubBody()
  seen = []
  real = body.go_to_routine

  def go(x, y, timeout=90.0, stop=None):
    seen.append(timeout)
    return (yield from real(x, y, timeout))
  body.go_to_routine = go
  life = stub_life("home", body=body)
  life.explore_routine = lambda *a, **kw: tick.result(None)
  body.run(life._after_decision_routine(Decision(action="explore", zone="lab")))
  assert seen == [lc.ZONE_PATIENCE_S] and lc.ZONE_PATIENCE_S > 60.0


# ---- the walk, flown whole ----------------------------------------------------------


@pytest.mark.endurance
def test_a_fresh_quadruped_walks_to_the_kitchen_by_the_hall():
  """The integration the rules above are pinned for: a fresh robot at its
  start pose, an empty map, one look round, and the kitchen (-8.47, 4.03).
  Before, it walked into the workshop and stood against its wall with the
  kitchen behind it, `no route` 2.4 m short at 47 s; it arrives by the hall
  door in 38.6 s (`scripts/unknown_spike.py`, SimNotes "Walking into the
  unknown"). Behind --endurance (~40 sim s of the walking policy): every
  rule it exercises is pinned fast in this file."""
  import mujoco
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  from pluggybot.lifecycle import world_config
  cfg = world_config(QUAD_HOME)
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=cfg["grid_bounds"])
  try:
    body.start_at(*cfg["start"])
    hall = []
    body.step_hooks.append(lambda: hall.append(-5.0 < body.true_pose()[0] < -2.0))
    body.run(body.look_around_routine())
    assert body.run(body.go_to_routine(-8.47, 4.03, timeout=120.0)), body.last_drive
    tx, ty, _ = body.true_pose()
    assert math.hypot(tx + 8.47, ty - 4.03) < 0.15 and any(hall), "by the hall"
  finally:
    body.close()
