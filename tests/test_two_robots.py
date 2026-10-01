"""Two robots in one world (issue #167): the namespacing, one loop, the
wire, and the pair as an obstacle.

A second robot is the same body attached with a prefix; a `RobotHandle` is
that prefix, and every class that resolves a robot element does so through
it. The first robot keeps the bare names, so a single-robot world is what
it was. The world is the quadruped pair's (`legs.world.home_spec`); where a
rule is the loop's bookkeeping it is pinned on the stub, and where it is
the drive's, on a stub that walks.
"""

import math
import pathlib
import re

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.body import StubBody
from pluggybot.legs import rack as rk
from pluggybot.legs import world as lw
from pluggybot.legs.body import QuadMission
from pluggybot.lifecycle import QUAD_HOME, world_config
from pluggybot.navigator import CLOSE_ENOUGH_M
from pluggybot.robot import FIRST, SECOND, RobotHandle
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_unknown import STEPPER, _Walk

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
#: Where a second robot is parked out of the way: the kitchen.
PARK = (-8.0, 4.0)
BOUNDS = world_config(QUAD_HOME)["grid_bounds"]


def _pair(second_at=PARK, **kw):
  return lw.home_spec(second_at=second_at, **kw).compile()


def test_two_namespaced_robots_compile():
  model = _pair()
  names = {model.body(i).name for i in range(model.nbody)}
  assert FIRST.root in names and SECOND.root in names
  for el in ("FL_knee", "arm_shoulder", "arm_elbow"):
    model.actuator(SECOND.el(el))
  for el in ("nav_eye", "depth_eye"):
    model.camera(SECOND.el(el))
  model.sensor(SECOND.el("imu_ang_vel"))
  for el in ("lidar", "arm_seat"):
    model.site(SECOND.el(el))
  # the robots come after the house, the second after the first
  assert 0 < FIRST.qpos_adr(model) < SECOND.qpos_adr(model)
  # ...and the rack, the dock and the tools are the WORLD's, not prefixed
  assert {rk.RACK_BODY, "dock", "module_pen"} <= names
  assert not any(n.startswith(("r2_module", "r2_tool", "r2_dock")) for n in names)


def test_the_second_robot_walks_through_its_handle_and_touches_nothing_of_the_firsts():
  """The second robot's mission resolves every element by its prefix: its
  commands reach `r2_`'s drivers and none of the first robot's."""
  model = _pair(second_at=(2.0, -0.6))
  data = mujoco.MjData(model)
  second = QuadMission(model, data, realtime=False, grid_bounds=BOUNDS, handle=SECOND)
  try:
    second.start_at(2.0, -0.6, math.pi)
    firsts = [i for i in range(model.nu)
              if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("r2_")]
    assert firsts
    x0 = second.true_pose()[0]
    second.run(second._drive_routine(1.5, 0.5, 0.0))
    assert x0 - second.true_pose()[0] > 0.1, "the second robot did not walk"
    assert not data.ctrl[firsts].any(), "the second robot drove the first's joints"
  finally:
    second.close()


def _seat(model, data, module, handle):
  """`module` dropped onto `handle`'s fork at the carry pose, as a pick
  leaves it, a tenth of a second on: seated, before the unheld arm sags."""
  from pluggybot.legs import arm as am
  from pluggybot.legs.model import CHOSEN
  from pluggybot.rack.coupling import PEG_ABOVE_BODY
  for joint, q in zip(("shoulder", "elbow", "wrist"), (*am.CARRY_Q, -sum(am.CARRY_Q))):
    data.qpos[model.jnt_qposadr[model.joint(handle.el(f"arm_{joint}")).id]] = q
  mujoco.mj_forward(model, data)
  rot = data.xmat[model.body(handle.root).id].reshape(3, 3)
  peg = data.site_xpos[model.site(handle.el("arm_seat")).id] + rot @ np.array(
    [0.0, 0.0, CHOSEN.arm.fork.seat_rise() + 0.0003])
  yaw = math.atan2(rot[1, 0], rot[0, 0]) + math.pi
  q = model.jnt_qposadr[model.body(module).jntadr[0]]
  data.qpos[q:q + 3] = peg - np.array([0.0, 0.0, PEG_ABOVE_BODY])
  data.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  mujoco.mj_forward(model, data)
  for _ in range(50):
    mujoco.mj_step(model, data)


def test_the_second_robots_fork_powers_a_tool_and_the_firsts_does_not():
  """The coupling through the handle: the second robot's fork plates are
  `r2_arm_v*`, so `tool_power` with its prefix says powered while the first
  robot's says not -- a tool on the other robot's fork is not this one's."""
  from pluggybot.legs.dock import dock_charge_contact
  model = _pair(second_at=(0.0, 1.0))
  data = mujoco.MjData(model)
  lw.stand(model, data, FIRST.prefix, 1.5, 0.5, 0.0)
  lw.stand(model, data, SECOND.prefix, 0.0, 1.0, 0.0)
  _seat(model, data, "module_claw", SECOND)
  assert rk.tool_power(model, data, "module_claw", SECOND.prefix)["powered"]
  assert not rk.tool_power(model, data, "module_claw", FIRST.prefix)["powered"]
  assert not dock_charge_contact(model, data, SECOND.prefix)


# ---- the fence: no bare robot name in mission code -----------------------------

#: Element names that belong to a robot, which mission code may only reach
#: through a handle. Tool, rack, bay, dock and board names are the WORLD's.
ROBOT_ELEMENTS = ("pluggybot", "torso", "lidar", "lidar_mast", "depth_eye", "nav_eye",
                  "imu", "imu_ang_vel", "imu_lin_acc", "arm_seat", "arm_shoulder",
                  "arm_elbow", "arm_wrist", "belly_pad_l", "belly_pad_r",
                  "FL_hip_abd", "FL_hip_flex", "FL_knee")
#: The mission stack: what a second robot runs a copy of.
MISSION_CODE = ("lifecycle.py", "power.py", "body.py", "navigator.py", "mission/errand.py",
                "tools/screen.py", "procedure/steps.py", "procedure/axes.py",
                "procedure/lang.py", "perception/lidar.py", "perception/depth.py",
                "perception/imu.py", "rack/tags.py", "economy/census.py",
                "mapping/scan_match.py", "legs/body.py", "legs/swap.py",
                "legs/places.py", "legs/odometry.py", "legs/posture.py",
                "legs/arm.py", "legs/dock.py", "legs/rack.py", "legs/drivers.py",
                "legs/policy.py", "legs/scan.py")
BARE = re.compile(r'(?<!body)\.(body|geom|joint|actuator|site|camera|sensor)\("('
                  + "|".join(ROBOT_ELEMENTS) + r')"\)')
#: A robot's root read at a fixed `qpos` index: the robots are attached
#: after the house, so none of them is at `qpos[0]`.
ROOT_INDEX = re.compile(r"qpos\[(0|1|2|3:7|:7|:3)\]")


def test_mission_code_resolves_every_robot_element_through_the_handle():
  bad = []
  for rel in MISSION_CODE:
    text = (SRC / rel).read_text()
    for n, line in enumerate(text.splitlines(), 1):
      if BARE.search(line) or ROOT_INDEX.search(line):
        bad.append(f"{rel}:{n}: {line.strip()}")
  assert not bad, "bare robot names in mission code:\n  " + "\n  ".join(bad)
  # ...and the fence fires on what it is for
  assert BARE.search('self.model.site("lidar").id') and ROOT_INDEX.search("d.qpos[:7]")
  assert not BARE.search('self.model.site(handle.el("lidar")).id')


# ---- one loop, two controllers ----------------------------------------------------


class _Swap:
  """A stepper (tick.py): records the order the loop calls it in; raises
  from a hook on cue."""

  STILL = (0.0, 0.0)

  def __init__(self, log, name, raise_at=None):
    self.log, self.name, self.raise_at, self.after = log, name, raise_at, 0

  def apply(self, command):
    self.log.append((self.name, "before", command))

  def after_step(self):
    self.after += 1
    self.log.append((self.name, "after"))
    if self.raise_at is not None and self.after == self.raise_at:
      raise RuntimeError("hook")


def test_run_many_applies_every_command_steps_once_then_books_every_robot():
  log: list = []
  a, b = _Swap(log, "a"), _Swap(log, "b")

  def ra():
    yield 0.1, 0.0
    yield 0.1, 0.0
    return "a-done"

  def rb():
    yield 0.2, 0.0
    return "b-done"
  results = tick.run_many([(a, ra()), (b, rb())],
                          step=lambda: log.append(("world", "step")))
  assert results == ["a-done", "b-done"]
  # step 1: both commands, one world step, both bookkeepings -- in order
  assert log[:5] == [("a", "before", (0.1, 0.0)), ("b", "before", (0.2, 0.0)),
                     ("world", "step"), ("a", "after"), ("b", "after")]
  # step 2: b has returned and holds STILL while a finishes
  assert log[5:10] == [("a", "before", (0.1, 0.0)), ("b", "before", (0.0, 0.0)),
                       ("world", "step"), ("a", "after"), ("b", "after")]
  assert len(log) == 10


def test_run_many_throws_a_hook_exception_into_every_live_routine():
  log: list = []
  a, b = _Swap(log, "a", raise_at=1), _Swap(log, "b")
  seen: list = []

  def ra():
    try:
      yield 0.1, 0.0
      yield 0.1, 0.0
    finally:
      seen.append("a-finally")

  def rb():
    try:
      yield 0.1, 0.0
      yield 0.1, 0.0
    finally:
      seen.append("b-finally")
  with pytest.raises(RuntimeError, match="hook"):
    tick.run_many([(a, ra()), (b, rb())], step=lambda: None)
  assert sorted(seen) == ["a-finally", "b-finally"]


def _errand(used, station=0.0):
  from pluggybot.mission.errand import Errand
  return Errand(name="carry:test", module="module_lcd", station_y=station,
                use_at=(1.0, 1.0), use=lambda _l: used.append(1) or {},
                needs_use_pose=False)


def _swaps(body, why="missed"):
  """The body's two swaps recorded, and a pick that takes nothing."""
  swaps = []
  body.fetch_tool_routine = lambda station, module: (swaps.append("pick"), tick.result(why))[1]
  body.stow_tool_routine = lambda station, module: (swaps.append("return"), tick.result("arrived"))[1]
  return swaps


def test_a_failed_pick_ends_the_errand_at_the_rack():
  """Issue #298. On the deployed pair a pick fails whenever the other robot
  holds the tool, and the errand used to carry on regardless: go to the use
  pose with nothing on the fork, skip the use, come back and attempt to
  RETURN a tool it never had. That phantom trip parked the robot at the
  rack exactly when the other came back to stow. Now the errand ends at the
  rack, scored as it stands, and History says why."""
  life = stub_life()
  body = life.body
  swaps = _swaps(body)
  drives = []
  body.go_to_routine = lambda *a, **kw: (drives.append(a), tick.result(True))[1]
  # the other robot has the LCD: not on this fork, not on its bay
  body.module_state = lambda *a, **kw: {"on_fork": False, "hung": False, "bay": 0}
  used = []
  result = life.run_errand(_errand(used))
  assert swaps == ["pick"], swaps                      # no return of nothing
  assert drives == [] and used == []                  # no trip, no use
  assert result["picked"] is False and result["stowed"] is False
  assert result["error"] == "never picked up module_lcd"
  assert any("could not pick up module_lcd" in ln and "not on its bay" in ln
             for ln in life.thoughts.read("History.md").splitlines()), \
      life.thoughts.read("History.md")
  # ...and a pick that MISSED with the tool still hanging says that instead
  body.module_state = lambda *a, **kw: {"on_fork": False, "hung": True, "bay": 0}
  result = life.run_errand(_errand(used))
  assert result["picked"] is False and result["stowed"] is True
  assert any("the pick missed and it is still on its bay" in ln
             for ln in life.thoughts.read("History.md").splitlines())


def test_a_pick_lost_to_a_peer_names_it_in_history():
  """The narration #313 asks for: which robot, how far off. A robot
  diagnosed its pen, told the other robot to move out of a pose measured
  harmless, and started declining pen jobs -- off a correlation in its own
  History that no line ever named."""
  life = stub_life()
  body = life.body
  station = 0.0
  sx, sy, _ = body.bay_standoff(station)

  class Peer:                                # the public surface, no more
    robot_name, root = "Rowan", SECOND.root
    down = staticmethod(lambda: False)       # standing (issue #365)

    class body:                              # ...and a fork with nothing on it
      pose_xy = staticmethod(lambda: (sx + 0.26, sy))
      module_state = staticmethod(lambda t: {"on_fork": False, "hung": True})

  life.peers = [Peer()]

  def blocked(station_y, module):            # what the body's swap just did
    body.peer_at_bay_m = 0.26
    return tick.result("peer-at-bay")

  body.fetch_tool_routine = blocked
  body.module_state = lambda *a, **kw: {"on_fork": False, "hung": True, "bay": 0}
  result = life.run_errand(_errand([], station))
  assert result["error"] == "never picked up module_lcd"
  assert result["peerAtBayM"] == 0.26
  history = life.thoughts.read("History.md")
  assert "Rowan was standing 0.26 m from the bay" in history, history
  assert "the pick missed" not in history, "blamed the tool for a blocked bay"


def _mission(at=(1.0, 1.0)):
  """A quadruped's mission in the house, on a mapped and empty floor."""
  model = lw.home_spec().compile()
  m = QuadMission(model, mujoco.MjData(model), realtime=False, grid_bounds=BOUNDS)
  m.start_at(*at, 0.0)
  m.grid.grid[:] = -5.0
  return m


def _plan(m, x, y):
  """A fresh plan: the quadruped keeps its last one for a moment from the
  same spot (`QuadMission.REPLAN_HOLD_S`)."""
  m._plan_memo = None
  return m._plan_to(x, y)


def test_the_planner_routes_round_the_other_robots_reported_pose():
  """On the grid, nothing stepped: a free room, the other robot reported on
  the straight line, and every waypoint keeps clear of it."""
  m = _mission()
  try:
    goal = (4.0, 1.0)
    straight = _plan(m, *goal)
    assert straight is not None
    m.others = [lambda: (2.5, 1.0)]
    bent = _plan(m, *goal)
    assert bent is not None
    clearance = m.OTHER_ROBOT_CELLS * m.grid.resolution
    assert all(math.hypot(x - 2.5, y - 1.0) > clearance - 2 * m.grid.resolution
               for x, y in bent), "a waypoint passes through the other robot"
    assert any(math.hypot(x - 2.5, y - 1.0) < clearance for x, y in straight)
    assert m._other_in_the_way(*goal) is False
    assert m._other_in_the_way(2.6, 1.2) is True
  finally:
    m.close()


def test_a_blocked_drive_waits_for_the_other_robot_instead_of_giving_up():
  """The other robot reported ON the goal: the drive stagnates, sees the
  other in the way, and waits out its timeout rather than giving up after
  its stagnation clock."""
  walk = _Walk()
  walk.others = [lambda: (1.6, 0.0)]
  arrived = tick.run(STEPPER, walk.drive_to_routine(1.6, 0.0, 14.0))
  assert not arrived
  assert walk.data.time >= 13.5, "gave up before its timeout"


def test_explore_does_not_spin_when_a_drive_steps_nothing():
  """Flown in the deployed pair: the frontier planner, which does not mask
  the other robot, called a frontier "ok" that the drive planner, which
  does, could not route to -- so the drive returned without stepping and
  explore asked again at the same sim instant, forever. A body whose every
  drive steps nothing is that; the plan count is the tripwire, because a
  regression here is a hang, not a failure."""
  from pluggybot import lifecycle as lc
  body = StubBody(grid_bounds=(0.0, 0.0, 4.0, 2.0))
  life = stub_life(body=body)
  life.max_sim_time, life.blacklist = 3600.0, set()   # what run() sets up
  body.start_at(1.0, 1.0, 0.0)
  g = body.grid
  g.grid[:] = 5.0                                  # walls everywhere...
  x0, y0 = g.world_to_cell(0.2, 0.2)
  x1, y1 = g.world_to_cell(3.0, 1.8)
  g.grid[y0:y1, x0:x1] = -5.0                      # ...but a known-free corridor
  g.grid[y0:y1, x1:g.world_to_cell(3.6, 0.0)[0]] = 0.0   # the only frontier
  real, plans = body.plan_frontier, []

  def counted(blacklist):
    plans.append(life.data.time)
    if len(plans) > 50:
      raise AssertionError(f"explore replanned {len(plans)} times at sim "
                           f"time {life.data.time:.3f} without stepping")
    return real(blacklist)

  body.plan_frontier = counted
  step = tick.Step(life.explore_routine(budget=30.0))
  assert step.tick() is None and step.done, "explore neither stepped nor ended"
  assert len(plans) == lc.STRIKES_TO_FINISH
  assert life.floor_explored


# ---- the wire ------------------------------------------------------------------


def test_the_census_and_the_scene_key_every_robot_by_its_root():
  from pluggybot.telemetry.protocol import body_census, robot_roots
  from pluggybot.telemetry.scene import scene_dict
  model = _pair()
  assert robot_roots(model) == [FIRST.root, SECOND.root]
  alone = lw.home_spec().compile()
  assert robot_roots(alone) == [FIRST.root]
  r1, world = body_census(model)
  r2, world2 = body_census(model, SECOND.root)
  assert r1 and r2 == [SECOND.el(n) for n in r1]
  assert world == world2 and not any(n.startswith("r2_") for n in world)
  assert body_census(alone) == (r1, world), "a single-robot census moved"
  scene = scene_dict(model, QUAD_HOME)
  owners = {b["name"]: b["robot"] for b in scene["bodies"]}
  assert owners[SECOND.root] == SECOND.root and owners[FIRST.root] == FIRST.root
  assert owners[SECOND.el("lidar_mast")] == SECOND.root
  assert owners[rk.RACK_BODY] is None and owners["module_pen"] is None
  assert owners["dock"] is None


def test_a_pair_recording_carries_both_robots_and_keys_every_event(tmp_path):
  """The stream with a second robot on it, off the recorder `run_pair` wires
  (`record_pair`), driven by hand rather than flown: the header names both
  robots and advertises the game's referee, a frame carries
  `robots[<root>]` for each, and the thoughts say whose they are. Nothing
  steps -- the frame is the world as built."""
  import json
  from pluggybot.pair import arrange_game, build_pair, record_pair
  lives = build_pair(QUAD_HOME, pack="hosting", errands=("none", "none"),
                     tasks=True, overseer=None, thoughts_root=str(tmp_path / "t"))
  arrange_game(lives)
  lives[0].thoughts.pin("the other one is quick", t=0.0)
  lives[1].thoughts.pin("the first one is slow", t=0.0)
  path = str(tmp_path / "pair.jsonl")
  recorder = record_pair(lives, path)
  try:
    mujoco.mj_forward(lives[0].model, lives[0].data)
    recorder.step_hook()
  finally:
    recorder.close()
    for life in lives:
      life.body.close()
  assert float(lives[0].data.time) == 0.0, "something stepped the physics"
  rows = [json.loads(line) for line in open(path)]
  header = rows[0]
  assert header["type"] == "header"
  assert set(header["robots"]) == {FIRST.root, SECOND.root}
  assert header["robotNames"] == {FIRST.root: "Pluggy", SECOND.root: "Rowan"}
  assert header["robots"][SECOND.root] == [SECOND.el(n) for n in header["robots"][FIRST.root]]
  assert header["ledger"] == [FIRST.root, SECOND.root]
  assert header["model"] == "home_quad_pair"
  goals = [r for r in rows if r.get("type") == "goals"]
  assert [g["robot"] for g in goals] == [FIRST.root, SECOND.root]
  frames = [r for r in rows if "type" not in r]
  assert frames and all(set(f["robots"]) == {FIRST.root, SECOND.root} for f in frames)
  assert frames[0]["robots"][SECOND.root]["bodies"]
  thoughts = [r for r in rows if r.get("type") == "thought"]
  assert {t["robot"] for t in thoughts} == {FIRST.root, SECOND.root}
  assert {"encounters", "hide_and_seek"} <= set(header["activities"]), \
      "the referee must be advertised"


class _Block:
  """A `snapshot()` duck (a spend book, an appetite) that returns what it is
  told to, so the builder's per-robot change detection is the only thing
  under test."""
  def __init__(self, value: dict):
    self.value = value

  def snapshot(self) -> dict:
    return dict(self.value)


class _Docs:
  """A `ThoughtFiles` duck: one document, keyed by its robot."""
  def __init__(self, robot: str):
    self.robot = robot

  def messages(self, t: float = 0.0) -> list[dict]:
    return [{"type": "thought", "t": t, "robot": self.robot, "name": "Main.md",
             "text": f"I am {self.robot}"}]


def _two_robot_builder(**second):
  from pluggybot.telemetry.recorder import FrameBuilder, StreamRobot
  model = _pair()
  data = mujoco.MjData(model)
  other = StreamRobot(SECOND.root, "Rowan", lambda: {"state": "IDLE"}, **second)
  return model, data, other, FrameBuilder


def test_spend_and_metabolism_ride_each_robots_own_record():
  """0.20.0: the two blocks are keyed by robot like everything else on the
  wire, and each robot's change detection is its own -- the second robot
  eating does not re-ship the first robot's unchanged block."""
  model, data, other, FrameBuilder = _two_robot_builder(
    metabolism=_Block({"state": "starving", "points": 0}))
  hunger1 = _Block({"state": "satisfied", "points": 15})
  purse = _Block({"spentUsd": 0.0})
  b = FrameBuilder(model, data, metabolism=hunger1, spend=purse, others=[other])
  first = b.build()
  assert "metabolism" not in first and "spend" not in first, \
    "a top-level block is the first robot's alone -- the pre-0.20.0 shape"
  assert first["robots"][FIRST.root]["metabolism"]["state"] == "satisfied"
  assert first["robots"][FIRST.root]["spend"] == {"spentUsd": 0.0}
  assert first["robots"][SECOND.root]["metabolism"]["state"] == "starving"
  assert "spend" not in first["robots"][SECOND.root], "no purse, no block"
  other.metabolism.value = {"state": "hungry", "points": 3}
  data.time = 0.1
  second = b.build()
  assert second["robots"][SECOND.root]["metabolism"]["state"] == "hungry"
  assert "metabolism" not in second["robots"][FIRST.root], \
    "the first robot's unchanged appetite was re-sent"
  b.reset()
  data.time = 0.2
  key = b.build()
  assert key["robots"][FIRST.root]["metabolism"] and key["robots"][SECOND.root]["metabolism"], \
    "a keyframe must re-ship every robot's block"
  assert b.header()["hungerStates"], "any robot with an appetite advertises the vocabulary"


def test_goals_thoughts_and_the_map_are_one_message_per_robot():
  """0.20.0: `goals`, `thought` and `grid` each carry the ROOT of the robot
  they belong to, and there is one per robot that has the thing."""
  from pluggybot.telemetry.recorder import GridSampler, grid_samplers
  grid2 = object()
  model, data, other, FrameBuilder = _two_robot_builder(
    thoughts=_Docs(SECOND.root), goals="find the other one", steering=True,
    grid=grid2)
  b = FrameBuilder(model, data, thoughts=_Docs(FIRST.root), goals="",
                   steering=False, others=[other])
  goals = b.goals_messages(1.0)
  assert [(g["robot"], g["text"], g["steering"]) for g in goals] == \
    [(FIRST.root, "", False), (SECOND.root, "find the other one", True)]
  assert [(m["robot"], m["text"]) for m in b.thought_messages(1.0)] == \
    [(FIRST.root, f"I am {FIRST.root}"), (SECOND.root, f"I am {SECOND.root}")]
  samplers = grid_samplers(None, [other], hz=1.0, dedupe=False)
  assert [(s.root, s.grid) for s in samplers] == [(FIRST.root, None), (SECOND.root, grid2)]
  assert GridSampler(None).root == FIRST.root, "a single-robot caller's map is the first robot's"
  assert b.header()["robotNames"] == {FIRST.root: "Pluggy", SECOND.root: "Rowan"}


def test_a_pair_recording_is_named_for_the_pair_world():
  """A replayer picks its scene off the header's `model`, and the second
  robot's bodies are in no single-robot scene: the pair world has a name of
  its own, and the scene generator knows it."""
  from pluggybot.robot import pair_model_name
  assert pair_model_name(QUAD_HOME) == "home_quad_pair"
  import inspect
  from pluggybot import pair
  assert "pair_model_name(cfg[\"model_name\"])" in inspect.getsource(pair.record_pair)


# ---- standing by, clear of the rack ----------------------------------------------


def _near_the_rack(life):
  """A bay's standoff: where a robot stands after a swap."""
  sx, sy, _ = life.body.bay_standoff(0.0)
  return sx, sy


def test_a_robot_standing_by_for_work_clears_the_rack_first(monkeypatch):
  """Measured on the pair: the second robot stood by at the bay standoff
  after a stow and the first robot's next pick failed 0.4 m from it.
  Standing by within `RACK_CLEAR_M` of the rack prior walks home first;
  standing by anywhere else stays put -- the branch is checked on the loop
  that actually runs it, on the stub."""
  from pluggybot import lifecycle as lc
  monkeypatch.setattr(lc, "WAIT_FOR_WORK_S", 0.2)   # the slice length is not the claim
  start = world_config(QUAD_HOME)["start"]

  def life_at(where):
    life = stub_life()
    body = life.body
    life.expects_work = True
    drives = []

    def explored(*a, **kw):
      # Where the robot BELIEVES it ended up exploring, without walking.
      body.x, body.y = where(life)
      life.floor_explored = True
      return tick.result(None)

    life.explore_routine = explored
    body.go_to_routine = lambda *a, **kw: (drives.append(a), tick.result(True))[1]
    life.run(start, max_sim_time=1.5)
    return drives

  near = life_at(_near_the_rack)
  assert near and near[0][:2] == start[:2], \
    f"a robot idling at the rack did not clear it: drives={near}"
  far = life_at(lambda life: start[:2])
  assert not far, f"a robot already clear of the rack walked anyway: {far}"


def test_a_decided_idle_clears_the_rack_first():
  """Issue #298. With a mind, the loop never reaches the standby branch
  above: a decided `idle` -- the model's, a map row's, the fallback floor's
  -- stood still wherever the robot was, and after a failed pick or a
  charge that is the bay standoff. Same rule as standing by for work:
  within `RACK_CLEAR_M`, go home first; elsewhere, stay put."""
  from pluggybot.mind.overseer import Decision
  start = tuple(float(v) for v in world_config(QUAD_HOME)["start"])

  def idle_at(where):
    life = stub_life()
    body = life.body
    life.home_pose = start
    body.x, body.y = where(life)
    drives = []
    body.go_to_routine = lambda *a, **kw: (drives.append(a), tick.result(True))[1]
    life._after_decision(Decision(action="idle", reason="standing by"))
    return drives

  near = idle_at(_near_the_rack)
  assert near and near[0][:2] == start[:2], \
    f"a robot that decided to idle at the rack did not clear it: drives={near}"
  assert not idle_at(lambda life: start[:2]), "already clear, walked anyway"


# ---- encounters ----------------------------------------------------------------


def test_an_encounter_is_met_on_the_way_in_and_parted_on_the_way_out_with_hysteresis():
  """`activity/encounter.py`: `met` at <= 1.5 m, `parted` at >= 2.0 m, and
  NOTHING in between -- a pair hovering at 1.7 m does not chatter. Poses
  are written straight into the data: what is under test is the rule, not
  the walk."""
  from pluggybot.activity.encounter import ENCOUNTER_M, PARTED_M, Encounters
  model = _pair()
  data = mujoco.MjData(model)
  meetings = Encounters(model, FIRST, SECOND)
  events = []
  meetings.on_event.append(events.append)

  def apart(d):
    q1, q2 = FIRST.qpos_adr(model), SECOND.qpos_adr(model)
    data.qpos[q1:q1 + 2] = [0.0, 0.0]
    data.qpos[q2:q2 + 2] = [d, 0.0]
    mujoco.mj_forward(model, data)
    data.time += 0.1
    meetings.sense(model, data)

  for d in (3.0, 1.7, 1.51):
    apart(d)
  assert events == [], "met before the threshold"
  apart(1.5)
  assert [e["phase"] for e in events] == ["met"]
  assert events[0]["robots"] == [FIRST.root, SECOND.root] and events[0]["distanceM"] == 1.5
  for d in (1.7, 1.9, 1.99):
    apart(d)
  assert len(events) == 1, "a pair hovering between the thresholds chattered"
  apart(2.0)
  assert [e["phase"] for e in events] == ["met", "parted"]
  assert meetings.flags == {"near": False, "distanceM": 2.0, "met": 1,
                            "touching": False, "bumps": 0}
  assert (ENCOUNTER_M, PARTED_M) == (1.5, 2.0)


def test_the_second_robot_wears_its_own_livery_and_nothing_else_is_repainted():
  """An attached robot is repainted on exactly the geoms that carry the
  first robot's livery colour, so two robots in one room are told apart at
  a glance -- on the site, which keeps the producer's colour per geom, and
  in each other's cameras. Everything else keeps its colour: the tags are
  how a reader tells parts apart, and a hint never rides as a colour."""
  from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA
  model = _pair()
  name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)  # noqa: E731
  purple = {name(i) for i in range(model.ngeom)
            if np.allclose(model.geom_rgba[i], SECOND_CHASSIS_RGBA, atol=1e-6)}
  blue = {name(i) for i in range(model.ngeom)
          if np.allclose(model.geom_rgba[i], CHASSIS_RGBA, atol=1e-6)}
  assert purple == {SECOND.el("torso")}
  assert blue == {"torso"}, "the first robot keeps its paint"
  # ...and the paint is opt-out: `None` attaches an identical twin.
  twin = _pair(second_rgba=None)
  assert np.allclose(twin.geom(SECOND.el("torso")).rgba, CHASSIS_RGBA)


# ---- the pair as an OBSTACLE (issues #316, #313, #328) -----------------------------


def _pitched(model, data, handle: RobotHandle, x, y, yaw, pitch) -> None:
  """`handle`'s robot standing at (x, y, yaw), its torso pitched `pitch`
  rad nose-down, as on a flight."""
  lw.stand(model, data, handle.prefix, x, y, yaw)
  q = handle.qpos_adr(model)
  tilted = np.zeros(4)
  mujoco.mju_mulQuat(tilted, data.qpos[q + 3:q + 7].copy(),
                     np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
  data.qpos[q + 3:q + 7] = tilted
  mujoco.mj_forward(model, data)


def test_a_scan_answers_the_room_and_the_peer_apart():
  """`Lidar.scan_split`: one set of casts, the room's returns and the
  peer's, and `scan()` still answers the map's. The other quadruped stands
  under a level scan's plane, so the first is pitched as on a flight and
  its scan cuts the other's torso a metre ahead."""
  from pluggybot.perception.lidar import Lidar
  model = _pair(second_at=(2.5, 0.5))
  data = mujoco.MjData(model)
  lw.stand(model, data, SECOND.prefix, 2.5, 0.5, 0.0)
  _pitched(model, data, FIRST, 1.5, 0.5, 0.0, 0.2)
  lidar = Lidar(model, seed=3)
  lidar.exclude_robot(SECOND.root)
  angles, ranges, peer_angles, peer_ranges = lidar.scan_split(data)
  ahead = np.abs(peer_angles) < 0.15
  assert ahead.any() and peer_ranges[ahead].min() < 1.2, \
    "the peer's own returns are missing"
  room = np.abs(angles) < 0.15
  assert not room.any() or ranges[room].min() > 1.2, \
    "the other robot is still in the map's scan"
  # ⚠ A PEER RETURN MAY NOT DRAW ON THE MAP'S NOISE STREAM (`Lidar.peer_rng`):
  # with one stream, giving the reflex its peer returns would move every
  # reading the MAP takes in a pair world. Counted: at `dropout` 0 every
  # room ray that HIT draws exactly twice (the dropout roll and the noise),
  # a ray that returned nothing draws not at all, and a peer ray neither.
  class Counting:
    def __init__(self, inner):
      self.inner, self.draws = inner, 0

    def random(self, *a, **kw):
      self.draws += 1
      return self.inner.random(*a, **kw)

    def normal(self, *a, **kw):
      self.draws += 1
      return self.inner.normal(*a, **kw)

  lidar.dropout = 0.0
  lidar.rng = counted = Counting(np.random.default_rng(7))
  _, room_r, peer_a, _ = lidar.scan_split(data)
  assert peer_a.size, "no peer returns in this scan at all"
  hits = int((room_r < lidar.max_range).sum())
  assert counted.draws == 2 * hits, \
    f"the map's stream was drawn {counted.draws} times for {hits} of its own returns"
  # A world with nobody else in it splits into nothing
  lidar._other_geoms = set()
  lidar._index_geoms()
  _, _, alone_a, alone_r = lidar.scan_split(data)
  assert alone_a.size == 0 and alone_r.size == 0


def test_a_robot_that_touches_its_pair_leaves_a_row():
  """Issue #316's fourth question: contact between the pair is a phase of
  the encounter -- sensed off the contact array, latched like the plate's
  press. ⚠ The served quadrupeds' geoms are `contype 1 conaffinity 0`, so
  they never collide with each other; the second robot's body is given an
  affinity here so the rule has a contact to read."""
  from pluggybot.activity.encounter import Encounters
  spec = lw.home_spec(second_at=(0.3, 0.0))
  for g in spec.geoms:
    if g.name.startswith(SECOND.prefix) and g.contype:
      g.conaffinity = 1
  model = spec.compile()
  data = mujoco.MjData(model)
  lw.stand(model, data, FIRST.prefix, 0.0, 0.0, 0.0)
  lw.stand(model, data, SECOND.prefix, 3.0, 0.0, 0.0)
  enc = Encounters(model, FIRST, SECOND)
  events = []
  enc.on_event.append(events.append)
  enc.sense(model, data)
  assert enc.bumps == 0 and not enc.flags["touching"]
  lw.stand(model, data, SECOND.prefix, 0.3, 0.0, 0.0)      # torsos overlapping
  enc.sense(model, data)
  touched = [e for e in events if e["phase"] == "touched"]
  assert touched, "two robots in contact and nothing on the wire"
  assert set(touched[0]["robots"]) == {FIRST.root, SECOND.root}
  assert enc.bumps == 1 and enc.flags["touching"]
  enc.sense(model, data)
  assert enc.bumps == 1, "one contact, counted twice"
  # ...and it is over once they are apart, past the bumper's own hold.
  lw.stand(model, data, SECOND.prefix, 3.0, 0.0, 0.0)
  data.time += 1.0
  enc.sense(model, data)
  assert not enc.flags["touching"]
  assert [e["phase"] for e in events][-1] == "separated"


def test_a_peer_beside_the_goal_puts_it_out_of_reach_by_the_planners_own_arithmetic():
  """Issue #313, as arithmetic rather than a correlation: the planner takes
  a disc round the other robot's reported pose out of the floor, and a
  stagnated drive counts as arrived only inside CLOSE_ENOUGH_M -- so a peer
  nearer the goal than the difference makes arrival impossible however
  many attempts are spent on it. `peer_on_the_goal` says so, and agrees
  with what the planner does (a stand-in short of the goal)."""
  m = _mission()
  try:
    gx, gy = 2.5, 1.0
    res = m.grid.resolution
    assert m.OTHER_ROBOT_CELLS * res - CLOSE_ENOUGH_M < 0.75
    for beside, reachable in ((0.45, False), (0.75, True)):
      m.others = [lambda d=beside: (gx + d, gy)]
      near = m.peer_on_the_goal(gx, gy)
      path = _plan(m, gx, gy)
      assert path is not None
      reached = m._stand_in is None or math.dist(path[-1], (gx, gy)) < CLOSE_ENOUGH_M
      assert (near is None) is reachable, f"{beside} m: {near}"
      assert reached is reachable, f"{beside} m: {math.dist(path[-1], (gx, gy)):.3f} m short"
    m.others = []
    assert m.peer_on_the_goal(gx, gy) is None
  finally:
    m.close()


def _pair_with_the_peer_across_the_bow(across: float, ahead: float, near_field=True):
  """The quadruped pair with the near-field camera on, the second robot
  standing `ahead` and `across` of the first, and NOTHING BROADCAST between
  them: the only thing standing between the two is what the first robot
  can sense."""
  from pluggybot.pair import build_pair
  lives = build_pair(QUAD_HOME, near_field=near_field, errands=("none", "none"))
  me, peer = lives
  me.body.start_at(0.0, -0.5, 0.0)
  lw.stand(me.model, me.data, SECOND.prefix, ahead, -0.5 + across, 0.0)
  me.body.grid.grid[:] = -5.0            # a mapped, empty room
  me.body.mission.others = []            # the broadcast is what we are not using
  return me, peer


def test_the_seam_hands_the_frames_peers_to_the_drive():
  """One line of wiring, pinned on its own: `_near_field_step` is where
  the peer channel reaches the mission, and a world without the camera
  hears of no peer at all."""
  me, _ = _pair_with_the_peer_across_the_bow(across=0.25, ahead=0.9)
  me._next_near_field = 0.0
  me._near_field_step()
  seen = me.body.mission.peer_sighting()
  assert seen is not None, "the frame's peers never arrived"
  assert 0.3 < seen < 0.8, seen
  # ...and the seam only SEES: what to do about it is the drive's, and
  # nothing has held yet.
  assert me.body.peer_holds == 0
  # a sighting ages out rather than standing for ever
  me.data.time += 1.0
  assert me.body.mission.peer_sighting() is None
  dark, _ = _pair_with_the_peer_across_the_bow(across=0.25, ahead=0.9, near_field=False)
  assert dark.depth_camera is None and dark.body.mission.peer_sighting() is None


class _Sees(_Walk):
  """The drive's own `self` over a stubbed walk, with another robot's body
  seen 0.5 m ahead on every look."""

  def peer_sighting(self):
    return 0.5


def test_a_peer_the_drive_stops_short_of_is_not_held_for():
  """The hold is against the TRAVEL LEFT, not against the camera (issue
  #328). A robot with 0.1 m still to walk cannot reach a body 0.5 m ahead,
  and holding for one is not caution -- MEASURED, it took a charge approach
  from 96 s and a dock to 201 s and none."""
  short = _Sees(step=0.01)
  assert tick.run(STEPPER, short.drive_to_routine(0.1, 0.0, 20.0))
  assert short.peer_holds == 0, "held for a body it stops short of"
  # ...and the same sighting with the goal beyond it does hold
  far = _Sees(step=0.01)
  tick.run(STEPPER, far.drive_to_routine(2.0, 0.0, 2.0))
  assert far.peer_holds > 0, "walked on with a body in the way"
