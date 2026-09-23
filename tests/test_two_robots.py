"""Two robots in one world (issue #167, M12), slice A: the namespacing.

A second robot is the same model file attached with a prefix; a
`RobotHandle` is that prefix, and every class that resolves a robot element
does so through it. The first robot keeps the bare names, so a single-robot
world is byte-for-byte what it was -- the parity flight is the proof at
mission scale, and the first test here is it at the scale of a spin and a
drive.
"""

import math
import pathlib
import re

import mujoco
import numpy as np
import pytest

from pluggybot.control import wheel_targets
from pluggybot.mission.mission import HubMission
from pluggybot.rack.coupling import (
  HUB_STATION_YS, module_power_contact, rack_charge_contact,
)
from pluggybot.rack.swap import HubSwap
from pluggybot.robot import FIRST, SECOND, RobotHandle, world_with_robots
from pluggybot.tools.gripper import CLAW_MODULE

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
PARK = (5.5, 5.5)          # room_hub: a far corner, off every route


@pytest.mark.parametrize("world", ["models/room_hub.xml", "models/home_world.xml"])
def test_two_namespaced_robots_compile(world):
  model = world_with_robots(world, second_at=(9.0, -5.0) if "home" in world else PARK)
  names = {model.body(i).name for i in range(model.nbody)}
  assert FIRST.root in names and SECOND.root in names
  for el in ("lift", "arm", "left_motor", "right_motor"):
    model.actuator(SECOND.el(el))
  for el in ("dock_eye", "left_eye"):
    model.camera(SECOND.el(el))
  model.sensor(SECOND.el("imu_gyro"))
  for el in ("lidar", "fork_vertex"):
    model.site(SECOND.el(el))
  assert FIRST.qpos_adr(model) == 0 and SECOND.qpos_adr(model) > 7
  # ...and the rack, the bays and the modules are the WORLD's, not prefixed
  assert "rack" in names and "module_pen" in names
  assert not any(n.startswith("r2_module") or n.startswith("r2_rack") for n in names)


def _subtree_hash(model, data, handle: RobotHandle):
  root = model.body(handle.root).id
  ids = [i for i in range(model.nbody) if model.body_rootid[i] == root]
  return np.concatenate([data.xpos[ids].ravel(), data.xquat[ids].ravel()]).tobytes()


def test_a_parked_second_robot_leaves_the_first_robots_trajectory_byte_identical():
  """The parity claim at test scale. MuJoCo 3.10's solver is island-
  separable, so an extra robot resting on the floor changes nothing in the
  first robot's numbers -- measured near and far before this was built."""
  def fly(model):
    data = mujoco.MjData(model)
    m = HubMission(model, data, viewer=None, realtime=False)
    m.start_at(0.5, 3.0, 1.5708)
    m._spin()
    m.drive_to(1.5, 2.0, timeout=12.0)
    return _subtree_hash(model, data, FIRST), m.pose, float(data.time)
  alone = fly(mujoco.MjModel.from_xml_path("models/room_hub.xml"))
  with_r2 = fly(world_with_robots("models/room_hub.xml", second_at=PARK))
  assert alone == with_r2


def test_the_second_robot_drives_through_its_handle_and_the_first_stays_put():
  model = world_with_robots("models/room_hub.xml", second_at=PARK)
  data = mujoco.MjData(model)
  first = HubMission(model, data, viewer=None, realtime=False, handle=FIRST)
  second = HubMission(model, data, viewer=None, realtime=False, handle=SECOND)
  first.start_at(0.5, 3.0, 1.5708)
  second.start_at(3.0, 3.0, 0.0)
  before = _subtree_hash(model, data, FIRST)
  q = SECOND.qpos_adr(model)
  x0 = float(data.qpos[q])
  second._drive(2.0, 0.15, 0.0)
  assert second.face(1.2)
  assert float(data.qpos[q]) - x0 > 0.2, "the second robot did not move"
  assert abs(second.pose[2] - 1.2) < 0.05
  assert abs(first.pose[0] - 0.5) < 1e-6 and abs(first.pose[1] - 3.0) < 1e-6
  # the first robot's bodies did not move while the second drove (it was
  # held by its own brake; the hash is over its whole subtree)
  after = _subtree_hash(model, data, FIRST)
  assert np.frombuffer(before, float).round(3).tolist() == \
    np.frombuffer(after, float).round(3).tolist()


def test_the_second_robot_picks_a_module_and_only_its_own_fork_powers_it():
  """The coupling through the handle: the second robot's fork plates are
  `r2_fork_v*`, so `module_power_contact` with its prefix says powered while
  the first robot's says not -- a module on the other robot's fork is not
  this robot's."""
  model = world_with_robots("models/hub_world.xml", second_at=(1.5, 1.5))
  data = mujoco.MjData(model)
  first = HubSwap(model, data, handle=FIRST)
  first.place_at_standoff(HUB_STATION_YS[0])          # out of the way, bay A
  second = HubSwap(model, data, handle=SECOND)
  second.place_at_standoff(HUB_STATION_YS[3])         # the claw's bay
  second.pick()
  assert second.module_state(CLAW_MODULE)["on_fork"]
  assert module_power_contact(model, data, CLAW_MODULE, SECOND.prefix)
  assert not module_power_contact(model, data, CLAW_MODULE, FIRST.prefix)
  assert not first.module_state(CLAW_MODULE)["on_fork"]
  assert not rack_charge_contact(model, data, SECOND.prefix)


# ---- the fence: no bare robot name in mission code -----------------------------

#: Element names that belong to a robot, which mission code may only reach
#: through a handle. Module, rack, bay and board names are the WORLD's.
ROBOT_ELEMENTS = ("chassis", "lift", "arm", "left_motor", "right_motor",
                  "left_wheel_joint", "right_wheel_joint", "lift_joint",
                  "arm_joint", "imu_gyro", "dock_eye", "left_eye", "lidar",
                  "fork_vertex", "pluggybot")
#: The mission stack: what a second robot runs a copy of.
MISSION_CODE = ("lifecycle.py", "power.py", "rack/swap.py", "mission/mission.py",
                "mission/errand.py", "tools/drawing.py", "tools/gripper.py",
                "tools/dispenser.py", "tools/screen.py", "procedure/steps.py",
                "procedure/axes.py", "procedure/lang.py", "perception/lidar.py",
                "rack/localize.py", "rack/tags.py", "economy/census.py")


def test_mission_code_resolves_every_robot_element_through_the_handle():
  bare = re.compile(r'\.(body|geom|joint|actuator|site|camera|sensor)\("('
                    + "|".join(ROBOT_ELEMENTS) + r')"\)')
  root_index = re.compile(r"qpos\[(0|1|2|3:7|:7|:3)\]")
  bad = []
  for rel in MISSION_CODE:
    text = (SRC / rel).read_text()
    for n, line in enumerate(text.splitlines(), 1):
      if bare.search(line) or root_index.search(line):
        bad.append(f"{rel}:{n}: {line.strip()}")
  assert not bad, "bare robot names in mission code:\n  " + "\n  ".join(bad)


# ---- slice B: one loop, two controllers, mutual awareness ----------------------


class _Swap:
  """Records the order the loop calls it in; raises from a hook on cue."""

  def __init__(self, log, name, raise_at=None):
    self.log, self.name, self.raise_at, self.after = log, name, raise_at, 0

  def _before_step(self, tl, tr):
    self.log.append((self.name, "before", round(tl, 3)))

  def _after_step(self):
    self.after += 1
    self.log.append((self.name, "after"))
    if self.raise_at is not None and self.after == self.raise_at:
      raise RuntimeError("hook")


def test_run_many_applies_every_command_steps_once_then_books_every_robot():
  from pluggybot import tick
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
  assert log[:5] == [("a", "before", round(wheel_targets(0.1, 0.0)[0], 3)),
                     ("b", "before", round(wheel_targets(0.2, 0.0)[0], 3)),
                     ("world", "step"), ("a", "after"), ("b", "after")]
  # step 2: b has returned and holds zero while a finishes
  assert log[5:10] == [("a", "before", round(wheel_targets(0.1, 0.0)[0], 3)),
                       ("b", "before", 0.0),
                       ("world", "step"), ("a", "after"), ("b", "after")]
  assert len(log) == 10


def test_run_many_throws_a_hook_exception_into_every_live_routine():
  from pluggybot import tick
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


def test_a_failed_pick_ends_the_errand_at_the_rack():
  """Issue #298. On the deployed pair a pick fails whenever the other robot
  holds the tool, and the errand used to carry on regardless: drive to
  the use pose with nothing on the fork, skip the use, drive back and
  attempt to RETURN a module it never had. That phantom trip parked the
  robot at the rack exactly when the other came back to stow, whose stow
  then failed, whose next pick then failed -- Rowan's correct answers
  paid 3 of 27, and it filed a ticket about its pen. Now the errand ends
  at the rack, scored as it stands, and History says why.
  """
  from test_overseer import _lifecycle
  from pluggybot import tick
  from pluggybot.mission.errand import Errand
  life = _lifecycle("room_hub", errand=False)
  drives, swaps = [], []
  life.mission.drive_to_routine = lambda *a, **kw: (drives.append(a), tick.result(True))[1]
  life.mission.swap_at_bay_routine = lambda *a, **kw: (swaps.append(a[1]), tick.result(None))[1]
  # the other robot has the LCD: not on this fork, not on its bay
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False, "hung": False}
  used = []
  errand = Errand(name="carry:test", module="module_lcd", station_y=0.0,
                  use_at=(1.0, 1.0), use=lambda _l: used.append(1) or {},
                  needs_use_pose=False)
  result = life.run_errand(errand)
  assert swaps == ["pick"], swaps                      # no return of nothing
  assert drives == [] and used == []                  # no trip, no use
  assert result["picked"] is False and result["stowed"] is False
  assert result["error"] == "never picked up module_lcd"
  assert any("could not pick up module_lcd" in ln and "not on its bay" in ln
             for ln in life.thoughts.read("History.md").splitlines()), \
      life.thoughts.read("History.md")
  # ...and a pick that MISSED with the module still hanging says that instead
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False, "hung": True}
  result = life.run_errand(errand)
  assert result["picked"] is False and result["stowed"] is True
  assert any("the pick missed and it is still on its bay" in ln
             for ln in life.thoughts.read("History.md").splitlines())


def test_the_planner_routes_round_the_other_robots_reported_pose():
  """Unit test on the grid, no mission: a free room, the other robot
  reported on the straight line, and every waypoint keeps clear of it."""
  from pluggybot.mission.mission import OTHER_ROBOT_CELLS
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  m.grid.grid[:] = -5.0                      # everything known free
  goal = (4.0, 1.0)
  straight = m._plan_to(*goal)
  assert straight is not None
  m.others = [lambda: (2.5, 1.0)]
  bent = m._plan_to(*goal)
  assert bent is not None
  clearance = OTHER_ROBOT_CELLS * m.grid.resolution
  assert all(math.hypot(x - 2.5, y - 1.0) > clearance - m.grid.resolution
             for x, y in bent), "a waypoint passes through the other robot"
  assert any(math.hypot(x - 2.5, y - 1.0) < clearance for x, y in straight)
  assert m._other_in_the_way(*goal) is False
  assert m._other_in_the_way(2.6, 1.2) is True


def test_a_blocked_drive_waits_for_the_other_robot_instead_of_giving_up():
  """The other robot reported ON the goal: the drive stagnates, sees the
  other in the way, and waits out its timeout rather than returning False
  after ten seconds of no progress."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  m.grid.grid[:] = -5.0
  m.others = [lambda: (1.6, 1.0)]
  t0 = data.time
  arrived = m.drive_to(1.6, 1.0, timeout=14.0)
  assert not arrived
  assert data.time - t0 >= 13.5, "gave up before its timeout"


def test_explore_does_not_spin_when_the_other_robot_blocks_the_only_route(monkeypatch):
  """Flown in the deployed pair: the frontier planner, which does not mask
  the other robot, called a frontier "ok" that the drive planner, which
  does, could not route to -- so the drive returned without stepping and
  explore asked again at the same sim instant, forever. A walled corridor
  with the other robot standing in it is that geometry; the plan count is
  the tripwire, because a regression here is a hang, not a failure."""
  from pluggybot import lifecycle as lc, tick
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         world="room_hub", errand=False,
                         battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                         grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"])
  life.max_sim_time, life.blacklist = 3600.0, set()   # what run() sets up
  m = life.mission
  m.start_at(1.0, 1.0, 0.0)
  g = m.grid
  g.grid[:] = 5.0                                  # walls everywhere...
  x0, y0 = g.world_to_cell(0.2, 0.2)
  x1, y1 = g.world_to_cell(3.0, 1.8)
  g.grid[y0:y1, x0:x1] = -5.0                      # ...but a known-free corridor
  g.grid[y0:y1, x1:g.world_to_cell(3.6, 0.0)[0]] = 0.0   # the only frontier
  m.others = [lambda: (2.2, 1.0)]                  # standing across it

  # The premise: the two planners disagree about this frontier.
  path, status = lc.plan(g, m.pose, set())
  assert status == "ok"
  assert m._plan_to(*g.cell_to_world(*path[-1])) is None

  real, plans = lc.plan, []

  def counted(*a, **kw):
    plans.append(life.data.time)
    if len(plans) > 50:
      raise AssertionError(f"explore replanned {len(plans)} times at sim "
                           f"time {life.data.time:.3f} without stepping")
    return real(*a, **kw)

  monkeypatch.setattr(lc, "plan", counted)
  # The spin is 7 s of physics and not the claim; stubbed, nothing here steps.
  m._spin_routine = lambda *a, **kw: tick.result(None)
  step = tick.Step(life.explore_routine(budget=30.0))
  assert step.tick() is None and step.done, "explore neither stepped nor ended"
  assert len(plans) == lc.STRIKES_TO_FINISH
  assert life.map_done


def test_the_lidar_drops_the_other_robots_body_from_the_scan():
  model = world_with_robots("models/room_hub.xml", second_at=(2.0, 3.0))
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(0.5, 3.0, 0.0)                  # facing +x, the other 1.5 m ahead
  mujoco.mj_forward(model, data)
  angles, ranges = m.lidar.scan(data)
  ahead = np.abs(angles) < 0.15
  assert ranges[ahead].min() < 1.6, "the other robot is not in the scan"
  m.lidar.exclude_robot(SECOND.root)
  angles, ranges = m.lidar.scan(data)
  ahead = np.abs(angles) < 0.15
  assert not ahead.any() or ranges[ahead].min() > 3.0, \
    "the other robot's body is still in the scan"


# ⚠ BEHIND `--endurance` (suite budget, 2026-09-13): 137 s under the
# parallel suite. Its RULE -- every robot's command applied, one step, every
# robot booked, an exception thrown into every live routine -- is
# `test_run_many_*` above in milliseconds; what only this proves is that a
# real fetch survives being ticked beside another robot's day.
@pytest.mark.slow
@pytest.mark.endurance
def test_two_robots_run_from_one_loop_and_the_first_fetches_its_tool():
  """The pair, flown: one `mj_step` loop, two days; robot 1 fetches the LCD
  while robot 2 explores, and both are alive when the pick lands. Stops on
  its claim."""
  from pluggybot.pair import build_pair, run_pair
  lives = build_pair("room_hub", pack="hosting", errands=("carry", "none"))
  assert lives[0].mission.others and lives[1].mission.others
  results = run_pair(lives, max_sim_time=200.0,
                     stop_when=lambda ls: ls[0].swaps_done >= 1)
  assert results[0]["swaps_done"] >= 1 and results[0]["aborted"]
  assert all(r["dead"] is None for r in results)
  assert results[0]["collision_steps"] == 0 and results[1]["collision_steps"] == 0
  assert lives[0].data.time == lives[1].data.time


# ---- slice E: the wire ------------------------------------------------------


def test_the_census_and_the_scene_key_every_robot_by_its_root():
  from pluggybot.telemetry.protocol import body_census, robot_roots
  from pluggybot.telemetry.scene import scene_dict
  model = world_with_robots("models/room_hub.xml", second_at=PARK)
  assert robot_roots(model) == [FIRST.root, SECOND.root]
  alone = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  assert robot_roots(alone) == [FIRST.root]
  r1, world = body_census(model)
  r2, world2 = body_census(model, SECOND.root)
  assert r1 and r2 == [SECOND.el(n) for n in r1]
  assert world == world2 and not any(n.startswith("r2_") for n in world)
  assert body_census(alone) == (r1, world), "a single-robot census moved"
  scene = scene_dict(model, "room_hub")
  owners = {b["name"]: b["robot"] for b in scene["bodies"]}
  assert owners[SECOND.root] == SECOND.root and owners[FIRST.root] == FIRST.root
  assert owners[SECOND.el("head")] == SECOND.root
  assert owners["rack"] is None and owners["module_pen"] is None


# ⚠ BEHIND `--endurance` (suite budget, 2026-09-13): 226 s under the
# parallel suite -- `build_pair` + `arrange_game` + two lifecycles for 12 s
# of sim. The wire SHAPE it flies for is proved in a second by the vendored
# pair fixture (`test_the_pair_recording_gives_every_robot_the_same_shape`)
# and the per-role claim by `test_a_two_role_task_is_claimed_per_role_*`;
# what only this proves is the recorder wired live into a running pair.
@pytest.mark.slow
@pytest.mark.endurance
def test_a_pair_recording_carries_both_robots_and_keys_every_event(tmp_path):
  """The stream with a second robot on it: the header names both, every
  frame carries `robots[<root>]` for each, and the ledger's, thoughts',
  encounters' and referee's events say whose they are. A single-robot
  stream is what it was (the fixture test)."""
  import json
  from pluggybot.pair import arrange_game, build_pair, run_pair
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     tasks=True, overseer=None, thoughts_root=str(tmp_path / "t"))
  arrange_game(lives)
  # `pin`, not `learn`: the verb was retired with the memory tiers (issue
  # #221) and this line has raised `AttributeError` at setup ever since --
  # an endurance test nobody ran is an endurance test nobody has.
  lives[0].thoughts.pin("the other one is quick", t=0.0)
  lives[1].thoughts.pin("the first one is slow", t=0.0)
  path = str(tmp_path / "pair.jsonl")
  run_pair(lives, max_sim_time=12.0, record=path)
  rows = [json.loads(line) for line in open(path)]
  header = rows[0]
  assert header["type"] == "header"
  assert set(header["robots"]) == {FIRST.root, SECOND.root}
  assert header["robotNames"] == {FIRST.root: "Pluggy", SECOND.root: "Rowan"}
  assert header["robots"][SECOND.root] == [SECOND.el(n) for n in header["robots"][FIRST.root]]
  assert header["ledger"] == [FIRST.root, SECOND.root]
  assert header["model"] == "room_hub_pair"
  goals = [r for r in rows if r.get("type") == "goals"]
  assert [g["robot"] for g in goals] == [FIRST.root, SECOND.root]
  frames = [r for r in rows if "type" not in r]
  assert frames and all(set(f["robots"]) == {FIRST.root, SECOND.root} for f in frames)
  assert frames[0]["robots"][SECOND.root]["bodies"]
  thoughts = [r for r in rows if r.get("type") == "thought"]
  assert {t["robot"] for t in thoughts} == {FIRST.root, SECOND.root}
  claims = [r for r in rows if r.get("type") == "task_claimed"]
  assert claims and claims[-1]["claims"] == {"hider": FIRST.root, "seeker": SECOND.root}
  assert header["activities"] == ["encounters", "hide_and_seek"], "the referee must be advertised"


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
  model = world_with_robots("models/room_hub.xml", second_at=PARK)
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
  assert pair_model_name("room_hub") == "room_hub_pair"
  import inspect
  from pluggybot import pair
  assert "pair_model_name(cfg[\"model_name\"])" in inspect.getsource(pair.record_pair)


def test_a_robot_standing_by_for_work_clears_the_rack_first(monkeypatch):
  """Measured on the pair fixture: the second robot stood by at the bay
  standoff after a stow and the first robot's next pick failed 0.4 m from
  it. Standing by within `RACK_CLEAR_M` of the rack prior drives home first;
  standing by anywhere else stays put -- the branch is checked with the
  drive stubbed, on the loop that actually runs it."""
  from pluggybot import lifecycle as lc, tick
  from pluggybot.rack.coupling import RACK_ROOM_POS
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  monkeypatch.setattr(lc, "WAIT_FOR_WORK_S", 0.2)   # the slice length is not the claim

  def life_at(x, y):
    life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                           world="room_hub", errand=False,
                           battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                           grid_bounds=cfg["grid_bounds"],
                           low_battery_wh=cfg["low_battery_wh"])
    life.expects_work = True
    drives = []

    def explored(*a, **kw):
      # Where the robot BELIEVES it ended up exploring, without driving.
      life.mission.swap.reckoner.x, life.mission.swap.reckoner.y = x, y
      life.map_done = True
      return tick.result(None)

    life.explore_routine = explored
    life.mission.drive_to_routine = lambda *a, **kw: (drives.append(a), tick.result(True))[1]
    # The opening spin is 7 s of real physics and says nothing about this
    # branch; stubbed, two stand-by slices are the whole day.
    life.mission._spin_routine = lambda *a, **kw: tick.result(None)
    life.run(cfg["start"], max_sim_time=1.5)
    return drives

  rx, ry = RACK_ROOM_POS
  near = life_at(rx + 0.3, ry - 0.6)               # the bay standoff
  assert near and near[0][:2] == cfg["start"][:2], \
    f"a robot idling at the rack did not clear it: drives={near}"
  far = life_at(cfg["start"][0], cfg["start"][1])
  assert not far, f"a robot already clear of the rack drove anyway: {far}"


def test_a_decided_idle_clears_the_rack_first(monkeypatch):
  """Issue #298. With a mind, the loop never reaches the standby branch
  above: a decided `idle` -- the model's, a map row's, the fallback floor's
  -- stood still wherever the robot was, and after a failed pick or a
  charge that is the bay standoff. On the deployed pair Rowan idled there
  from 550 s ("waiting out the pen"); Luca's next stow failed, its picks
  failed, and its drive home planned "no route to the charge bay" in 0 s,
  inside Rowan's mask against the south wall. Same rule as standing by
  for work: within `RACK_CLEAR_M`, drive home first; elsewhere, stay put.
  """
  from pluggybot import lifecycle as lc, tick
  from pluggybot.mind.overseer import Decision
  from pluggybot.rack.coupling import RACK_ROOM_POS
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])

  def idle_at(x, y):
    life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                           world="room_hub", errand=False,
                           battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                           grid_bounds=cfg["grid_bounds"],
                           low_battery_wh=cfg["low_battery_wh"])
    life.home_pose = tuple(float(v) for v in cfg["start"])
    life.mission.swap.reckoner.x, life.mission.swap.reckoner.y = x, y
    drives = []
    life.mission.drive_to_routine = lambda *a, **kw: (drives.append(a), tick.result(True))[1]
    life.mission._drive_routine = lambda *a, **kw: tick.result(None)   # the idle itself
    life._after_decision(Decision(action="idle", reason="standing by"))
    return drives

  rx, ry = RACK_ROOM_POS
  near = idle_at(rx + 0.3, ry - 0.6)               # the bay standoff
  assert near and near[0][:2] == cfg["start"][:2], \
    f"a robot that decided to idle at the rack did not clear it: drives={near}"
  assert not idle_at(cfg["start"][0], cfg["start"][1]), "already clear, drove anyway"


def test_an_encounter_is_met_on_the_way_in_and_parted_on_the_way_out_with_hysteresis():
  """`activity/encounter.py`: `met` at <= 1.5 m, `parted` at >= 2.0 m, and
  NOTHING in between -- a pair hovering at 1.7 m does not chatter. Poses
  are written straight into the data (the referee test's `_place`): what is
  under test is the rule, not the drive."""
  from pluggybot.activity.encounter import ENCOUNTER_M, PARTED_M, Encounters
  model = world_with_robots("models/room_hub.xml", second_at=PARK)
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


def test_a_second_robots_strokes_are_its_own_on_the_wire():
  """A `draw` / `board_cleared` message names the robot that drew (0.20.0),
  and the drawing errand is the one place strokes are made -- it must pass
  THIS robot's root, or a second robot's drawing is the first's on the wire
  (found flying `serve.py --pair`, issue #181: Rowan took a draw task off the
  board and every stroke said `pluggybot`)."""
  import inspect
  from pluggybot.mission import errand
  src = inspect.getsource(errand.drawing_errand)
  assert src.count("by=life.root") == 2, "clear AND stroke name the robot"


def test_the_second_robot_wears_its_own_livery_and_nothing_else_is_repainted():
  """An attached robot is repainted on exactly the geoms that carry the
  first robot's livery colour (the chassis and the head mount), so two
  robots in one room are told apart at a glance -- on the site, which keeps
  the producer's colour per geom, and in each other's cameras. Everything
  else keeps its colour: the battery's red and the tags are how a reader
  tells parts apart, and a hint never rides as a colour."""
  import numpy as np
  from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, world_with_robots
  model = world_with_robots("models/room_hub.xml", second_at=PARK)
  name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)  # noqa: E731
  purple = {name(i) for i in range(model.ngeom)
            if np.allclose(model.geom_rgba[i], SECOND_CHASSIS_RGBA, atol=1e-6)}
  blue = {name(i) for i in range(model.ngeom)
          if np.allclose(model.geom_rgba[i], CHASSIS_RGBA, atol=1e-6)}
  assert purple == {SECOND.el("chassis"), SECOND.el("head_mount")}
  assert blue == {"chassis", "head_mount"}, "the first robot keeps its paint"
  # ...and the paint is opt-out: `None` attaches an identical twin.
  twin = world_with_robots("models/room_hub.xml", second_at=PARK, chassis_rgba=None)
  assert np.allclose(twin.geom(SECOND.el("chassis")).rgba, CHASSIS_RGBA)


# ---- the pair as an OBSTACLE (issues #316, #313) -----------------------------


def test_the_front_stop_sees_the_other_robot_and_the_map_still_does_not():
  """Issue #316. One scan fed two consumers with opposite needs -- the map,
  which must not contain the other robot (#167: painted in and inflated, a
  robot driving past walls in the robot it passed), and the 0.25 m reflex,
  which must -- and the exclusion silenced both. The only moving obstacle
  in the world was the one thing no sensor on board could see: `met` 382
  against nine `stuck` deaths in the seven days that found it.

  Drive one robot at a stationary peer and it stops. Take `scan_split`
  away -- feed `grid.update` and the reflex the same filtered scan -- and
  it drives on into it.
  """
  model = world_with_robots("models/room_hub.xml", second_at=(2.0, 3.0))
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(0.5, 3.0, 0.0)                  # facing +x, the other 1.5 m ahead
  m.lidar.exclude_robot(SECOND.root)         # as `pair.build_pair` does
  mujoco.mj_forward(model, data)
  for _ in range(40):                        # stop on the claim: ~0.2 s a go
    m.run(m._drive_routine(0.2, 0.3, 0.0))
    if m.backoff_until > 0.0:
      break
  assert m.backoff_until > 0.0, "the reflex never armed for the other robot"
  assert m.pose[0] < 1.6, "stopped before it could have seen anything"
  # ...and the map is still innocent of it: no cell under the other robot
  # is an obstacle, which is the half #167 measured and this must not undo.
  ix, iy = m.grid.world_to_cell(2.0, 3.0)
  around = m.grid.grid[iy - 3:iy + 4, ix - 3:ix + 4]
  assert (around <= 0.0).all(), "the other robot was painted into the map"


def test_a_scan_answers_the_room_and_the_peer_apart():
  """`Lidar.scan_split` is the whole of the fix: one set of casts, the
  room's returns and the peer's, and `scan()` still answers the map's."""
  model = world_with_robots("models/room_hub.xml", second_at=(2.0, 3.0))
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(0.5, 3.0, 0.0)
  mujoco.mj_forward(model, data)
  m.lidar.exclude_robot(SECOND.root)
  angles, ranges, peer_angles, peer_ranges = m.lidar.scan_split(data)
  ahead = np.abs(peer_angles) < 0.15
  assert ahead.any() and peer_ranges[ahead].min() < 1.6, \
    "the peer's own returns are missing"
  room = np.abs(angles) < 0.15
  assert not room.any() or ranges[room].min() > 3.0, \
    "the other robot is still in the map's scan"
  # ⚠ A PEER RETURN MAY NOT DRAW ON THE MAP'S NOISE STREAM (`Lidar.peer_rng`).
  # With one stream, giving the reflex its peer returns would have moved
  # every reading the MAP takes in a pair world -- and then no pair flight
  # could say whether it changed because the reflex brakes now or because
  # the sensor rolled different numbers. MEASURED both ways: with two
  # streams a pair world's 20-scan map hash is identical to the scan
  # before issue #316; with one it is not.
  #
  # Counted rather than hashed: at `dropout` 0 every room ray that HIT
  # draws exactly twice (the dropout roll and the noise), a ray that
  # returned nothing draws not at all, and a peer ray must draw neither.
  class Counting:
    def __init__(self, inner):
      self.inner, self.draws = inner, 0

    def random(self, *a, **kw):
      self.draws += 1
      return self.inner.random(*a, **kw)

    def normal(self, *a, **kw):
      self.draws += 1
      return self.inner.normal(*a, **kw)

  m.lidar.dropout = 0.0
  m.lidar.rng = counted = Counting(np.random.default_rng(7))
  room_a, room_r, peer_a, _ = m.lidar.scan_split(data)
  assert peer_a.size, "no peer returns in this frame at all"
  hits = int((room_r < m.lidar.max_range).sum())
  assert counted.draws == 2 * hits, \
    f"the map's stream was drawn {counted.draws} times for {hits} of its own returns"
  # A world with nobody else in it splits into nothing: the single-robot
  # path is the same ray loop it always was.
  m.lidar._other_geoms = set()
  _, _, alone_a, alone_r = m.lidar.scan_split(data)
  assert alone_a.size == 0 and alone_r.size == 0


def test_a_robot_that_drives_into_its_pair_leaves_a_row():
  """Issue #316's fourth question: a collision left nothing behind but
  `HubSwap.collision_steps`, which is on no record and no wire, so a week
  of `stuck` deaths could not be attributed either way. Contact is now a
  phase of the encounter -- sensed off the contact array, latched like the
  plate's press."""
  from pluggybot.activity.encounter import Encounters
  model = world_with_robots("models/room_hub.xml", second_at=(2.0, 3.0))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  enc = Encounters(model, FIRST, SECOND)
  events = []
  enc.on_event.append(events.append)
  enc.sense(model, data)
  assert enc.bumps == 0 and not enc.flags["touching"]
  # ...now put the second robot where its body meets the first's, which
  # starts at the origin: the chassis is 0.24 m long, so 0.2 m of
  # separation is two chassis in contact (measured: 24 contact points; at
  # 0.25 m, none).
  adr = SECOND.qpos_adr(model)
  data.qpos[adr:adr + 2] = [0.2, 0.0]
  mujoco.mj_forward(model, data)
  assert data.ncon, "no contacts at all: the world is not touching anything"
  enc.sense(model, data)
  touched = [e for e in events if e["phase"] == "touched"]
  assert touched, "two robots in contact and nothing on the wire"
  assert set(touched[0]["robots"]) == {FIRST.root, SECOND.root}
  assert enc.bumps == 1 and enc.flags["touching"]
  enc.sense(model, data)
  assert enc.bumps == 1, "one contact, counted twice"
  # ...and it is over once they are apart, past the bumper's own hold.
  data.qpos[adr:adr + 2] = [2.0, 3.0]
  mujoco.mj_forward(model, data)
  data.time += 1.0
  enc.sense(model, data)
  assert not enc.flags["touching"]
  assert [e["phase"] for e in events][-1] == "separated"


def test_a_peer_at_the_neighbouring_bay_puts_the_standoff_out_of_reach():
  """Issue #313, as arithmetic rather than a correlation.

  `_mask_others` takes 0.60 m out of the traversable mask around the other
  robot's reported pose, and a stagnated drive counts as arrived only
  inside 0.15 m -- so a peer nearer the goal than the difference makes
  arrival impossible however many attempts are spent on it. That is the
  measured table in #313: a robot at one bay's standoff is 0.26 m from its
  neighbour's and the pick failed 0/3; 0.56 m away it landed 3/3.
  """
  from pluggybot.mission.mission import CLOSE_ENOUGH_M
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  m.grid.grid[:] = -5.0                      # everything known free
  gx, gy = 2.0, 1.0
  for beside, reachable in ((0.26, False), (0.56, True)):
    m.others = [lambda d=beside: (gx + d, gy)]
    near = m.peer_on_the_goal(gx, gy)
    path = m._plan_to(gx, gy)
    assert path is not None
    short = math.hypot(path[-1][0] - gx, path[-1][1] - gy)
    assert (near is None) is reachable, f"{beside} m: {near}"
    # ...and the rule agrees with what the planner actually does, which is
    # the only reason to trust it: the best cell A* may use is that far off.
    assert (short < CLOSE_ENOUGH_M) is reachable, f"{beside} m: {short:.3f} m short"
  m.others = []
  assert m.peer_on_the_goal(gx, gy) is None


def test_a_bay_given_up_for_a_peer_says_so_instead_of_trying_again():
  """A drive that cannot arrive is not a drive worth a second attempt
  (issue #313): the swap spent another 90 s timeout and a spin to reach the
  same unreachable point, then reported "no route" -- and the robot read
  that as a fault in its own tool."""
  from pluggybot import tick
  from pluggybot.mission.mission import bay_standoff
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  station = HUB_STATION_YS[0]
  sx, sy, _ = bay_standoff(station, m.rack)
  spins = []
  m._spin_routine = lambda *a, **kw: (spins.append(1), tick.result(None))[1]
  m.drive_to_routine = lambda *a, **kw: tick.result(False)   # never arrives
  m.others = [lambda: (sx + 0.26, sy)]
  why = m.run(m.swap_at_bay_routine(station, "pick", module="module_lcd"))
  assert why == "peer-at-bay" and spins == []
  assert m.peer_at_bay_m == pytest.approx(0.26, abs=0.02)
  # ...and a bay nobody is standing on fails exactly as it always did.
  m.others = [lambda: (sx + 2.0, sy)]
  why = m.run(m.swap_at_bay_routine(station, "pick", module="module_lcd"))
  assert why == "no-route" and len(spins) == 2
  assert m.peer_at_bay_m is None


def test_a_pick_lost_to_a_peer_names_it_in_history():
  """The narration #313 asks for: which robot, how far off. Rowan
  diagnosed its pen, told the other robot to move out of a pose measured
  harmless, and started declining pen jobs -- off a correlation in its own
  History that no line ever named."""
  from test_overseer import _lifecycle
  from pluggybot import tick
  from pluggybot.mission.errand import Errand
  from pluggybot.mission.mission import bay_standoff
  life = _lifecycle("room_hub", errand=False)
  station = HUB_STATION_YS[0]
  sx, sy, _ = bay_standoff(station, life.mission.rack)

  class Peer:                                # the public surface, no more
    robot_name, root = "Rowan", SECOND.root

    class mission:
      pose_xy = staticmethod(lambda: (sx + 0.26, sy))

  life.peers = [Peer()]
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(True)

  def blocked(*a, **kw):                     # what the real swap just did
    life.mission.peer_at_bay_m = 0.26
    return tick.result("peer-at-bay")

  life.mission.swap_at_bay_routine = blocked
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False, "hung": True}
  errand = Errand(name="carry:test", module="module_lcd", station_y=station,
                  use_at=(1.0, 1.0), use=lambda _l: {}, needs_use_pose=False)
  result = life.run_errand(errand)
  assert result["error"] == "never picked up module_lcd"
  assert result["peerAtBayM"] == 0.26
  history = life.thoughts.read("History.md")
  assert "Rowan was standing 0.26 m from the bay" in history, history
  assert "the pick missed" not in history, "blamed the tool for a blocked bay"


def test_a_charge_bay_blocked_by_a_peer_is_named_but_not_given_up_on():
  """The asymmetry (issue #313): the SAME arithmetic covers the charge
  bay -- and the neighbouring tool bay's standoff is 0.200 m from it on
  the rack prior, so a robot swapping there blocks the approach outright
  -- but a tool bay given up is a lost errand and a charge given up is a
  death. So `go_charge_routine` says whose robot it was and still spends
  both of its attempts."""
  from test_overseer import _lifecycle
  from pluggybot import tick
  from pluggybot.mission.mission import charge_standoff
  life = _lifecycle("room_hub", errand=False)
  sx, sy, _ = charge_standoff(life.mission.rack)

  class Peer:
    robot_name, root = "Rowan", SECOND.root

    class mission:
      pose_xy = staticmethod(lambda: (sx + 0.2, sy))

  life.peers = [Peer()]
  life.mission.others = [Peer.mission.pose_xy]
  spins = []
  life.mission._spin_routine = lambda *a, **kw: (spins.append(1), tick.result(None))[1]
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
  assert life.mission.run(life.go_charge_routine()) is False
  assert len(spins) == 2, "the charge approach gave up an attempt early"
  assert any("Rowan is standing 0.20 m from it" in line for line in life.log), life.log[-3:]


# ---- the pair's BODIES (issue #328) ------------------------------------------


def _pair_with_the_peer_across_the_bow(across: float, ahead: float = 1.3):
  """A pair with the near-field camera on, the second robot placed `ahead`
  and `across` of the first, and NOTHING BROADCAST between them: no
  reported pose, so the plan-time mask has nothing to route around and the
  only thing standing between the two is what the first robot can sense."""
  from pluggybot.pair import build_pair
  lives = build_pair("room_hub", near_field=True, errands=("none", "none"))
  me, peer = lives
  model, data = me.model, me.data
  px, py, _ = me.mission.pose
  adr = SECOND.qpos_adr(model)
  data.qpos[adr:adr + 2] = [px + ahead, py + across]
  mujoco.mj_forward(model, data)
  me.mission.grid.grid[:] = -5.0            # a mapped, empty room
  me.mission.others = []                    # the broadcast is what we are not using
  return me, peer


def _closest_approach(me, peer) -> float:
  """The nearest the two chassis got, sampled on the physics seam."""
  a = me.model.body(FIRST.root).id
  b = me.model.body(SECOND.root).id
  seen = []

  def watch() -> None:
    pa, pb = me.data.xpos[a], me.data.xpos[b]
    seen.append(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))

  me.mission.step_hooks.append(watch)
  return seen


def test_a_peer_across_the_bow_stops_the_drive_with_nothing_broadcast():
  """Issue #328, the whole of it. The LIDAR's front stop sees the other
  robot's MAST and only inside a +/-0.35 rad cone, so a peer 0.25 m across
  the bow puts ZERO rays in it while its chassis -- 0.15 m of half-width
  the scan plane passes clean over -- is still wide enough to clip. The
  depth camera's peer channel sees the body: measured 685 points at a
  0.5 m gap and 0.25 m across, where the LIDAR has none.

  Flown with NOTHING BROADCAST, because that is the claim: avoidance must
  not rest on the other robot's reported pose. MEASURED both ways, closest
  approach between the two body origins (they meet at 0.24 m):

  | peer across the bow | with the peer channel | without |
  |---|---|---|
  | 0.00 m | 0.551 m | 0.311 m (the LIDAR, on the mast) |
  | 0.25 m | 0.594 m | **0.225 m** |
  | 0.40 m | 0.312 m | 0.275 m |
  """
  me, peer = _pair_with_the_peer_across_the_bow(across=0.25)
  seen = _closest_approach(me, peer)
  px, py, _ = me.mission.pose
  me.mission.drive_to(px + 2.6, py, timeout=25.0)
  assert seen, "the seam never ran"
  assert min(seen) > 0.24, f"the chassis met: closest {min(seen):.3f} m"
  assert me.mission.collision_steps == 0
  assert me.mission.peer_holds > 0, "it kept clear without ever holding"


def test_the_seam_hands_the_frames_peers_to_the_drive():
  """One line of wiring, pinned on its own: `_near_field_step` is where
  the peer channel reaches the mission, and a world without the camera
  keeps the LIDAR-only behaviour of issue #316."""
  from pluggybot.pair import build_pair
  # ...inside the stop range, which the drive test approaches from beyond:
  # at a 0.6 m gap the peer's nearest point is ~0.48 m ahead.
  me, _ = _pair_with_the_peer_across_the_bow(across=0.25, ahead=0.6)
  me._next_near_field = 0.0
  me._near_field_step()
  assert me.mission.peer_sighting() is not None, "the frame's peers never arrived"
  assert 0.3 < me.mission.peer_sighting() < 0.6, me.mission.peer_sighting()
  # ...and the seam only SEES: what to do about it is the drive's, and
  # nothing has held yet.
  assert me.mission.peer_holds == 0
  # a sighting ages out rather than standing for ever
  me.data.time += 1.0
  assert me.mission.peer_sighting() is None
  dark = build_pair("room_hub", near_field=False, errands=("none", "none"))[0]
  assert dark.depth_camera is None and dark.mission.peer_sighting() is None


def test_a_peer_the_drive_stops_short_of_is_not_held_for():
  """The hold is against the TRAVEL LEFT, not against the camera (issue
  #328). A robot with 0.1 m still to drive cannot reach a body 0.5 m
  ahead, and holding for one is not caution -- MEASURED, it took the
  charge approach from 96 s and a dock to 201 s and none, because arriving
  at the standoff turns the robot to face the rack and sweeps the robot
  parked BESIDE the bay through the corridor."""
  me, _ = _pair_with_the_peer_across_the_bow(across=0.25, ahead=0.6)
  me._next_near_field = 0.0
  me._near_field_step()
  seen = me.mission.peer_sighting()
  assert seen is not None and seen < 0.6
  px, py, _ = me.mission.pose
  # a goal it reaches well short of the sighting: no hold, and it arrives
  me.mission.drive_to(px + 0.1, py, timeout=20.0)
  assert me.mission.peer_holds == 0, "held for a body it stops short of"
  # ...and the same sighting with the goal beyond it does hold
  me.mission.step_hooks.append(lambda: None)
  px, py, _ = me.mission.pose
  me.mission.drive_to(px + 2.0, py, timeout=20.0)
  assert me.mission.peer_holds > 0, "drove on with a body in the way"


def test_the_fine_step_is_counted_so_the_first_swap_out_cannot_end_the_others():
  """The swap's step is the MODEL's, and a pair shares one (issue #264): the
  robot whose swap ended first put the cruise step back under the other's
  terminal approach -- peg contacts at twice the step, on a failure one robot
  alone can never show."""
  from types import SimpleNamespace
  from pluggybot.mission import mission as ms
  model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
  ms.fine_step_begin(model)                  # one robot's swap
  ms.fine_step_begin(model)                  # the other's, overlapping
  ms.fine_step_end(model, 0.002)             # ...which ends first
  assert model.opt.timestep == ms.SWAP_TIMESTEP
  ms.fine_step_end(model, 0.002)
  assert model.opt.timestep == 0.002


def test_the_bay_swap_takes_the_fine_step_only_through_the_count():
  import inspect
  src = inspect.getsource(HubMission.swap_at_bay_routine)
  assert "fine_step_begin(self.model)" in src and "fine_step_end(" in src
  assert "opt.timestep =" not in src, "writes the shared step itself"
