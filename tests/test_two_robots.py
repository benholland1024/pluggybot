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


@pytest.mark.slow
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


@pytest.mark.slow
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
  lives[0].thoughts.learn("the other one is quick", t=0.0)
  lives[1].thoughts.learn("the first one is slow", t=0.0)
  path = str(tmp_path / "pair.jsonl")
  run_pair(lives, max_sim_time=12.0, record=path)
  rows = [json.loads(line) for line in open(path)]
  header = rows[0]
  assert header["type"] == "header"
  assert set(header["robots"]) == {FIRST.root, SECOND.root}
  assert header["robotNames"] == {FIRST.root: "Pluggy", SECOND.root: "Bolt"}
  assert header["robots"][SECOND.root] == [SECOND.el(n) for n in header["robots"][FIRST.root]]
  assert header["ledger"] == [FIRST.root, SECOND.root]
  frames = [r for r in rows if "type" not in r]
  assert frames and all(set(f["robots"]) == {FIRST.root, SECOND.root} for f in frames)
  assert frames[0]["robots"][SECOND.root]["bodies"]
  thoughts = [r for r in rows if r.get("type") == "thought"]
  assert {t["robot"] for t in thoughts} == {FIRST.root, SECOND.root}
  claims = [r for r in rows if r.get("type") == "task_claimed"]
  assert claims and claims[-1]["claims"] == {"hider": FIRST.root, "seeker": SECOND.root}
  assert header["activities"] == ["encounters"]
