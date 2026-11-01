"""Two robots in one world (issue #167): the namespacing, one loop, the
wire, and the pair as an obstacle.

A second robot is the same body attached with a prefix; a `RobotHandle` is
that prefix, and every class that resolves a robot element does so through
it. The first robot keeps the bare names, so a single-robot world is what
it was. The world is the humanoid pair's (`humanoid.world.home_spec`);
where a rule is the loop's bookkeeping it is pinned on the stub, and where
it is the drive's, on a stub that walks.
"""

import math
import pathlib
import re

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.body import RackPose, StubBody
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import world as hw
from pluggybot.humanoid.body import HumanoidBody
from pluggybot.lifecycle import HUMANOID_HOME, world_config
from pluggybot.navigator import CLOSE_ENOUGH_M
from pluggybot.robot import FIRST, SECOND, RobotHandle
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_unknown import STEPPER, _Walk

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
#: Where a second robot is parked out of the way: the kitchen.
PARK = (-8.0, 4.0)
BOUNDS = world_config(HUMANOID_HOME)["grid_bounds"]


def _pair(second_at=PARK, **kw):
  return hw.home_spec(second_at=second_at, **kw).compile()


def _stand(model, data, handle: RobotHandle, x, y, yaw=0.0) -> None:
  """`handle`'s humanoid upright at (x, y) facing `yaw`, as
  `HumanoidBody.start_at` stands it: nothing stepped."""
  q = handle.qpos_adr(model)
  data.qpos[q:q + 3] = [x, y, hm.stand_height()]
  data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  stand = hm.stand_qpos()
  for name in hm.qpos_order():
    data.qpos[model.jnt_qposadr[model.joint(handle.el(name)).id]] = stand.get(name, 0.0)
  mujoco.mj_forward(model, data)


def test_two_namespaced_robots_compile():
  model = _pair()
  names = {model.body(i).name for i in range(model.nbody)}
  assert FIRST.root in names and SECOND.root in names
  for el in ("left_knee", "right_shoulder_pitch", "left_elbow"):
    model.actuator(SECOND.el(el))
  for el in ("nav_eye", "depth_eye", "color_eye"):
    model.camera(SECOND.el(el))
  model.sensor(SECOND.el("imu_ang_vel"))
  for el in ("lidar", "charge_port", "left_grip"):
    model.site(SECOND.el(el))
  # the robots come after the house, the second after the first
  assert 0 < FIRST.qpos_adr(model) < SECOND.qpos_adr(model)
  # ...and the station, its plug and the counter's props are the WORLD's,
  # not prefixed
  assert {"station", "station_plug", "store_counter", "chest"} <= names
  assert not any(n.startswith(("r2_station", "r2_store", "r2_chest")) for n in names)


def test_the_second_robot_walks_through_its_handle_and_touches_nothing_of_the_firsts():
  """The second robot's mission resolves every element by its prefix: its
  commands reach `r2_`'s drivers and none of the first robot's."""
  model = _pair(second_at=(2.0, -0.6))
  data = mujoco.MjData(model)
  second = HumanoidBody(model, data, realtime=False, grid_bounds=BOUNDS, handle=SECOND)
  try:
    second.start_at(2.0, -0.6, math.pi)
    firsts = [i for i in range(model.nu)
              if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("r2_")]
    assert firsts
    x0 = second.true_pose()[0]
    second.run(second.mission._drive_routine(1.0, 0.5, 0.0))
    assert x0 - second.true_pose()[0] > 0.1, "the second robot did not walk"
    assert not data.ctrl[firsts].any(), "the second robot drove the first's joints"
  finally:
    second.close()


# ---- the fence: no bare robot name in mission code -----------------------------

#: Element names that belong to a robot, which mission code may only reach
#: through a handle. The station, the counter's props and board names are
#: the WORLD's.
ROBOT_ELEMENTS = ("pluggybot", "pelvis", "torso", "head", "lidar", "depth_eye", "nav_eye",
                  "color_eye", "imu", "imu_ang_vel", "imu_lin_acc", "pelvis_quat",
                  "charge_port", "left_foot", "right_foot", "left_grip", "right_grip",
                  "left_sole", "right_sole", "left_knee", "right_knee",
                  "left_foot_touch", "right_foot_touch")
#: The mission stack: what a second robot runs a copy of. (The humanoid's
#: model of itself -- `arms.py`, `reach.py`, `odometry.py`, `getup.py` --
#: is its own drawing compiled alone, whose root IS `qpos[0]`.)
MISSION_CODE = ("lifecycle.py", "power.py", "body.py", "navigator.py", "mission/errand.py",
                "procedure/steps.py", "procedure/axes.py", "procedure/lang.py",
                "perception/lidar.py", "perception/depth.py", "perception/imu.py",
                "perception/tags.py", "perception/watch.py", "mapping/scan_match.py",
                "humanoid/body.py", "humanoid/mission.py", "humanoid/places.py",
                "humanoid/way.py", "humanoid/charge.py", "humanoid/walk.py",
                "humanoid/policy.py", "humanoid/scan.py", "humanoid/senses.py",
                "humanoid/drivers.py")
#: ...read through the model of itself (`sm.m`, `sm.d`) is not the world's.
BARE = re.compile(r'(?<!body)(?<!sm\.m)\.(body|geom|joint|actuator|site|camera|sensor)\("('
                  + "|".join(ROBOT_ELEMENTS) + r')"\)')
#: A robot's root read at a fixed `qpos` index: the robots are attached
#: after the house, so none of them is at `qpos[0]`.
ROOT_INDEX = re.compile(r"(?<!sm\.d\.)qpos\[(0|1|2|3:7|:7|:3)\]")


def test_mission_code_resolves_every_robot_element_through_the_handle():
  bad = []
  for rel in MISSION_CODE:
    text = (SRC / rel).read_text()
    for n, line in enumerate(text.splitlines(), 1):
      if BARE.search(line) or ROOT_INDEX.search(line):
        bad.append(f"{rel}:{n}: {line.strip()}")
  assert not bad, "bare robot names in mission code:\n  " + "\n  ".join(bad)
  # ...and the fence fires on what it is for, and not on the model of itself
  assert BARE.search('self.model.site("lidar").id') and ROOT_INDEX.search("d.qpos[:7]")
  assert ROOT_INDEX.search("self.data.qpos[3:7]")
  assert not BARE.search('self.model.site(handle.el("lidar")).id')
  assert not BARE.search('sm.m.site("charge_port").id')
  assert not ROOT_INDEX.search("sm.d.qpos[3:7]")


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


def _mission(at=(1.0, 1.0)):
  """A humanoid in the house on a mapped and empty floor: its body, to
  close, and its mission, the planner's own `self`."""
  model = hw.home_spec().compile()
  body = HumanoidBody(model, mujoco.MjData(model), realtime=False, grid_bounds=BOUNDS)
  body.start_at(*at, 0.0)
  body.grid.grid[:] = -5.0
  return body, body.mission


def _plan(m, x, y):
  """A fresh plan: a body keeps its last one for a moment from the same
  spot (`Navigator.REPLAN_HOLD_S`)."""
  m._plan_memo = None
  return m._plan_to(x, y)


def test_the_planner_routes_round_the_other_robots_reported_pose():
  """On the grid, nothing stepped: a free room, the other robot reported on
  the straight line, and every waypoint keeps clear of it."""
  body, m = _mission()
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
    body.close()


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
  alone = hw.home_spec().compile()
  assert robot_roots(alone) == [FIRST.root]
  r1, world = body_census(model)
  r2, world2 = body_census(model, SECOND.root)
  assert r1 and r2 == [SECOND.el(n) for n in r1]
  assert world == world2 and not any(n.startswith("r2_") for n in world)
  assert body_census(alone) == (r1, world), "a single-robot census moved"
  scene = scene_dict(model, HUMANOID_HOME)
  owners = {b["name"]: b["robot"] for b in scene["bodies"]}
  assert owners[SECOND.root] == SECOND.root and owners[FIRST.root] == FIRST.root
  assert owners[SECOND.el("lidar_mount")] == SECOND.root
  assert owners["station"] is None and owners["station_plug"] is None
  assert owners["chest"] is None


def test_a_pair_recording_carries_both_robots_and_keys_every_event(tmp_path):
  """The stream with a second robot on it, off the recorder `run_pair` wires
  (`record_pair`), driven by hand rather than flown: the header names both
  robots and advertises their encounters, a frame carries `robots[<root>]`
  for each, and the thoughts say whose they are. Nothing steps -- the frame
  is the world as built."""
  import json
  from pluggybot.pair import build_pair, record_pair
  lives = build_pair(HUMANOID_HOME, pack="hosting", errands=("none", "none"),
                     tasks=True, overseer=None, thoughts_root=str(tmp_path / "t"))
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
  assert header["model"] == "home_humanoid_pair"
  goals = [r for r in rows if r.get("type") == "goals"]
  assert [g["robot"] for g in goals] == [FIRST.root, SECOND.root]
  frames = [r for r in rows if "type" not in r]
  assert frames and all(set(f["robots"]) == {FIRST.root, SECOND.root} for f in frames)
  assert frames[0]["robots"][SECOND.root]["bodies"]
  thoughts = [r for r in rows if r.get("type") == "thought"]
  assert {t["robot"] for t in thoughts} == {FIRST.root, SECOND.root}
  assert "encounters" in header["activities"], "the pair's encounters must be advertised"


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
  assert pair_model_name(HUMANOID_HOME) == "home_humanoid_pair"
  import inspect
  from pluggybot import pair
  assert "pair_model_name(cfg[\"model_name\"])" in inspect.getsource(pair.record_pair)


# ---- standing by, clear of the rack ----------------------------------------------


#: The frame the charge logic works round (`Body.rack_prior`): the
#: humanoid's charging station, 3.1 m from its start.
RACK = RackPose(*world_config(HUMANOID_HOME)["dock"])


def _by_the_rack():
  """A life on a stub whose rack is the house's."""
  return stub_life(body=StubBody(rack=RACK, grid_bounds=BOUNDS))


def _near_the_rack(life):
  """Half a metre out from the station, where a robot stands after a
  charge."""
  r = life.body.rack_prior
  return r.x + 0.5, r.y


def test_a_robot_standing_by_for_work_clears_the_rack_first(monkeypatch):
  """Measured on the pair: the second robot stood by at the bay standoff
  after a stow and the first robot's next pick failed 0.4 m from it.
  Standing by within `RACK_CLEAR_M` of the rack prior walks home first;
  standing by anywhere else stays put -- the branch is checked on the loop
  that actually runs it, on the stub."""
  from pluggybot import lifecycle as lc
  monkeypatch.setattr(lc, "WAIT_FOR_WORK_S", 0.2)   # the slice length is not the claim
  start = world_config(HUMANOID_HOME)["start"]

  def life_at(where):
    life = _by_the_rack()
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
  start = tuple(float(v) for v in world_config(HUMANOID_HOME)["start"])

  def idle_at(where):
    life = _by_the_rack()
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
  assert {"pelvis", "torso", "head"} <= blue, "the first robot keeps its paint"
  assert purple == {SECOND.el(n) for n in blue}
  # ...and the paint is opt-out: `None` attaches an identical twin.
  twin = _pair(second_rgba=None)
  assert np.allclose(twin.geom(SECOND.el("torso")).rgba, CHASSIS_RGBA)


# ---- the pair as an OBSTACLE (issues #316, #313, #328) -----------------------------


def _pitched(model, data, handle: RobotHandle, x, y, yaw, pitch) -> None:
  """`handle`'s robot standing at (x, y, yaw), its pelvis pitched `pitch`
  rad forward, as a walking trunk rocks."""
  _stand(model, data, handle, x, y, yaw)
  q = handle.qpos_adr(model)
  tilted = np.zeros(4)
  mujoco.mju_mulQuat(tilted, data.qpos[q + 3:q + 7].copy(),
                     np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
  data.qpos[q + 3:q + 7] = tilted
  mujoco.mj_forward(model, data)


def test_a_scan_answers_the_room_and_the_peer_apart():
  """`Lidar.scan_split`: one set of casts, the room's returns and the
  peer's, and `scan()` still answers the map's. The other humanoid's head is
  under a level ring (1.47 m), so the first is pitched forward and its ring
  cuts the other's head a metre ahead."""
  from pluggybot.perception.lidar import Lidar
  model = _pair(second_at=(2.5, 0.5))
  data = mujoco.MjData(model)
  _stand(model, data, SECOND, 2.5, 0.5, 0.0)
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
  # room ray that HIT draws exactly three numbers (the dropout roll, its
  # range's noise and its bearing's), a ray that returned nothing none, and
  # a peer ray none.
  class Counting:
    def __init__(self, inner):
      self.inner, self.draws = inner, 0

    def random(self, size=None):
      self.draws += 1 if size is None else int(np.prod(size))
      return self.inner.random(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
      self.draws += 1 if size is None else int(np.prod(size))
      return self.inner.normal(loc, scale, size)

  lidar.dropout = 0.0
  lidar.rng = counted = Counting(np.random.default_rng(7))
  _, room_r, peer_a, _ = lidar.scan_split(data)
  assert peer_a.size, "no peer returns in this scan at all"
  hits = int((room_r < lidar.max_range).sum())
  assert counted.draws == 3 * hits, \
    f"the map's stream drew {counted.draws} numbers for {hits} of its own returns"
  # A world with nobody else in it splits into nothing
  lidar._other_geoms = set()
  lidar._index_geoms()
  _, _, alone_a, alone_r = lidar.scan_split(data)
  assert alone_a.size == 0 and alone_r.size == 0


def _owners(model) -> np.ndarray:
  """Each geom's robot by its root body's name, "" for the world's."""
  out = []
  for g in range(model.ngeom):
    b = int(model.geom_bodyid[g])
    while int(model.body_parentid[b]) != 0:
      b = int(model.body_parentid[b])
    name = model.body(b).name
    out.append(name if name in (FIRST.root, SECOND.root) else "")
  return np.array(out)


def _collides(model) -> np.ndarray:
  """MuJoCo's own filter, every geom against every geom."""
  ct, ca = model.geom_contype.astype(int), model.geom_conaffinity.astype(int)
  return ((ct[:, None] & ca[None, :]) | (ct[None, :] & ca[:, None])) != 0


def test_the_pair_collides_with_each_other_and_never_with_itself(monkeypatch):
  """Ben, 2026-10-14 (#455): the two robots can collide, and their bodies
  never clip into each other. A robot's geoms are `contype 1, conaffinity
  0` (`humanoid/model.py`: no contact of its own but the pairs its model
  names), which alone would keep the pair from ever touching -- on afb473a
  the quadrupeds lay on each other on the dock. `PAIR_BITS`: every solid
  geom of one meets every solid geom of the other, none meets its own
  robot's, and the world meets each robot exactly as it did."""
  model = hw.home_spec(second_at=(2.5, 0.5)).compile()
  monkeypatch.setattr(hw, "PAIR_BITS", ((0, 0), (0, 0)))
  apart = hw.home_spec(second_at=(2.5, 0.5)).compile()
  who, hits, was = _owners(model), _collides(model), _collides(apart)
  a, b, w = who == FIRST.root, who == SECOND.root, who == ""
  solid = (model.geom_contype | model.geom_conaffinity) != 0
  assert not hits[np.ix_(a, a)].any() and not hits[np.ix_(b, b)].any()
  assert hits[np.ix_(a & solid, b & solid)].all() and (a & solid).sum() > 15
  assert not was[np.ix_(a, b)].any(), "the premise: before, they never met"
  assert np.array_equal(hits[np.ix_(a | b, w)], was[np.ix_(a | b, w)])
  alone = hw.home_spec().compile()            # ...and a robot alone, as it was
  mine = _owners(alone) == FIRST.root
  assert set(zip(alone.geom_contype[mine].tolist(),
                 alone.geom_conaffinity[mine].tolist())) == {(0, 0), (1, 0)}


def test_a_robot_that_touches_its_pair_leaves_a_row():
  """Issue #316's fourth question: contact between the pair is a phase of
  the encounter -- sensed off the contact array, latched like the plate's
  press -- in the world as served (#455: the pair collides)."""
  from pluggybot.activity.encounter import Encounters
  model = hw.home_spec(second_at=(0.3, 0.0)).compile()
  data = mujoco.MjData(model)
  _stand(model, data, FIRST, 0.0, 0.0, 0.0)
  _stand(model, data, SECOND, 3.0, 0.0, 0.0)
  enc = Encounters(model, FIRST, SECOND)
  events = []
  enc.on_event.append(events.append)
  enc.sense(model, data)
  assert enc.bumps == 0 and not enc.flags["touching"]
  _stand(model, data, SECOND, 0.15, 0.0, 0.0)              # trunks overlapping
  enc.sense(model, data)
  touched = [e for e in events if e["phase"] == "touched"]
  assert touched, "two robots in contact and nothing on the wire"
  assert set(touched[0]["robots"]) == {FIRST.root, SECOND.root}
  assert enc.bumps == 1 and enc.flags["touching"]
  enc.sense(model, data)
  assert enc.bumps == 1, "one contact, counted twice"
  # ...and it is over once they are apart, past the bumper's own hold.
  _stand(model, data, SECOND, 3.0, 0.0, 0.0)
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
  body, m = _mission()
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
    body.close()


def _pair_with_the_peer_across_the_bow(across: float, ahead: float, near_field=True):
  """The humanoid pair with the near-field camera on, the second robot
  standing `ahead` and `across` of the first, and NOTHING BROADCAST between
  them: the only thing standing between the two is what the first robot
  can sense."""
  from pluggybot.pair import build_pair
  lives = build_pair(HUMANOID_HOME, near_field=near_field, errands=("none", "none"))
  me, peer = lives
  me.body.start_at(0.0, -0.5, 0.0)
  peer.body.start_at(ahead, -0.5 + across, 0.0)
  me.body.grid.grid[:] = -5.0            # a mapped, empty room
  me.body.mission.others = []            # the broadcast is what we are not using
  return me, peer


def test_the_seam_hands_the_frames_peers_to_the_drive():
  """One line of wiring, pinned on its own: `_near_field_step` is where
  the peer channel reaches the mission, and a world without the camera
  hears of no peer at all."""
  me, _ = _pair_with_the_peer_across_the_bow(across=0.25, ahead=0.7)
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
  dark, _ = _pair_with_the_peer_across_the_bow(across=0.25, ahead=0.7, near_field=False)
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
