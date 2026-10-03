"""The rack at the quadruped's arm's reach, and the swap (issue #405, stage
B): each rule it rests on, pinned as cheaply as it can be while still
failing for the right reason -- the world's rack and its three tools, the
nose camera's near plane, a carried tool as the body's own to its senses,
where a tool is, the fold and the carry pose, the walk-in's stop against
the settle's drift, the verbs, and the rack view. A fetch and a stow flown
whole are behind `--endurance`. SimNotes, "The rack at the arm's reach"."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.legs import arm as am
from pluggybot.legs import body as qb
from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.legs import swap as sw
from pluggybot.legs import world as lw
from pluggybot.lifecycle import QUAD_HOME, tool_places, world_config
from pluggybot.procedure import steps as st
from pluggybot.rack.coupling import PEG_ABOVE_BODY, STATION_YS, bay_switches


@pytest.fixture(scope="module")
def quad_world():
  """The home world with one quadruped, its dock and its rack, compiled once."""
  return lw.home_spec().compile()


def _quad(model, at=(1.5, 0.5, 0.0)):
  body = qb.QuadBody(model, mujoco.MjData(model), realtime=False,
                     grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  body.start_at(*at)
  return body


def _settle(body, seconds=0.5):
  body.run(body.mission._drive_routine(seconds, 0.0, 0.0))


# ---- the world --------------------------------------------------------------------


def test_the_world_has_its_own_rack_and_every_tool_is_compiled_hanging(quad_world):
  m = quad_world
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  names = {m.body(i).name for i in range(m.nbody)}
  assert rk.RACK_BODY in names and not {"rack", "rack_built", "module_plug"} & names
  x, y, yaw = lw.rack_pose()
  assert (float(m.body(rk.RACK_BODY).pos[0]), float(m.body(rk.RACK_BODY).pos[1])) \
    == pytest.approx((x, y))
  # qpos0 is every tool on its bay: what a lost one is put back to
  for _ in range(250):
    mujoco.mj_step(m, d)
  for module, bay in rk.TOOL_BAYS.items():
    assert rk.on_bay(m, d, module, rk.DEFAULT, bay), module
  # ...and the rover's switch reader reads this rack's bays (the rack view)
  assert bay_switches(m, d)[:3] == (True, True, True)


def test_the_nose_camera_reads_a_bays_tags_from_its_working_pose(quad_world):
  # The house pins its extent for its cameras, and MuJoCo's near plane is a
  # hundredth of it: 0.37 m, and a bay's tags are 0.31 m from the nose. The
  # robot found the rack from a metre off and no bay once it stood at one.
  body = _quad(quad_world, dk.compose(lw.rack_pose(), rk.work_pose(rk.DEFAULT, 0)))
  try:
    assert body.mission.bay_aim(0) is not None
    quad_world.vis.map.znear = 0.01                       # the house's own
    assert body.mission.bay_aim(0) is None
  finally:
    quad_world.vis.map.znear = lw.NEAR_M / quad_world.stat.extent
    body.close()


# ---- where a tool is ----------------------------------------------------------------


def _mount(body, module, carried=True):
  """The tool seated on the fork at the carry pose, as a pick leaves it --
  or, `carried` False, as a pick that did not claim it would."""
  mis, m, d = body.mission, body.model, body.data
  mis.arm.hold_at(*am.CARRY_Q)
  j = {n: m.jnt_qposadr[m.joint(f"arm_{n}").id] for n in ("shoulder", "elbow", "wrist")}
  d.qpos[j["shoulder"]], d.qpos[j["elbow"]] = am.CARRY_Q
  d.qpos[j["wrist"]] = -sum(am.CARRY_Q)
  mujoco.mj_forward(m, d)
  seat = m.site("arm_seat").id
  rot = d.xmat[mis.root].reshape(3, 3)
  peg = d.site_xpos[seat] + rot @ np.array([0.0, 0.0, mis.arm_spec.fork.seat_rise() + 0.0003])
  yaw = math.atan2(rot[1, 0], rot[0, 0]) + math.pi
  q = m.jnt_qposadr[m.body(module).jntadr[0]]
  d.qpos[q:q + 3] = peg - np.array([0.0, 0.0, PEG_ABOVE_BODY])
  d.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  mujoco.mj_forward(m, d)
  if carried:
    body.mission.carry(module)
  _settle(body)


def test_a_tool_on_the_fork_is_seated_on_this_robot_and_off_its_bay(quad_world):
  body = _quad(quad_world)
  try:
    hung = body.module_state("module_lcd")
    assert hung["hung"] and not hung["on_fork"] and hung["bay"] == 0
    assert body.seated_on("module_lcd") is None and not body.tool_powered("module_lcd")
    assert not body.tool_powered(None)
    _mount(body, "module_claw")
    st_ = body.module_state("module_claw")
    assert st_["on_fork"] and not st_["hung"] and body.tool_powered("module_claw")
    assert body.seated_on("module_claw") == body.handle.root
    # ...and the rack view says so, off the bays' switches and the fork
    life = SimpleNamespace(model=body.model, data=body.data, peers=[], body=body,
                           rack_inventory=dict(rk.TOOL_BAYS))
    assert tool_places(life) == {"module_lcd": "on bay A", "module_pen": "on bay B",
                                 "module_claw": "on your fork"}
  finally:
    body.close()


def test_a_carried_tool_is_the_bodys_own_to_its_senses(quad_world):
  body = _quad(quad_world)
  try:
    mis = body.mission
    tool = mis.tool_gids("module_pen")
    bid = body.model.body("module_pen").id
    mis.carry("module_pen")
    assert np.isin(tool, mis.lidar._self_ids).all()
    assert np.isin(tool, mis.depth._mine).all()
    assert mis._is_ignored[tool].all()
    assert mis.walker.scan_exclude == bid and mis.arm.payload[0] > 0.0
    mis.carry(None)
    assert not np.isin(tool, mis.lidar._self_ids).any()
    assert not np.isin(tool, mis.depth._mine).any()
    assert not mis._is_ignored[tool].any() and mis._is_ignored[mis.body_gids].all()
    assert mis.walker.scan_exclude == -1 and mis.arm.payload[0] == 0.0
  finally:
    body.close()


def test_a_carried_tool_turns_slower(quad_world):
  # At the drive's full 1.0 rad/s a pivot swung a carried tool off its V's
  # for up to 160 ms; at W_CARRY, 14 (`arm_spike.py --served --carry`).
  body = _quad(quad_world)
  try:
    mis = body.mission
    assert next(mis._twist_routine(0.0, 0.0, -1.0))[2] == pytest.approx(-1.0)
    mis.arm.hold_at(*am.CARRY_Q)
    mis.carry("module_claw")
    assert next(mis._twist_routine(0.0, 0.0, -1.0))[2] == pytest.approx(-qb.W_CARRY)
    assert next(mis._twist_routine(0.5, 0.0, 0.3))[2] == pytest.approx(0.3)
  finally:
    body.close()


def test_a_walk_keeps_a_carried_tool_at_the_carry_pose(quad_world):
  # Every walk folds an arm a program left out, and folded, a carried tool
  # is thrown: the walk's driving pose is the carry pose (found in review).
  body = _quad(quad_world)
  try:
    mis = body.mission
    carry = [am.CARRY_Q[0], sum(am.CARRY_Q)]
    _mount(body, "module_pen")
    body.run(mis._twist_routine(0.3, 0.0, 0.0))
    assert np.allclose(mis.arm.goal, carry)
    mis.arm.aim(1.0, 0.0)                     # a program's pose, carrying
    body.run(mis._twist_routine(0.3, 0.0, 0.0))
    assert mis.carrying == "module_pen"
    assert np.allclose(mis.arm.goal, carry) and mis.arm.arrived(qb.ARM_TOL)
  finally:
    body.close()


def test_a_tool_against_the_fork_is_a_bump_unless_it_is_carried(quad_world):
  # A tool is the body's own to the bumper only while it rides this fork:
  # the bump is what backs the pair apart when one's carried tool meets
  # the other's fork. Ignoring every tool against the fork, the pair's
  # tools knocked each other off at the rack (4 of 12 hung back, #405).
  body = _quad(quad_world)
  try:
    _mount(body, "module_lcd", carried=False)
    assert body.mission.on_this_fork("module_lcd") and body.mission._press_now()
    body.mission.carry("module_lcd")
    assert not body.mission._press_now()
  finally:
    body.close()


def test_a_tool_that_falls_away_is_let_go_of(quad_world):
  # Knocked off by the other robot's tool at the rack, or sent home by an
  # admin, a carried tool left the body "carrying" it: walking with its arm
  # up at W_CARRY and blind to the tool, until a fall (found in review).
  body = _quad(quad_world)
  try:
    mis, m, d = body.mission, body.model, body.data
    _mount(body, "module_pen")
    assert mis.carrying == "module_pen"
    q = m.jnt_qposadr[m.body("module_pen").jntadr[0]]
    d.qpos[q:q + 7] = m.qpos0[q:q + 7]                  # as `_return_module` does
    mujoco.mj_forward(m, d)
    _settle(body, 0.05)
    assert mis.carrying is None and mis.arm.payload[0] == 0.0
    assert not mis._is_ignored[mis.tool_gids("module_pen")].any()
  finally:
    body.close()


def test_a_fall_under_a_fork_move_stops_it(quad_world):
  # A fall folds the arm; re-aimed along its line, the fork was held out
  # through the get-up (found in review).
  body = _quad(quad_world)
  try:
    mis = body.mission
    q = body.handle.qpos_adr(body.model)
    step = tick.Step(mis._fork_to_routine(0.45, 0.10), "fork")
    cmd, k = step.tick(), 0
    while cmd is not None:
      if k == 25:                                       # knocked onto its side
        body.data.qpos[q + 3:q + 7] = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
        mujoco.mj_forward(body.model, body.data)
      body.stepper.step(cmd)
      cmd, k = step.tick(), k + 1
    stow = mis.arm_spec.stow
    assert step.result is False and mis.posture != qb.STANDING
    assert np.allclose(mis.arm.goal, (stow[0], stow[0] + stow[1]))
  finally:
    body.close()


def test_a_fall_lets_go_and_a_rest_keeps_the_carry_pose(quad_world):
  body = _quad(quad_world)
  try:
    mis = body.mission
    carry = [am.CARRY_Q[0], sum(am.CARRY_Q)]
    stow = [mis.arm_spec.stow[0], sum(mis.arm_spec.stow)]
    mis.carry("module_lcd")
    mis.fold_arm(fall=False)                  # lying down to rest, carrying
    assert np.allclose(mis.arm.goal, carry) and mis.carrying == "module_lcd"
    mis.fold_arm(fall=True)                   # a fall throws it
    assert np.allclose(mis.arm.goal, stow) and mis.carrying is None
    assert mis.arm.payload[0] == 0.0
  finally:
    body.close()


# ---- the swap's rules ---------------------------------------------------------------


def test_a_save_waits_out_a_swap(quad_world, tmp_path):
  # A save mid-swap holds a fork half under a peg, which no file keeps.
  from pluggybot import continuation
  body = _quad(quad_world)
  try:
    life = SimpleNamespace(_standing_up=False, body=body, world_fingerprint="x",
                           data=body.data)
    keeper = continuation.Keeper([life], tmp_path / "world.npz")
    assert not keeper.busy()
    body.mission.working = True
    assert keeper.busy()
  finally:
    body.close()


def _stub(value, seen=None, mis=None):
  def routine(*a, **kw):
    if seen is not None:
      seen.append(mis.working)
    return tick.result(value)
  return routine


def test_a_swap_is_working_at_the_bay_and_not_on_the_walk(quad_world):
  # A save held off through the whole swap waited out a 45 s walk, past a
  # stop's grace (found in review): only the part at the bay is `working`.
  body = _quad(quad_world)
  try:
    mis, walk, bay = body.mission, [], []
    mis._to_the_bay_routine = _stub("ok", walk, mis)
    mis._fetch_at_routine = _stub("arrived", bay, mis)
    assert body.run(mis.fetch_routine(0, "module_lcd")) == "arrived"
    assert walk == [False] and bay == [True] and not mis.working
  finally:
    body.close()


def test_a_move_inside_a_swap_leaves_it_working(quad_world):
  body = _quad(quad_world)
  try:
    body.mission.working = True
    body.run(body.ramp_routine(body.actuator("arm_elbow"), -1.0, 1.5))
    assert body.mission.working
  finally:
    body.close()


def test_a_pick_that_left_the_tool_on_the_fork_carries_it(quad_world):
  # Folded, a tool the fork came out still holding is thrown; left
  # unclaimed, it was a bump on every step (found in review).
  body = _quad(quad_world)
  try:
    mis, ran = body.mission, []
    aim = rk.BayAim(x=0.54, z=0.08, across=0.0, yaw=0.0)
    mis._lined_up_routine = _stub(aim)
    mis._pick_routine = _stub(False)
    mis._rack_back_out_routine = _stub(None)
    mis.on_this_fork = lambda module: True
    mis.carry_routine = lambda: (ran.append("carry"), tick.result(True))[1]
    mis.stow_arm_routine = lambda: (ran.append("fold"), tick.result(True))[1]
    rec = {"attempts": []}
    assert body.run(mis._fetch_at_routine(0, "module_pen", rec)) == "arrived"
    assert ran == ["carry"] and mis.carrying == "module_pen"
    assert rec["attempts"][0]["why"] == "on the fork, not seated"
    # ...and one that came out empty folds carrying nothing
    mis.on_this_fork = lambda module: False
    assert body.run(mis._fetch_at_routine(0, "module_pen", {"attempts": []})) == "arrived"
    assert ran == ["carry", "fold"] and mis.carrying is None
  finally:
    body.close()


def test_a_bay_is_named_by_its_station():
  assert [sw.bay_of(STATION_YS[i]) for i in range(3)] == [0, 1, 2]


def test_the_walk_in_stops_turned_against_the_settles_drift(quad_world, monkeypatch):
  # Square on the line, it turns clockwise toward -SETTLE_DRIFT: the settle
  # turns it counter-clockwise, 1.2 to 2.7 deg on the served body.
  tw = rk.walk_in_twist(-0.3, 0.0, 0.0, heading=-rk.SETTLE_DRIFT)
  assert tw.vx > 0.0 and tw.yaw_rate < 0.0
  assert rk.walk_in_twist(-0.3, 0.0, 0.0).yaw_rate == 0.0     # the spike's
  # ...and the swap's walk-in asks for it
  from pluggybot.legs.policy import Twist
  asked = []
  monkeypatch.setattr(rk, "walk_in_twist",
                      lambda *a, **kw: (asked.append(kw.get("heading")), Twist())[1])
  body = _quad(quad_world)
  try:
    body.mission.tool_rack_seen = body.mission.tool_rack_prior
    assert body.run(body.mission._rack_walk_in_routine(0)) == "stopped"
    assert asked == [-rk.SETTLE_DRIFT]
  finally:
    body.close()


def test_a_verb_that_walks_carries_a_tool_at_the_carry_pose(quad_world):
  body = _quad(quad_world)
  try:
    life = SimpleNamespace(body=body, model=body.model, data=body.data)
    sh, el = body.actuator("arm_shoulder"), body.actuator("arm_elbow")
    assert [(a, t) for a, t, _ in st.travel_pose(life, "module_pen")] == \
      [(sh, am.CARRY_Q[0]), (el, am.CARRY_Q[1])]
    # ...and a stow's carrying configuration is the body's own, not a lift
    ran = []
    body.mission.carry_routine = lambda: (ran.append("carry"), tick.result(True))[1]
    body.mission.carrying = "module_pen"
    assert body.run(st.carry_configuration_routine(life, "module_pen")) == \
      {"setDown": None, "dropped": None}
    assert ran == ["carry"]
  finally:
    body.close()


# ---- the swap, flown whole ----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.endurance(when=(
  "src/pluggybot/legs/", "models/quadruped", "models/home_world.xml",
  "src/pluggybot/navigator.py", "src/pluggybot/mapping/", "src/pluggybot/perception/",
  "src/pluggybot/rack/coupling.py", "src/pluggybot/rack/tags.py", "src/pluggybot/tick.py"))
def test_the_served_quadruped_fetches_a_tool_and_hangs_it_back(quad_world):
  """From a metre north of the rack's east end the served body walks to bay
  A, takes the LCD, carries it and hangs it back: the swap as the served
  world runs it (`scripts/arm_spike.py --served` flies it from the dock and
  from across the house). Flown (~60 sim s of the walking policy): every
  rule it rests on is pinned fast above -- the nose camera's near plane, the
  tools compiled at rest, a carried tool as the body's own, the fork judged
  by the world, the walk-in's stop against the drift; only this flies them
  together. Shown to fail with the house's own near plane: no walk-in finds
  the bay."""
  body = _quad(quad_world, (2.4, -0.3, -math.pi / 2))
  try:
    assert body.run(body.fetch_tool_routine(STATION_YS[0], "module_lcd")) == "arrived"
    assert body.module_state("module_lcd")["on_fork"] and body.tool_powered("module_lcd")
    assert body.mission.carrying == "module_lcd"
    assert body.run(body.stow_tool_routine(STATION_YS[0], "module_lcd")) == "arrived"
    st_ = body.module_state("module_lcd")
    assert st_["hung"] and not st_["on_fork"] and body.mission.carrying is None
  finally:
    body.close()
