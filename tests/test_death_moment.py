"""A death says where the robot was and what it was doing (issue #362).

Seven topples in the week to 2026-09-24 could not be explained, because the
`death` event carried the cause and nothing else: "at the rack, mid-fetch"
was inferred from how long after a claim each robot fell. The event now
carries `at` -- the true and believed pose, the lean, the state, what is on
the fork and what every axis is commanded to, the errand and the procedure
step, the bay a swap is working and the nearest peer. A topple reads it AS
THE ROBOT FALLS, because the death is TOPPLE_HOLD_S later and the errand
has had that long to react.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot import lifecycle as lc
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.mission.errand import programmed_errand
from pluggybot.mission.mission import MissionAborted
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st
from pluggybot.rack.coupling import CLAW_JAW_TRAVEL
from pluggybot.rack.swap import align_lift


def _life(world: str = "room_hub") -> HubLifecycle:
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, world=world,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      mortal=True)
  life.mission.start_at(*cfg["start"])
  life.home_pose = tuple(cfg["start"])
  life.survival_since = float(data.time)
  return life


def _deaths(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(lambda e: seen.append(e) if e.get("type") == "death" else None)
  return seen


def _stop_on_death(life) -> None:
  """End the run the step the robot dies: what comes after is the errand's
  business, and this test is about the row."""
  def hook() -> None:
    if life.dead is not None:
      raise MissionAborted("dead")
  life.mission.step_hooks.append(hook)


def test_a_robot_killed_mid_errand_says_where_it_was_and_what_it_was_running(
    monkeypatch):
  """The acceptance test: a procedure errand, the pack emptied during its
  second step. The row names the errand, the step by count and source line,
  the state, the module on the fork and every axis's setpoint -- the body's
  and that tool's, no other tool's -- and the true pose beside the believed
  one, which are read apart (the belief is pushed 0.25 m off)."""
  life = _life()
  seen = _deaths(life)
  cfg = world_config("room_hub")
  src = "def pen_check():\n  wait(1)\n  wait(5)\n"
  proc = lang.compile_procedure(src, lc.world_facts("room_hub"))
  errand = programmed_errand(proc, task="program", name="procedure")
  # the fork holds the pen, as far as anything that asks can tell
  monkeypatch.setattr(st, "_carried", lambda life: "module_pen")
  life.data.ctrl[life.model.actuator("pen_carriage").id] = 0.012
  life.mission.swap.reckoner.x += 0.25
  life.mission.swap.reckoner.theta += 2 * math.pi   # a heading is never wrapped
  t0 = float(life.data.time)

  def starve() -> None:
    if life.data.time >= t0 + 2.0:
      life.battery.energy_wh = 0.0
  life.mission.step_hooks.append(starve)
  _stop_on_death(life)
  with pytest.raises(MissionAborted):
    life.mission.run(life.run_errand_routine(errand))

  assert len(seen) == 1 and seen[0]["cause"] == "flat"
  at = seen[0]["at"]
  assert at["t"] == seen[0]["t"], "a flat death is read as it happens"
  assert at["errand"]["name"] == "procedure"
  assert at["step"] == {"procedure": "pen_check", "n": 2, "line": 3,
                        "verb": "wait", "args": {"seconds": 5.0}}
  assert at["state"] == "USE_TOOL"
  assert at["carrying"] == "module_pen"
  assert at["setpoints"] == {"lift": round(align_lift(), 4), "arm": 0.0,
                             "pen.carriage": 0.012}
  sx, sy, syaw = cfg["start"]
  assert at["pose"]["x"] == pytest.approx(sx, abs=0.02)
  assert at["pose"]["y"] == pytest.approx(sy, abs=0.02)
  assert at["pose"]["yawDeg"] == pytest.approx(math.degrees(syaw), abs=1.0)
  assert at["believed"]["x"] - at["pose"]["x"] == pytest.approx(0.25, abs=0.02)
  assert at["believed"]["yawDeg"] == pytest.approx(at["pose"]["yawDeg"], abs=1.0)
  assert at["tilt"]["deg"] < 2.0
  assert at["swapping"] is None and "peer" not in at
  # ...and the step is let go when it ends, however it ended
  assert life.step_now is None


def _lay_on_its_right_side(life, yaw: float) -> None:
  """Roll the robot 90 deg onto its right side, facing `yaw`."""
  d = life.data
  q = life.mission.swap.root_qadr
  facing = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  roll = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
  quat = np.zeros(4)
  mujoco.mju_mulQuat(quat, facing, roll)
  d.qpos[q + 2] += 0.15
  d.qpos[q + 3:q + 7] = quat
  d.qvel[:6] = 0.0
  mujoco.mj_forward(life.model, d)


def test_a_topple_is_read_as_the_robot_falls_not_when_it_is_declared_dead():
  """A step is running when the robot goes over, and has ended by the time
  the death is called TOPPLE_HOLD_S later: the row names the step that was
  running AS IT FELL, at the fall's time. The lean is in the robot's OWN
  frame -- onto its right side reads `right` whichever way it faced."""
  life = _life()
  seen = _deaths(life)
  t0 = float(life.data.time)
  fell: list[float] = []

  def push() -> None:
    if not fell and life.data.time >= t0 + 0.5:
      _lay_on_its_right_side(life, math.radians(120.0))
      fell.append(float(life.data.time))
  life.mission.step_hooks.append(push)
  where = {"procedure": "reach", "n": 1, "line": 2}
  life.mission.run(st.run_verb(life, st.VERBS["wait"], {"seconds": 1.0}, where))
  assert life.dead is None and life.step_now is None
  # ...the fall is kept across a restart, as the tilt clock is (#345)
  state, _ = life.kept_state()
  assert state["fall"] is not None and state["fall"] == life._fall
  life.mission._drive(lc.TOPPLE_HOLD_S + 0.5, 0.0, 0.0)

  assert len(seen) == 1 and seen[0]["cause"] == "stuck"
  at = seen[0]["at"]
  assert fell[0] <= at["t"] <= fell[0] + 2 * lc.DEATH_CHECK_S
  assert seen[0]["t"] - at["t"] >= lc.TOPPLE_HOLD_S - 1e-6
  assert at["step"] == {**where, "verb": "wait", "args": {"seconds": 1.0}}
  assert at["tilt"]["deg"] == pytest.approx(90.0, abs=3.0)
  assert at["tilt"]["towardDeg"] == pytest.approx(-90.0, abs=3.0)
  assert at["tilt"]["toward"] == "right"
  assert at["pose"]["yawDeg"] == pytest.approx(120.0, abs=3.0)


def test_the_lean_word_is_the_nearest_of_four():
  assert [lc.lean_word(a) for a in (0, 44, -44, 46, 134, -46, -134, 136, -180)] \
      == ["forward", "forward", "forward", "left", "left", "right", "right",
          "back", "back"]


def test_a_death_on_a_pair_names_the_nearest_peer_and_how_far_off():
  """Measured between the two chassis off the root joints -- what the
  encounter rows measure -- and each robot names the other."""
  from pluggybot.pair import build_pair
  from pluggybot.robot import SECOND
  me, peer = build_pair("room_hub", errands=("none", "none"), mortal=True)
  seen = _deaths(me)
  adr = SECOND.qpos_adr(me.model)
  me.data.qpos[adr:adr + 2] = me.data.qpos[me.mission.swap.root_qadr:
                                           me.mission.swap.root_qadr + 2] + [0.6, 0.8]
  mujoco.mj_forward(me.model, me.data)
  me._die("flat", "the pack reached zero")
  assert seen[-1]["at"]["peer"] == {"name": peer.robot_name or peer.root,
                                    "robot": peer.root, "distanceM": 1.0,
                                    "state": peer.state}
  assert peer._moment()["peer"]["robot"] == me.root


def test_setpoints_are_the_body_and_the_carried_tool_and_skip_what_is_gone(
    monkeypatch):
  """The claw's jaws are two actuators read as one axis, 0 wide .. 1 shut;
  an axis whose actuator the world no longer has (a retired built tool) is
  left out rather than failing the death that reads it."""
  life = _life()
  for act in ("claw_l", "claw_r"):
    life.data.ctrl[life.model.actuator(act).id] = -0.5 * CLAW_JAW_TRAVEL
  assert axes.setpoints(life, "module_claw")["claw.jaws"] == 0.5
  assert set(axes.setpoints(life, None)) == {"lift", "arm"}
  monkeypatch.setitem(axes.AXES, "gone.hinge", axes.Axis(
    "gone.hinge", 0.0, 1.0, 1.0, "rad", "retired", requires="tool_gone",
    actuator="tool_gone_hinge"))
  assert set(axes.setpoints(life, "tool_gone")) == {"lift", "arm"}


def test_a_moment_that_cannot_be_read_never_costs_the_death(monkeypatch):
  """`_moment` runs on the physics seam inside `_die`: a diagnostic that
  raised there would lose the very death it describes."""
  life = _life()
  seen = _deaths(life)

  def broken():
    raise KeyError("module_gone")
  monkeypatch.setattr(life, "_read_moment", broken)
  life._die("flat", "the pack reached zero")
  assert life.dead is not None and seen[-1]["cause"] == "flat"
  assert seen[-1]["at"] == {"t": seen[-1]["t"], "error": "KeyError: 'module_gone'"}
