"""A death says where the robot was and what it was doing (issue #362).

Seven topples in the week to 2026-09-24 could not be explained, because the
`death` event carried the cause and nothing else. The event now carries
`at` -- the true and believed pose, the lean, the state, what is on the
fork and what every axis is commanded to, the errand and the procedure
step, the bay a swap is working and the nearest peer. A topple reads it AS
THE ROBOT FALLS, because the death is the body's `stuck_after_s` later and
the errand has had that long to react.

The row's bookkeeping is pinned on the stub body (`tests/test_body.py`);
the true pose beside the believed one, and the arm's setpoints, on the
served quadruped.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot import lifecycle as lc
from pluggybot.body import StubBody
from pluggybot.legs import model as qm
from pluggybot.lifecycle import QUAD_HOME, HubLifecycle, world_config
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st
from pluggybot.robot import SECOND, world_spec
from pluggybot.tick import MissionAborted
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _born(life) -> HubLifecycle:
  start = world_config(QUAD_HOME)["start"]
  life.body.start_at(*start)
  life.home_pose = tuple(start)
  life.survival_since = float(life.data.time)
  return life


def _life(**kw) -> HubLifecycle:
  return _born(stub_life(mortal=True, **kw))


def _quad() -> HubLifecycle:
  cfg = world_config(QUAD_HOME)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  return _born(HubLifecycle(model, mujoco.MjData(model), realtime=False,
                            world=QUAD_HOME, spec=spec,
                            battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                            grid_bounds=cfg["grid_bounds"],
                            low_battery_wh=cfg["low_battery_wh"], mortal=True))


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
  life.body.step_hooks.append(hook)


def test_a_robot_killed_mid_errand_says_where_it_was_and_what_it_was_running(
    monkeypatch):
  """The acceptance test: a procedure errand, the pack emptied during its
  second step. The row names the errand, the step by count and source line,
  the state, the module on the fork and every axis's setpoint -- the arm's
  -- and the true pose beside the believed one, which are read apart (the
  belief is pushed 0.25 m off)."""
  life = _quad()
  try:
    seen = _deaths(life)
    sx, sy, syaw = world_config(QUAD_HOME)["start"]
    src = "def pen_check():\n  wait(0.5)\n  wait(5)\n"
    proc = lang.compile_procedure(src, lc.world_facts(QUAD_HOME))
    errand = programmed_errand(proc, task="program", name="procedure")
    # the fork holds the pen, as far as anything that asks can tell
    monkeypatch.setattr(st, "_carried", lambda life: "module_pen")
    # ...and the belief kept 0.25 m off: matched (#386), the scans would take
    # it out before the death this reads it at
    life.body.mission.matcher = None
    life.body.mission.odo.x += 0.25
    t0 = float(life.data.time)

    def starve() -> None:
      if life.data.time >= t0 + 0.7:
        life.battery.energy_wh = 0.0
    life.body.step_hooks.append(starve)
    _stop_on_death(life)
    with pytest.raises(MissionAborted):
      life.body.run(life.run_errand_routine(errand))

    assert len(seen) == 1 and seen[0]["cause"] == "flat"
    at = seen[0]["at"]
    assert at["t"] == seen[0]["t"], "a flat death is read as it happens"
    assert at["errand"]["name"] == "procedure"
    assert at["step"] == {"procedure": "pen_check", "n": 2, "line": 3,
                          "verb": "wait", "args": {"seconds": 5.0}}
    assert at["state"] == "USE_TOOL"
    assert at["carrying"] == "module_pen"
    stow = qm.CHOSEN.arm.stow
    assert at["setpoints"] == {"shoulder": round(stow[0], 4), "elbow": round(stow[1], 4)}
    assert at["pose"]["x"] == pytest.approx(sx, abs=0.03)
    assert at["pose"]["y"] == pytest.approx(sy, abs=0.03)
    assert at["pose"]["yawDeg"] == pytest.approx(math.degrees(syaw), abs=1.5)
    assert at["believed"]["x"] - at["pose"]["x"] == pytest.approx(0.25, abs=0.03)
    assert at["believed"]["yawDeg"] == pytest.approx(at["pose"]["yawDeg"], abs=1.5)
    # ...and a standing robot has no lean to name: the direction of a
    # fraction of a degree is noise, and "back" would count it among falls
    assert at["tilt"]["deg"] < 1.5
    assert at["tilt"]["towardDeg"] is None and at["tilt"]["toward"] is None
    assert at["swapping"] is None and "peer" not in at
    # ...and the step is let go when it ends, however it ended
    assert life.step_now is None
  finally:
    life.body.close()


def _lay_on_its_right_side(life, yaw: float) -> None:
  """Roll the stub 90 deg onto its right side, facing `yaw`."""
  facing = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  roll = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
  quat = np.zeros(4)
  mujoco.mju_mulQuat(quat, facing, roll)
  life.body.theta = yaw
  life.body.attitude = tuple(quat)


def test_a_topple_is_read_as_the_robot_falls_not_when_it_is_declared_dead():
  """A step is running when the robot goes over, and has ended by the time
  the death is called `stuck_after_s` later: the row names the step that
  was running AS IT FELL, at the fall's time. The lean is in the robot's
  OWN frame -- onto its right side reads `right` whichever way it faced."""
  life = _life()
  seen = _deaths(life)
  t0 = float(life.data.time)
  fell: list[float] = []

  def push() -> None:
    if not fell and life.data.time >= t0 + 0.5:
      _lay_on_its_right_side(life, math.radians(120.0))
      fell.append(float(life.data.time))
  life.body.step_hooks.append(push)
  where = {"procedure": "reach", "n": 1, "line": 2}
  life.body.run(st.run_verb(life, st.VERBS["wait"], {"seconds": 1.0}, where))
  assert life.dead is None and life.step_now is None
  # ...the fall is kept across a restart, as the tilt clock is (#345)
  state, _ = life.kept_state()
  assert state["fall"] is not None and state["fall"] == life._fall
  hold = life.body.stuck_after_s
  life.body.run(life.body.hold_routine(hold + 0.5))

  assert len(seen) == 1 and seen[0]["cause"] == "stuck"
  at = seen[0]["at"]
  assert fell[0] <= at["t"] <= fell[0] + 2 * lc.DEATH_CHECK_S
  assert seen[0]["t"] - at["t"] >= hold - 1e-6
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
  """Measured between the two bodies' roots -- what the encounter rows
  measure -- and each robot names the other, with whether it is dead: a
  peer knocked over mid-errand keeps its errand's state."""
  model, data = StubBody.world()
  me = _life(body=StubBody(model, data))
  peer = _life(body=StubBody(model, data, handle=SECOND), robot_name="Rowan")
  me.peers, peer.peers = [peer], [me]
  seen = _deaths(me)
  peer.body.x, peer.body.y = me.body.x + 0.6, me.body.y + 0.8
  me._die("flat", "the pack reached zero")
  assert seen[-1]["at"]["peer"] == {"name": "Rowan", "robot": peer.root,
                                    "distanceM": 1.0, "state": peer.state,
                                    "dead": None}
  assert peer._moment()["peer"]["robot"] == me.root
  assert peer._moment()["peer"]["dead"] == "flat"


def test_setpoints_are_the_bodys_and_skip_what_is_gone(monkeypatch):
  """An axis whose actuator the world no longer has (a retired built tool)
  is left out rather than failing the death that reads it; the body's own
  are there."""
  from types import SimpleNamespace
  life = _quad()
  try:
    view = SimpleNamespace(body=life.body, model=life.model, data=life.data)
    assert set(axes.setpoints(view, None)) == {"shoulder", "elbow"}
    monkeypatch.setitem(axes.AXES, "gone.hinge", axes.Axis(
      "gone.hinge", 0.0, 1.0, 1.0, "rad", "retired", requires="tool_gone",
      actuator="tool_gone_hinge"))
    assert set(axes.setpoints(view, "tool_gone")) == {"shoulder", "elbow"}
  finally:
    life.body.close()


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
