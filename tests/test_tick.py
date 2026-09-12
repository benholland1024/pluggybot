"""The physics seam (pluggybot/tick.py, issue #58): a routine yields one
drive command per physics step, and the loop above does the stepping."""

import ast
import pathlib

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.control import wheel_targets
from pluggybot.mission.mission import HubMission
from pluggybot.rack.swap import HubSwap

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"


class FakeSwap:
  """Records what it was asked to step with; raises when told to."""

  def __init__(self, raise_on: int | None = None):
    self.steps: list = []
    self.raise_on = raise_on

  def _step_once(self, tl, tr):
    self.steps.append((tl, tr))
    if self.raise_on is not None and len(self.steps) == self.raise_on:
      raise RuntimeError("hook")


def test_a_step_hands_back_commands_then_the_result():
  def routine():
    yield 0.1, 0.0
    yield 0.0, 0.5
    return "arrived"
  step = tick.Step(routine(), "x")
  assert step.tick() == (0.1, 0.0)
  assert step.tick() == (0.0, 0.5)
  assert step.tick() is None and step.done and step.result == "arrived"
  assert step.tick() is None, "a finished step stays finished"


def test_run_steps_the_swap_with_each_command_and_returns_the_value():
  swap = FakeSwap()

  def routine():
    yield 0.2, 0.0
    yield 0.0, 0.0
    return 7
  assert tick.run(swap, routine()) == 7
  assert swap.steps == [wheel_targets(0.2, 0.0), wheel_targets(0.0, 0.0)]


def test_hold_and_result_are_the_two_shapes_a_test_stubs_with():
  swap = FakeSwap()
  assert tick.run(swap, tick.hold(0.01, 0.001, v=0.3)) is None
  assert len(swap.steps) == 10
  assert tick.run(FakeSwap(), tick.result(False)) is False


def test_an_exception_from_the_step_is_thrown_into_the_routine():
  """`stop_when` ends a run by raising from a step hook, and the manoeuvre
  in flight may hold a `finally` (the swap restores the physics timestep,
  the charge clears the press flag). Raised past the routine those would run
  at garbage collection; thrown in, they run where they always ran."""
  swap = FakeSwap(raise_on=2)
  seen: list = []

  def routine():
    try:
      yield 0.1, 0.0
      yield 0.1, 0.0
      yield 0.1, 0.0
    finally:
      seen.append("finally")
  with pytest.raises(RuntimeError, match="hook"):
    tick.run(swap, routine())
  assert seen == ["finally"], "the routine's cleanup did not run in the driver"
  assert len(swap.steps) == 2

  # ...and a routine may CATCH it and carry on, which is `run_errand`'s
  # `except Exception` around a use-phase.
  swap = FakeSwap(raise_on=1)

  def survives():
    try:
      yield 0.1, 0.0
    except RuntimeError:
      pass
    yield 0.0, 0.0
    return "ok"
  assert tick.run(swap, survives()) == "ok"
  assert len(swap.steps) == 2


# ---- parity on real physics, in miniature ------------------------------------


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def _hash(data):
  return np.ascontiguousarray(np.concatenate([data.qpos, data.qvel, data.ctrl])).tobytes()


def test_the_blocking_twin_and_the_ticked_routine_are_one_trajectory(hub_model):
  """The whole refactor's claim, at the scale a test can afford: driving a
  manoeuvre by its blocking name and ticking its routine from outside step
  the same commands in the same order and leave the world in the same state
  -- byte for byte. The scripted day's version of this is
  `scripts/determinism_spike.py --compare` (Evaluation.md)."""
  a = mujoco.MjData(hub_model)
  ma = HubMission(hub_model, a, viewer=None, realtime=False)
  ma.start_at(0.5, 3.0, 0.0)
  ma._drive(0.4, 0.15, 0.6)
  ma.face(1.0)

  b = mujoco.MjData(hub_model)
  mb = HubMission(hub_model, b, viewer=None, realtime=False)
  mb.start_at(0.5, 3.0, 0.0)
  for routine in (mb._drive_routine(0.4, 0.15, 0.6), mb.face_routine(1.0)):
    step = tick.Step(routine)
    cmd = step.tick()
    while cmd is not None:
      mb.swap._step_once(*wheel_targets(*cmd))
      cmd = step.tick()
  assert _hash(a) == _hash(b)
  assert ma.step_count == mb.step_count and ma.pose == mb.pose


def test_swap_manoeuvres_are_the_same_ticked(hub_model):
  a = mujoco.MjData(hub_model)
  sa = HubSwap(hub_model, a)
  sa.place_at_standoff(0.125)
  sa.pick()
  b = mujoco.MjData(hub_model)
  sb = HubSwap(hub_model, b)
  sb.place_at_standoff(0.125)
  step = tick.Step(sb.pick_routine())
  cmd = step.tick()
  while cmd is not None:
    sb._step_once(*wheel_targets(*cmd))
    cmd = step.tick()
  assert _hash(a) == _hash(b)


# ---- the fence: a routine call is nothing until it is driven ---------------


ROUTINE_DRIVERS = {"run", "Step"}


def _routine_calls(tree):
  """Every call to a `*_routine` name, with its parent node."""
  for parent in ast.walk(tree):
    for child in ast.iter_child_nodes(parent):
      child._parent = parent  # type: ignore[attr-defined]
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      f = node.func
      name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
      if name.endswith("_routine"):
        yield node, name


def _driven(call):
  """A routine call is DRIVEN when it is the operand of `yield from`, an
  argument to a driver (`run`, `Step`), or a lambda's body (a factory
  handed to something that will drive it)."""
  parent = call._parent  # type: ignore[attr-defined]
  if isinstance(parent, ast.YieldFrom):
    return True
  if isinstance(parent, ast.Lambda):
    return True
  if isinstance(parent, ast.Call) and call in parent.args:
    f = parent.func
    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
    return name in ROUTINE_DRIVERS
  return False


def test_every_routine_call_in_src_is_driven():
  """⚠ THE BUG CLASS THE SEAM INVITES. `self.drive_to_routine(x, y)` builds
  a generator and moves nothing; without `yield from` (or a driver) it is a
  truthy object that silently did not happen -- `if self.drive_to_routine(
  ...)` is always true and the robot never left. Walked off the syntax tree
  because it is a mistake no flown test reliably reaches."""
  bad = []
  for path in sorted(SRC.rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))
    for call, name in _routine_calls(tree):
      if not _driven(call):
        bad.append(f"{path.relative_to(SRC.parent)}:{call.lineno} {name}(...)")
  assert not bad, "undriven routine calls:\n  " + "\n  ".join(bad)


def test_the_fence_fails_on_an_undriven_call():
  src = "def f(self):\n  if self.drive_to_routine(1, 2):\n    pass\n"
  tree = ast.parse(src)
  calls = list(_routine_calls(tree))
  assert calls and not _driven(calls[0][0])
  ok = "def f(self):\n  if (yield from self.drive_to_routine(1, 2)):\n    pass\n"
  calls = list(_routine_calls(ast.parse(ok)))
  assert calls and _driven(calls[0][0])
