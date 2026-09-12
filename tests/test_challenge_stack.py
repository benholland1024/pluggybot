"""The three-block tower: the first predicate-graded challenge (issue #120).

docs/Challenges.md is the decision; this file is the worked example's proof:
the criteria were written into `challenge/stack.py` first, a run that meets
them passes, and the runs that do not -- unstacked, propped up, fallen,
merely reported -- each fail for the reason the criteria give.
"""

from types import SimpleNamespace

import mujoco
import pytest

from pluggybot.challenge import stack
from pluggybot.economy import scoring
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.swap import HubSwap
from pluggybot.tools.gripper import GRIP_Z, ClawTool

TABLE = scoring.challenge_table()
TOWER_XY = (1.10, 0.30)


@pytest.fixture(scope="module")
def model():
  return stack.world_with_blocks()


def tower(model, data, offsets=((0, 0), (0, 0), (0, 0)), xy=TOWER_XY):
  """A test's hand: put the blocks in a stack, each layer offset (dx, dy)
  from the one below, cumulatively. Not a solution -- there is no robot in
  this and the grader cannot tell the difference, which is the point."""
  x, y = xy
  for i, (name, (dx, dy)) in enumerate(zip(stack.BLOCKS, offsets)):
    x, y = x + dx, y + dy
    stack.place(model, data, name,
                (x, y, stack.BLOCK_HALF + i * (stack.PITCH_M + 0.0005)))
  mujoco.mj_forward(model, data)


def hold(model, data, seconds=stack.HOLD_S):
  t0 = data.time
  while data.time - t0 < seconds:
    mujoco.mj_step(model, data)


def grade(model, data, at_done=None, seconds=stack.HOLD_S):
  """The seam as the lifecycle would run it: snapshot at the call, hold,
  sample, evaluate. `result` is the errand's account and is unread."""
  before = at_done if at_done is not None else stack.measure(model, data)
  hold(model, data, seconds)
  life = SimpleNamespace(model=model, data=data)
  errand = SimpleNamespace(task="stack")
  return scoring.score_errand(life, errand, {"stacked": True, "layers": 3},
                              before, table=TABLE)


# ---- the criteria are data the robot cannot move ---------------------------


def test_the_challenge_row_is_scoreable_and_not_in_the_shipped_table():
  """Every challenge row has an evaluator (a price with no judge is a promise
  the robot cannot collect), and none of them is in rewards.json: that file
  is what the overseer is shown and what every committed result hashes, so a
  challenge moves across in the PR that offers it, not before."""
  for name, row in TABLE.tasks.items():
    assert name in scoring.EVALUATORS and name in scoring.SAMPLERS
    assert row.tier in scoring.TIERS
  assert not set(TABLE.names) & set(scoring.default_table().names)


def test_the_criteria_are_the_issues_own():
  assert stack.LAYERS == 3
  assert stack.HOLD_S == 10.0
  assert stack.REST_OFFSET_M == stack.BLOCK_HALF


# ---- the evaluator is pure, and absences fail --------------------------------


def test_an_unmeasured_tower_fails():
  ok, _, reason = stack.eval_stack({})
  assert not ok and "never measured" in reason
  ok, _, reason = stack.eval_stack(stack.measurements(
    {"t": 0.0, "layers": 3, "freeStanding": True}, None))
  assert not ok and "never measured" in reason


def test_a_hold_shorter_than_the_criterion_fails_even_a_perfect_tower():
  done = {"t": 0.0, "layers": 3, "freeStanding": True, "offsetMm": 0.0,
          "heightMm": 77.7}
  early = {**done, "t": stack.HOLD_S - 0.5}
  ok, _, reason = stack.eval_stack(stack.measurements(done, early))
  assert not ok and "9.5 s" in reason
  ok, _, _ = stack.eval_stack(stack.measurements(done, {**done, "t": stack.HOLD_S}))
  assert ok


# ---- the runs: real physics, the grader reading the world --------------------


def test_a_free_standing_tower_passes_and_is_paid(model):
  """THE PASSING RUN. Three blocks placed in a stack, ten sim-seconds of
  physics, the grader reads the poses and contacts off the world."""
  data = mujoco.MjData(model)
  tower(model, data)
  verdict = grade(model, data)
  assert verdict.ok, verdict.reason
  assert verdict.metrics["layers"] == 3 and verdict.metrics["freeStanding"]
  assert verdict.metrics["holdS"] == pytest.approx(stack.HOLD_S, abs=0.05)
  assert 75.0 < verdict.metrics["heightMm"] < 80.0
  assert verdict.points == TABLE["stack"].base + TABLE["stack"].bonus


def test_a_leaning_tower_still_passes_with_less_bonus(model):
  """Method-agnostic, and neatness is a bonus not a gate: 6 mm of lean per
  layer is a tower whose centre of mass is over its base, and it stands.
  ⚠ On MuJoCo's default soft contact it fell at 2.6 s (Challenges.md §4)."""
  data = mujoco.MjData(model)
  tower(model, data, offsets=((0, 0), (0.006, 0), (0.006, 0)))
  verdict = grade(model, data)
  assert verdict.ok, verdict.reason
  assert 5.0 < verdict.metrics["offsetMm"] < 8.0
  assert TABLE["stack"].base < verdict.points < (TABLE["stack"].base
                                                  + TABLE["stack"].bonus)


def test_an_overhung_tower_falls_during_the_hold_and_fails(model):
  """THE FAILING RUN. 9 mm of lean per layer keeps every block within half
  an edge of the one below and puts the upper pair's centre of mass past the
  base block's edge: a tower at the instant it is let go -- every geometric
  check passes at the call -- and on the floor 0.3 s later. Without the hold
  this would be paid."""
  data = mujoco.MjData(model)
  tower(model, data, offsets=((0, 0), (0.009, 0), (0.009, 0)))
  at_done = stack.measure(model, data)
  assert at_done["layers"] == 3 and at_done["freeStanding"], \
    "the premise: it looks stacked when the robot calls it"
  verdict = grade(model, data, at_done=at_done)
  assert not verdict.ok
  assert "it fell" in verdict.reason, verdict.reason
  assert verdict.metrics["layersAtDone"] == 3 and verdict.metrics["layers"] == 1
  assert verdict.points == 0


def test_two_stacked_and_one_beside_fails_at_the_call(model):
  data = mujoco.MjData(model)
  tower(model, data, offsets=((0, 0), (0, 0), (0.10, 0)))
  stack.place(model, data, "block_2", (TOWER_XY[0] + 0.10, TOWER_XY[1],
                                       stack.BLOCK_HALF))
  mujoco.mj_forward(model, data)
  verdict = grade(model, data, seconds=0.5)
  assert not verdict.ok
  assert verdict.reason == "2 of 3 blocks stacked at the call"


def test_two_blocks_balanced_on_one_is_not_a_tower(model):
  """`layers` is a CHAIN: two blocks side by side on top of a third rest on
  it, but neither rests on the other."""
  data = mujoco.MjData(model)
  stack.place(model, data, "block_0", (*TOWER_XY, stack.BLOCK_HALF))
  for name, dx in (("block_1", -0.012), ("block_2", 0.012)):
    stack.place(model, data, name, (TOWER_XY[0] + dx, TOWER_XY[1],
                                    stack.BLOCK_HALF + stack.PITCH_M + 0.0005))
  mujoco.mj_forward(model, data)
  assert stack.measure(model, data)["layers"] == 2


def test_a_tower_the_claw_is_holding_up_fails(model):
  """Criterion 4. The claw on the fork, jaws closed on the top block of a
  three-high stack: the geometry passes, the contacts do not."""
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  claw = ClawTool(model, data, swap)
  claw.jaws(0.0)
  claw.lower_grip_to(GRIP_Z + 2 * stack.PITCH_M)
  gx, gy, _ = claw.grip_world()
  tower(model, data, xy=(gx, gy))
  claw.jaws(1.0, settle=1.2)
  assert claw.holding("block_2_box"), "the premise: the claw has the top block"
  at_done = stack.measure(model, data)
  assert at_done["layers"] == 3 and not at_done["freeStanding"]
  assert at_done["touchedBy"] == ["module_claw_pad_l", "module_claw_pad_r"]
  ok, _, reason = stack.eval_stack(stack.measurements(
    at_done, {**at_done, "t": at_done["t"] + stack.HOLD_S}))
  assert not ok and "holding the tower up" in reason


def test_the_report_is_not_read(model):
  """scoring.py's founding rule, on the one kind an agent is most tempted to
  break it on: the sampler is handed an errand result that says the tower
  is built, and the world says the blocks are where they started."""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  verdict = grade(model, data, seconds=0.5)
  assert not verdict.ok and verdict.points == 0
  assert verdict.reason == "1 of 3 blocks stacked at the call"
