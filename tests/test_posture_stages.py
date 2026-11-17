"""#719, Ben's option 1: a posture a reach, a carry or a pull asks further than
a stage from the one the body is in -- as its senses have it -- is taken in
stages, each held before the next is asked (or given its wait), the plan's
asked after the last. Asked in one go, Luca's trunk came up 16.5 deg in 0.6 s
out of a bow it had leant on a pressed hand, and it went over backward."""
import ast
import inspect
import math
import textwrap
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc


class _Stand:
  """The standing policy as a reach meets it: what it was asked, whether
  the posture it is shown has arrived, and the trunk's gyro."""

  def __init__(self):
    self.asks, self.settled, self.trunk = [], True, NS(rate=np.zeros(3))

  def ask(self, height=0.0, pitch=0.0):
    self.asks.append((round(float(height), 6), round(float(pitch), 6)))


def _reach(now=(0.0, 0.0, 0.0)):
  """A `Reach` with no world: its posture's own methods over a policy that
  records what it is asked, the body in posture `now` by its senses."""
  r = object.__new__(rc.Reach)
  r.posture, r.trim = np.zeros(3), np.zeros(2)
  r.stand, r.stand_home = _Stand(), False
  r.sensed = np.zeros(len(hm.JOINT_NAMES))
  r.now = np.array(now, dtype=float)
  r.posture_now = lambda: r.now.copy()
  r.waist_cmd, r.waist_speed = np.zeros(len(hm.WAIST)), np.zeros(len(hm.WAIST))
  r.waist_sent = []
  r.waist = NS(command=lambda target, feed: r.waist_sent.append(np.array(target)))
  return r


def test_a_posture_further_than_a_stage_is_taken_in_even_stages():
  r = _reach((-0.22, 0.8, 0.0))
  goal = np.array([-0.08, 0.0, math.radians(60)])
  stages = r._stages_to(goal)
  # ...as many as its pitch's or its waist's change takes, whichever more
  n = max(math.ceil(0.8 / rc.STAGE_PITCH), math.ceil(math.radians(60) / rc.STAGE_WAIST))
  assert len(stages) == n - 1 > 0
  for k, s in enumerate(stages, 1):
    assert np.allclose(s, r.now + (goal - r.now) * k / n)
  # ...none where its pitch and its waist are each within a stage, the height
  # whatever it is
  near = r.now + np.array([-0.2, -0.9 * rc.STAGE_PITCH, 0.9 * rc.STAGE_WAIST])
  assert r._stages_to(near) == []


def test_a_reachs_posture_asks_its_first_stage_and_turns_the_waist_toward_it():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, math.radians(45))
  first = r.stages[0]
  assert r.stand.asks[-1] == (round(float(first[0]), 6), round(float(first[1]), 6))
  assert np.array_equal(r.asked(), first)
  assert np.allclose(r.posture, (-0.16, 0.3, math.radians(45)))
  for _ in range(400):
    r._turn_waist(0.01, np.zeros(len(hm.WAIST)))
  assert np.allclose(r.waist_sent[-1], first[2:])
  # ...and one within a stage is asked at once, as `hold_posture` asks it
  r = _reach((0.0, 0.3, 0.0))
  r.take_posture(0.0, 0.3 - 0.9 * rc.STAGE_PITCH, 0.0)
  assert r.stages == [] and r.stand.asks[-1] == (0.0, round(0.3 - 0.9 * rc.STAGE_PITCH, 6))


def test_each_stage_held_asks_the_next_and_the_plans_after_the_last():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  n = len(r.stages)
  assert n > 1
  for k in range(n):
    stage = r.stages[0]
    # ...not until it is held: there, still and the policy shown it
    r.now = stage + np.array([0.0, 0.1, 0.0])
    r._stage_on(0.01)
    assert len(r.stages) == n - k
    r.now = stage.copy()
    r.stand.trunk.rate = np.array([0.0, 0.2, 0.0])
    r._stage_on(0.01)
    assert len(r.stages) == n - k
    r.stand.trunk.rate = np.zeros(3)
    r._stage_on(0.01)
    assert len(r.stages) == n - k - 1
  assert np.allclose(r.posture, (-0.16, 0.3, 0.0)) and np.array_equal(r.asked(), r.posture)
  assert r.stand.asks[-1] == (-0.16, 0.3)


def test_a_stage_not_held_is_given_its_wait_then_the_next_is_asked():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.08, 0.0, math.radians(60))
  n = len(r.stages)
  # ...something keeps the trunk bowed short of the stage
  r.now = np.array([-0.18, 0.75, 0.0])
  t = 0.0
  while t < rc.STAGE_WAIT_S - 0.02:
    r._stage_on(0.01)
    t += 0.01
  assert len(r.stages) == n
  r._stage_on(0.01)
  r._stage_on(0.01)
  assert len(r.stages) == n - 1
  assert np.allclose(r.posture, (-0.08, 0.0, math.radians(60)))
  first = r.stages[0]
  assert r.stand.asks[-1] == (round(float(first[0]), 6), round(float(first[1]), 6))


def test_a_reach_goes_in_and_a_carried_hand_moves_only_once_every_stage_is_held():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  # ...whatever the time: the stages' own waits bound it, never POSTURE_WAIT_S
  r._posture_wait = 0.0
  assert not any(r._posture_there(1.0) for _ in range(int(rc.POSTURE_WAIT_S) + 2))
  assert not r._posture_held()
  r.stages, r.now = [], np.array([-0.16, 0.3, 0.0])
  assert r._posture_held() and r._posture_there(0.01)


#: The asks of a posture that go through its stages, by the method that makes
#: them -- and the ones that never do: a move stopped is held where it got
#: to (`halt`), as is #699's back-off, which goes back to where its pull
#: began, and an arm's way home asks the stand (#693's).
STAGED = {"reach": 1, "carry": 1, "pull": 1, "_look": 1, "_held": 1, "_pulled": 1}


def _calls(fn, name):
  tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
  return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
          and isinstance(n.func, ast.Attribute) and n.func.attr == name]


@pytest.mark.parametrize("method", sorted(STAGED))
def test_a_reach_a_carry_a_pull_and_their_new_postures_ask_through_their_stages(method):
  fn = getattr(rc.Reach, method)
  assert len(_calls(fn, "take_posture")) == STAGED[method]
  # ...none of their postures asked in one go, but #699's give-up and back-off
  held = [ast.unparse(c) for c in _calls(fn, "hold_posture")]
  assert all("began" in h or "posture_now" in h for h in held), held


def test_a_carried_hand_holds_still_through_its_stages_however_long():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  at = np.array([0.4, 0.1, 0.8])
  r.cmd = {"left": rc.HandGoal(at.copy(), None, 1.0)}
  r.goal = {"left": rc.HandGoal(at + [0.0, 0.0, 0.1], None, 1.0)}
  r.speed, r.q_cmd = {"left": rc.HOLD_SPEED}, {"left": np.zeros(len(hm.ARM))}
  r.miss, r._off, r.phase = {"left": 0.0}, {"left": 0.0}, {"left": "carry"}
  r.self_model = NS(solve=lambda side, pos, rot: (np.zeros(len(hm.ARM)), 0.0, 0.0))
  r.pulls = {"left": {"start": at.copy(), "t": 0.0, "ways": [], "fixed": False, "postured": 0.0}}
  # ...past the time a posture asked in one go is given: the stages' waits bound it
  for _ in range(int(2 * rc.POSTURE_WAIT_S / 0.1)):
    r._held("left", 0.1, at, None)
  assert np.array_equal(r.cmd["left"].pos, at)
  r.stages, r.now = [], np.array([-0.16, 0.3, 0.0])
  r._held("left", 0.1, at, None)
  assert r.cmd["left"].pos[2] > at[2]


def test_a_way_outs_stop_waits_out_a_transient_only_while_the_body_takes_a_stage():
  # #708's stop (P1's gate): at once with the body in the posture it was asked
  # -- the stage asked, where it comes in stages -- else held `OUT_TIP_S`
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.08, 0.0, math.radians(60))
  r._tip, r._tipping = {"right": 0.0}, lambda side: rc.OUT_TIP_NM + 5.0
  r.now = r.stages[0].copy()
  assert r._tipped("right", 0.01)
  r._tip["right"], r.now = 0.0, r.stages[0] + np.array([0.0, 0.1, 0.0])
  assert not r._tipped("right", 0.01)


def test_no_new_posture_is_planned_for_a_carried_hand_while_stages_are_to_come():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  at = np.array([0.4, 0.1, 0.8])
  r.cmd = {"left": rc.HandGoal(at.copy(), None, 1.0)}
  r.goal = {"left": rc.HandGoal(at + [0.0, 0.0, 0.1], None, 1.0)}
  r.speed, r.q_cmd = {"left": rc.HOLD_SPEED}, {"left": np.zeros(len(hm.ARM))}
  r.miss, r._off, r.phase = {"left": 0.0}, {"left": 0.0}, {"left": "carry"}
  # ...its arm missing its command, as a body on its way between stages makes it
  r.self_model = NS(solve=lambda side, pos, rot: (np.zeros(len(hm.ARM)), 2 * rc.REPLAN_M, 0.0))
  planned = []
  r.posture_for = lambda *a, **kw: planned.append(a) or (0.0, 0.0, 0.0)
  r.pulls = {"left": {"start": at.copy(), "t": 0.0, "ways": [], "fixed": False, "postured": 0.0}}
  for _ in range(int(3 * rc.REPLAN_S / 0.1)):
    r._held("left", 0.1, at, None)
  assert planned == []
  r.stages = []
  r._held("left", 0.1, at, None)
  assert len(planned) == 1


def test_the_plan_asked_again_keeps_its_stages_and_never_stages_a_body_held_short():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  stages, asks = [s.copy() for s in r.stages], len(r.stand.asks)
  r.now = stages[0].copy()
  r.take_posture(-0.16, 0.3, 0.0)
  assert len(r.stages) == len(stages) and all(np.array_equal(a, b) for a, b in zip(r.stages, stages))
  assert len(r.stand.asks) == asks
  # ...and with none to come, a body held 0.2 rad short of it by what its
  # other hand grips is asked the plan, as `hold_posture` asks it -- never a
  # shallower stage that would unload what it grips
  r.stages, r.now = [], np.array([-0.16, 0.5, 0.0])
  r.take_posture(-0.16, 0.3, 0.0)
  assert r.stages == [] and r.stand.asks[-1] == (-0.16, 0.3)


def test_a_carry_on_the_same_plan_with_stages_to_come_holds_its_hand_still():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  at = np.array([0.4, 0.1, 0.8])
  r.goal = {"left": rc.HandGoal(at.copy(), np.eye(3), 1.0)}
  r.cmd = {"left": rc.HandGoal(at.copy(), np.eye(3), 1.0)}
  r.phase, r.speed, r._off = {"left": "hold"}, {"left": rc.HOLD_SPEED}, {"left": 0.0}
  r.pulls, r.report = {"left": None}, {"left": None}
  r.grip_now = lambda side: (at.copy(), np.eye(3))
  r.posture_for = lambda *a, **kw: (-0.16, 0.3, 0.0)
  assert r.carry("left", at + [0.0, 0.0, 0.05])
  assert r.stages and "postured" in r.pulls["left"]


def test_a_carry_ended_or_an_arm_sent_home_ends_the_stages_where_the_body_got_to():
  r = _reach((-0.22, 0.8, 0.0))
  r.take_posture(-0.16, 0.3, 0.0)
  at = np.array([0.4, 0.1, 0.8])
  r.now = np.array([-0.2, 0.62, 0.0])
  r.goal = {"left": rc.HandGoal(at + [0.0, 0.0, 0.05], None, 1.0)}
  r.cmd = {"left": rc.HandGoal(at.copy(), None, 1.0)}
  r.pulls, r.report, r.phase = {"left": {"start": at.copy(), "t": 1.0}}, {"left": None}, {"left": "carry"}
  r._end("left", "held back", at, np.eye(3))
  assert r.stages == [] and np.allclose(r.posture, r.now)
  # ...and a withdraw cut short keeps no plan still walking in stages: the stand at once
  r.take_posture(-0.25, 0.95, 0.0)
  assert r.stages
  r.phase = {s: "rest" for s in hm.SIDES}
  r.phase["left"] = "retreat"
  r.goal, r.cmd = {s: None for s in hm.SIDES}, {s: None for s in hm.SIDES}
  r.outs, r.looking = {s: [] for s in hm.SIDES}, {s: False for s in hm.SIDES}
  r.judged_home, r.homes = {s: None for s in hm.SIDES}, {s: [] for s in hm.SIDES}
  r.q_to, r.q_cmd = {s: np.zeros(len(hm.ARM)) for s in hm.SIDES}, {s: np.zeros(len(hm.ARM)) for s in hm.SIDES}
  r.way_home = lambda side: [np.zeros(len(hm.ARM))]
  r.carry_q, r._ahead = {s: None for s in hm.SIDES}, lambda side: False
  r._go_home("left")
  assert r.stages == [] and not r.stand_home and r.stand.asks[-1] == (0.0, 0.0)
