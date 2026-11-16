"""#719's option 2: while one hand's gripper reads it holds something, a reach,
a carry or a pull of the other hand takes no posture but the one the body is
in, and says so where that one does not reach. Asked a rise out of a deep bow
and a turn of the waist with the left jaws on a cupboard's lever, Rowan's
right reach tipped the body over backward (door_p3, 6 of 48 flights at its two
live draws, 3 of them stuck deaths); taken in stages, it fell as often."""
import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.body import StubBody
from pluggybot.procedure import steps as st

from test_hands import _bare, _bare_reach, _fly


def _holding(left=True):
  r = _bare_reach()
  r.posture = np.array([-0.22, 0.8, 0.0])
  r.grippers["left"].holding = left
  return r


def test_a_reach_plans_only_in_the_posture_it_is_in_while_the_other_hand_holds():
  for left in (True, False):
    r = _holding(left)
    asked = []
    r._planned = lambda side, goal, approach, postures, flip: (asked.append(postures), (None, goal))[1]
    assert not r.reach("right", rc.HandGoal(np.array([0.4, -0.1, 0.8])))
    assert (asked[0] == ((-0.22, 0.8, 0.0),)) == left
    assert asked[0] == rc.REACH_POSTURES or left
    assert r.kept_by["right"] == ("left" if left else None)


def test_a_carry_and_a_pull_take_no_other_posture_while_the_other_hand_holds():
  r = _holding()
  r.goal["right"], r.phase["right"] = rc.HandGoal(np.array([0.4, -0.1, 0.8])), "hold"
  r.grip_now = lambda side: (np.array([0.4, -0.1, 0.8]), np.eye(3))
  planned = []
  r.posture_for = r.path_for = lambda *a, **kw: planned.append(a) or (0.0, 0.0, 0.0)
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  assert r.carry("right", np.array([0.4, -0.1, 0.9]))
  assert planned == [] and r.postures == [(-0.22, 0.8, 0.0)]
  r.phase["right"] = "hold"
  assert r.pull("right", np.array([0.0, 0.0, 1.0]), 10.0, 0.1)
  assert planned == [] and "postured" not in r.pulls["right"]
  # ...and with nothing held, as before
  r.grippers["left"].holding = False
  r.phase["right"] = "hold"
  assert r.carry("right", np.array([0.4, -0.1, 0.9]))
  assert len(planned) == 1


class _Kept(StubBody):
  """A stub whose left hand holds something as its right reaches, the right
  out of reach from the posture the body is in."""

  def posture_kept_by(self, side):
    return {"right": "left"}.get(side)


def test_a_reach_out_of_reach_while_the_other_hand_holds_says_so():
  body = _Kept(*_bare())
  body.out_of_reach = {"right"}
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n', body)["steps"][-1]
  assert not step["ok"] and step["why"] == "out of reach"
  assert step["reason"] == st.REACH_KEPT.format(hand="right", other="left") == (
    "your left hand holds something, so your body keeps the posture it is in, and from it "
    "your right hand does not reach there")


def test_a_move_out_of_reach_while_the_other_hand_holds_says_so():
  class Short(_Kept):
    def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0,
                          path=None, turns=None):
      return {"why": "out of reach", "seconds": 0.5, "moved": 0.0, "dx": 0.0, "dy": 0.0,
              "dz": 0.0}
      yield
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n    move_hand("right", 0.3, -0.2, 1.1)\n',
              Short(*_bare()))["steps"][-1]
  assert not step["ok"]
  assert step["reason"] == st.MOVE_KEPT.format(hand="right", other="left") == (
    "your right hand could reach no further that way from the posture you are in: your left "
    "hand holds something, so your body keeps it")


def test_the_verbs_say_the_body_keeps_its_posture_while_the_other_hand_holds():
  for verb in ("reach", "move_hand", "pull"):
    assert "while your other hand holds something, your body keeps the posture it is in" in (
      st.HANDS_VERBS[verb].doc.replace("While", "while")), verb
