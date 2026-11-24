"""#719's option 2: while one hand is out at its work and its gripper reads it
holds something, a reach, a carry or a pull of the other hand takes no
posture but the one the body is in, and says so where that one does not
reach. Asked a rise out of a deep bow and a turn of the waist with the left
jaws on a cupboard's lever, Rowan's right reach tipped the body over backward
(door_p3, 6 of 48 flights at its two live draws, 3 of them stuck deaths);
taken in stages, it fell as often."""
import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.procedure import steps as st

from test_hands import _bare, _bare_reach, _fly

BOWED = (-0.22, 0.8, 0.0)


def _holding(left="hold", held=True):
  """A bare reach bowed, its left hand in phase `left` -- out at its work
  ("hold"), carried at its side ("carried") or home ("rest") -- its gripper
  reading `held`."""
  r = _bare_reach()
  r.posture = np.array(BOWED)
  # ...the posture the body is in, as it reads itself (`posture_now`)
  r.posture_now = lambda: np.array(BOWED)
  r.phase["left"] = left
  r.goal["left"] = None if left == "rest" else rc.HandGoal(np.array([0.4, 0.2, 0.8]))
  r.grippers["left"].holding = held
  return r


@pytest.mark.parametrize("left,held,verb,kept", [
  ("hold", True, True, True),
  ("hold", False, True, False),
  ("carried", True, True, False),     # a piece carried at the side (#682) holds the body in nothing
  ("rest", True, True, False),
  ("hold", True, False, False),       # the loop's own reach, a charge's: never kept
])
def test_a_reach_plans_only_in_the_posture_it_is_in_while_the_other_hand_holds(left, held, verb, kept):
  r = _holding(left, held)
  asked = []
  r._planned = lambda side, goal, approach, postures, flip, **kw: (
    asked.append(postures), (None, goal, None, None))[1]
  assert not r.reach("right", rc.HandGoal(np.array([0.4, -0.1, 0.8])), verb=verb)
  assert asked[0] == ((BOWED,) if kept else rc.REACH_POSTURES)
  assert r.kept_by["right"] == ("left" if kept else None)


def test_a_carry_and_a_pull_take_no_other_posture_while_the_other_hand_holds():
  r = _holding()
  r.goal["right"], r.phase["right"] = rc.HandGoal(np.array([0.4, -0.1, 0.8])), "hold"
  r.grip_now = lambda side: (np.array([0.4, -0.1, 0.8]), np.eye(3))
  asked = []
  r.posture_for = r.path_for = lambda *a, postures=None, **kw: asked.append(postures) or BOWED
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  assert r.carry("right", np.array([0.4, -0.1, 0.9]), verb=True)
  assert asked == [(BOWED,)] and r.postures == [BOWED]
  # ...and one the posture it is in does not reach is refused, nothing moved
  r.posture_for = lambda *a, postures=None, **kw: asked.append(postures) or None
  r.phase["right"] = "hold"
  assert not r.carry("right", np.array([0.4, -0.1, 1.2]), verb=True)
  assert r.report["right"]["why"] == "out of reach" and r.postures == [BOWED]
  # a pull takes none, and says whose grip kept it, pressed or not
  r.phase["right"] = "hold"
  r.force["right"] = np.array([0.0, 0.0, -100.0])
  r.kept_by["right"] = None
  assert r.pull("right", np.array([0.0, 0.0, 1.0]), 10.0, 0.1, verb=True)
  assert len(asked) == 2 and "postured" not in r.pulls["right"] and r.kept_by["right"] == "left"
  # ...but the loop's own carry and pull -- a charge's, a walk's lift -- keep nothing
  r.phase["right"] = "hold"
  r.posture_for = lambda *a, postures=None, **kw: asked.append(postures) or BOWED
  assert r.carry("right", np.array([0.4, -0.1, 0.9]), verb=False)
  assert asked[-1] is None and r.kept_by["right"] is None
  r.phase["right"] = "hold"
  r.force["right"] = np.zeros(3)
  r.path_for = lambda *a, postures=None, **kw: asked.append(postures) or BOWED
  assert r.pull("right", np.array([0.0, 0.0, 1.0]), 10.0, 0.1, verb=False)
  assert asked[-1] is None and r.kept_by["right"] is None
  # ...and with nothing held, as before
  r.grippers["left"].holding = False
  r.phase["right"] = "hold"
  r.posture_for = lambda *a, postures=None, **kw: asked.append(postures) or BOWED
  assert r.carry("right", np.array([0.4, -0.1, 0.9]), verb=True)
  assert asked[-1] is None and r.kept_by["right"] is None   # (no `postures` passed)


def test_a_carried_or_pulled_hand_plans_no_new_posture_while_the_grip_kept_it():
  r = _holding()
  at = np.array([0.4, -0.1, 0.8])
  r.cmd = {"right": rc.HandGoal(at.copy(), None, 1.0)}
  # ...a goal it does not reach in the test's time
  r.goal["right"] = rc.HandGoal(at + [0.0, 0.0, 1.0], None, 1.0)
  r.speed, r.q_cmd = {"right": rc.HOLD_SPEED}, {"right": np.zeros(len(hm.ARM))}
  r.miss, r._off, r.phase["right"] = {"right": 0.0}, {"right": 0.0}, "carry"
  r.self_model = type("SM", (), {"solve": staticmethod(
    lambda side, pos, rot: (np.zeros(len(hm.ARM)), 2 * rc.REPLAN_M, 0.0))})()
  planned = []
  r.posture_for = lambda *a, **kw: planned.append(a) or (0.0, 0.0, 0.0)
  r.pulls = {"right": {"start": at.copy(), "t": 0.0, "ways": [], "fixed": False}}
  r.kept_by["right"] = "left"
  for _ in range(int(3 * rc.REPLAN_S / 0.1)):
    r._held("right", 0.1, at, None)
  assert planned == []
  # ...the value kept at the ask, never the gripper read again: a grip
  # reading empty for an instant opens no posture change
  r.grippers["left"].holding = False
  for _ in range(int(3 * rc.REPLAN_S / 0.1)):
    r._held("right", 0.1, at, None)
  assert planned == []
  r.kept_by["right"] = None
  r._held("right", 0.1, at, None)
  assert len(planned) == 1


class _Kept(StubBody):
  """A stub whose left hand holds something as its right reaches, the right
  out of reach from the posture the body is in -- and, `elsewhere`, in
  reach from another."""

  elsewhere = True

  def posture_kept_by(self, side):
    return {"right": "left"}.get(side)

  def hand_reaches(self, side, pos, rot=None, verb=False):
    return self.elsewhere if side == "right" else super().hand_reaches(side, pos, rot, verb)


def test_a_reach_out_of_reach_while_the_other_hand_holds_says_so():
  body = _Kept(*_bare())
  body.out_of_reach = {"right"}
  # ...where no posture would reach either, it says so as before -- but
  # never that the hand out holding something could
  body.elsewhere = False
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n', body)["steps"][-1]
  assert step["reason"] == st.REACH_WHY["out of reach"].format(hand="right")
  body.elsewhere = True
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n', body)["steps"][-1]
  assert not step["ok"] and step["why"] == "out of reach"
  assert step["reason"] == st.REACH_KEPT.format(hand="right", other="left") == (
    "your left hand is out holding something, so your body keeps the posture it is in, and "
    "from it your right hand does not reach there")


def test_a_move_out_of_reach_while_the_other_hand_holds_says_so():
  class Short(_Kept):
    def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0,
                          path=None, turns=None, verb=True):
      return {"why": "out of reach", "seconds": 0.5, "moved": 0.0, "dx": 0.0, "dy": 0.0,
              "dz": 0.0}
      yield
  body = Short(*_bare())
  body.elsewhere = False
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n    move_hand("right", 0.3, -0.2, 1.1)\n',
              body)["steps"][-1]
  assert step["reason"] == st.MOVE_WHY["out of reach"].format(hand="right")
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n    move_hand("right", 0.3, -0.2, 1.1)\n',
              Short(*_bare()))["steps"][-1]
  assert not step["ok"]
  assert step["reason"] == st.MOVE_KEPT.format(hand="right", other="left") == (
    "your right hand could reach no further that way from the posture you are in: your left "
    "hand is out holding something, so your body keeps it")


def test_the_verbs_say_the_body_keeps_its_posture_while_the_other_hand_holds():
  for verb in ("reach", "move_hand", "pull"):
    assert "while your other hand is out holding something, your body keeps the posture it is in" in (
      st.HANDS_VERBS[verb].doc.replace("While", "while")), verb


def _calls(fn, name):
  import ast
  import inspect
  import textwrap
  tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
  return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
          and isinstance(n.func, ast.Attribute) and n.func.attr == name]


def _kw(call, name):
  import ast
  return next((ast.unparse(k.value) for k in call.keywords if k.arg == name), None)


def test_a_procedures_moves_are_kept_and_the_loops_own_never_are():
  """The one line of wiring each: a reach's `verb` is its `refuse` (a
  procedure's, #678's), the body hands `verb` down to the reach's carry and
  pull, a procedure's `move_hand` and `pull` pass it, and the loop's own
  moves -- a charge's, a walk's lift -- take the default: never kept."""
  import inspect
  from pluggybot.humanoid import body as hb
  from pluggybot.humanoid import charge
  from pluggybot.humanoid import mission as hmi
  reached = _calls(hb.HumanoidBody.reach_routine, "_reached")
  assert reached and all(_kw(c, "verb") == "refuse" for c in reached)
  assert all(_kw(c, "verb") == "verb" for c in _calls(hb.HumanoidBody._reached, "reach"))
  assert _kw(_calls(hb.HumanoidBody.move_hand_routine, "carry")[0], "verb") == "verb"
  assert _kw(_calls(hb.HumanoidBody.pull_routine, "pull")[0], "verb") == "verb"
  assert _kw(_calls(st._move_hand, "move_hand_routine")[0], "verb") == "True"
  assert _kw(_calls(st._pull, "pull_routine")[0], "verb") == "True"
  for fn in (hb.HumanoidBody.move_hand_routine, hb.HumanoidBody.pull_routine, rc.Reach.reach,
             rc.Reach.carry, rc.Reach.pull, rc.Reach.keep):
    assert inspect.signature(fn).parameters["verb"].default is False, fn
  src = inspect.getsource(charge) + inspect.getsource(hmi)
  assert "verb=" not in src


def test_a_pull_kept_holds_a_posture_still_coming_where_the_body_got_to():
  r = _holding()
  r.goal["right"], r.phase["right"] = rc.HandGoal(np.array([0.4, -0.1, 0.8])), "hold"
  r.grip_now = lambda side: (np.array([0.4, -0.1, 0.8]), np.eye(3))
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  # ...a posture still coming: the body stands short of the one asked
  r.posture = np.array([-0.16, 0.3, 0.5])
  r.path_for = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("planned"))
  assert r.pull("right", np.array([0.0, 0.0, 1.0]), 10.0, 0.1, verb=True)
  assert r.postures == [BOWED] and r.pulls["right"]["postured"] == 0.0


def test_a_reach_kept_off_another_robot_says_the_other_hand_is_out_holding():
  class Off(_Kept):
    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                      turns=None, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "kept off"

    def hand_kept(self, side):
      return {"m": 0.9, "x": 0.5, "y": -0.6, "home": False, "back": False, "other": "holds"}
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n', Off(*_bare()))["steps"][-1]
  assert step["reason"].endswith("; your left hand is out holding something")


def test_the_posture_kept_is_the_plans_where_the_body_is_near_it_never_a_swaying_reading():
  r = _holding()
  r.posture_now = lambda: np.array(BOWED) + [0.01, 0.03, 0.01]
  assert r.keep("right", True) == BOWED
  r.posture_now = lambda: np.array(BOWED) + [0.0, -0.2, 0.0]
  assert np.allclose(r.keep("right", True), (-0.22, 0.6, 0.0))


def test_whose_grip_kept_the_posture_rides_a_restart():
  """A move's way reads which hand's grip kept its posture (`_held`,
  `_pulled`, `_look`), so a restart keeps it with the reach (`Reach.KEPT`),
  beside #693's way home (`test_carry`'s)."""
  assert "kept_by" in rc.Reach.KEPT


# ---- the review of loop-719c: what a refusal says is planned as the reach would be ----


def test_the_posture_a_move_would_be_kept_in_is_read_without_recording_it():
  r = _holding()
  assert r.keeps("right", True) == BOWED and r.kept_by["right"] is None
  assert r.keeps("right", False) is None and r.keeps("left", True) is None
  assert r.keep("right", True) == BOWED and r.kept_by["right"] == "left"


def test_a_hand_reaches_as_its_reach_would_plan_it_kept_or_not():
  """`Body.hand_reaches(verb=True)` plans as a procedure's reach would now:
  in the posture the body is in where the other hand is out holding
  something, else in any -- recording nothing."""
  from types import SimpleNamespace as NS
  from pluggybot.humanoid import body as hb
  r = _holding()
  asked = []
  r.plans = lambda side, goal, postures=rc.REACH_POSTURES, **kw: asked.append(postures) or True
  body = NS(reach=r, stance=np.zeros(3), to_stance=lambda p: np.asarray(p, dtype=float),
            mission=NS(arms_seen=lambda: np.zeros((0, 3))))
  assert hb.HumanoidBody.hand_reaches(body, "right", (0.4, -0.1, 0.8), verb=True)
  assert hb.HumanoidBody.hand_reaches(body, "right", (0.4, -0.1, 0.8))
  assert hb.HumanoidBody.hand_reaches(body, "left", (0.4, 0.2, 0.8), verb=True)
  assert asked == [(BOWED,), rc.REACH_POSTURES, rc.REACH_POSTURES]
  assert r.kept_by == {"left": None, "right": None}


def test_what_a_reach_out_of_reach_says_of_either_hand_is_planned_as_its_reach_would_be():
  """The turn it might take and what the other hand could do are asked as
  a procedure's reach would plan them (`verb`): offered from any posture,
  a hand whose reach the holding hand would keep was offered, and refused."""
  class Asked(StubBody):
    def hand_reaches(self, side, pos, rot=None, verb=False):
      self.asked.append((side, rot is not None, verb))
      return False
  body = Asked(*_bare())
  body.asked, body.out_of_reach = [], {"right"}
  _fly('def a():\n    reach("right", 0.3, -0.2, 0.9, fx=1.0, jy=1.0)\n', body)
  assert body.asked == [("right", False, True), ("left", True, True), ("left", False, True)]


def test_a_kept_reach_out_of_reach_as_turned_says_whether_another_turn_reaches_from_it():
  class Turned(_Kept):
    def hand_reaches(self, side, pos, rot=None, verb=False):
      # ...from another posture as asked; from the one kept, only turned another way
      return side == "right" and (not verb or rot is None)
  body = Turned(*_bare())
  body.out_of_reach = {"right"}
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9, fx=1.0, jy=1.0)\n', body)["steps"][-1]
  assert step["reason"] == st.REACH_KEPT.format(hand="right", other="left") + st.TURNED_ELSE
  body.hand_reaches = lambda side, pos, rot=None, verb=False: side == "right" and not verb
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9, fx=1.0, jy=1.0)\n', body)["steps"][-1]
  assert step["reason"] == st.REACH_KEPT.format(hand="right", other="left")


def test_a_reach_kept_off_judges_the_other_hands_way_as_its_reach_would_be_planned():
  """Kept off another robot (#678), a reach says what the other hand's way
  would do: judged in the posture the body is in where this hand holds
  something (#719), and said off a copy of what the reach kept."""
  from types import SimpleNamespace as NS
  from pluggybot.humanoid import body as hb
  r = _holding()                      # the LEFT hand out, holding
  off = {"m": 0.9, "x": 0.5, "y": -0.6, "home": False, "back": False}
  r.kept["left"] = off
  r.reach = lambda *a, **kw: False
  judged = []
  r.clear_way = lambda side, goal, postures=rc.REACH_POSTURES, **kw: judged.append((side, postures))
  body = NS(reach=r, _why={}, _kept={},
            mission=NS(keep_out=lambda seen: [], look_round=lambda: None,
                       arms_seen=lambda: np.zeros((0, 3))))
  goal = rc.HandGoal(np.array([0.4, 0.2, 0.9]))
  assert not hb.HumanoidBody._reached(body, "left", goal, rc.REACH_POSTURES, False, words=True,
                                      verb=True)
  assert judged == [("right", (BOWED,))] and body._why["left"] == "kept off"
  assert not hb.HumanoidBody._reached(body, "left", goal, rc.REACH_POSTURES, False, words=False)
  assert body._kept["left"] == off and body._kept["left"] is not r.kept["left"]
