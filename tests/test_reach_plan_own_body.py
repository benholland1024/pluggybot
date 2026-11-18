"""A reach's plan keeps its arm off the robot's own trunk at its pre-grasp
and at its goal (issue #726; docs/Hands.md, SimNotes "A reach's plan kept
off the robot's own trunk"), each rule pinned without a flight -- the
robot's own drawing and a bare reach (docs/Testing.md). The flights that
measured it are the SimNotes section's.

  1. where today's plan keeps clear of the trunk it is today's, untouched;
  2. where it lies in the trunk, the goal's other solves in that posture are
     taken in turn, then the next posture's;
  3. where none keeps clear, a verb's reach is refused and nothing moves,
     `own` saying how near; the loop's own reach takes today's plan; asked
     whether it would plan, the reach says no, and `clear_way` "own";
  4. where the grip asked plans only in the trunk, the grip half a turn
     round is planned;
  5. a look at the pre-grasp passes over a posture whose plan from the arm
     as it is lies in the trunk -- with none clear, today's;
  6. an arm lies in the trunk past the slack two readings part by; and the
     body's why for the refusal is "own body".
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.humanoid import body as hb
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal, SelfModel

from test_hands import _standing_reach, _stepped
from test_reach_own_body import CUBE_7_PRE

#: The left hand to a point 12 cm ahead of the stance's middle, across to
#: its right, at no turn (drawn on the robot's own drawing, with nothing
#: under it): at the stand today's plan puts the arm 8.3 cm into the torso
#: and none of its other solves keeps clear of it...
INTO_TORSO = HandGoal(np.array([0.12, -0.10, 1.0]), None, 1.0)
STAND = ((0.0, 0.0, 0.0),)
#: ...15 cm higher, today's 4.4 cm into the torso at the stand, and the
#: next posture's plan 2.4 cm clear of it...
HIGHER = HandGoal(np.array([0.12, -0.10, 1.15]), None, 1.0)
TWO = ((0.0, 0.0, 0.0), (-0.08, 0.0, 0.0))
#: ...and low across, bowed: today's solve 6.6 cm into the torso, another of
#: its seeds' 1.7 cm clear in the same posture.
LOW = HandGoal(np.array([0.12, -0.10, 0.55]), None, 1.0)
BOWED = ((-0.2, 0.55, 0.0),)
#: A goal out in front whose plan keeps well clear.
AHEAD = HandGoal(np.array([0.35, 0.20, 0.9]), None, 1.0)


def _room(q) -> float:
  return rc.own_room(SelfModel(), "left", q, [])[0]


def _today(r, goal, postures):
  """Today's plan for a bare reach's left hand: `plan` as `Reach._planned`
  asks it, judging nothing of the trunk."""
  return rc.plan("left", goal, SelfModel(), postures=postures, arms=r._arms(but="left"),
                 start=r.q_cmd["left"])


def _pre(r) -> np.ndarray:
  """The arm at the pre-grasp a reach's way out ends at."""
  return ([r.q_to["left"]] + r.outs["left"])[-1]


# ---- 1-2: today's where it keeps clear, else another solve or posture ------------------------


def test_a_plan_that_keeps_clear_of_the_trunk_is_todays():
  r = _standing_reach()
  today = _today(r, AHEAD, rc.REACH_POSTURES)
  assert min(_room(today[1]), _room(today[2])) > 0.0, "the premise: it keeps clear"
  assert r.reach("left", AHEAD, verb=True)
  assert r.postures == [today[0]] and np.array_equal(_pre(r), today[1]) and r.own["left"] is None


def test_a_plan_in_the_trunk_takes_another_solve_in_the_same_posture():
  r = _standing_reach()
  today = _today(r, LOW, BOWED)
  assert _room(today[1]) < 0.0 or _room(today[2]) < 0.0, "the premise: today's is in the torso"
  assert r.reach("left", LOW, postures=BOWED, verb=True)
  assert r.postures == [BOWED[0]] and r.own["left"] is None
  assert _room(_pre(r)) >= 0.0 and not np.allclose(_pre(r), today[1], atol=0.1)


def test_a_plan_in_the_trunk_with_no_other_solve_clear_takes_the_next_posture():
  r = _standing_reach()
  today = _today(r, HIGHER, TWO)
  assert today[0] == TWO[0] and _room(today[1]) < 0.0, "the premise: the stand's is in the torso"
  assert r.reach("left", HIGHER, postures=TWO, verb=True)
  assert r.postures == [TWO[1]] and _room(_pre(r)) >= 0.0 and r.own["left"] is None


def test_the_other_solves_are_taken_in_turn_as_solve_seeded_ranks_them(monkeypatch):
  """Stubbed: today's arm lies in the trunk; of the goal's other solves, the
  one that is today's again (within `SAME_ARM_RAD`) is not planned twice,
  the next lies in the trunk too, and the one after it, clear, is taken --
  never a later one."""
  today, again, inside, clear, later = (np.full(6, v) for v in (0.0, 0.002, 1.0, 2.0, 3.0))
  monkeypatch.setattr(rc, "statue", lambda *a, **kw: None)
  monkeypatch.setattr(rc, "_solutions", lambda *a, **kw: [today, again, inside, clear, later])
  planned = []

  def reached_from(sm, side, posture, q0, goal, pre, slack, lean, arms, start, seeded=True):
    planned.append(float(start[0]) if start is not None else None)
    return (start, start) if start is not None else (today, today)
  monkeypatch.setattr(rc, "_reached_from", reached_from)
  monkeypatch.setattr(rc, "in_trunk", lambda side, *arms: (
    {"part": "torso", "m": -0.01} if float(arms[0][0]) in (0.0, 0.002, 1.0) else None))
  got = rc._reached(None, "left", (0.0, 0.0, 0.0), None, LOW, LOW, True, None, None, None, clear=True)
  assert float(got[0][0]) == 2.0 and got[2] is None and planned == [None, 1.0, 2.0]


# ---- 3: none clear --------------------------------------------------------------------------


def test_a_verbs_reach_whose_every_plan_lies_in_the_trunk_moves_nothing():
  r = _standing_reach()
  today = _today(r, INTO_TORSO, STAND)
  assert today is not None and _room(today[1]) < -0.05, "the premise: 8 cm into the torso"
  assert not r.reach("left", INTO_TORSO, postures=STAND, verb=True)
  assert r.phase["left"] == "rest" and r.postures == [] and r.goal["left"] is None
  assert r.own["left"] == {"part": "torso", "m": pytest.approx(round(_room(today[1]), 3))}
  # ...asked whether it would plan, it says no; whether its way would be clear, "own"
  assert not r.plans("left", INTO_TORSO, postures=STAND)
  assert r.clear_way("left", INTO_TORSO, postures=STAND) == "own"


def test_the_loops_own_reach_whose_every_plan_lies_in_the_trunk_takes_todays():
  """A charge's reach is never refused for its own trunk (a failed charge
  is a death): today's plan, `own` saying how near it comes."""
  r = _standing_reach()
  today = _today(r, INTO_TORSO, STAND)
  assert r.reach("left", INTO_TORSO, postures=STAND, refuse=True, verb=False)
  assert r.postures == [today[0]] and np.array_equal(_pre(r), today[1])
  assert r.own["left"]["part"] == "torso" and r.own["left"]["m"] < -0.05


# ---- 4: the grip half a turn round ------------------------------------------------------------


def test_a_grip_that_plans_only_in_the_trunk_is_planned_half_a_turn_round(monkeypatch):
  """`plan` stubbed: the grip asked plans only in the trunk, half a turn
  round it plans clear -- that is taken; and where that plans nothing, the
  loop's own reach takes the grip asked, as planned without the trunk."""
  r = _standing_reach()
  asked = np.eye(3)
  inside = _today(r, INTO_TORSO, STAND)
  clear = _today(r, AHEAD, rc.REACH_POSTURES)
  turned = {"plans": True}

  def plan(side, goal, sm, postures=rc.REACH_POSTURES, own=None, **kw):
    if np.allclose(goal.rot, asked):
      if own is None:
        return inside
      own.update(part="torso", m=-0.083, plan=inside)
      return None
    return clear if turned["plans"] else None
  monkeypatch.setattr(rc, "plan", plan)
  assert r.reach("left", HandGoal(INTO_TORSO.pos, asked, 1.0), verb=True)
  assert np.allclose(r.goal["left"].rot, np.diag([-1.0, -1.0, 1.0])) and np.array_equal(_pre(r), clear[1])
  turned["plans"] = False
  r = _standing_reach()
  assert not r.reach("left", HandGoal(INTO_TORSO.pos, asked, 1.0), verb=True)
  assert r.reach("left", HandGoal(INTO_TORSO.pos, asked, 1.0), verb=False)
  assert np.allclose(r.goal["left"].rot, asked) and np.array_equal(_pre(r), inside[1])


# ---- 5: a look at the pre-grasp --------------------------------------------------------------


SQUARE = [(0.0, 0.0, 0.0), (-0.08, 0.0, 0.0), (-0.16, 0.0, 0.0)]


def _looking(monkeypatch, inside, here=SQUARE[0]):
  """A bare reach's left hand at its pre-grasp in posture `here`; every
  posture reaches, and its plan from the arm as it is lies in the trunk
  where `inside` says so (`_reached` and `in_trunk` stubbed)."""
  r = _stepped(_standing_reach())
  r.stand = NS(settled=True)
  r._posture_there = lambda dt: True
  r._lean = lambda: rc.Lean()

  def hold(height, pitch, waist=0.0):
    r.posture = np.array([height, pitch, waist], dtype=float)
    r.postures.append((height, pitch, waist))
  r.hold_posture = hold
  monkeypatch.setattr(rc, "stands", lambda sm, posture, q0: True)
  monkeypatch.setattr(rc, "out_of_reach", lambda *a, **kw: False)
  marks = {p: np.full(len(hm.ARM), k + 1.0) for k, p in enumerate(SQUARE)}
  monkeypatch.setattr(rc, "_reached", lambda sm, side, posture, *a, **kw: (marks[tuple(posture)],) * 2)
  monkeypatch.setattr(rc, "in_trunk", lambda side, *arms: (
    {"part": "torso", "m": -0.01} if any(tuple(p) in inside and np.array_equal(arms[0], marks[tuple(p)])
                                          for p in SQUARE) else None), raising=False)
  q = r.self_model.solve("left", np.array([0.32, 0.2, 0.9]), None)[0]
  r.q_cmd["left"], r.q_to["left"] = q.copy(), q.copy()
  r.self_model._set_arm("left", q)
  r.goal["left"] = HandGoal(np.array([0.40, 0.2, 0.9]), None, 1.0)
  r.allowed["left"] = list(SQUARE)
  r.posture = np.array(here, dtype=float)
  r.phase["left"] = "move"
  return r


def test_a_look_passes_over_a_posture_whose_plan_lies_in_the_trunk(monkeypatch):
  r = _looking(monkeypatch, inside={SQUARE[0], SQUARE[1]})
  r._update(0.01)
  assert r.looking["left"] and r.postures == [SQUARE[2]]


def test_a_look_with_no_posture_clear_of_the_trunk_takes_todays(monkeypatch):
  """...and with none clear, the least bent that reaches, as before -- one
  less bent than the posture it holds is taken; at the one it holds, in it
  goes."""
  r = _looking(monkeypatch, inside=set(SQUARE), here=SQUARE[1])
  r._update(0.01)
  assert r.looking["left"] and r.postures == [SQUARE[0]]
  r = _looking(monkeypatch, inside=set(SQUARE))
  r._update(0.01)
  assert r.phase["left"] == "approach" and not r.looking["left"] and not r.postures


# ---- 6: in the trunk, and the body's why ------------------------------------------------------


def test_an_arm_lies_in_the_trunk_past_the_slack_two_readings_part_by(monkeypatch):
  """Our cube's set-out 7's pre-grasp, 3.5 mm into the head's back on the
  robot's model of itself, lies in it; the arm at rest does not; and a
  reading inside `OWN_SLACK_M` of touching is no depth."""
  assert rc.in_trunk("left", CUBE_7_PRE)["part"] == "head"
  assert rc.in_trunk("left", CUBE_7_PRE)["m"] == pytest.approx(-0.0035, abs=0.0005)
  rest = np.array([hm.stand_qpos().get(f"left_{j}", 0.0) for j in hm.ARM])
  assert rc.in_trunk("left", rest) is None and rc.in_trunk("left", rest, CUBE_7_PRE) is not None
  n = len(hm.ARM_GEOMS) * len(rc.UPPER_GEOMS)
  for depth, inside in ((0.5, False), (2.0, True)):
    monkeypatch.setattr(rc, "own_rooms", lambda sm, side, q0, ways, d=depth: np.full(n, -d * rc.OWN_SLACK_M))
    assert (rc.in_trunk("left", rest) is not None) is inside


def test_the_body_says_own_body_where_every_plan_lies_in_the_trunk():
  """Through the body (`HumanoidBody._reached`, as `reach_routine` asks
  it): a procedure's reach is refused, "own body"; the loop's own goes."""
  for verb in (True, False):
    r = _standing_reach()
    me = NS(reach=r, _why={}, _kept={}, mission=NS(look_round=lambda: None, keep_out=lambda seen: []))
    went = hb.HumanoidBody._reached(me, "left", INTO_TORSO, STAND, False, words=verb, verb=verb)
    assert went is (not verb) and me._why["left"] == ("own body" if verb else "out of reach")
