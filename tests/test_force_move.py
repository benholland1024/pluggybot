"""The hands' vocabulary (issue #525; docs/Hands.md): the force move and the
verbs around it, each rule pinned by a direct call on a `Reach` with no
world, or on the stub body (docs/Testing.md) -- the flights that measured
them are `scripts/hands_check.py`'s.

  1. along its way a pull's command walks at its speed while the hand
     pushes under half its force, slows as it nears it, stands at its
     force, and backs off past it, faster than it walked in -- while its
     body takes a posture too: the force is CLOSED on what the hand feels
     (admittance), a moment ahead at the rate it rises, and what it reports
     is a slower reading's;
  2. across its way, giving, the command follows the hand, no faster than
     its cap; holding, it keeps its line (hybrid force and position);
  3. its turn gives toward what it holds, but never about the jaws' own
     line;
  4. it takes its posture first and stands meanwhile; it ends at its
     distance, held fast where the hand stays put pushing its whole force
     or goes nowhere with its command as far ahead as it may go, out of
     reach, or -- a grip it began with lost -- where the hand is, and
     reports the way the hand was last going;
  5. a hand going on takes another posture from its arm as it is, never
     from a seed, never one it has left -- and a hand HELD, its arm reaching
     it, takes none: its command stays where the arm last reached, or backs
     off toward the hand, and pinned there it is held fast only pushing its
     force, else out of reach;
  6. a grip is shut with the force asked, inside the gripper's sheet, and
     judged against it, on its squeeze's running mean;
  7. `spot` bows until it sees and stands up again, and says which way the
     tag's edges run; the humanoid keeps its posture to look with a hand
     out;
  8. a path's point with no turn keeps the one before.
"""

import math

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.procedure import steps as st

from test_hands import _bare, _bare_reach, _fly, _out_hand

H = np.array([0.30, 0.20, 0.95])
# a command as long as the slowest reading: every reading is the force as set
DT = max(rc.ADMIT_SMOOTH_S, rc.MEAN_S, rc.RISE_SMOOTH_S)


def _pulling(way, force=10.0, distance=0.2, give=True, rot=None, held=False):
  """A bare reach's left hand out at `H` and pulling along `way`, holding
  something where `held`: its senses answer `H` and `rot` unless a test
  moves them."""
  r = _bare_reach()
  rot = np.eye(3) if rot is None else rot
  _out_hand(r, H, rot)
  r.phase["left"] = "hold"
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  r.grip_now = lambda side: (H.copy(), rot.copy())
  r.path_for = lambda side, poses, rot=None: tuple(float(v) for v in r.posture)
  r.posture_for = lambda side, pos, rot=None, onward=False, past=(): None
  r.grippers["left"].holding = held
  # ...and its arm reaching every command: the rules here are the pull's
  r.self_model.solve = lambda side, pos, rot=None, iters=None: (np.zeros(len(hm.ARM)), 0.0, 0.0)
  assert r.pull("left", way, force, distance, give=give)
  return r


def _along(r, way):
  return float((r.cmd["left"].pos - H) @ (np.asarray(way) / np.linalg.norm(way)))


def test_a_pull_walks_while_it_pushes_light_stands_at_its_force_and_backs_off_past_it():
  way = np.array([0.0, 0.0, 1.0])
  r = _pulling(way)
  r._pulled("left", DT, H, np.eye(3))
  assert _along(r, way) == pytest.approx(rc.HOLD_SPEED * DT)
  most = 0.0
  for _ in range(40):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H, np.eye(3))
    most = max(most, _along(r, way)) if r.pulls["left"] is not None else most
  assert most == pytest.approx(rc.LEAD_MAX_M)
  r = _pulling(way)
  r.force["left"] = 10.0 * way
  r._pulled("left", DT, H, np.eye(3))
  assert _along(r, way) == pytest.approx(0.0)
  # ...past it faster than it walks in (#547: no faster, a body leaning on
  # a held hand kept the force over its ask)
  for force, pace in ((12.0, -0.4 * rc.BACK_SPEED), (20.0, -rc.BACK_SPEED), (40.0, -rc.BACK_SPEED)):
    r = _pulling(way)
    r.force["left"] = force * way
    r._pulled("left", DT, H, np.eye(3))
    assert _along(r, way) == pytest.approx(pace * rc.HOLD_SPEED * DT), force
  assert rc.BACK_SPEED > 1.0
  # ...at its speed to half its force, and slowing past it
  for share, pace in ((0.5, 1.0), (0.75, 0.5)):
    r = _pulling(way)
    r.force["left"] = share * 10.0 * way
    r._pulled("left", DT, H, np.eye(3))
    assert _along(r, way) == pytest.approx(pace * rc.HOLD_SPEED * DT), share


def test_a_pull_is_closed_on_its_force_a_moment_ahead_at_the_rate_it_rises():
  """Closed on its force as it stood, the move built 100-200 N/s on a
  locked bar and put 1.7 times its ask on it before it read the ask, even
  closed on the bar's true force (#547)."""
  way = np.array([0.0, 0.0, 1.0])
  assert rc.AHEAD_S > 0.0
  # 5 N rising from 3 slows it where 5 stood would not, 8 rising from 2 backs
  # it off under its ask of 10, and 4 falling from 6 walks it on at its speed
  for before, now in ((3.0, 5.0), (2.0, 8.0), (6.0, 4.0)):
    r = _pulling(way)
    r.force["left"] = before * way
    r._pulled("left", DT, H, np.eye(3))
    went = _along(r, way)
    r.force["left"] = now * way
    r._pulled("left", DT, H, np.eye(3))
    ahead = now + rc.AHEAD_S * (now - before) / DT
    x = 2.0 * (10.0 - ahead) / 10.0
    pace = min(1.0, x) if x >= 0.0 else rc.BACK_SPEED * max(-1.0, x)
    assert _along(r, way) - went == pytest.approx(pace * rc.HOLD_SPEED * DT), (before, now)


def test_a_hand_going_along_its_way_is_only_slowed_by_where_its_force_is_going():
  """A hand going along its way builds no force against what holds it: the
  whole body's reading rides 4-5 N on a hand moving in free air, and backed
  off on where that went, a 4 N pull up 0.3 m stopped after 1.1 cm (#547).
  So going, the force a moment ahead stands it at most, and it backs off
  past its force as read, at its speed."""
  way = np.array([0.0, 0.0, 1.0])
  for now, pace in ((8.0, 0.0), (12.0, -0.4)):
    r = _pulling(way, distance=1.0)
    r.force["left"] = 2.0 * way
    step = 1.1 * rc.GO_ON_M / 3.0
    for k in range(1, 4):
      r._pulled("left", DT, H + step * k * way, np.eye(3))
    went = _along(r, way)
    # rising from 2 N to `now` in one command: well past its ask, a moment on
    r.force["left"] = now * way
    r._pulled("left", DT, H + step * 4 * way, np.eye(3))
    assert _along(r, way) - went == pytest.approx(pace * rc.HOLD_SPEED * DT), now


def test_a_pull_reports_a_slower_reading_than_it_is_closed_on():
  """The loop reads the hand over `ADMIT_SMOOTH_S`; the hardest it pushed --
  `pull.force`, a procedure's "it pressed" -- and held fast over `MEAN_S`
  (in free air the whole body's reading rides 3.6 N on a 4 N pull, #547)."""
  assert rc.ADMIT_SMOOTH_S < rc.MEAN_S
  way = np.array([0.0, 0.0, 1.0])
  r = _pulling(way)
  r._pulled("left", rc.ADMIT_SMOOTH_S, H, np.eye(3))
  r.force["left"] = 10.0 * way
  r._pulled("left", rc.ADMIT_SMOOTH_S, H, np.eye(3))
  st = r.pulls["left"]
  assert float(st["f"] @ way) == pytest.approx(10.0)
  assert st["peak"] == pytest.approx(10.0 * rc.ADMIT_SMOOTH_S / rc.MEAN_S)


def test_a_giving_pull_follows_the_hand_across_its_way_and_a_holding_one_keeps_its_line():
  way, moved = np.array([0.0, 0.0, 1.0]), H + [0.0, 0.01, 0.0]
  r = _pulling(way)
  r._pulled("left", DT, moved, np.eye(3))
  assert r.cmd["left"].pos[1] == pytest.approx(moved[1])
  r = _pulling(way, give=False)
  r._pulled("left", DT, moved, np.eye(3))
  assert r.cmd["left"].pos[1] == pytest.approx(H[1])
  # ...no faster than its cap: unbounded, a lid let go of its weight fell
  r = _pulling(way)
  r._pulled("left", DT, H + [0.0, 0.05, 0.0], np.eye(3))
  assert np.linalg.norm(r.cmd["left"].pos - H) == pytest.approx(
    rc.ACROSS_SPEED * rc.HOLD_SPEED * DT)


def test_a_pulls_turn_gives_to_what_it_holds_but_never_about_the_jaws_line():
  """A lid's weight bent the wrist about the jaws' line, and a turn that
  gave there followed it 70 deg down and out of the arm's reach (#525)."""
  from pluggybot.legs.probe import rot_about
  for axis, gives in ((np.array([1.0, 0.0, 0.0]), False), (np.array([0.0, 0.0, 1.0]), True)):
    for give in (True, False):
      r = _pulling(np.array([0.0, 0.0, 1.0]), give=give)
      before = r.cmd["left"].rot.copy()
      r._pulled("left", DT, H, rot_about(axis, math.radians(10.0)) @ np.eye(3))
      turned = math.degrees(math.acos(min(1.0, (np.trace(before.T @ r.cmd["left"].rot) - 1) / 2)))
      want = rc.GIVE_RATE * 10.0 if gives and give else 0.0
      assert turned == pytest.approx(want, abs=1e-6), (axis, give)
  # ...toward what it holds, in the world's frame, off a grip turned as a
  # grip is: never away from it, never about its jaws' line
  from pluggybot.humanoid.arms import _rot_error
  tilt = rot_about(np.array([0.0, 1.0, 0.0]), math.radians(30.0))
  r = _pulling(np.array([0.0, 0.0, 1.0]), rot=tilt)
  before = r.cmd["left"].rot.copy()
  sensed = rot_about(np.array([0.0, 0.0, 1.0]), math.radians(10.0)) @ tilt
  r._pulled("left", DT, H, sensed)
  e = _rot_error(sensed, before)
  perp = e - (e @ before[:, 0]) * before[:, 0]
  want = rot_about(perp, rc.GIVE_RATE * float(np.linalg.norm(perp))) @ before
  assert np.allclose(r.cmd["left"].rot, want, atol=1e-9)


def test_a_pull_ends_at_its_distance_or_held_fast_pushing_its_whole_force():
  way = np.array([0.0, 0.0, 1.0])
  r = _pulling(way, distance=0.05)
  for k in range(1, 20):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H + 0.01 * k * way, np.eye(3))
  assert r.report["left"]["why"] == "distance" and r.report["left"]["went"] >= 0.05 - 1e-9
  r = _pulling(way)
  r.force["left"] = 9.0 * way
  for _ in range(int(rc.FAST_S / DT) + 2):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H, np.eye(3))
  assert r.report["left"]["why"] == "held fast" and r.report["left"]["force"] == 9.0
  # ...and a hand that moves, or pushes light, is not held fast
  r = _pulling(way)
  r.force["left"] = 5.0 * way
  for _ in range(int(rc.FAST_S / DT) + 2):
    r._pulled("left", DT, H, np.eye(3))
  assert r.pulls["left"] is not None
  r = _pulling(way, distance=0.5)
  r.force["left"] = 9.0 * way
  for k in range(1, int(rc.FAST_S / DT) + 3):
    r._pulled("left", DT, H + 0.01 * k * way, np.eye(3))
  assert r.pulls["left"] is not None
  # ...it ends out of reach where its arm stays out of reach
  r = _pulling(way)
  r.self_model.solve = lambda side, pos, rot=None, iters=None: (np.zeros(len(hm.ARM)), 0.05, 0.0)
  for _ in range(int((rc.SETTLE_S + rc.BLOCKED_S) / DT) + 3):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H, np.eye(3))
  assert r.report["left"]["why"] == "out of reach"
  # ...and its report's way is the way the hand was last going, its own
  r = _pulling(way, distance=0.05)
  for k in range(1, 20):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H + 0.01 * k * np.array([1.0, 0.0, 1.0]), np.eye(3))
  assert r.report["left"]["way"] == pytest.approx([0.7071, 0.0, 0.7071], abs=1e-3)


def test_the_reachs_own_loop_runs_a_pull_as_a_force_move(monkeypatch):
  """Every rule above is `_pulled`'s: the loop must hand it a pull, and a
  carry to `_held`."""
  from types import SimpleNamespace as NS
  r = _pulling(np.array([0.0, 0.0, 1.0]))
  monkeypatch.setattr(rc.friction, "taken", lambda *a: 0.0)
  r._pose, r._trim = (lambda arms=None: None), (lambda dt: None)
  r.self_model.hand_force = lambda side, tau: np.zeros(3)
  r.tau = r.qd = np.zeros(64)
  r.friction = None
  r.arms = {s: NS(command=lambda *a: None) for s in hm.SIDES}
  r.phase["right"], r.cmd["right"] = "blocked", r.cmd["left"]
  ran = []
  r._pulled = lambda side, dt, h, rot: ran.append("pulled") or 1.0
  r._held = lambda side, dt, h, rot: ran.append("held") or 1.0
  r._update(0.01)
  r.pulls["left"]["pull"] = None
  r._update(0.01)
  assert ran == ["pulled", "held"]


def test_a_pull_takes_its_posture_first_and_stands_still_meanwhile():
  """A posture changed partway jerks what the hand holds: the pull takes
  its first, the hand held still -- and not held fast -- while it comes."""
  r = _bare_reach()
  _out_hand(r, H, np.eye(3))
  r.phase["left"] = "hold"
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  r.grip_now = lambda side: (H.copy(), np.eye(3))
  r.path_for = lambda side, poses, rot=None: (0.0, 0.15)
  r.posture_for = lambda side, pos, rot=None, onward=False, past=(): None
  r.self_model.solve = lambda side, pos, rot=None, iters=None: (np.zeros(len(hm.ARM)), 0.0, 0.0)
  r._posture_held = lambda: False
  assert r.pull("left", np.array([0.0, 0.0, 1.0]), 10.0, 0.2)
  assert r.postures == [(0.0, 0.15)]
  r.force["left"] = np.array([0.0, 0.0, 9.0])
  for _ in range(int(rc.FAST_S / DT) + 2):
    r._pulled("left", DT, H, np.eye(3))
  assert np.allclose(r.cmd["left"].pos, H) and r.pulls["left"] is not None


def _reaching_only(hand):
  """A solve that reaches `hand` (a list holding where the hand is) and
  misses every command a millimetre or more off it by 2 cm."""
  def solve(side, pos, rot=None, iters=None):
    off = float(np.linalg.norm(np.asarray(pos) - hand[0]))
    return np.zeros(len(hm.ARM)), (0.02 if off >= 1e-3 else 0.0), 0.0
  return solve


def test_a_held_hand_whose_arm_reaches_it_takes_no_posture_and_ends_held_fast():
  """Against a hand held, the command's lead is the solve's miss: postures
  taken for it bowed, crouched and stood the body up with the hand on a
  locked bar, 30-100 N on it for a pull asked 10-20, and 7 ways in 14 never
  ended (#547). Pinned there pushing its force it is held fast; pushing
  nothing -- in free air, past the arm's reach -- it can reach no further
  (it was "held fast" after 1.2 cm, #547)."""
  way = np.array([0.0, 0.0, 1.0])
  for force, why in ((9.0, "held fast"), (0.0, "out of reach")):
    r = _pulling(way)
    hand = [H.copy()]
    r.posture_for = lambda side, pos, rot=None, onward=False, past=(): (0.0, 0.3)
    r._pulled("left", DT, H, np.eye(3))
    reached = r.cmd["left"].pos.copy()
    r.self_model.solve = _reaching_only(hand)
    # pushing a force under its ask that does not move the command (its
    # reading as it stands)
    r.force["left"] = force * way
    r.pulls["left"]["f"] = r.pulls["left"]["fm"] = r.force["left"].copy()
    for _ in range(int((rc.REPLAN_S + rc.FAST_S) / DT) + 2):
      if r.pulls["left"] is None:
        break
      r._pulled("left", DT, H, np.eye(3))
      if r.pulls["left"] is not None:
        assert np.allclose(r.cmd["left"].pos, reached)
    assert r.postures == [] and r.report["left"]["why"] == why, force
    assert r.report["left"]["force"] == force
  # ...but a hand going takes one, as one its arm cannot hold where it is
  for going, reaches in ((True, True), (False, False)):
    r = _pulling(way, distance=1.0)
    r.posture_for = lambda side, pos, rot=None, onward=False, past=(): (0.0, 0.3)
    r._posture_held = lambda: True
    for k in range(1, int(rc.REPLAN_S / DT) + 2):
      at = H + (1.1 * rc.GO_ON_M * k * way if going else 0.0)
      hand[0] = at if reaches else at + 1.0
      r.self_model.solve = _reaching_only(hand)
      r._pulled("left", DT, at, np.eye(3))
      if r.postures:
        break
    assert r.postures == [(0.0, 0.3)], (going, reaches)


def test_a_held_hand_backs_off_past_its_force_toward_where_it_is():
  """Held where the arm last reached, a pull on a locked bar stood at 21-24
  N for 20 while its body leaned on the hand (#547): backing off, its
  command goes back a share of the way to the hand, and its arm's joints as
  far toward those that reach it."""
  way = np.array([0.0, 0.0, 1.0])
  # (the solve that reaches the hand answers zeros: its arm a joint gap off)
  for gap in (0.01, 1.0):
    r = _pulling(way)
    r._pulled("left", DT, H, np.eye(3))
    lead = _along(r, way)
    reached = r.cmd["left"].pos.copy()
    r.self_model.solve = _reaching_only([H.copy()])
    r.q_cmd["left"] = np.full(len(hm.ARM), gap)
    # 11 N on an ask of 10, its readings as they stand: back off at
    # `BACK_SPEED` times its speed, a fifth of the way past its ask -- no
    # joint faster than `JOINT_RATE` (a setpoint is never written across a
    # gap)
    r.force["left"] = 11.0 * way
    r.pulls["left"]["f"] = r.pulls["left"]["fm"] = r.force["left"].copy()
    r._pulled("left", DT, H, np.eye(3))
    share = min(rc.BACK_SPEED * 0.2 * rc.HOLD_SPEED * DT / lead, rc.JOINT_RATE * DT / gap)
    assert 0.0 < share < 1.0
    assert np.allclose(r.cmd["left"].pos, reached + share * (H - reached)), gap
    assert np.allclose(r.q_cmd["left"], gap * (1.0 - share)), gap
  assert share == pytest.approx(rc.JOINT_RATE * DT)
  # ...and under its ask it stays where the arm last reached
  r.force["left"] = 5.0 * way
  r.pulls["left"]["f"] = r.pulls["left"]["fm"] = r.force["left"].copy()
  r.pulls["left"]["rise"] = 0.0
  at, q = r.cmd["left"].pos.copy(), r.q_cmd["left"].copy()
  r._pulled("left", DT, H, np.eye(3))
  assert np.allclose(r.cmd["left"].pos, at) and np.allclose(r.q_cmd["left"], q)


def test_a_pull_backs_off_past_its_force_while_its_body_takes_a_posture():
  """Stood still, it held on reading 15 times its force as a posture stood
  the body up with the hand on a locked bar (#547)."""
  way = np.array([0.0, 0.0, 1.0])
  for force, pace in ((20.0, -rc.BACK_SPEED), (5.0, 0.0)):
    r = _pulling(way)
    r.pulls["left"]["postured"] = 0.0
    r._posture_held = lambda: False
    r.force["left"] = force * way
    r._pulled("left", DT, H, np.eye(3))
    assert _along(r, way) == pytest.approx(pace * rc.HOLD_SPEED * DT), force


def test_a_pinned_pull_is_held_fast_by_how_far_it_goes_along_its_way():
  """Pinned on a locked bar, the hand's sensed place moved 4 mm in a tenth
  of a second and its grip slid 3-7 mm a second: held fast by its place
  alone, such a pull never ended (#547)."""
  way = np.array([0.0, 0.0, 1.0])
  r = _pulling(way)
  for k in range(int(rc.LEAD_MAX_M / (rc.HOLD_SPEED * DT)) + int(rc.FAST_S / DT) + 3):
    if r.pulls["left"] is None:
      break
    # sliding across its way, and along it slower than a hand goes
    r._pulled("left", DT, H + 0.006 * k * np.array([0.0, 1.0, 0.0])
              + 0.3 * rc.GO_ON_M * DT * k * way, np.eye(3))
  assert r.report["left"]["why"] == "held fast"


def test_a_pull_that_loses_its_grip_ends_where_the_hand_is():
  """Ended at its distance alone, a knob slipped from the jaws leaves the
  hand pulling on through the air, its report that free motion's (#525)."""
  way = np.array([0.0, 0.0, 1.0])
  r = _pulling(way, held=True)
  for k in range(1, 6):
    r._pulled("left", DT, H + 0.01 * k * way, np.eye(3))
  assert r.pulls["left"] is not None
  r.grippers["left"].holding = False
  r._pulled("left", DT, H + 0.06 * way, np.eye(3))
  assert r.pulls["left"] is None and r.report["left"]["why"] == "let go"
  assert r.cmd["left"].pos == pytest.approx(H + 0.06 * way)
  # ...one that began holding nothing is not ended for it
  r = _pulling(way)
  r._pulled("left", DT, H, np.eye(3))
  assert r.pulls["left"] is not None
  # ...and a pull pushes with some force
  with pytest.raises(ValueError):
    _pulling(way, force=0.0)


def test_a_held_hand_going_on_takes_another_posture_from_its_arm_as_it_is(monkeypatch):
  """Up a lid's arc an arm stood at its elbow's stop 2 cm short of its way
  for 9 s, while a fresh seed "reached" in the posture it held (#525)."""
  r = _bare_reach()
  posed, starts = [], []
  monkeypatch.setattr(rc, "_posed", lambda sm, h, p, q0, arms=None: posed.append((h, p)))

  class Planner:
    def _set_arm(self, side, q):
      starts.append(np.asarray(q, dtype=float).copy())

    def solve(self, side, pos, rot, iters=80):
      return None, (0.0 if posed[-1] == (0.0, 0.15) else 1.0), 0.0
  r._planner = Planner()
  r.q_cmd["left"] = np.full(len(hm.ARM), 0.3)
  assert r.posture_for("left", H, onward=True) == (0.0, 0.15)
  assert (0.0, 0.0) not in posed
  posed.clear()
  starts.clear()
  assert r.posture_for("left", H) == (0.0, 0.15)
  assert posed[0] == (0.0, 0.0) and len(starts) > len(posed)
  # ...from the arm as it is alone, never a seed: where a seed alone
  # reaches, going on finds nothing
  Planner.solve = lambda self, side, pos, rot, iters=80: (
    None, 1.0 if np.allclose(starts[-1], 0.3) else 0.0, 0.0)
  assert r.posture_for("left", H, onward=True) is None
  assert r.posture_for("left", H) is not None


def test_a_pull_whose_arm_misses_goes_on_and_never_back_to_a_posture_it_left():
  """Free to go back, two postures could take turns, each switch a fresh
  wait, and the pull never end out of reach (#525)."""
  r = _pulling(np.array([0.0, 0.0, 1.0]))
  held, asked = [tuple(float(v) for v in r.posture)], []

  def hold(height, pitch):
    r.posture = np.array([height, pitch])
    held.append((height, pitch))

  def posture_for(side, pos, rot=None, onward=False, past=()):
    asked.append(onward)
    here = tuple(float(v) for v in r.posture)
    return next((p for p in ((0.0, 0.15), (0.0, 0.0)) if p != here and p not in past), None)
  r.hold_posture, r.posture_for, r._posture_held = hold, posture_for, lambda: True
  r.self_model.solve = lambda side, pos, rot=None, iters=None: (np.zeros(len(hm.ARM)), 0.02, 0.0)
  for _ in range(int(3 * rc.REPLAN_S / DT) + 2):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H, np.eye(3))
  assert asked and all(asked) and held == [(0.0, 0.0), (0.0, 0.15)]


def test_a_paths_point_with_no_turn_keeps_the_one_before():
  """`via`'s point with no turn keeps the one before -- the hand's own,
  first -- never the next one given (#525)."""
  from pluggybot.legs.probe import rot_about
  r = _bare_reach()
  own = rot_about(np.array([0.0, 1.0, 0.0]), math.radians(20.0))
  _out_hand(r, H, own)
  r.phase["left"] = "hold"
  r.grip_now = lambda side: (H.copy(), own.copy())
  r.path_for = lambda side, poses, rot=None: tuple(float(v) for v in r.posture)
  turn = rot_about(np.array([0.0, 0.0, 1.0]), 0.3)
  up = np.array([0.0, 0.0, 0.02])
  assert r.carry("left", H + 3 * up, path=[H + up, H + 2 * up], turns=[None, turn, None])
  assert np.allclose(r.goal["left"].rot, own)
  assert [np.allclose(t, turn) for _, t in r.pulls["left"]["ways"]] == [True, True]


def test_a_grip_is_shut_with_the_force_asked_inside_its_sheet_and_judged_against_it():
  from types import SimpleNamespace as NS

  from pluggybot.humanoid import arms as ha
  from pluggybot.humanoid import parts as hp
  g = object.__new__(ha.Gripper)
  g.m, g.d, g.act = NS(actuator_forcerange=np.zeros((1, 2))), NS(ctrl=np.zeros(1)), 0
  squeeze = [0.0]
  g.read = lambda: (0.02, squeeze[0])
  sheet = hp.GRIPPER.grip_force_n
  for asked, gets in ((15.0, 15.0), (1.0, ha.GRIP_MIN_N), (500.0, sheet), (None, sheet)):
    g.shut(asked)
    assert g.force == gets and tuple(g.m.actuator_forcerange[0]) == (-2 * gets, 2 * gets)
  g.shut(15.0)
  squeeze[0] = 0.5 * 15.0 + 0.1
  assert g.held()
  squeeze[0] = 0.5 * 15.0 - 0.1
  assert not g.held()
  g.open()
  assert g.force == sheet
  # ...on its squeeze's running mean, sampled every physics step: one
  # reading's noise, the current sense's 5.1 N, read a 15 N grip as nothing
  # held 7 % of the time (#525)
  g = object.__new__(ha.Gripper)
  g.m, g.d = NS(opt=NS(timestep=0.002)), None
  readings = iter([(0.02, 0.0), (0.02, 30.0)] * 200)
  g.sense = NS(read=lambda d: next(readings))
  g.shut_asked, g.force, g.gap, g.squeeze = True, 15.0, 0.02, 15.0
  for _ in range(300):
    g.sample()
    assert g.held()
  # ...every physics step, the reach's own
  r = _bare_reach()
  sampled = []
  for side, grip in r.grippers.items():
    grip.sample = lambda side=side: sampled.append(side)
  r.steps, r.every = 1, 10
  r.step()
  assert sorted(sampled) == sorted(hm.SIDES)
  # ...and the verb hands the body the force it was asked, inside the sheet's
  body = StubBody(*_bare())
  _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    close_hand("left", 15)\n', body)
  assert body.squeezes["left"] == 15.0
  assert st.HANDS_VERBS["close_hand"].args["force"].lo == ha.GRIP_MIN_N
  assert st.HANDS_VERBS["close_hand"].args["force"].hi == sheet


def test_spot_bows_until_it_sees_stands_up_again_and_says_which_way_the_edges_run():
  class Bowing(StubBody):
    def __init__(self, *a):
      super().__init__(*a)
      self.gazes, self.down = [], 0.0

    def gaze_routine(self, down):
      self.gazes.append(down)
      self.down = down
      return
      yield

    def spot(self, tag, at_height=None):
      if self.down < st.SPOT_GAZES[2] - 1e-9:
        return None
      return {"x": 0.35, "y": 0.2, "z": 0.85, "normal": (-1.0, 0.0, 0.0),
              "edge": (0.0, 1.0, 0.0), "range": 0.6}
  body = Bowing(*_bare())
  ran = _fly('def a():\n    spot(55)\n    ex = read("spot.ex")\n    ey = read("spot.ey")\n',
             body)
  assert ran["ok"] and (ran["locals"]["ex"], ran["locals"]["ey"]) == pytest.approx((0.0, 1.0))
  # ...in the frame it stands in: turned a quarter left, the world's +y is
  # its ahead
  body.gazes, body.down = [], 0.0
  body.start_at(0.0, 0.0, math.pi / 2)
  ran = _fly('def a():\n    spot(55)\n    ex = read("spot.ex")\n    ey = read("spot.ey")\n',
             body)
  assert (ran["locals"]["ex"], ran["locals"]["ey"]) == pytest.approx((1.0, 0.0), abs=1e-9)
  body.start_at(0.0, 0.0, 0.0)
  assert body.gazes == [st.SPOT_GAZES[1], st.SPOT_GAZES[2], 0.0]
  # ...and one it never sees is no reading, the body stood up again
  body.spot = lambda tag, at_height=None: None
  body.gazes = []
  ran = _fly('def a():\n    spot(55)\n', body)
  assert not ran["ok"] and body.gazes[-1] == 0.0


def test_the_humanoids_grip_shuts_with_the_force_asked():
  from types import SimpleNamespace as NS

  from pluggybot.humanoid import body as hb

  class Grip:
    def shut(self, force=None):
      self.force = force

    def held(self):
      return True

    def read(self):
      return 0.026, 15.0
  body = object.__new__(hb.HumanoidBody)
  grip = Grip()
  body.mission = NS(to_stand_routine=lambda: iter(()), reach=NS(grippers={"left": grip}))
  body.hold_routine = lambda seconds: iter(())
  for _ in body.grip_routine("left", True, 15.0):
    pass
  assert grip.force == 15.0


def test_the_humanoid_keeps_its_posture_to_look_with_a_hand_out():
  """A bow carries a hand that is out with it: a look from a low reach's
  bow lifted the hand 43 cm (#525)."""
  from types import SimpleNamespace as NS

  from pluggybot.humanoid import body as hb
  body = object.__new__(hb.HumanoidBody)
  r = _bare_reach()
  body.mission = NS(to_stand_routine=lambda: iter(()), reach=r)
  r._posture_held = lambda: True
  r.phase["left"] = "hold"
  for _ in body.gaze_routine(st.SPOT_GAZES[2]):
    pass
  assert r.postures == []
  r.phase["left"] = "rest"
  for _ in body.gaze_routine(st.SPOT_GAZES[2]):
    pass
  assert r.postures == [(0.0, st.SPOT_GAZES[2])]
