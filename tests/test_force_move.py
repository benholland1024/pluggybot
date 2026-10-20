"""The hands' vocabulary (issue #525; docs/Hands.md): the force move and the
verbs around it, each rule pinned by a direct call on a `Reach` with no
world, or on the stub body (docs/Testing.md) -- the flights that measured
them are `scripts/hands_check.py`'s.

  1. along its way a pull's command walks at its speed while the hand
     pushes under half its force, stands at its force, and backs off past
     it: the force is CLOSED on what the hand feels (admittance);
  2. across its way, giving, the command follows the hand; holding, it
     keeps its line (hybrid force and position);
  3. its turn gives to what it holds, but never about the jaws' own line;
  4. it ends at its distance, or held fast where the hand stays put pushing
     its whole force;
  5. a held hand going on takes another posture from its arm as it is;
  6. a grip is shut with the force asked, inside the gripper's sheet, and
     judged against it;
  7. `spot` bows until it sees and stands up again, and says which way the
     tag's edges run.
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
DT = rc.ADMIT_SMOOTH_S


def _pulling(way, force=10.0, distance=0.2, give=True, rot=None):
  """A bare reach's left hand out at `H` and pulling along `way`: its senses
  answer `H` and `rot` unless a test moves them."""
  r = _bare_reach()
  rot = np.eye(3) if rot is None else rot
  _out_hand(r, H, rot)
  r.phase["left"] = "hold"
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  r.grip_now = lambda side: (H.copy(), rot.copy())
  r.path_for = lambda side, poses, rot=None: tuple(float(v) for v in r.posture)
  r.posture_for = lambda side, pos, rot=None, onward=False: None
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
  for _ in range(40):
    r._pulled("left", DT, H, np.eye(3))
  assert _along(r, way) == pytest.approx(rc.LEAD_MAX_M)
  r = _pulling(way)
  r.force["left"] = 10.0 * way
  r._pulled("left", DT, H, np.eye(3))
  assert _along(r, way) == pytest.approx(0.0)
  r = _pulling(way)
  r.force["left"] = 20.0 * way
  r._pulled("left", DT, H, np.eye(3))
  assert _along(r, way) == pytest.approx(-rc.HOLD_SPEED * DT)


def test_a_giving_pull_follows_the_hand_across_its_way_and_a_holding_one_keeps_its_line():
  way, moved = np.array([0.0, 0.0, 1.0]), H + [0.0, 0.01, 0.0]
  r = _pulling(way)
  r._pulled("left", DT, moved, np.eye(3))
  assert r.cmd["left"].pos[1] == pytest.approx(moved[1])
  r = _pulling(way, give=False)
  r._pulled("left", DT, moved, np.eye(3))
  assert r.cmd["left"].pos[1] == pytest.approx(H[1])


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
  assert (0.0, 0.0) not in posed and all(np.allclose(s, 0.3) for s in starts)
  posed.clear()
  starts.clear()
  assert r.posture_for("left", H) == (0.0, 0.15)
  assert posed[0] == (0.0, 0.0) and len(starts) > len(posed)


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
  assert body.gazes == [st.SPOT_GAZES[1], st.SPOT_GAZES[2], 0.0]
  # ...and one it never sees is no reading, the body stood up again
  body.spot = lambda tag, at_height=None: None
  body.gazes = []
  ran = _fly('def a():\n    spot(55)\n', body)
  assert not ran["ok"] and body.gazes[-1] == 0.0
