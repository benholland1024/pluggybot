"""A hand's way keeps off another robot's body (issue #678; docs/Hands.md,
SimNotes "The arms keep off the other robot"), each rule pinned without a
flight -- the robot's own drawing, a fake reach on the bare humanoid, the
stub (docs/Testing.md). The flights that measured it are the SimNotes
section's.

  1. a reach's way out, and its way home from there, are judged against a
     cylinder round the other robot's middle: where none keeps clear the
     reach is refused and nothing moves; an arm out's way home likewise,
     from where it truly begins (its line out first);
  2. another way is taken where one keeps clear and dips no deeper over the
     work (#539's and #560's rule kept): at a counter, none;
  3. the robot kept off is one the LIDAR saw in ONE look as the move began,
     placed where `others` places one seen (#675), within `ARMS_NEAR_M` --
     never where it says it is; alone, no look at all;
  4. a refusal reads false in words saying where that robot stands, and a
     reach's what the other hand's way would do;
  5. a withdraw refused keeps its gripper as it was: judged before #669's
     let-go;
  6. the walk verbs refuse to walk with a hand out that could not come home;
     the loop's own walks wait for it, `ARM_WAIT_S` at most, looking every
     `ARM_LOOK_S`, then bring it home today's way -- the wait counted against
     neither a walk's patience nor its stagnation clock, nor a step's -- and
     the loop's own hand moves (a charge's) wait so too, never refused;
  7. nothing of it reads the simulator's truth.
"""

import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot.body import KeepClear, StubBody, body_for
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import mission as hmi
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal
from pluggybot.perception.lidar import Scan
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st

from test_hands import _Life, _bare, _fly, _standing_reach
from test_humanoid_body import _floor

#: 11:48's lever grip on 11-08 (#670), in the stance frame Rowan reached it
#: from: 0.43 m ahead at a counter's height.
LEVER = (0.434, 0.089, 0.866)


def _at(m, bearing_deg):
  return (m * math.cos(math.radians(bearing_deg)), m * math.sin(math.radians(bearing_deg)))


# ---- 1, 2: the ways judged, on the robot's own drawing ---------------------------------


def test_a_reach_whose_every_way_meets_another_robots_body_is_refused_and_moves_nothing():
  """11:48 on 11-08 (#670, #678): Rowan's reach for the lever swung out by its
  side, its jaw met Luca's thigh, and both fell. Its way out, and its way
  home from there, are judged against a cylinder round the other robot's
  middle: a robot 0.6 m to the right (where Luca stood), or as far as 1.0 m,
  refuses the right hand -- nothing moves, no posture is taken -- and its
  record says where that robot stood. The same reach with nobody by, or the
  robot 1.3 m off, or to the left, goes."""
  goal = HandGoal(np.array(LEVER), None, 1.0)
  for where in (_at(0.6, -90), _at(1.0, -90)):
    r = _standing_reach()
    r.keep_out = [where]
    assert not r.reach("right", goal), where
    assert r.phase["right"] == "rest" and r.postures == [] and r.outs["right"] == []
    kept = r.kept["right"]
    assert kept["home"] is False and kept["m"] == pytest.approx(math.hypot(*where), abs=0.01)
    assert (kept["x"], kept["y"]) == pytest.approx(where, abs=0.01)
  for keep in ([], [_at(1.3, -90)], [_at(0.6, 90)]):
    r = _standing_reach()
    r.keep_out = keep
    assert r.reach("right", goal), keep
    assert r.phase["right"] == "move" and r.kept["right"] is None


#: An arm raised high and out to its right (drawn, `probe/other_way.py`):
#: straight home its forearm sweeps down through a robot 0.7 m ahead and to
#: the right; the side way keeps 16 cm clear of it.
RAISED_OUT = np.array([-1.91, -2.07, 1.24, 1.07, 0.02, -1.09])


def test_an_arm_takes_another_way_home_where_one_keeps_clear():
  """Today's way home is straight where nothing it works over is ahead of
  the robot; with another robot where straight would meet its body, the
  first of straight, the side way and the way drawn in that keeps clear is
  taken -- here the side way -- and with nobody by, today's."""
  r = _standing_reach()
  r.q_cmd["right"] = RAISED_OUT.copy()
  today = r.way_home("right")
  assert len(today) == 1 and np.allclose(today[0], r.rest["right"]), "the premise: straight home"
  r.keep_out = [_at(0.7, -30)]
  sm = r._planner_model()
  r._pose_planner(sm, None)
  assert rc.room(sm, "right", RAISED_OUT, today, r.keep_out) < 0.0, "the premise: straight meets it"
  aside = rc._side_way("right", RAISED_OUT, r.rest["right"])
  way = r.way_home("right", refuse=True)
  assert way is not None and rc._same_way(way, aside), "it did not take the side way"
  assert rc.room(sm, "right", RAISED_OUT, way, r.keep_out) >= rc.OTHER_CLEAR_M
  assert r.home_clear("right") and r.kept["right"] is None
  # ...and the arm sent home walks it (`_go_home`, by `retreat`)
  r.phase["right"], r.cmd["right"] = "joints", None
  r.retreat("right")
  got = [r.q_to["right"]] + r.homes["right"]
  assert rc._same_way(got, aside)


def test_the_way_home_taken_is_the_way_judged_clear():
  """A way home judged clear (`home_clear`) is the way the arm takes
  (`_go_home`), begun where it was judged from: judged again from a start a
  few mm off -- the line out's end beside the solved pre-grasp -- a way clear
  by a hair could cross the other robot, taken unrefused as the one that
  comes least near."""
  r = _standing_reach()
  r.q_cmd["right"], r.phase["right"] = RAISED_OUT.copy(), "joints"
  r.keep_out = [_at(0.7, -30)]
  assert r.home_clear("right")
  judged = r.judged_home["right"][1]
  asked = []
  real = rc.Reach.way_home
  r.way_home = lambda side, refuse=False: (asked.append(side), real(r, side, refuse))[1]
  r.q_cmd["right"] = RAISED_OUT + 1e-3
  r.retreat("right")
  got = [r.q_to["right"]] + r.homes["right"]
  assert asked == [] and rc._same_way(got, judged), "judged afresh where it went"
  assert r.judged_home["right"] is None
  # ...and never from a start it was not judged from
  assert r.home_clear("right")
  r.q_cmd["right"], r.phase["right"] = RAISED_OUT + 0.2, "joints"
  r.retreat("right")
  assert asked == ["right"]


def test_at_a_counter_no_way_that_dips_deeper_is_taken():
  """#539, #560: straight out to a counter swings the arm low in front, into
  its face; today's way there is the side way. With another robot 0.75 m to
  the right the straight way would keep 15 cm clear of it -- and is not
  taken, being deeper over the work: the reach is refused (#678's stage 1:
  at a counter there is no other way)."""
  r = _standing_reach()
  goal = HandGoal(np.array(LEVER), None, 1.0)
  posture, q_pre = r._planned("right", goal, rc.APPROACH_M, rc.REACH_POSTURES, True)[0][:2]
  r.keep_out = [_at(0.75, -90)]
  sm = r._planner_model()
  r._pose_planner(sm, None)
  straight = [np.asarray(q_pre, dtype=float)]
  assert rc.room(sm, "right", r.q_cmd["right"], straight, r.keep_out) >= rc.OTHER_CLEAR_M, \
    "the premise: straight keeps clear of it"
  assert r.way_out("right", q_pre, posture) is None
  assert not r.reach("right", goal) and r.kept["right"]["home"] is False


#: A hand already out over the work going on to another place (drawn,
#: `probe/out_hand_search.py`): straight there its arm meets a robot 0.65 m
#: ahead and to its right; the side way keeps clear of it, 3 cm deeper.
OUT_FROM = np.array([-0.08, -0.21, 0.77, 0.73, 0.2, -1.51])
OUT_TO = np.array([-0.95, -1.19, 0.37, 0.0, 0.48, -0.99])


def test_a_hand_already_out_takes_no_way_on_that_dips_deeper_than_straight():
  """A hand already out goes on straight today, its way's dip unjudged (it
  sweeps nothing aside); kept off another robot, another way it takes dips
  no deeper over the work than straight would have -- judged as the arm
  stands at its end -- or none is taken."""
  r = _standing_reach()
  r.q_cmd["right"] = OUT_FROM.copy()
  today = r.way_out("right", OUT_TO)
  assert len(today) == 1 and np.allclose(today[0], OUT_TO), "the premise: straight on"
  r.keep_out = [_at(0.65, -30)]
  sm = r._planner_model()
  r._pose_planner(sm, None)
  aside = rc._side_way("right", OUT_TO, OUT_FROM)[::-1][1:] + [OUT_TO]
  start = rc.room(sm, "right", OUT_FROM, [], r.keep_out)
  assert rc.room(sm, "right", OUT_FROM, aside, r.keep_out) - min(rc.OTHER_CLEAR_M, start) > 0.0, \
    "the premise: the side way keeps clear of it"
  assert r.way_out("right", OUT_TO) is None, "it took a way deeper over the work"


def test_a_hand_holding_is_judged_home_from_where_its_line_out_ends():
  """A hand at its goal comes home along its line first, to its pre-grasp
  `APPROACH_M` (8 cm) back along its fingers (`retreat`), and its way home
  begins there (`_go_home`): judged from the grip, a way clear from there
  could meet the other robot from where it truly begins, and was taken
  unrefused. `home_clear` judges from the pre-grasp, solved on the robot's
  model of itself."""
  from pluggybot.humanoid.arms import HandGoal
  r = _standing_reach()
  sm = r.self_model
  # ...a hand out in front of the robot, holding at its goal
  q_at = np.array([-1.2, -0.2, 0.0, 1.2, 0.0, 0.0])
  sm._set_arm("right", q_at)
  at, rot = sm.grip_pose("right")
  sm._set_arm("right", r.rest["right"])
  pre_at = rc.pregrasp(HandGoal(at, rot, 1.0)).pos
  assert np.linalg.norm(pre_at - at) == pytest.approx(rc.APPROACH_M)
  r.q_cmd["right"], r.phase["right"] = q_at.copy(), "hold"
  r.goal["right"] = r.cmd["right"] = HandGoal(at, rot, 1.0)
  judged = []
  real = rc.Reach._judged

  def spied(self, side, q0, *a, **kw):
    judged.append(np.asarray(q0, dtype=float).copy())
    return real(self, side, q0, *a, **kw)
  r._judged = spied.__get__(r)
  r.keep_out = [_at(1.4, -90)]
  assert r.home_clear("right")
  m = r._planner_model()
  m._set_arm("right", judged[-1])
  assert np.linalg.norm(m.grip_pose("right")[0] - pre_at) < rc.REPLAN_M, \
    "judged home from the grip, not from where its line out ends"


# ---- 3: whom it keeps off, and where -------------------------------------------------------


def _scanning(m, roots):
  """`m`'s LIDAR returning a scan that saw the robots `roots` (one return
  each, 0.6 m to its right), its own cadence off -- standing, it takes none
  -- every scan counted in `m.scans` and every look in `m.looks`."""
  m.scans, m.looks, m._next_scan, m._looked = 0, 0, math.inf, None
  look = type(m).look_round

  def scan():
    m.scans += 1
    n = len(roots)
    return Scan(np.zeros(1), np.full(1, 8.0), np.full(n, -math.pi / 2), np.full(n, 0.6),
                peer_roots=tuple(roots))

  def looked():
    m.looks += 1
    return look(m)
  m._scan, m.look_round = scan, looked


def test_a_look_scans_once_and_records_whom_it_saw_and_alone_takes_none():
  """ONE scan as a hand's move begins (Ben, 2026-11-12): standing still the
  LIDAR takes none of its own (#670), so the move looks, its sighting
  recorded as any scan's is (`seen_t`, `peer_returns`) and never into the
  map. With no other robot about it scans nothing -- drawing nothing on the
  LIDAR's noise -- so a robot alone flies as it did, bit for bit."""
  body = body_for(*_floor())
  m = body.mission
  _scanning(m, ("r2_",))
  assert m.look_round() == frozenset() and m.scans == 0, "it looked with nobody about"
  body.others = [lambda: KeepClear(0.0, -0.6, root="r2_")]
  t = float(body.data.time)
  grid = m.grid.grid.copy()
  assert m.look_round() == {"r2_"} and m.scans == 1
  assert m.seen_t["r2_"] == t and m.scanned_t == t and m.peer_returns[0] == t
  assert np.array_equal(m.grid.grid, grid), "the look went into the map"
  # ...one scan an instant, however many moves look
  assert m.look_round() == {"r2_"} and m.scans == 1
  _scanning(m, ())
  body.data.time += m.model.opt.timestep
  assert m.look_round() == frozenset() and m.scans == 1


def test_it_keeps_off_only_a_robot_its_look_saw_where_it_was_seen_and_near():
  """Only a robot the LIDAR sees (Ben, 2026-11-12): placed where `others`
  places one seen (#675, its body through this robot's own pose error) --
  never where it says it is, which at the counter lay 0.49 m from its body
  at the median -- within `ARMS_NEAR_M`, past which no part of an arm comes
  near it in any posture; and neither one lying down nor one dead."""
  body = body_for(*_floor())
  m = body.mission
  sx, sy, syaw = m._stance_frame()

  def world(x, y):
    c, s = math.cos(syaw), math.sin(syaw)
    return sx + c * x - s * y, sy + s * x + c * y
  near, far = world(0.1, -0.6), world(0.0, -(hmi.ARMS_NEAR_M + 0.05))
  body.others = [lambda: KeepClear(*near, root="r2_", seen=True)]
  assert m.keep_out({"r2_"}) == [pytest.approx((0.1, -0.6), abs=1e-9)]
  assert m.keep_out(set()) == [], "kept off a robot its look did not see"
  for other in (KeepClear(*far, root="r2_", seen=True), KeepClear(*near, root="r2_", down=True),
                KeepClear(*near, root="r2_", dead=True), (near[0], near[1])):
    body.others = [lambda other=other: other]
    assert m.keep_out({"r2_"}) == [], other


def test_a_reach_keeps_off_the_robot_its_own_look_sees_as_it_begins(monkeypatch):
  """Through `Body`: a verb's reach (`refuse`) looks as it begins, and its
  plan keeps off the robot that look saw -- its reach handed it where that
  robot stands -- and off none it did not, the robot's word
  notwithstanding; and once judged, none is kept for a way judged after."""
  body = body_for(*_floor())
  m = body.mission
  sx, sy, syaw = m._stance_frame()
  where = (sx + 0.6 * math.sin(syaw), sy - 0.6 * math.cos(syaw))   # 0.6 m to its right
  body.others = [lambda: KeepClear(*where, root="r2_", seen=True)]
  handed = []
  real = rc.Reach.reach

  def reach(self, side, goal, *a, **kw):
    handed.append(list(self.keep_out))
    return real(self, side, goal, *a, **kw)
  monkeypatch.setattr(rc.Reach, "reach", reach)
  for roots, kept in ((("r2_",), True), ((), False)):
    _scanning(m, roots)
    handed.clear()
    world = body.to_world(np.array(LEVER))
    got = body.run(body.reach_routine("right", tuple(world), patience=0.01, refuse=True))
    assert m.scans == 1
    assert handed and all(bool(h) is kept for h in handed), (roots, handed)
    if kept:
      assert got is False and body.hand_why("right") == "kept off"
      assert body.hand_kept("right")["m"] == pytest.approx(0.6, abs=0.03)
      assert m.reach.phase["right"] == "rest"
    else:
      assert body.hand_why("right") != "kept off"
    assert m.reach.keep_out == [], "a judgment was kept past its move"


# ---- 4: the words -------------------------------------------------------------------------


class _Kept(StubBody):
  """A stub whose reach and withdraw are refused for another robot's body,
  and whose hand stays out (`arm_kept`), as a test says."""

  reach_kept = withdraw_kept = arm_out = None

  def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                    turns=None, refuse=False):
    if self.reach_kept is None:
      return (yield from super().reach_routine(side, pos, rot, stiffness, patience, path, turns))
    self._why[side] = "kept off"
    return False

  def release_routine(self, side, refuse=False):
    if self.withdraw_kept is None:
      return (yield from super().release_routine(side))
    return "kept off"

  def hand_kept(self, side):
    return self.reach_kept or self.withdraw_kept

  def arm_kept(self):
    return self.arm_out


def test_a_refusal_says_where_the_other_robot_stands_and_what_the_other_hand_can_do():
  """The minds coordinate off these words (Ben, 2026-11-12): what did not
  move, why, where that robot stands from the robot's own middle, and for a
  reach what the other hand's way would do."""
  body = _Kept(*_bare())
  for kept, said in (
      ({"home": False, "m": 0.54, "x": 0.02, "y": -0.54, "other": True},
       "your right arm's way there would meet another robot, its middle 0.54 m from yours on "
       "your right: no way your arm takes there keeps clear of its body, so your arm did not "
       "move; your left hand's way there keeps clear of it"),
      ({"home": True, "back": True, "m": 0.73, "x": -0.6, "y": -0.42, "other": False},
       "your right arm could get there, but its way home from there would meet another robot, "
       "its middle 0.73 m from yours behind you: no way home keeps clear of its body, so your "
       "arm did not move; nor does your left hand's"),
      ({"home": False, "m": 0.9, "x": 0.85, "y": -0.3, "other": None},
       "your right arm's way there would meet another robot, its middle 0.90 m from yours ahead "
       "of you: no way your arm takes there keeps clear of its body, so your arm did not move; "
       "your left hand does not reach there")):
    body.reach_kept = kept
    ran = _fly('def a():\n    if not reach("right", 0.4, -0.1, 0.9):\n        wait(1)\n', body)
    step = ran["steps"][0]
    assert not step["ok"] and step["why"] == "kept off" and step["reason"] == said
  body.reach_kept = None
  body.withdraw_kept = {"home": True, "m": 0.62, "x": 0.1, "y": 0.61}
  ran = _fly('def a():\n    reach("left", 0.4, 0.1, 0.9)\n    withdraw("left")\n', body)
  step = ran["steps"][1]
  assert not step["ok"] and step["reason"] == (
    "your left arm's way home would meet another robot, its middle 0.62 m from yours on your "
    "left: no way home keeps clear of its body, so your hand stays where it is, its gripper as "
    "it was")


def test_the_walk_verbs_refuse_to_walk_with_a_hand_out_that_could_not_come_home():
  """Ben, 2026-11-12: a walk brings the arms home first, so with a hand out
  whose every way home would meet another robot's body, `step`, `drive_to`,
  `face` and `find` neither turn nor walk, and read false saying so;
  `step.*` reads where nothing went."""
  out = {"hand": "right", "home": True, "m": 0.7, "x": 0.0, "y": -0.7}
  facts = st.served_hand_facts(places=(60,))
  for source, verb in (('    if not step(0.3):\n        d = read("step.dx")\n', "step"),
                       ("    drive_to(3.0, 0.0)\n", "walk"),
                       ("    face(1.0)\n", "walk"),
                       ("    find(60, 5.0, 0.0)\n", "walk")):
    body = _Kept(*_bare())
    body.arm_out = out
    proc = lang.compile_procedure("def a():\n" + source, facts)
    ran = body.run(lang.run_procedure_routine(_Life(body), proc, facts))
    step = ran["steps"][0]
    assert not step["ok"] and step["reason"] == st.ARM_KEPT_OFF[verb].format(
      where="on your right", **out), source
    assert step["reason"].endswith(f"so you did not {verb}")
    assert body.pose == (0.0, 0.0, 0.0) and body.stepped == [] and body.went == []
    if verb == "step":
      assert ran["locals"]["d"] == 0.0
  # ...and with none out it walks as before
  body = _Kept(*_bare())
  proc = lang.compile_procedure("def a():\n    drive_to(3.0, 0.0)\n", facts)
  assert body.run(lang.run_procedure_routine(_Life(body), proc, facts))["steps"][0]["ok"]


def test_the_robot_is_told_its_arms_keep_off_another_robot_and_what_a_walk_does():
  """The language says it where the verbs are: a reach's ways, a withdraw's,
  and every walk with a hand out that could not come home."""
  from pluggybot.mind import practice as mp
  reach, withdraw = st.HANDS_VERBS["reach"].doc, st.HANDS_VERBS["withdraw"].doc
  assert ("Its way out, and its way home from there, keep clear of the body of another robot "
          "standing near you: where no way does, your arm does not move") in reach
  assert ("Its way home keeps clear of the body of another robot standing near you: where no "
          "way does, your gripper stays as it is and your hand where it is") in withdraw
  assert withdraw.startswith("open a hand's gripper, letting go of whatever it holds, then")
  body = " ".join(dict(mp.system_sections(st.hand_facts()))["your body"].split())
  assert ("Nor does a `step` or any other walk move while a hand is out whose every way home "
          "-- or to your side, carried -- would meet the body of another robot standing near "
          "you") in body


# ---- 5: a withdraw refused keeps its gripper ------------------------------------------------


class _Arms:
  """A reach whose hands `out` are out, their ways home keeping clear of
  another robot's body while `clear()` says so -- judged only against a
  robot kept off (`keep_out`), as `Reach.home_clear` judges -- its grippers
  the body's own, each hand sent home said in `log` with what it kept off."""

  def __init__(self, grippers, out, clear, log, at):
    self.grippers, self.log, self.at, self.clear = grippers, log, at, clear
    self.phase = {s: "hold" if s in out else "rest" for s in hm.SIDES}
    self.keep_out, self.kept, self.judged = [], {s: None for s in hm.SIDES}, []
    #: ...and what a walk carries (#682): none, until `carry_q` says; and
    #: where each judged way home ended (`toward`: a carry pose, or None home)
    self.carry_q = {s: None for s in hm.SIDES}
    self.toward = []
    self.rest = {s: np.array([0.0, 0.1 if s == "left" else -0.1, 0.0, 0.3, 0.0, 0.0])
                 for s in hm.SIDES}

  def out(self, side) -> bool:
    return self.phase[side] == "hold"

  def carried(self, side) -> bool:
    return self.phase[side] == "carried" and self.carry_q[side] is not None

  def drop_carry(self, side) -> None:
    self.carry_q[side] = None

  def carry_pose(self, side):
    return np.array([0.15, 0.25, 0.0, 2.0, -1.0, -1.0])

  def home_clear(self, side, carry=None) -> bool:
    self.judged.append((side, self.at()))
    # ...as `Reach.home_to` ends it: `carry`, else the carry it holds, else home
    end = carry if carry is not None else self.carry_q[side]
    self.toward.append((side, None if end is None else list(end)))
    if not self.keep_out or self.clear():
      self.kept[side] = None
      return True
    self.kept[side] = {"home": True, "m": 0.62, "x": 0.1, "y": -0.61, "short": 0.2}
    return False

  def retreat(self, side, approach=None) -> None:
    self.log.append(("retreat", side, self.at(), list(self.keep_out)))
    self.phase[side] = "rest" if self.carry_q[side] is None else "carried"

  def home(self, side) -> bool:
    return self.phase[side] == "rest"

  def step(self) -> None:
    pass

  def hold_posture(self, height, pitch) -> None:
    pass

  def waist_home(self) -> bool:
    return True

  def hold_waist(self) -> None:
    pass

  def held_pitch(self) -> float:
    return 0.0


def _beside(body, roots=("r2_",)):
  """Another robot 0.62 m to the body's right, its look seeing `roots`."""
  m = body.mission
  sx, sy, syaw = m._stance_frame()
  where = (sx + 0.61 * math.sin(syaw), sy - 0.61 * math.cos(syaw))
  body.others = [lambda: KeepClear(*where, root="r2_", seen=True)]
  _scanning(m, roots)


def _spied(monkeypatch, body, log):
  from pluggybot.humanoid.arms import Gripper
  opened = Gripper.open

  def spied(g, *args, **kwargs):
    log.append(("open", next((s for s, h in body.mission.reach.grippers.items() if h is g), "?"),
                float(body.data.time)))
    opened(g, *args, **kwargs)
  monkeypatch.setattr(Gripper, "open", spied)


def test_a_withdraw_refused_moves_nothing_and_keeps_its_gripper_as_it_was(monkeypatch):
  """#669 opens a withdraw's gripper before its hand moves; one whose every
  way home would meet another robot's body is refused BEFORE that (#678's
  stage 1, `findings.md` §8): its jaws stay shut on what they hold, its arm
  where it is, "kept off" -- and with that robot's way home clear, or
  nobody seen, it lets go and comes home as before."""
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spied(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  _beside(body)
  m.reach = _Arms(grippers, ("right",), lambda: False, log, at)
  grippers["right"].shut()
  assert body.run(body.release_routine("right", refuse=True)) == "kept off"
  assert log == [] and not grippers["right"].opened(), log
  assert m.reach.phase["right"] == "hold" and body.hand_kept("right")["m"] == 0.62
  assert m.reach.keep_out == []
  for clear, roots in ((lambda: True, ("r2_",)), (lambda: False, ())):
    log.clear()
    _beside(body, roots)
    m.reach = _Arms(grippers, ("right",), clear, log, at)
    grippers["right"].shut()
    assert body.run(body.release_routine("right", refuse=True)) == "home"
    assert [e[:2] for e in log] == [("open", "right"), ("retreat", "right")], log
    assert body.hand_kept("right") is None


# ---- 6: the loop's walks wait for it, a bounded while ----------------------------------------


def test_a_walks_hand_over_waits_for_an_arms_way_home_to_clear_then_brings_it_home(
    monkeypatch):
  """The loop's own walks -- a charge's among them -- are never refused
  (Ben, 2026-11-12): an arm out whose every way home would meet another
  robot's body waits, its gripper as it was and its legs the standing
  policy's, looking again every `ARM_LOOK_S`, until a look finds it clear --
  that robot gone -- and then lets go and comes home, judged against where
  that look saw it."""
  monkeypatch.setattr(hmi, "ARM_LOOK_S", 0.1)
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spied(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  _beside(body)
  t0 = at()
  m.reach = _Arms(m.reach.grippers, ("right",), lambda: at() - t0 >= 0.45, log, at)
  m.reach.grippers["right"].shut()
  body.run(m.to_walk_routine())
  assert m.legs == "walk"
  assert [e[:2] for e in log] == [("open", "right"), ("retreat", "right")], log
  assert log[0][2] - t0 == pytest.approx(0.5, abs=0.01), "it did not wait for the way to clear"
  assert m.looks == m.scans == 6 and len(m.reach.judged) == 6, \
    "it looked other than every ARM_LOOK_S"
  assert log[1][3], "its way home was judged against nobody"
  assert m.arm_waits[-1]["why"] == "clear" and m.arm_waits[-1]["s"] == pytest.approx(0.5, abs=0.01)
  assert m.reach.keep_out == []


def test_a_walks_hand_over_waits_its_bound_at_most_then_brings_the_arm_home_today(
    monkeypatch):
  """...and `ARM_WAIT_S` at most: then it lets go and the arm comes home its
  way judged against the robot still there -- none keeping clear, today's
  (`Reach._kept_off`) -- the wait kept as waited out; and alone, or with an
  arm at rest, it neither looks nor waits."""
  monkeypatch.setattr(hmi, "ARM_LOOK_S", 0.1)
  monkeypatch.setattr(hmi, "ARM_WAIT_S", 0.25)
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spied(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  _beside(body)
  t0 = at()
  m.reach = _Arms(m.reach.grippers, ("right",), lambda: False, log, at)
  m.reach.grippers["right"].shut()
  body.run(m.to_walk_routine())
  assert m.legs == "walk" and [e[:2] for e in log] == [("open", "right"), ("retreat", "right")]
  assert log[0][2] - t0 == pytest.approx(0.3, abs=0.01), "it did not wait its bound out"
  assert log[1][3], "its way home was judged against nobody"
  assert m.arm_waits[-1]["why"] == "waited"
  for out, others in ((("right",), False), ((), True)):
    body = body_for(*_floor())
    m = body.mission
    _beside(body)
    if not others:
      body.others = []
    m.reach = _Arms(m.reach.grippers, out, lambda: False, [], at)
    t0 = at()
    body.run(m.to_walk_routine())
    assert m.legs == "walk" and m.looks == 0 and not m.arm_waits, (out, others)


def test_the_loops_own_hand_moves_wait_for_a_way_then_go(monkeypatch):
  """A charge's plug-in and unplug are the loop's own hand moves, never a
  verb's: refused, the plug went unfetched, the charge failed, and a failed
  charge is a death (`HubLifecycle._strand`). So the loop's reach and
  release wait for a way that keeps clear as a walk's hand-over does,
  looking every `ARM_LOOK_S`, `ARM_WAIT_S` at most -- a release with its
  gripper as it was meanwhile -- then go: a reach the way out that comes
  least near that robot, a release the way home."""
  monkeypatch.setattr(hmi, "ARM_LOOK_S", 0.1)
  monkeypatch.setattr(hmi, "ARM_WAIT_S", 0.25)
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spied(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  _beside(body)
  t0 = at()
  m.reach = _Arms(m.reach.grippers, ("right",), lambda: at() - t0 >= 0.15, log, at)
  m.reach.grippers["right"].shut()
  assert body.run(body.release_routine("right")) == "home"
  assert [e[:2] for e in log] == [("open", "right"), ("retreat", "right")], log
  assert log[0][2] - t0 == pytest.approx(0.2, abs=0.01), "it did not wait for the way"
  assert m.arm_waits[-1]["why"] == "clear"
  # ...and a reach the loop asks, the robot 0.6 m to its right
  body = body_for(*_floor())
  m = body.mission
  sx, sy, syaw = m._stance_frame()
  where = (sx + 0.6 * math.sin(syaw), sy - 0.6 * math.cos(syaw))
  body.others = [lambda: KeepClear(*where, root="r2_", seen=True)]
  _scanning(m, ("r2_",))
  t0 = float(body.data.time)
  world = tuple(body.to_world(np.array(LEVER)))
  body.run(body.reach_routine("right", world, patience=0.01))
  assert m.reach.phase["right"] != "rest", "the loop's reach never went"
  assert float(body.data.time) - t0 >= 0.25 and m.looks >= 3
  assert m.arm_waits[-1]["move"] == "reach" and m.arm_waits[-1]["why"] == "waited"


def test_a_robot_whose_arm_waits_for_its_way_is_never_asked_aside(monkeypatch):
  """Standing still with no hand out, a robot whose reach waits for its way
  would read as resting after `REST_S`, and a robot resting is asked aside
  and walked off (#415): its reach then planned off the stance it had left.
  While an arm waits (`arm_wait_routine`) it is not resting."""
  from pluggybot.humanoid.way import REST_S
  monkeypatch.setattr(hmi, "ARM_LOOK_S", 0.1)
  monkeypatch.setattr(hmi, "ARM_WAIT_S", 0.25)
  body = body_for(*_floor())
  m = body.mission
  m.last_motion_t = float(body.data.time) - REST_S - 1.0
  assert m.resting, "the premise: standing still, no hand out, it rests"
  seen = []

  def clear():
    seen.append(m.resting)
    return False
  assert body.run(m.arm_wait_routine(clear, ("right",), "reach")) == "waited"
  assert seen and not any(seen), "it rested while its arm waited"
  assert m.resting and not m.arm_waiting


def test_a_walk_counts_no_wait_for_an_arms_way_home_against_its_patience():
  """A walk's hand-over that waited for an arm's way home (`held_s`) is
  time the walk stood for nothing of its own: counted, a minute's wait in a
  loop walk's first command left a 40 s walk out of time before it took a
  stride, and a longer one gave up "stalled" the moment the arm came home,
  its stagnation clock run out while it stood. Nor does a step count it."""
  from pluggybot import tick
  from pluggybot.humanoid import body as hb
  from test_step import Stepper, Walk, _step
  from test_unknown import STEPPER, _Walk

  class Held(_Walk):
    """`_Walk`, its first command's hand-over standing it `hold` s."""

    def __init__(self, hold, **kw):
      super().__init__(**kw)
      self.hold = hold

    def _nav_routine(self, v, w):
      if self.hold:
        self.data.time += self.hold
        self.held_s, self.hold = self.held_s + self.hold, 0.0
      yield from super()._nav_routine(v, w)
  for patience in (40.0, 120.0):
    walk = Held(60.0, step=0.05)
    assert tick.run(STEPPER, walk.drive_to_routine(5.0, 0.0, patience)), walk.last_drive
    assert walk.last_drive["why"] == ""

  class HeldWalk(Walk):
    """`Walk`, its hand-over standing it a minute for an arm's way home."""

    def to_walk_routine(self):
      if self.legs != "walk":
        self.t += 60.0
        self.held_s += 60.0
        yield hb.STILL
      yield from super().to_walk_routine()
  walk = HeldWalk()
  body = object.__new__(Stepper)
  body.mission, body._d, body._m = walk, NS(time=0.0), NS(opt=NS(timestep=0.002))
  rep = _step(body, 0.3)
  assert rep["why"] == "stepped", rep


# ---- 7: hardware-honest ------------------------------------------------------------------------


def test_the_ways_are_judged_on_the_robots_model_of_itself_never_the_simulators_truth():
  """Whatever it keeps off, its ways are judged on its own drawing: a reach
  refused, and an arm's way home judged, with the world's `MjData` a spy,
  read nothing of where the world says the bodies are or what they touch
  (`test_humanoid.TRUTH`); where that robot stands is the look's (#675)."""
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  from test_humanoid import TRUTH, _Spy
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  spy = _Spy(data)
  stand = StandDriver(model, spy)
  reach = Reach(model, spy, stand)
  for _ in range(int(0.3 / model.opt.timestep)):
    stand.step()
    reach.step()
    mujoco.mj_step(model, data)
  reach.keep_out = [_at(0.6, -90)]
  spy.read.clear()
  assert not reach.reach("right", HandGoal(np.array(LEVER), None, 1.0))
  assert reach.kept["right"] is not None, "the premise: it was judged and refused"
  reach.q_cmd["right"] = RAISED_OUT.copy()
  reach.way_home("right", refuse=True)
  assert reach.d is spy, "the premise: the reach meets the world through the spy"
  assert not spy.read & TRUTH, spy.read & TRUTH
