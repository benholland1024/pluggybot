"""The hands' capability check (issue #525; docs/Hands.md): what its jobs
stand on, each pinned without a flight (docs/Testing.md) -- the flights are
`scripts/hands_check.py`'s.

  1. every check's procedure is the language a mind is given, inside its
     caps, and nothing under `mind/` can reach one;
  2. each grader (`challenge/drawer.py`, `door.py`, `place.py`) passes only
     what its criteria say, each criterion failing alone, judged unrounded;
  3. a check pays nothing and is shown to no mind;
  4. the door's latch is a trip latch: bolted until the lever is pressed
     past most of its travel, then a kick sets it ajar, and it bolts again
     only shut with the lever up;
  5. the set-outs are seeded, inside their ranges, and the cube's mat lies
     straight on ahead of it;
  6. the tally reads a job's bar only where every set-out flew.
"""

import ast
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.activity import door as dr
from pluggybot.challenge import door, drawer, lid, place
from pluggybot.economy import scoring
from pluggybot.evaluation import counter as ct
from pluggybot.evaluation import hands_check as hc
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st

SRC = Path(__file__).parent.parent / "src" / "pluggybot"


# ---- the procedures: the language a mind is given --------------------------------


def test_every_checks_procedure_is_the_language_a_mind_is_given():
  facts = st.hand_facts()
  assert set(hc.JOBS) == {"lid", "drawer", "door", "place"}
  for job, source in hc.JOBS.items():
    proc = hc.compiled(job)
    assert len(source) <= lang.MAX_SOURCE_CHARS, job
    called = {n.func.id for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert called - {"budget", "read", "range", *lang.MATH} <= set(facts.verbs), job
    assert proc.verbs > 0


def test_the_mind_never_reaches_a_checks_procedure():
  """A procedure here is a solution: nothing a mind is built from imports
  the check or names it (`challenge/solutions.py`'s fence)."""
  for path in (SRC / "mind").rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      names = ([a.name for a in node.names] if isinstance(node, ast.Import)
               else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
      assert not any("hands_check" in n for n in names), path
    assert "hands_check" not in path.read_text(), path


# ---- the graders: each criterion fails alone ---------------------------------------


def _drawer(**kw):
  m = {"mostM": 0.18, "fastestMS": 0.1, "tiltDeg": 2.0, "falls": 0, "outAtCallM": 0.17,
       "outM": 0.17, "holdS": drawer.HOLD_S, "elapsedS": 20.0, "limitS": 60.0,
       "touchedDuringHold": []}
  return {**m, **kw}


def test_the_drawer_passes_out_15_cm_let_go_in_time_and_on_nothing_less():
  assert drawer.eval_drawer(_drawer())[0]
  ok, _, why = drawer.eval_drawer(_drawer(mostM=0.149, outAtCallM=0.149, outM=0.149))
  assert not ok and "came out 15 cm, short of the 15" in why
  ok, _, why = drawer.eval_drawer(_drawer(outAtCallM=0.10, outM=0.10))
  assert not ok and "when you finished" in why
  ok, _, why = drawer.eval_drawer(_drawer(touchedDuringHold=["left_jaw_a_pad"]))
  assert not ok and "let go" in why
  ok, _, why = drawer.eval_drawer(_drawer(outM=0.12))
  assert not ok and "s later" in why
  ok, _, why = drawer.eval_drawer(_drawer(elapsedS=61.0))
  assert not ok and "past the 60 s" in why
  assert drawer.eval_drawer(_drawer(outM=None))[2] == "the drawer was never measured"


def _door(**kw):
  m = {"mostDeg": 70.0, "pressedDeg": 45.0, "fastestRadS": 1.0, "tiltDeg": 2.0, "falls": 0,
       "angleAtCallDeg": 65.0, "angleDeg": 64.0, "holdS": door.HOLD_S, "elapsedS": 30.0,
       "limitS": 60.0, "touchedDuringHold": []}
  return {**m, **kw}


def test_the_door_passes_open_50_deg_let_go_in_time_and_on_nothing_less():
  assert door.eval_door(_door())[0]
  ok, _, why = door.eval_door(_door(mostDeg=0.5, angleAtCallDeg=0.5, angleDeg=0.5))
  assert not ok and "never opened" in why
  ok, _, why = door.eval_door(_door(mostDeg=49.9, angleAtCallDeg=49.9, angleDeg=49.9))
  assert not ok and "short of the 50" in why
  ok, _, why = door.eval_door(_door(angleAtCallDeg=19.0, angleDeg=19.0))
  assert not ok and "was 19 deg open when you finished" in why
  ok, _, why = door.eval_door(_door(touchedDuringHold=["left_gripper"]))
  assert not ok and "let go" in why
  ok, _, why = door.eval_door(_door(angleDeg=40.0))
  assert not ok and "s later" in why
  ok, _, why = door.eval_door(_door(elapsedS=60.5))
  assert not ok and "past the 60 s" in why


def _cube(**kw):
  m = {"liftedM": 0.05, "fastestDownMS": 0.2, "robotTiltDeg": 2.0, "falls": 0,
       "offAtCallM": 0.01, "offM": 0.01, "aboveM": 0.0005, "tiltDeg": 1.0,
       "holdS": place.HOLD_S, "elapsedS": 20.0, "limitS": 60.0, "touchedDuringHold": []}
  return {**m, **kw}


def test_the_cube_passes_set_down_on_the_mat_upright_let_go_and_on_nothing_less():
  assert place.eval_place(_cube())[0]
  ok, _, why = place.eval_place(_cube(fastestDownMS=place.DROP_M_S + 0.01))
  assert not ok and "dropped, not set down" in why
  ok, _, why = place.eval_place(_cube(offM=place.PLACE_M + 0.001))
  assert not ok and "must rest on the mat" in why
  ok, _, why = place.eval_place(_cube(aboveM=0.03))
  assert not ok and "over it" in why
  ok, _, why = place.eval_place(_cube(tiltDeg=math.degrees(place.UPRIGHT_RAD) + 0.5))
  assert not ok and "tipped" in why
  ok, _, why = place.eval_place(_cube(touchedDuringHold=["left_jaw_b_pad"]))
  assert not ok and "let go" in why
  ok, _, why = place.eval_place(_cube(elapsedS=61.0))
  assert not ok and "past the 60 s" in why


def test_a_check_pays_nothing_and_is_shown_to_no_mind():
  rows = json.loads((SRC / "economy" / "challenges.json").read_text())["tasks"]
  for job in ("drawer", "door", "place"):
    row = rows[job]
    assert row["base"] == 0 and row.get("bonus", 0) == 0 and row["shown"] is False, job
    assert job in scoring.EVALUATORS and job in scoring.SAMPLERS, job
  assert hc.JOBS.keys() <= {"lid", *scoring.EVALUATORS}
  assert lid.TIME_LIMIT_S == drawer.TIME_LIMIT_S == door.TIME_LIMIT_S == place.TIME_LIMIT_S


# ---- the door's latch ---------------------------------------------------------------


def _cupboard():
  """The cupboard alone on a floor, its tip untagged: the latch's physics
  without the house."""
  spec = mujoco.MjSpec.from_string(
    '<mujoco><option timestep="0.002"/><worldbody>'
    '<geom name="floor" type="plane" size="2 2 0.1"/></worldbody></mujoco>')
  dr.attach_cupboard(spec, dr.Hinge(), (0.0, 0.0), tags=False)
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  return m, d, dr.Door(m, d, dr.Hinge())


def _run(m, d, act, seconds, lever=None, door_torque=0.0):
  lv = m.jnt_dofadr[m.joint("door_lever_pivot").id]
  dd = m.jnt_dofadr[m.joint("door_hinge").id]
  for _ in range(int(seconds / m.opt.timestep)):
    act.sense(m, d)
    d.qfrc_applied[lv] = 0.0 if lever is None else lever
    d.qfrc_applied[dd] += door_torque
    mujoco.mj_step(m, d)


def test_the_door_is_bolted_until_its_lever_is_pressed_then_kicked_ajar_and_rebolted_shut():
  m, d, act = _cupboard()
  # bolted: pulled open with the lever up, it stays shut
  _run(m, d, act, 0.5, door_torque=0.5)
  assert act.angle(d) < dr.SHUT_RAD + 0.01 and act.bolted
  # pressed past the trip, the latch frees it and its kick sets it ajar
  _run(m, d, act, 0.5, lever=1.5)
  assert act.pressed(d) > dr.UNLATCH_RAD and not act.bolted
  _run(m, d, act, 2.0)
  assert dr.KICK_RAD / 2 < act.angle(d) < dr.KICK_RAD + math.radians(1.0)
  assert not act.bolted and act.flags["latched"] is False
  # ...the lever back up does not catch it while it stands open
  assert act.pressed(d) < dr.UNLATCH_RAD and not act.bolted
  # shut again with the lever up, it bolts
  _run(m, d, act, 1.0, door_torque=-0.6)
  assert act.bolted
  # ...and a door held shut while its lever is pressed and let go -- the
  # hand that pressed it holding it shut -- is free once let go: the latch
  # stays tripped until the door has stood ajar
  m, d, act = _cupboard()
  _run(m, d, act, 0.5, lever=1.5, door_torque=-0.5)
  _run(m, d, act, 0.5, door_torque=-0.5)
  assert act.pressed(d) < dr.UNLATCH_RAD and act.angle(d) <= dr.SHUT_RAD
  _run(m, d, act, 2.0)
  assert act.angle(d) > dr.KICK_RAD / 2 and not act.bolted


# ---- the set-outs ---------------------------------------------------------------------


def test_the_set_outs_are_seeded_and_inside_their_ranges():
  start = ct.START[0] + ct._stance_ahead()
  for k in hc.SET_OUTS:
    so = ct.door_set_out(k)
    assert so == ct.door_set_out(k)
    tip = np.array(dr.KNOB_AT)
    c, s = math.cos(so.cupboard[2]), math.sin(so.cupboard[2])
    x = so.cupboard[0] + c * tip[0] - s * tip[1] - start
    y = so.cupboard[1] + s * tip[0] + c * tip[1] - ct.START[1]
    assert ct.DOOR_AHEAD[0] - 1e-9 <= x <= ct.DOOR_AHEAD[1] + 1e-9
    assert ct.DOOR_LEFT[0] - 1e-9 <= y <= ct.DOOR_LEFT[1] + 1e-9
    cube = ct.cube_set_out(k)
    ahead, left = cube.mat[0] - cube.cube[0], cube.mat[1] - cube.cube[1]
    assert ct.MAT_AHEAD[0] - 1e-9 <= ahead <= ct.MAT_AHEAD[1] + 1e-9
    assert ct.MAT_LEFT[0] - 1e-9 <= left <= ct.MAT_LEFT[1] + 1e-9
    assert ct.CUBE_AHEAD[0] - 1e-9 <= cube.cube[0] - start <= ct.CUBE_AHEAD[1] + 1e-9


# ---- the tally ------------------------------------------------------------------------


def test_a_jobs_bar_is_read_only_where_every_set_out_flew():
  def row(job, k, ok):
    return {"task": {"grader": job, "chest": k}, "verdict": {"ok": ok}}
  rows = [row("door", k, k != 3) for k in hc.SET_OUTS]
  rows += [row("lid", k, k < hc.BAR - 1) for k in hc.SET_OUTS]
  rows += [row("drawer", 0, True), {"task": {"grader": "place", "chest": 0}, "error": "x"}]
  got = hc.tally(rows)
  assert got["door"]["met"] is True and 3 not in got["door"]["passed"]
  assert got["lid"]["met"] is False
  assert got["drawer"]["met"] is None and got["place"] == {"flown": [], "passed": [],
                                                           "met": None}


@pytest.mark.parametrize("job", ["drawer", "door", "place"])
def test_a_checks_watch_and_snapshot_are_its_graders(job):
  from pluggybot.evaluation import attempts as at
  watching, snapshot = at.WATCHED[job]
  assert watching.__module__ == f"pluggybot.challenge.{job}"
  assert snapshot.__module__ == f"pluggybot.challenge.{job}"
