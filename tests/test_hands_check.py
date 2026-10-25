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
from pluggybot.activity import drawer as dw
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


def test_a_check_is_held_its_whole_hold_and_failed_by_a_fall_never_by_a_tilt():
  """A fall fails the try (Ben, 2026-10-25, #544): the lid's rule, the
  checks' too -- said first, then what the world shows; falls never counted
  fail as any missing measurement does, and a tilt is judged by nothing."""
  for grade, row in ((drawer.eval_drawer, _drawer), (door.eval_door, _door),
                     (place.eval_place, _cube)):
    ok, _, why = grade(row(holdS=1.0))
    assert not ok and "measured 1.0 s after the call" in why, grade
    tilt = "robotTiltDeg" if grade is place.eval_place else "tiltDeg"
    assert grade(row(**{tilt: 80.0}))[0], grade
    ok, metrics, why = grade(row(falls=1))
    assert not ok and metrics["falls"] == 1, grade
    assert why == f"you fell, and a fall fails the try -- {grade(row())[2]}", grade
    assert grade(row(falls=None))[::2] == (False, "your falls were never counted"), grade
    assert grade(row(limitS=None, elapsedS=lid.TIME_LIMIT_S + 1.0))[0] is False, grade


def _stub_set_out(task):
  """A check's set-out on the stub body (`attempts.set_out`'s shape): its
  prop on a bare floor, its activity sensing every step."""
  from pluggybot.activity import cube as cb
  from pluggybot.body import StubBody
  from pluggybot.evaluation import attempts as at
  from test_lid import BARE
  spec = mujoco.MjSpec.from_string(BARE)
  k = task.chest
  if task.grader == "drawer":
    dw.attach_cabinet(spec, dw.draw(k), (0.5, 0.0), tags=False)
  elif task.grader == "door":
    dr.attach_cupboard(spec, dr.draw(k), (0.5, 0.0), tags=False)
  else:
    cb.attach_props(spec, cb.draw(k), (0.4, 0.0), (0.4, 0.2), tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data)
  act = {"drawer": lambda: dw.Drawer(model, data, dw.draw(k)),
         "door": lambda: dr.Door(model, data, dr.draw(k)),
         "place": lambda: cb.Cube(model, data, cb.draw(k), 0.0)}[task.grader]()
  body.step_hooks.append(lambda: act.sense(model, data))
  return at.World(task=task, model=model, data=data, body=body, chest=None,
                  start=(0.0, 0.0, 0.0), activity=act)


@pytest.mark.parametrize("grader", ["lid", "drawer", "door", "place"])
def test_a_stub_bodys_fall_fails_every_grader_through_its_attempt(grader):
  """Ben, 2026-10-25 (#544): every grader fails an attempt in which the
  body's own watch counted a fall (`Body.falls`, the stub's counter: what
  a fall is stays the body's to say) -- read through the attempt's own
  watch and sampler, said first -- and a body standing does not."""
  from pluggybot.evaluation import attempts as at
  from test_lid import stub_world
  task = (at.Task(chest=0) if grader == "lid" else
          at.Task(chest=0, body="humanoid", mount="counter", grader=grader))

  def skill(fall: bool):
    def routine(body):
      body.falls += int(fall)
      yield body.STILL
      return {"ok": True, "why": "it says it is done"}
    return routine
  build = stub_world if grader == "lid" else _stub_set_out
  fell = at.attempt(task, skill(True), build=build, budget_s=0.5)["verdict"]
  stood = at.attempt(task, skill(False), build=build, budget_s=0.5)["verdict"]
  assert not fell["ok"] and fell["metrics"]["falls"] == 1
  assert fell["reason"] == f"you fell, and a fall fails the try -- {stood['reason']}"
  assert stood["metrics"]["falls"] == 0 and not stood["reason"].startswith("you fell")


def test_a_check_is_judged_as_measured_and_shown_rounded():
  """Shown rounded, a number is judged as measured: a drawer 14.96 cm out
  is short of 15, though it shows as 0.150 (#525) -- and a cube under the
  mat's top is as far off it as one over it."""
  assert not drawer.eval_drawer(_drawer(mostM=0.1496, outAtCallM=0.1496, outM=0.1496))[0]
  assert not door.eval_door(_door(mostDeg=49.996, angleAtCallDeg=49.996, angleDeg=49.996))[0]
  assert not place.eval_place(_cube(offM=0.0254))[0]
  assert not place.eval_place(_cube(aboveM=-0.01))[0]


def test_a_cube_lies_flat_on_any_face():
  """It is a cube: every face is a bottom -- read off one, a cube set down
  flat on another rested "tipped 180 deg" (#525)."""
  from pluggybot.legs.probe import rot_about
  q = np.zeros(4)
  for axis, deg, want in (((1, 0, 0), 0, 0), ((1, 0, 0), 90, 0), ((1, 0, 0), 180, 0),
                          ((0, 1, 0), 270, 0), ((0, 0, 1), 30, 0), ((1, 0, 0), 45, 45),
                          ((0, 1, 0), 20, 20), ((0, 1, 0), 110, 20)):
    mujoco.mju_mat2Quat(q, rot_about(np.array(axis, dtype=float), math.radians(deg)).ravel())
    assert math.degrees(place.resting_tilt(q)) == pytest.approx(want, abs=1e-6), (axis, deg)
  # ...as the grader's snapshot reads it: a cube upside down on the mat
  m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><body name="cube_mat"><geom name="cube_mat" type="box" size=".04 .04 .002"/>'
    '</body><body name="cube" pos="0 0 .015" quat="0 1 0 0"><geom name="cube" type="box" '
    'size=".013 .013 .013"/></body></worldbody></mujoco>')
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  assert place.measure(m, d)["tiltDeg"] == pytest.approx(0.0, abs=1e-6)


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
    so = ct.drawer_set_out(k)
    assert so == ct.drawer_set_out(k)
    c, s = math.cos(so.cabinet[2]), math.sin(so.cabinet[2])
    knob = np.array(dw.KNOB_AT)
    x = so.cabinet[0] + c * knob[0] - s * knob[1] - start
    y = so.cabinet[1] + s * knob[0] + c * knob[1] - ct.START[1]
    assert ct.KNOB_AHEAD[0] - 1e-9 <= x <= ct.KNOB_AHEAD[1] + 1e-9
    assert ct.KNOB_LEFT[0] - 1e-9 <= y <= ct.KNOB_LEFT[1] + 1e-9


def test_a_set_outs_place_is_not_its_props_hidden_draw():
  """Off one seed, the furthest cube was always the heaviest, its weight
  read off where it stood (#525); every set-out draws off a stream of its
  own, never its prop's, nor its neighbour's."""
  from pluggybot.activity import cube as cb
  ks = range(16)

  def ahead(seed, *ranges):
    return [ct._drawn(k, seed, *ranges)[0][0] for k in ks]
  pairs = ((ahead(ct.CUBE_SEED, ct.CUBE_AHEAD, ct.CUBE_LEFT), [cb.draw(k).cube_kg for k in ks]),
           (ahead(ct.DRAWER_SEED), [dw.draw(k + 1).friction for k in ks]),
           (ahead(ct.DOOR_SEED, ct.DOOR_AHEAD, ct.DOOR_LEFT), [dr.draw(k + 1).friction for k in ks]))
  for placed, hidden in pairs:
    assert abs(np.corrcoef(placed, hidden)[0, 1]) < 0.9


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
  # ...met at the bar itself, and read nowhere a set-out did not fly
  rows = [row("door", k, k < hc.BAR) for k in hc.SET_OUTS]
  rows += [row("lid", k, True) for k in hc.SET_OUTS[:-1]]
  got = hc.tally(rows)
  assert got["door"]["met"] is True and got["lid"]["met"] is None
  # ...over the check's own set-outs alone, each once: one flown twice
  # passes only where both did
  rows = [row("drawer", k, k in (2, 5)) for k in hc.SET_OUTS]
  rows += [row("drawer", k, True) for k in range(8, 16)]
  rows += [row("drawer", 2, True), row("drawer", 0, True)]
  got = hc.tally(rows)
  assert got["drawer"] == {"flown": list(hc.SET_OUTS), "passed": [2, 5], "met": False}


def test_a_checks_activity_senses_and_its_grader_watches_its_attempt():
  """The door's latch is its activity's, sensed every step -- unsensed it
  stays bolted for good -- and the attempt is graded by the door's grader."""
  from pluggybot.evaluation import attempts as at
  world = at.set_out(at.Task(chest=0, body="humanoid", mount="counter", grader="door"))
  try:
    sensed, sense = [], world.activity.sense
    world.activity.sense = lambda m, d: sensed.append(1) or sense(m, d)
    row = at.attempt_in(world, lambda body: body.hold_routine(0.01), budget_s=1.0, hold_s=0.01)
    assert sensed and row["verdict"]["ok"] is False and "angleDeg" in row["verdict"]["metrics"]
  finally:
    world.close()


def test_spots_edges_read_a_turned_cubes_turn():
  """Every check's grip takes its turn off the tag's edges: read mirrored,
  a cube turned 28 deg read -30 (#525)."""
  from pluggybot.evaluation import attempts as at

  def fold(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4
  world = at.set_out(at.Task(chest=3, body="humanoid", mount="counter", grader="place"))
  try:
    body, got = world.body, None
    for down in st.SPOT_GAZES:
      if down:
        body.run(body.gaze_routine(down))
      body.run(body.hold_routine(st.SPOT_STILL_S))
      got = body.spot(56)
      if got:
        break
    q = world.data.xquat[world.model.body("cube").id]
    true = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    assert abs(fold(true)) > math.radians(10)          # the premise: a turned cube
    read = math.atan2(got["edge"][1], got["edge"][0])
    assert abs(fold(read - true)) < math.radians(3.0)
  finally:
    world.close()


@pytest.mark.parametrize("job", ["drawer", "door", "place"])
def test_a_checks_watch_and_snapshot_are_its_graders(job):
  from pluggybot.evaluation import attempts as at
  watching, snapshot = at.WATCHED[job]
  assert watching.__module__ == f"pluggybot.challenge.{job}"
  assert snapshot.__module__ == f"pluggybot.challenge.{job}"
