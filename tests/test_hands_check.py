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
  6. the tally reads a job's bar only where every set-out flew;
  7. the lid's grip, aimed short of the knob's middle, clears the lid up
     its arc and still holds the whole knob;
  8. our table's grips clear what they must (the cup's palm its lid, the
     plate's open pads the fork, the fork squeezed under the stall); its
     hand reaches past each grip and takes and leaves each piece by a pull
     straight up, reading where it goes once it is up; and a cup reach that
     loses the plate's tag goes home and tries again higher.
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
  assert set(hc.JOBS) == {"lid", "drawer", "door", "place", "table"}
  for job, source in hc.JOBS.items():
    proc = hc.compiled(job)
    assert len(source) <= lang.MAX_SOURCE_CHARS, job
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    # ...and its own defs, which the language calls as it does a verb (#560)
    own = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert called - {"budget", "read", "range", *lang.MATH} - own <= set(facts.verbs), job
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
  m = {"liftedM": 0.05, "fastestDownMS": 0.2, "fastestFreeMS": 0.2, "robotTiltDeg": 2.0,
       "falls": 0,
       "offAtCallM": 0.01, "offM": 0.01, "aboveM": 0.0005, "tiltDeg": 1.0,
       "holdS": place.HOLD_S, "elapsedS": 20.0, "limitS": 60.0, "touchedDuringHold": []}
  return {**m, **kw}


def test_the_cube_passes_set_down_on_the_mat_upright_let_go_and_on_nothing_less():
  assert place.eval_place(_cube())[0]
  ok, _, why = place.eval_place(_cube(fastestFreeMS=place.DROP_M_S + 0.01))
  assert not ok and "with nothing holding it -- dropped, not set down" in why
  # a cube held fast and carried down fast is set down, not dropped (#560:
  # the carry's posture swung two held cubes down at 0.6-1.0 m/s)
  assert place.eval_place(_cube(fastestDownMS=1.0))[0]
  assert not place.eval_place(_cube(fastestFreeMS=None))[0]
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


def test_a_cube_held_is_carried_down_never_dropped(monkeypatch):
  """Only a cube nothing of the robot touches can be dropped: coming down
  at 1 m/s it reads nothing while a geom of the robot's touches it, and the
  same speed with nothing touching it is a fall; the fastest at all is
  said beside."""
  from pluggybot.activity import cube as cb
  from pluggybot.body import StubBody
  from test_lid import BARE
  spec = mujoco.MjSpec.from_string(BARE)
  cb.attach_props(spec, cb.draw(0), (0.4, 0.0), (0.4, 0.2), tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data)
  hand = model.geom("cube_mat").id                  # stands in for a pad of the robot's
  body.geom_ids = np.array([hand], dtype=np.int32)
  watch = place.Watch(model, data, body)
  cube = model.geom("cube").id
  held = [True]
  monkeypatch.setattr(place, "contact_pairs", lambda d: (
    np.array([[cube, hand]], dtype=np.int32) if held[0] else np.zeros((0, 2), dtype=np.int32)))
  data.qvel[model.jnt_dofadr[model.joint("cube_free").id] + 2] = -1.0
  watch()
  assert watch.fastest_free == 0.0 and watch.fastest_down == pytest.approx(1.0)
  held[0] = False
  watch()
  assert watch.fastest_free == pytest.approx(1.0)
  # a watch kept before the free line restores its old reading, the dearer
  again = place.Watch(model, data, body)
  again.restore_kept({"fastestDown": 0.8, "falls": 0})
  assert again.fastest_free == 0.8


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
  from pluggybot.activity import dishware as dish
  from pluggybot.body import StubBody
  from pluggybot.evaluation import attempts as at
  from test_lid import BARE
  spec = mujoco.MjSpec.from_string(BARE)
  k = task.chest
  if task.grader == "drawer":
    dw.attach_cabinet(spec, dw.draw(k), (0.5, 0.0), tags=False)
  elif task.grader == "door":
    dr.attach_cupboard(spec, dr.draw(k), (0.5, 0.0), tags=False)
  elif task.grader == "table":
    dish.attach_card(spec, (0.6, 0.0), math.pi, 0.0, tags=False)
    for i, p in enumerate(dish.PIECES):
      dish.ATTACH[p](spec, (0.4, 0.3 * i - 0.3), 0.0, 0.0, 0.2, 0.7, tags=False)
  else:
    cb.attach_props(spec, cb.draw(k), (0.4, 0.0), (0.4, 0.2), tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data)
  act = {"drawer": lambda: dw.Drawer(model, data, dw.draw(k)),
         "door": lambda: dr.Door(model, data, dr.draw(k)),
         "place": lambda: cb.Cube(model, data, cb.draw(k), 0.0),
         "table": lambda: dish.Dishware(model, data, dish.draw(k), 0.0)}[task.grader]()
  body.step_hooks.append(lambda: act.sense(model, data))
  return at.World(task=task, model=model, data=data, body=body, chest=None,
                  start=(0.0, 0.0, 0.0), activity=act)


@pytest.mark.parametrize("grader", ["lid", "drawer", "door", "place", "table"])
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
  q = np.zeros(4)
  for axis, deg, want in (((1, 0, 0), 0, 0), ((1, 0, 0), 90, 0), ((1, 0, 0), 180, 0),
                          ((0, 1, 0), 270, 0), ((0, 0, 1), 30, 0), ((1, 0, 0), 45, 45),
                          ((0, 1, 0), 20, 20), ((0, 1, 0), 110, 20)):
    mujoco.mju_axisAngle2Quat(q, np.array(axis, dtype=float), math.radians(deg))
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


# ---- 7. the lid's grip clears its lid (#560's stage (c3)) -------------------------------


def _lid_scene():
  """The chest alone on a floor, its handle settled as it hangs, and a hand's two pads as its
  jaws close them on the knob (`humanoid.model._jaw`): a mocap body, the grip its origin."""
  from pluggybot.activity import chest as ch
  from pluggybot.humanoid import parts as hp
  spec = mujoco.MjSpec.from_string('<mujoco><option timestep="0.002"/><worldbody>'
                                   '<geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
  ch.attach_chest(spec, ch.draw(0), (0.0, 0.0), 0.0, tags=False)
  hand, pad = spec.worldbody.add_body(name="hand", mocap=True), hp.GRIPPER.pad_half
  for k in (1, -1):
    hand.add_geom(name=f"pad{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pad),
                  pos=[k * (ch.KNOB + pad[0]), 0.0, 0.0], contype=0, conaffinity=0)
  m = spec.compile()
  d = mujoco.MjData(m)
  for _ in range(3000):
    mujoco.mj_step(m, d)
  return m, d


def _boxes_overlap(m, d, g1, g2) -> bool:
  """Two box geoms overlapping, by separating axes (`mj_geomDistance` read two boxes 1.1 cm
  apart as touching)."""
  t = d.geom_xpos[g2] - d.geom_xpos[g1]
  ra, rb = d.geom_xmat[g1].reshape(3, 3), d.geom_xmat[g2].reshape(3, 3)
  axes = [*ra.T, *rb.T, *(np.cross(a, b) for a in ra.T for b in rb.T)]
  return all(abs(t @ ax) <= (np.abs(ra.T @ ax) @ m.geom_size[g1] + np.abs(rb.T @ ax) @ m.geom_size[g2])
             for ax in (a / np.linalg.norm(a) for a in axes if np.linalg.norm(a) > 1e-9))


def _pads_meet_lid_deg(m, d0, grip, rot):
  """The first lid angle, deg, at which a pad meets the lid or the pin's bracket, the hand at
  `grip` and turned `rot` off the knob at rest -- the lid raised with the handle kept at its
  turn in the world, as the hand holds it -- or None by 75."""
  d = mujoco.MjData(m)
  d.qpos[:] = d0.qpos
  hinge, pin = (m.jnt_qposadr[m.joint(f"chest_{j}").id] for j in ("hinge", "pin"))
  knob = m.geom("chest_knob").id
  off = np.asarray(grip) - d0.geom_xpos[knob]
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).ravel())
  pads = [m.geom(f"pad{k}").id for k in (1, -1)]
  hit = [m.geom("chest_lid").id, m.geom("chest_bracket").id]
  for deg in np.arange(0.0, 75.0, 0.5):
    d.qpos[hinge], d.qpos[pin] = math.radians(deg), d0.qpos[pin] - math.radians(deg)
    mujoco.mj_kinematics(m, d)
    d.mocap_pos[0], d.mocap_quat[0] = d.geom_xpos[knob] + off, quat
    mujoco.mj_kinematics(m, d)
    if any(_boxes_overlap(m, d, p, g) for p in pads for g in hit):
      return float(deg)
  return None


def _lid_grips(m, d0, far: float) -> list:
  """Each grip the lid's procedure reaches for (its place and turn, a tilt each, every reach
  refused so it tries them all), its knob's tag -- the top face, as `spot` reads it at the
  counter -- read `far` m further from the robot than it is, the robot square to the chest."""
  from pluggybot.body import StubBody
  from test_hands import _bare, _fly
  knob = m.geom("chest_knob").id
  c, r = d0.geom_xpos[knob], d0.geom_xmat[knob].reshape(3, 3)
  up = r[:, 2] if r[2, 2] > 0.5 else -r[:, 2]
  seen = {"x": c[0] + 0.013 * up[0] + far, "y": c[1] + 0.013 * up[1], "z": c[2] + 0.013 * up[2],
          "normal": list(-up), "edge": list(r[:, 1]), "range": 0.5}

  class Seeing(StubBody):
    def __init__(self, *a, **kw):
      super().__init__(*a, **kw)
      self.reaches = []

    def spot(self, tag, at_height=None):
      return dict(seen) if tag == 53 else None

    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                      turns=None):
      self.reaches.append((np.asarray(pos, dtype=float), np.asarray(rot, dtype=float)))
      self._why[side] = "out of reach"
      return False
      yield
  body = Seeing(*_bare())
  _fly(hc.JOBS["lid"], body)
  return body.reaches


def test_the_lids_grip_clears_its_lid_up_the_arc_and_holds_the_whole_knob():
  """⚠ AIMED AT THE KNOB'S MIDDLE, ITS PADS MET THE RISING LID (#560's stage (c3); SimNotes,
  "The chest's grip"). At every tilt the procedure tries, read 0.9 cm far -- the grips landed
  that much past their aim at the median -- its grip clears the lid past where its lift ends
  (61-63 deg); read true, the knob's far corner is still short of the pads' tips."""
  from pluggybot.activity import chest as ch
  from pluggybot.humanoid import parts as hp
  m, d0 = _lid_scene()
  knob = m.geom("chest_knob").id
  r = d0.geom_xmat[knob].reshape(3, 3)
  far, true = _lid_grips(m, d0, far=0.009), _lid_grips(m, d0, far=0.0)
  assert len(far) == len(true) == 6
  for grip, rot in far:
    met = _pads_meet_lid_deg(m, d0, grip, rot)
    assert met is None or met > 63.0, met
  for grip, rot in true:
    fingers = -rot[:, 2]
    tip = (d0.geom_xpos[knob] - grip) @ fingers + ch.KNOB * np.abs(r.T @ fingers).sum()
    assert tip <= hp.GRIPPER.pad_half[2], tip


# ---- 8. the table's reference: its grips and its moves (#560) ----------------------------


def _open_hands() -> dict:
  """Each hand's solid parts round its grip, its jaws open, as the robot's drawing compiles them
  (`models/humanoid.xml`): its palm and its two pads, each a box -- its middle and its axes in
  the grip's frame, its half sizes."""
  from pluggybot.humanoid import model as hm
  from pluggybot.humanoid import parts as hp
  m = mujoco.MjModel.from_xml_path(hm.MODEL_XML)
  d = mujoco.MjData(m)
  for side in hm.SIDES:
    for jaw in hm.JAWS:
      d.qpos[m.jnt_qposadr[m.joint(f"{side}_{jaw}").id]] = hp.GRIPPER.stroke / 2
  mujoco.mj_kinematics(m, d)
  hands = {}
  for side in hm.SIDES:
    grip = m.site(f"{side}_grip").id
    at, axes = d.site_xpos[grip], d.site_xmat[grip].reshape(3, 3)
    hands[side] = [(axes.T @ (d.geom_xpos[g] - at), axes.T @ d.geom_xmat[g].reshape(3, 3),
                    m.geom_size[g].copy())
                   for g in (m.geom(n).id for n in (f"{side}_gripper",
                                                    *(f"{side}_{jaw}_pad" for jaw in hm.JAWS)))]
  return hands


def _table_world():
  """The cup and the fork as the dishware draws them, where the set-out's middle puts them
  (the start's stance frame), and each open hand a mocap body, its origin its grip."""
  from pluggybot.activity import dishware as dish
  tags = _table_tags()
  spec = mujoco.MjSpec.from_string("<mujoco/>")
  dish.attach_cup(spec, tags[91][:2], 0.0, ct.TABLE_H, 0.25, 1.0, tags=False)
  dish.attach_fork(spec, tags[92][:2], 0.0, ct.TABLE_H, 0.08, 1.0, tags=False)
  for side, parts in _open_hands().items():
    hand = spec.worldbody.add_body(name=f"{side}_hand", mocap=True)
    for i, (at, axes, half) in enumerate(parts):
      quat = np.zeros(4)
      mujoco.mju_mat2Quat(quat, axes.ravel())
      hand.add_geom(name=f"{side}_part{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                    pos=list(at), quat=list(quat), contype=0, conaffinity=0)
  m = spec.compile()
  return m, mujoco.MjData(m)


def _box_gap(m, d, g1, g2) -> float:
  """How far apart two box geoms are, m, 0 where they meet: their nearest two points, a least
  squares bounded to each box (MuJoCo's own distance read boxes 1.1 cm apart as touching, and
  some 8 cm off)."""
  from scipy.optimize import lsq_linear
  edges = np.hstack([d.geom_xmat[g1].reshape(3, 3) * m.geom_size[g1],
                     -d.geom_xmat[g2].reshape(3, 3) * m.geom_size[g2]])
  t = d.geom_xpos[g2] - d.geom_xpos[g1]
  x = lsq_linear(edges, t, bounds=(-1.0, 1.0), method="bvls").x
  return float(np.linalg.norm(edges @ x - t))


def _hand_gaps(m, d, side, grip, rot, piece: str) -> list:
  """How far each part of the open hand at `grip`, turned `rot`, is from a piece, m -- its palm,
  then its pads: the cup's cylinder by MuJoCo's own distance, exact against a box and less than
  0 where the part cuts in; the fork's boxes by `_box_gap`."""
  mocap = m.body_mocapid[m.body(f"{side}_hand").id]
  d.mocap_pos[mocap] = grip
  mujoco.mju_mat2Quat(d.mocap_quat[mocap], np.asarray(rot, dtype=float).ravel())
  mujoco.mj_kinematics(m, d)
  parts = [m.geom(f"{side}_part{i}").id for i in range(3)]
  if piece == "cup":
    return [mujoco.mj_geomDistance(m, d, g, m.geom("cup").id, 0.5, np.zeros(6)) for g in parts]
  return [min(_box_gap(m, d, g, m.geom(f"fork_{p}").id) for p in ("handle", "head"))
          for g in parts]


def _table_tags() -> dict:
  """Each tag where the table's set-out's middle puts it, the start's stance frame (the
  soles `TABLE_GAP` out of the table's edge): the card's on its face, the plate's on its
  floor, the cup's on its lid, the fork's on its handle (`counter.TABLE_SOURCES`)."""
  from pluggybot.activity import dishware as dish
  gap, h = ct.TABLE_GAP, ct.TABLE_H
  at = {n: (x + gap, y) for n, ((x, y), _, _) in ct.TABLE_SOURCES.items()}
  return {93: (ct.CARD_IN + gap - dish.CARD[1], 0.0, h + dish.CARD[2]),
          90: (*at["plate"], h + dish.PLATE_FLOOR_T),
          91: (*at["cup"], h + dish.CUP_H),
          92: (*at["fork"], h + 2 * dish.HANDLE[2])}


def _table_flown(hide=None, drift=0.0):
  """Our table procedure on a stub that does every move at once, as asked, and sees each tag
  where the set-out's middle puts it -- but where `hide(tag, body)` says -- each step turning
  `drift` rad more than it asked, as a live one does: each grip it shut (the hand, where,
  turned how, at what squeeze), each verb as it began (its arguments and where each hand was),
  and the run."""
  from pluggybot.body import StubBody
  from test_hands import _bare, _Life
  tags = _table_tags()

  class Seeing(StubBody):
    def __init__(self, *a, **kw):
      super().__init__(*a, **kw)
      self.turns, self.shuts, self.reached = {}, [], {}

    def spot(self, tag, at_height=None):
      if int(tag) not in tags or (hide is not None and hide(int(tag), self)):
        return None
      x, y, z = tags[int(tag)]
      return {"x": x, "y": y, "z": z, "normal": [0.0, 0.0, -1.0], "edge": [1.0, 0.0, 0.0],
              "range": 0.5}

    def step_routine(self, ahead, left=0.0, turn=0.0, patience=15.0):
      return (yield from super().step_routine(ahead, left, turn + drift, patience))

    def reach_routine(self, side, pos, rot=None, **kw):
      self.turns[side] = np.asarray(rot, dtype=float)
      self.reached[side] = self.reached.get(side, 0) + 1
      return (yield from super().reach_routine(side, pos, rot, **kw))

    def grip_routine(self, side, closed, force=None):
      if closed:
        self.shuts.append((side, np.asarray(self.hand_at(side), dtype=float), self.turns[side],
                           force))
      return (yield from super().grip_routine(side, closed, force))
  body = Seeing(*_bare())
  life, log = _Life(body), []
  life.verb_began = lambda count, line, verb, values: log.append(
    (verb, dict(values), {s: np.asarray(body.hand_at(s), dtype=float) for s in ("left", "right")}))
  # ...each arm's roll-out (`move`) left out: it turns a live stance, and a stub has no joint
  source = "".join(line for line in hc.JOBS["table"].splitlines(keepends=True)
                   if not line.strip().startswith("move("))
  facts = st.hand_facts()
  ran = body.run(lang.run_procedure_routine(life, lang.compile_procedure(source, facts), facts))
  return body, ran, log


def test_the_tables_grips_clear_what_they_must():
  """⚠ EACH GRIP ITS OWN CLEARANCE (#560; SimNotes, "The table's reference on the waist"), the
  set-out at its middle and the hand as its drawing compiles: aimed 3 cm short of the cup's
  middle, no part of the open hand is in the cup -- at the middle its palm cuts into the lid,
  and the cup tipped; the plate's wall gripped 1.5 cm in, the open pads stay 2 cm off the fork
  waiting beside it -- at the wall they came within 5 mm and brushed the fork over."""
  body, ran, _ = _table_flown()
  assert ran["ok"], ran
  (side_c, cup, rot_c, _), (side_p, plate, rot_p, _), (side_f, _, _, _) = body.shuts[:3]
  assert (side_c, side_p, side_f) == ("right", "left", "left")
  tags = _table_tags()
  m, d = _table_world()
  assert min(_hand_gaps(m, d, side_c, cup, rot_c, "cup")) > 0.0
  assert _hand_gaps(m, d, side_c, (*tags[91][:2], cup[2]), rot_c, "cup")[0] < 0.0  # its middle
  assert min(_hand_gaps(m, d, side_p, plate, rot_p, "fork")[1:]) >= 0.015
  wall = (plate[0], tags[90][1] + 0.081, plate[2])    # its wall's inner face, as before
  assert min(_hand_gaps(m, d, side_p, wall, rot_p, "fork")[1:]) < 0.01


def _moves(log, side, start):
  """The hand's moves from `start` in the log: (index, verb, arguments)."""
  return [(i, v, a) for i, (v, a, _) in enumerate(log)
          if i >= start and a.get("hand") == side and v in ("reach", "move_hand", "pull",
                                                            "withdraw")]


def _straight(a, sign) -> bool:
  """A pull straight up (`sign` 1) or down (-1)."""
  return math.hypot(a["dx"], a["dy"]) < 1e-6 and a["dz"] * sign > 0


def test_the_tables_hand_reaches_its_grip_and_takes_and_leaves_each_piece_by_a_pull():
  """⚠ HOW OUR TABLE'S HAND MOVES (#560; SimNotes, "The table's reference on the waist"), each
  a fix flown: the plate's and the fork's reaches are aimed at their grips, the hand shut
  where the reach left it (#627: the reach plans where the robot stands; planned on a statue
  ahead of it, they were aimed 3 cm on and pulled back) -- and the cup's 10 cm over it, its
  hand brought down straight onto it by pulls (the reach's
  own way in tipped it); each piece is lifted straight up by a pull (a carry takes only a
  posture its statue reaches), where it goes is read again once it is up (the stance turns
  0.7-1.9 deg between a read and its set-down), and the hand leaves it straight up by a pull,
  its jaws opened first (carried back as well, it took another posture and dragged the plate
  by its wall; lifted as the jaws still opened, a pad rolled the fork over)."""
  body, ran, log = _table_flown()
  assert ran["ok"], ran
  shuts = [i for i, (v, _, _) in enumerate(log) if v == "close_hand"]
  assert [log[i][1]["hand"] for i in shuts] == ["right", "left", "left"]
  for i, landmark in zip(shuts, (90, 93, 93)):
    side = log[i][1]["hand"]
    grip = log[i][2][side]
    reach = max(j for j, v, _ in _moves(log, side, 0) if j < i and v == "reach")
    aimed = log[reach + 1][2][side] - grip            # where the reach left the hand
    way_in = [(v, a) for j, v, a in _moves(log, side, reach + 1) if j < i]
    if side == "right":
      # ...what the reach's band left to go, under a pull's least 1 cm, a `move_hand`
      assert way_in and all(v in ("pull", "move_hand") for v, _ in way_in), way_in
      assert way_in[-1][0] == "pull", way_in
      # ...over its grip, or as near it as the reach's shallow posture holds (a pull brings it)
      assert aimed[2] >= 0.09 and np.hypot(aimed[0], aimed[1]) < 0.02, aimed
      assert _straight(way_in[-1][1], -1), way_in[-1]
    else:
      assert not way_in and np.abs(aimed).max() < 1e-6, (way_in, aimed)
    after = _moves(log, side, i + 1)
    lift, v, a = after[0]
    assert v == "pull" and _straight(a, 1), (v, a)
    down = next(j for j, v_, a_ in after if v_ == "pull" and _straight(a_, -1))
    assert any(v_ == "spot" and a_["tag"] == landmark for v_, a_, _ in log[lift + 1:down])
    left = [(v_, a_) for v_, a_, _ in log[down + 1:down + 4]]
    assert [v_ for v_, _ in left] == ["open_hand", "wait", "pull"], left
    assert left[0][1]["hand"] == left[2][1]["hand"] == side and _straight(left[2][1], 1)


def _cup_laid_at(log) -> np.ndarray:
  """Where the right hand put the cup down: the hand as its jaws opened after they shut."""
  shut = next(i for i, (v, a, _) in enumerate(log) if v == "close_hand" and a["hand"] == "right")
  opened = next(i for i, (v, a, _) in enumerate(log)
                if i > shut and v == "open_hand" and a["hand"] == "right")
  return log[opened][2]["right"]


def test_the_lay_carries_what_it_read_through_the_turn_its_step_took():
  """⚠ WHAT THE LAY READ BEFORE ITS STEP IT CARRIES THROUGH THE STEP'S OWN TURN (#560; SimNotes,
  "The lay from the tour"): the cup is out of view from the stance, so where it is is read
  before the step -- its place from the plate's tag, as far from it as the cup's reach is long
  -- and a step aside turns the stance 0.5-2.5 deg, 0.8 cm at the cup a degree: a turn the
  step's own report gives to 0.2 deg. The cup's grip and where it is put down, turned 3.4 deg by
  the step here, land where a step that turned nothing puts them, to the 3 cm its grip stands
  short of the cup's middle (2 mm turned)."""
  straight, ran, log = _table_flown()
  assert ran["ok"], ran
  turned, ran2, log2 = _table_flown(drift=0.06)
  assert ran2["ok"], ran2
  assert np.linalg.norm(turned.shuts[0][1][:2] - straight.shuts[0][1][:2]) < 0.003
  assert np.linalg.norm(_cup_laid_at(log2)[:2] - _cup_laid_at(log)[:2]) < 0.004


def test_the_lay_turns_to_find_the_cup_its_first_look_loses():
  """⚠ THE CUP OUT OF VIEW AT THE FIRST LOOK (#560; SimNotes, "The lay from the tour"; it ended
  9 flights in 32 before anything moved): the lay turns a little to its right, reads the three
  tags from there, and carries what it read through the turn back -- the cup's grip where it
  is, the last millimetres the 3 cm its grip stands short of the cup's middle, turned."""
  body, ran, log = _table_flown(hide=lambda tag, b: tag == 91 and b.theta > -0.1)
  assert ran["ok"], ran
  assert body.stepped[0] == (0.0, 0.0, -0.17)
  turn_back = body.stepped[1]
  assert turn_back[2] == pytest.approx(0.17)
  straight, _, log0 = _table_flown()
  assert np.linalg.norm(body.shuts[0][1][:2] - straight.shuts[0][1][:2]) < 0.006
  assert np.linalg.norm(_cup_laid_at(log)[:2] - _cup_laid_at(log0)[:2]) < 0.006


def test_a_cup_reach_that_bows_out_of_the_plates_tags_view_goes_home_and_reaches_again_higher():
  """⚠ THE CUP'S SECOND TRY (#560; SimNotes, "The table's reference on the waist"): where the
  planner's pockets bowed the reach 0.55-0.8 rad, no tag was in view; with the plate's tag
  unseen while the first reach is out, the hand goes home and reaches 1.5 cm higher -- and
  grips where the reads put it, as the first try would."""
  body, ran, log = _table_flown(hide=lambda tag, b: tag == 90 and b.reached.get("right") == 1
                                and "right" in b.hands)
  assert ran["ok"], ran
  first, again = [(i, a) for i, v, a in _moves(log, "right", 0) if v == "reach"][:2]
  between = [(v, a.get("hand", a.get("tag"))) for v, a, _ in log[first[0] + 1:again[0]]]
  assert between == [("spot", 90), ("withdraw", "right")], between
  assert again[1]["z"] - first[1]["z"] >= 0.01
  seen, _, _ = _table_flown()
  assert np.allclose(body.shuts[0][1], seen.shuts[0][1], atol=1e-9)


# ---- the cube's procedure: its tries ------------------------------------------------------


#: The served cube's set-out where our walk leaves it, the start's stance frame: the cube's
#: top face's middle and the mat's (`storeroom.HANDLE_FROM_SPOT`, the mat 7 cm on).
CUBE_AT, MAT_AT = (0.43, 0.15, 0.806), (0.50, 0.15, 0.784)


def _cube_flown(fail_shut=0, lose_lift=0, held_lag=0.0, feet=(0.0, 0.0), unseen=()):
  """Our cube procedure on a stub that does every move at once, as asked, and sees the cube's
  tag and the mat's where the world has them (the start's frame: `spot` reads its own): its
  first `fail_shut` shuts hold nothing and its first `lose_lift` lifts lose what they held,
  its grip read held `held_lag` s on (the gripper's running mean); a try that failed moves
  its feet by `feet` (m, the start's frame) and leaves the tags in `unseen` out of view.
  Each verb as it began, and the run."""
  from pluggybot.body import StubBody
  from test_hands import _bare, _Life

  class Seeing(StubBody):
    def __init__(self, *a, **kw):
      super().__init__(*a, **kw)
      self.shuts = self.lifts = self.failed = 0
      self.lost_at = None

    def spot(self, tag, at_height=None):
      if int(tag) not in (56, 57) or (self.failed and int(tag) in unseen):
        return None
      x, y, z = CUBE_AT if int(tag) == 56 else MAT_AT
      return {"x": x, "y": y, "z": z, "normal": [0.0, 0.0, -1.0], "edge": [1.0, 0.0, 0.0],
              "range": 0.5}

    def _fail(self):
      if not self.failed:
        self.x, self.y = self.x + feet[0], self.y + feet[1]
      self.failed = 1

    def grip_routine(self, side, closed, force=None):
      held = yield from super().grip_routine(side, closed, force)
      if closed:
        self.shuts += 1
        if self.shuts <= fail_shut:
          self.grips[side] = False
          self._fail()
          return False
      return held

    def pull_routine(self, side, direction, limit_n, distance_m, speed=0.06, patience=30.0,
                     give=True):
      out = yield from super().pull_routine(side, direction, limit_n, distance_m, speed,
                                            patience, give)
      if direction[2] > 0.5:
        self.lifts += 1
        if self.lifts <= lose_lift:
          self.grips[side], self.lost_at = False, float(self.data.time)
          self._fail()
      return out

    def gripped(self, side):
      lagging = self.lost_at is not None and float(self.data.time) < self.lost_at + held_lag
      return super().gripped(side) or lagging
  body = Seeing(*_bare())
  life, log = _Life(body), []
  life.verb_began = lambda count, line, verb, values: log.append((verb, dict(values)))
  facts = st.hand_facts()
  ran = body.run(lang.run_procedure_routine(life, lang.compile_procedure(hc.PLACE, facts), facts))
  return body, ran, log


def _reaches(log):
  return [a for v, a in log if v == "reach"]


def _in_frame(at, body):
  """A point of the start's frame in the frame the stub stands in at the end (it never turns)."""
  assert body.theta == 0.0
  return at[0] - body.x, at[1] - body.y


def _tilt(reach) -> float:
  return round(math.degrees(math.asin(-reach["fz"])))


def test_the_cubes_first_try_aims_at_the_read_its_step_carried():
  """⚠ THE FIRST TRY READS NOTHING AGAIN (#560; SimNotes, "The cube from the tour"): its aim is
  the read from 0.4 m, carried through the step by the step's own report, which lands 0.2-1.6 cm
  on from the cube's middle; read again first, the aim was true and the first reach took worse
  postures -- 13 placed of 22 where 17 had been."""
  body, ran, log = _cube_flown()
  assert ran["ok"], ran
  verbs = [v for v, _ in log]
  between = verbs[verbs.index("step") + 1:verbs.index("reach")]
  assert between == ["open_hand"], between


def test_a_cube_try_that_failed_reads_the_mat_and_the_cube_again_and_aims_where_it_now_is():
  """⚠ A FAILED TRY READS THE MAT AND THE CUBE AGAIN (#560; SimNotes, "The cube from the
  tour"): a reach that stops short or a shut on nothing moved the feet 3-7 cm, and every try
  after it aimed at the first read -- on the tree before, no flight that tried twice placed the
  cube at its second try, 15 of them. Read again, the next aims where the cube now is, 1 cm
  on along the fingers as the first try's read aims it, at 60 deg again, and the carry goes
  from the mat read in the same frame, 0.6 cm on and 0.5 cm right of the grip's offset (the
  jaws push the cube along them as they shut)."""
  body, ran, log = _cube_flown(fail_shut=1, feet=(0.02, -0.03))
  assert ran["ok"], ran
  first, again = _reaches(log)[:2]
  verbs = [(v, a.get("tag", a.get("hand"))) for v, a in log]
  between = verbs[verbs.index(("reach", "left")) + 1:]
  between = between[:between.index(("reach", "left"))]
  assert ("spot", 57) in between and ("spot", 56) in between, between
  x, y = _in_frame(CUBE_AT, body)
  assert (again["x"], again["y"]) == pytest.approx((x + 0.01, y), abs=1e-6)
  assert _tilt(again) == _tilt(first) == 60
  assert body.shuts == 2
  (carry,) = [a for v, a in log if v == "move_hand"]
  mx, my = _in_frame(MAT_AT, body)
  assert (carry["x"], carry["y"]) == pytest.approx((mx + 0.01 + 0.006, my - 0.005), abs=1e-6)


@pytest.mark.parametrize("unseen", [(57,), (56,)])
def test_a_retry_that_reads_one_tag_moves_the_other_as_that_one_seems_to_have_moved(unseen):
  """⚠ ONE TAG READ AGAIN CARRIES THE OTHER (#560's review): the feet's move after a failed
  try moves the cube and the mat alike in the frame, so the one read gives the other's place
  -- the cube's aim with the mat out of view, and the mat with the cube out of view. Read
  again only where both were seen, a fresh draw's mat out of view left the next try aiming
  where the cube had been, and its cube was set down 6 cm off."""
  body, ran, log = _cube_flown(fail_shut=1, feet=(0.02, -0.03), unseen=unseen)
  assert ran["ok"], ran
  again = _reaches(log)[1]
  x, y = _in_frame(CUBE_AT, body)
  assert (again["x"], again["y"]) == pytest.approx((x + 0.01, y), abs=1e-6)
  (carry,) = [a for v, a in log if v == "move_hand"]
  mx, my = _in_frame(MAT_AT, body)
  assert (carry["x"], carry["y"]) == pytest.approx((mx + 0.01 + 0.006, my - 0.005), abs=1e-6)


def test_a_lift_that_lost_the_cube_is_a_try_that_failed():
  """⚠ A LIFT THAT LEFT THE CUBE BEHIND TRIES AGAIN (#560; SimNotes, "The cube from the tour"):
  a shut on the cube's edge read held, its lift let go, and the procedure carried and set down
  nothing. Its hand read after the lift -- once the grip's running mean has had its 0.1 s
  (`arms.SQUEEZE_S`: a grip lost read held up to 40 ms on, #646) -- reading nothing held, it
  lets go, goes home, reads again and reaches again."""
  body, ran, log = _cube_flown(lose_lift=1, held_lag=0.04)
  assert ran["ok"], ran
  assert len(_reaches(log)) == 2 and body.shuts == 2, [v for v, _ in log]
  carried = [a for v, a in log if v == "move_hand"]
  assert len(carried) == 1                          # one carry: the cube the second try lifted


def test_the_cubes_tries_go_60_60_46_and_75_deg_and_its_budget_lets_the_fourth_end():
  """The tries' tilts (#560; SimNotes, "The cube from the tour"): second at 75 deg a try took
  12 s and moved the feet most, so 75 comes last; and 150 s of budget lets a fourth try end
  (the check grades its own 68 s)."""
  body, ran, log = _cube_flown(fail_shut=4)
  assert [_tilt(a) for a in _reaches(log)] == [60, 60, 46, 75]
  assert lang.compile_procedure(hc.PLACE, st.hand_facts()).budget_s == 150
