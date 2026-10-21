"""Demo 1's verdict and its harness (issue #493, stage 1): the lid opened and
set down, graded by the world (`challenge/lid.py`), and the attempt
protocol that tries a skill from a set-out and writes a row of the record
(`evaluation/attempts.py`).

The criteria were written into the grader first; here a run that meets them
passes and each run that does not -- never opened, let fall, still held,
left open, fallen over, the tool lost, merely reported -- fails for the
reason they give. Every grade but the last two is a bare chest moved by a
test's hand, or the stub body: there is no robot in it, and the grader
cannot tell, which is the point.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.activity import chest as ch
from pluggybot.body import StubBody
from pluggybot.challenge import lid
from pluggybot.economy import scoring
from pluggybot.evaluation import attempts as at

BARE = """<mujoco model="lid"><option integrator="implicitfast"/>
  <worldbody><geom name="floor" type="plane" size="3 3 0.1"/></worldbody></mujoco>"""
#: Chest 35 is the drawn lid that closes slowest let go 2 deg over the shut
#: line (`scripts/attempts.py --drops`: 1.45 rad/s, the 128's least); 0 and
#: 1 a lid with no spring and one with.
SLOWEST, PLAIN, SPRUNG = 35, 0, 1


def stub_world(task: at.Task, hand: bool = False) -> at.World:
  """A set-out on the stub body (`attempts.set_out`'s shape): the chest on a
  bare floor, its catch on, and the tool its set-out gives put on -- with
  `hand`, a mocap box parked far off, which `place_hand` puts onto the
  lid."""
  spec = mujoco.MjSpec.from_string(BARE)
  ch.attach_chest(spec, ch.draw(task.chest), (0.0, 0.0), 0.0, tags=False)
  if hand:
    b = spec.worldbody.add_body(name="hand", pos=[2.0, 2.0, 0.5], mocap=True)
    b.add_geom(name="hand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.01])
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data)
  chest = ch.Chest(model, data, ch.draw(task.chest))
  body.step_hooks.append(lambda: chest.sense(model, data))
  if task.tool is not None:
    body.run(body.place_tool_routine(task.tool))
  return at.World(task=task, model=model, data=data, body=body, chest=chest,
                  start=(0.0, 0.0, 0.0))


def place_hand(body) -> None:
  """The mocap box onto the shut lid's top, 0.5 mm into it."""
  top = ch.BOX_H + ch.LID_T
  body.data.mocap_pos[body.model.body("hand").mocapid[0]] = [ch.BOX_D / 2, 0.0, top + 0.0095]


def lift(up_to=1.10, down_to=math.radians(2.0), rate=0.5, down_rate=None, then=None):
  """A test's hand on the lid, as a skill: its angle written each step --
  up at `rate` to `up_to`, held 0.4 s, down at `down_rate` to `down_to` --
  then let go (`then` runs, e.g. the hand left on the lid), saying it did
  the job. Not a solution: the grader cannot tell, which is the point."""
  down_rate = rate if down_rate is None else down_rate

  def skill(body):
    m, d = body.model, body.data
    q = int(m.jnt_qposadr[m.joint("chest_hinge").id])
    v = int(m.jnt_dofadr[m.joint("chest_hinge").id])

    def to(target, speed):
      a = float(d.qpos[q])
      step = speed * float(m.opt.timestep)
      while abs(target - a) > 1e-9:
        a = target if abs(target - a) <= step else a + math.copysign(step, target - a)
        d.qpos[q], d.qvel[v] = a, math.copysign(speed, target - a) if a != target else 0.0
        yield body.STILL

    def routine():
      yield from to(up_to, rate)
      for _ in range(round(0.4 / m.opt.timestep)):
        d.qpos[q], d.qvel[v] = up_to, 0.0
        yield body.STILL
      if down_to is not None:
        yield from to(down_to, down_rate)
      if then is not None:
        then(body)
      return {"ok": True, "why": "set it down"}
    return routine()
  return skill


def run(skill, chest=PLAIN, hand=False, **kw):
  return at.attempt(at.Task(chest=chest), skill, build=lambda t: stub_world(t, hand), **kw)


# ---- the row: graded, unpaid, shown to no mind ---------------------------------


def test_the_lid_is_graded_through_the_one_door_paid_nothing_and_shown_to_no_mind():
  """A demo's row (#493): an evaluator and a sampler on the registry, base
  0 in challenges.json -- and `shown: false`, so a mind with a library,
  shown every other challenge row, is never shown this one: the chest is
  in no served world, and a row in that table is the deployed prompt
  changed. The catch is a hidden number, secret on the row."""
  assert scoring.EVALUATORS["lid"] is lid.eval_lid and scoring.SAMPLERS["lid"] is lid.sample_lid
  row = scoring.default_table()["lid"]
  assert (row.base, row.bonus, row.offered, row.shown) == (0, 0, False, False)
  assert row.secret == ("catchN",)
  shown = [r["task"] for r in scoring.default_table().as_context(challenges=True)]
  assert "lid" not in shown and "mass" in shown and "stack" in shown
  # the flag is the row's alone: the same row without it would be shown
  spec = {"tier": "auto", "base": 0}
  assert scoring.TaskReward.from_json("lid", spec, offered=False).shown


def test_the_criteria_are_the_issues_and_their_measurements():
  """The numbers the criteria are written in, each read off its measurement
  (`scripts/attempts.py --drops`; the probe's 128 records)."""
  assert lid.OPEN_RAD == 1.0 and math.degrees(lid.SHUT_RAD) == pytest.approx(5.0)
  assert lid.SHUT_RAD == pytest.approx(ch.OPEN_OFF)
  assert lid.LOWER_RAD_S == 1.0 and lid.HOLD_S == 5.0
  # the job's time is Ben's (2026-10-17), and the harness calls it there
  assert lid.TIME_LIMIT_S == 300.0 and at.BUDGET_S == lid.TIME_LIMIT_S


# ---- the evaluator is pure, and absences fail -------------------------------------

PASSED = {"topDeg": 63.8, "downRadS": 0.51, "tiltDeg": 2.8, "falls": 0,
          "touchedDuringHold": [], "angleAtCallDeg": -0.2, "angleDeg": -0.2, "caught": True,
          "holdS": 5.0, "elapsedS": 84.0, "tool": "module_claw", "toolSeated": True,
          "catchN": 3.89}


@pytest.mark.parametrize("change, why", [
  ({"topDeg": None}, "never measured"),
  ({"elapsedS": None}, "never measured"),
  ({"holdS": 4.0}, "not the 5 s"),
  ({"elapsedS": 301.0}, "301 s after the start, past the 300 s the job allows"),
  ({"topDeg": 50.0}, "rose to 50 deg, short of the 57"),
  ({"downRadS": 1.4}, "let fall, not lowered"),
  ({"touchedDuringHold": ["pad_l"]}, "pad_l still touched"),
  ({"angleDeg": 8.0}, "8 deg open"),
])
def test_each_criterion_fails_for_its_own_reason(change, why):
  ok, metrics, reason = lid.eval_lid(PASSED | change)
  assert not ok and why in reason, reason
  assert "3.89" not in reason and "catch" not in reason


def test_a_number_is_judged_as_measured_and_only_shown_rounded():
  """A lid a hair short of the line or a hair over the speed fails, though
  either rounds to a pass (#493's review)."""
  short = lid.eval_lid(PASSED | {"topDeg": math.degrees(0.99999)})
  assert not short[0] and short[1]["topDeg"] == 57.3
  fast = lid.eval_lid(PASSED | {"downRadS": 1.0004})
  assert not fast[0] and fast[1]["downRadS"] == 1.0


def test_a_pass_says_how_and_pays_nothing_and_its_catch_stays_secret():
  verdict = scoring.evaluate("lid", PASSED)
  assert verdict.ok and verdict.points == 0, verdict.reason
  assert "set down" in verdict.reason and "caught" in verdict.reason
  assert "catchN" not in verdict.public_metrics() and verdict.metrics["catchN"] == 3.89
  assert "set down in 84 s" in verdict.reason


def test_a_fall_a_tilt_and_a_lost_tool_are_said_and_never_judged():
  """Ben, 2026-10-17: the job is the lid, and a fall is lost time -- what a
  fall is is the body's to say, and the humanoid's watch takes its own bow
  for one (#520)."""
  ok, metrics, _ = lid.eval_lid(PASSED | {"falls": 2, "tiltDeg": 80.0, "toolSeated": False})
  assert ok and (metrics["falls"], metrics["tiltDeg"], metrics["toolSeated"]) == (2, 80.0, False)
  assert lid.eval_lid(PASSED | {"tool": None, "toolSeated": None})[0]


# ---- the world, not the report ------------------------------------------------------


def test_a_lid_lifted_lowered_and_let_go_on_the_shut_line_passes():
  row = run(lift())
  v = row["verdict"]
  assert v["ok"], v["reason"]
  assert v["metrics"]["topDeg"] == pytest.approx(63.0, abs=0.5)
  assert v["metrics"]["downRadS"] < 0.7 and v["metrics"]["caught"]


def test_a_lid_let_fall_from_open_fails_even_shut_and_caught():
  """Let go at the top: on its stop and in its catch well inside the hold,
  so only the trajectory tells -- criterion 2, read every step."""
  row = run(lift(down_to=None))
  v = row["verdict"]
  assert not v["ok"] and "let fall" in v["reason"], v["reason"]
  assert v["metrics"]["caught"] and v["metrics"]["downRadS"] > 2.5


@pytest.mark.parametrize("chest", [SLOWEST, PLAIN, SPRUNG])
def test_a_lid_let_go_two_degrees_over_the_shut_line_was_let_fall(chest):
  """The line's meaning, on the slowest drawn lid: let go 2 deg over the
  shut line, a lid crosses it faster than `LOWER_RAD_S` -- a lowering that
  stops short is a drop."""
  row = run(lift(down_to=lid.SHUT_RAD + math.radians(2.0)), chest=chest)
  assert not row["verdict"]["ok"] and "let fall" in row["verdict"]["reason"]
  ok = run(lift(down_to=lid.SHUT_RAD - math.radians(0.5)), chest=chest)
  assert ok["verdict"]["ok"], ok["verdict"]["reason"]


def test_a_lid_pushed_past_upright_and_let_go_is_left_open():
  """Past upright a lid leans back on its stop, never coming down: only the
  end state tells -- criterion 3."""
  row = run(lift(up_to=1.75, down_to=None))
  assert not row["verdict"]["ok"] and "deg open 5 s after" in row["verdict"]["reason"]
  assert row["verdict"]["metrics"]["angleDeg"] > 100.0


def test_lowered_too_fast_is_let_fall_and_just_under_the_line_is_lowered():
  assert "let fall" in run(lift(down_rate=1.3))["verdict"]["reason"]
  assert run(lift(down_rate=0.85))["verdict"]["ok"]


def test_a_lid_still_touched_after_the_call_was_not_let_go():
  row = run(lift(then=place_hand), hand=True)
  assert not row["verdict"]["ok"]
  assert "hand still touched" in row["verdict"]["reason"], row["verdict"]["reason"]


def test_never_opened_and_merely_reported_fails_on_the_untouched_world():
  """The skill says it did the job and the world says it did not: the
  sampler reads nothing off the skill's word."""
  def says_so(body):
    return iter(())  # a routine that returns at once, having done nothing

  def word(body):
    def routine():
      return {"ok": True, "why": "opened and set down", "topDeg": 64.0}
      yield
    return routine()
  for skill in (says_so, word):
    row = run(skill)
    assert not row["verdict"]["ok"] and "never opened" in row["verdict"]["reason"]
  assert row["own"] == {"ok": True, "why": "opened and set down"}
  world = stub_world(at.Task(chest=PLAIN))
  m = lid.sample_lid(world, None, {"opened": True}, {})
  assert not lid.eval_lid(m)[0] and lid.eval_lid(m)[2] == "the lid was never measured"


def test_the_body_says_its_falls_tilt_and_tool_and_the_lid_is_judged():
  """Each asked of the body and none of them named here: the falls its own
  watch counted, its attitude, the tool its set-out gave
  (`Body.tool_powered`) -- recorded on the row, and a lid set down passes."""
  def tipped_fell_and_lost_its_tool(body):
    def routine():
      body.falls += 1
      body.attitude = (math.cos(math.radians(35)), math.sin(math.radians(35)), 0.0, 0.0)
      got = yield from lift(then=lambda b: setattr(b, "holding", None))(body)
      return got
    return routine()
  row = run(tipped_fell_and_lost_its_tool)
  m = row["verdict"]["metrics"]
  assert row["verdict"]["ok"], row["verdict"]["reason"]
  assert (m["falls"], m["tiltDeg"], m["toolSeated"]) == (1, 70.0, False)


# ---- the protocol ----------------------------------------------------------------------


def test_an_attempt_is_a_row_of_the_record_and_a_second_is_a_reset_the_world_made():
  row = run(lift(), n=2, info={"process": "code", "skill": "hand", "tokens": 0,
                               "imaginationCpuS": 0.0, "commit": "abc1234"})
  assert row["task"] == {"chest": PLAIN, "body": "quadruped", "mount": "floor", "grader": "lid"}
  assert row["attempt"] == 2 and row["help"] == {"reset": True, "demonstrations": 0,
                                                 "interventions": 0}
  assert row["process"] == "code" and row["tokens"] == 0 and row["steps"] is None
  assert row["own"] == {"ok": True, "why": "set it down"} and not row["cut"]
  assert row["holdS"] == pytest.approx(lid.HOLD_S, abs=0.01)
  assert row["secret"] == {"catchN": pytest.approx(ch.draw(PLAIN).catch_n, abs=1e-3)}
  assert row["run"] == {"commit": "abc1234", "budgetS": at.BUDGET_S, "holdS": lid.HOLD_S}
  assert run(lift())["help"]["reset"] is False


def test_the_budget_ends_a_skill_closing_it_and_the_attempt_is_graded_as_it_stands():
  closed = []

  def forever(body):
    def routine():
      try:
        while body.data.time < 5.0:       # a budget that never came still ends
          yield body.STILL
        return {"ok": False, "why": "the budget never came"}
      finally:
        closed.append(True)
    return routine()
  row = run(forever, budget_s=1.0)
  assert row["cut"] and closed and row["simS"] == pytest.approx(1.0, abs=0.01)
  assert row["own"] == {"ok": None, "why": ""}
  #  ...and a cut attempt that failed says so first, what it left after
  assert row["verdict"]["reason"].startswith("out of time (1 s) -- the lid never opened")


def test_an_attempt_that_raises_or_whose_world_was_reset_has_no_verdict():
  """No row, so no verdict: the caller records what it raised (`scripts/
  attempts.py`), and MuJoCo resetting an unstable world mid-attempt is
  caught by its warnings -- its readings are not the attempt's."""
  def broken(body):
    def routine():
      yield body.STILL
      raise ZeroDivisionError("a skill's own bug")
    return routine()
  with pytest.raises(ZeroDivisionError):
    run(broken)

  def blew_up(body):
    def routine():
      dof = body.model.jnt_dofadr[body.model.joint("chest_hinge").id]
      body.data.qvel[dof] = float("nan")
      yield body.STILL
      return {"ok": True, "why": "fine, it says"}
    return routine()
  with pytest.raises(at.Unstable, match="BADQVEL"):
    run(blew_up)


def test_a_task_is_data_and_a_set_out_is_measured_for_its_body():
  t = at.Task(chest=7, body="humanoid", mount="counter")
  assert at.Task.from_dict(t.as_dict()) == t and t.tool is None
  assert at.Task(chest=7).tool == "module_claw"
  with pytest.raises(ValueError, match="no set-out for the humanoid at a chest on the floor"):
    at.Task(chest=0, body="humanoid", mount="floor")
  with pytest.raises(ValueError):
    at.Task(chest=-1)
  with pytest.raises(ValueError, match="no grader"):
    at.Task(chest=0, grader="stack")


def test_held_out_chests_are_none_ever_flown_and_their_seed_is_a_frozen_builds():
  ks = at.held_out(at.seed_of("95fcbfc"), 32)
  assert len(set(ks)) == 32 and min(ks) >= at.HELD_OUT_FROM > max(at.PUBLIC)
  assert ks == at.held_out(at.seed_of("95fcbfc"), 32)
  assert set(ks).isdisjoint(at.held_out(at.seed_of("95fcbfd"), 32))
  assert at.Task(chest=ks[0]).held_out and not at.Task(chest=0).held_out
  with pytest.raises(ValueError, match="frozen build"):
    at.seed_of("unknown")
  # however long the sha it is handed: git's abbreviation is a setting
  assert at.seed_of("95FCBFC") == at.seed_of("95fcbfc8a1b2") == at.seed_of("95fcbfc")
  assert at.commit_of("95fcbfc8a1b2") == "95fcbfc"
  with pytest.raises(ValueError):
    at.commit_of("95fcb")


# ---- the bodies' own set-outs ----------------------------------------------------------


def test_the_quadrupeds_set_out_seats_its_claw_with_the_chest_shut_and_caught():
  """The set-out the probe's reference flies (#466's placement): the claw
  put on by the set-out (`Body.place_tool_routine`, a staging), the lid on
  its stop with its catch applied, the body upright."""
  world = at.set_out(at.Task(chest=0))
  try:
    m, d, body = world.model, world.data, world.body
    assert body.tool_powered("module_claw")
    shut = lid.measure(m, d)
    assert shut["caught"] and abs(shut["angleDeg"]) < 0.5
    dof = m.jnt_dofadr[m.joint("chest_hinge").id]
    assert d.qfrc_applied[dof] < 0.0                 # the catch, pulling it shut
    assert lid.tilt(body.orientation()) < math.radians(5.0)
  finally:
    world.close()


def test_the_grader_reads_the_humanoids_set_out_unchanged():
  """The same grader in #507's counter world: the chest's parts by the
  same names, the humanoid's attitude, and no tool to keep -- nothing in
  the grader is either body's."""
  world = at.set_out(at.Task(chest=0, body="humanoid", mount="counter"))
  try:
    watch = lid.Watch(world.model, world.data, world.body)
    watch()
    m = lid.sample_lid(world, None, None, {"at_call": lid.measure(world.model, world.data),
                                          "watch": watch, "tool": world.task.tool})
    assert m["tool"] is None and m["toolSeated"] is None and m["tiltDeg"] < 5.0
    assert m["topDeg"] == pytest.approx(0.0, abs=0.5) and m["caught"]
    assert np.any(watch.moving) and np.all(watch.own[watch.moving])
  finally:
    world.close()
