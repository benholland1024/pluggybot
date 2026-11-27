"""The humanoids' first jobs (issue #559, stage 3): the hands' four checks on
the storeroom's counter (`home.storeroom`), offered, set out, watched from
the claim and graded after `done` -- each rule pinned on a stub whose world
is the counter alone (`StubBody`), nothing flown. The references flown from
the station are `scripts/solve.py --feature drawer|door|cube|chest`'s.
"""

import json
import math
import re
from dataclasses import replace

import mujoco
import pytest

from pluggybot.activity import chest as ch
from pluggybot.activity import door as dd
from pluggybot.activity import drawer as dr
from pluggybot.activity.base import ActivitySet
from pluggybot.body import STUB_WORLD, StubBody
from pluggybot.challenge import door, drawer, lid, place
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.scoring import default_table
from pluggybot.economy.tasks import CUBE_MINUTES, HAND_JOBS, HAND_MINUTES, KINDS, Task, TaskBoard
from pluggybot.home import storeroom
from pluggybot.lifecycle import (HUMANOID_HOME, HubLifecycle, task_producer, world_config,
                                 world_targets)
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import HISTORY


class _Mind:
  """A mind as the claim reads one: a library, so a procedure job may be
  taken, and nothing that decides."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  library = object()
  decisions: list = []
  model, backend = "none", "stub"                 # what the day's first lines name
  reasoning_effort = None                         # ...and how hard it thinks


def _counter_world():
  """The stub's floor with the storeroom's counter on it, compiled."""
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.texturedir = "models/"
  storeroom.attach(spec)
  model = spec.compile()
  return model, mujoco.MjData(model)


def _life(tmp_path, rebase: bool = True):
  """A humanoid world's lifecycle on a stub at the counter: its board (as a
  start-over loads it, or `rebase=False` as a continuation does), its
  ledger, a mind, the counter's activities on its seam."""
  model, data = _counter_world()
  body = StubBody(model, data, grid_bounds=(-3.0, -8.0, 30.0, 8.0))
  life = HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME, body=body,
                      tasks=TaskBoard(path=str(tmp_path / "tasks.json"), rebase=rebase),
                      ledger=Ledger(path=str(tmp_path / "ledger.json")), overseer=_Mind())
  acts = ActivitySet(storeroom.activities(model, data))
  body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  # ...its offers drawing past the world's compiled draw (0): a draw the
  # world already has proves nothing was drawn
  life.tasks.seq = max(life.tasks.seq, 5)
  return life


def _offer(life, kind: str, **params):
  """`kind` offered as the cadence offers it in the humanoids' house, its
  `params` as given over the cadence's."""
  maker = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  target = KINDS[kind].target_kind
  made, secret = maker._build(kind, target)
  return life.tasks.offer(kind, target, params={**made, **params}, secret=secret, ttl=3000.0,
                          t=float(life.data.time))


def _joint_q(life, name: str) -> int:
  return int(life.model.jnt_qposadr[life.model.joint(name).id])


def _grade(life, task_id: str) -> dict:
  life._done(ov.Decision(action="idle", reason="", done=task_id))
  life.body.run(life._grade_routine())
  return dict(life.tasks.get(task_id).verdict or {})


def test_an_offer_draws_its_props_dynamics_and_sets_it_back_out(tmp_path):
  """The offer is the moment the world is made to match it (#559): the
  drawer pushed back shut, its runners and its load drawn afresh off the
  board's own sequence -- the secret -- and its activity's truth with them."""
  life = _life(tmp_path)
  slide = _joint_q(life, "drawer_slide")
  life.data.qpos[slide] = 0.2                      # left open by the last job
  task = _offer(life, "open_drawer")
  k = task.secret["draw"]
  want = dr.draw(k)
  dof = int(life.model.jnt_dofadr[life.model.joint("drawer_slide").id])
  assert life.data.qpos[slide] == 0.0, "pushed back shut"
  assert life.model.dof_damping[dof] == pytest.approx(want.damping)
  assert life.model.dof_frictionloss[dof] == pytest.approx(want.friction)
  assert life._target_activity("drawer").runners == want
  # ...and the next offer of it draws another
  life.tasks.expire_due(1e9)
  again = _offer(life, "open_drawer")
  assert again.secret["draw"] != k


def test_a_redraw_is_the_prop_compiled_over_that_draw_and_keeps_the_extent():
  """`storeroom.redraw(k)` leaves the compiled world's dynamics where a world
  compiled over draw k has them -- masses, inertias, damping, friction,
  springs -- and the pinned statistic every camera's near plane is scaled
  by where it was (`bench.set_unknown_mass`'s trap)."""
  from pluggybot.humanoid import world as hw
  base = hw.home_spec().compile()
  data = mujoco.MjData(base)
  extent = float(base.stat.extent)
  for prop in storeroom.PROPS:
    storeroom.redraw(base, data, prop, 7)
  drawn = mujoco.MjSpec.from_string(STUB_WORLD)
  drawn.texturedir = "models/"
  storeroom.attach(drawn, draws={p: 7 for p in storeroom.PROPS})
  want = drawn.compile()
  for prop in storeroom.PROPS:
    for b in storeroom._subtree(want, prop):
      name = want.body(b).name
      got = base.body(name).id
      assert base.body_mass[got] == pytest.approx(want.body_mass[b]), name
      assert base.body_inertia[got] == pytest.approx(want.body_inertia[b]), name
      for j in range(int(want.body_jntadr[b]), int(want.body_jntadr[b] + want.body_jntnum[b])):
        jn = want.joint(j).name
        gj = base.joint(jn).id
        assert base.dof_damping[base.jnt_dofadr[gj]] == pytest.approx(
          want.dof_damping[want.jnt_dofadr[j]]), jn
        assert base.dof_frictionloss[base.jnt_dofadr[gj]] == pytest.approx(
          want.dof_frictionloss[want.jnt_dofadr[j]]), jn
        assert base.jnt_stiffness[gj] == pytest.approx(want.jnt_stiffness[j]), jn
  assert base.geom_friction[base.geom("cube").id][0] == pytest.approx(
    want.geom_friction[want.geom("cube").id][0])
  assert float(base.stat.extent) == extent, "the house's pinned extent"


def test_the_drawer_pulled_out_and_let_go_is_paid(tmp_path):
  """The grade on the seam (`_grade_hands_routine`): the drawer out at the
  call and still out 5 s later, nothing touching it -- the check's own
  verdict through `scoring.evaluate`, banked on the job's own row."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id) and task.id in life._job_watches
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  verdict = _grade(life, task.id)
  assert verdict["ok"], verdict["reason"]
  assert life.tasks.get(task.id).state == "done"
  assert life.ledger.robots[life.root]["balance"] == default_table()["open_drawer"].base
  assert task.id not in life._job_watches and not any(
    isinstance(h, drawer.Watch) for h in life.body.step_hooks), "the watch ends with its hold"


def test_a_drawer_left_shut_fails_and_pays_nothing(tmp_path):
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and "came out 0 cm" in verdict["reason"]
  assert life.ledger.robots.get(life.root, {}).get("balance", 0) == 0


def test_a_fall_since_the_claim_fails_the_job(tmp_path):
  """The watch is the job's from its CLAIM: a fall on the way there -- the
  body's own count -- fails a drawer that ends out (#544's rule), and ends
  the job (#701): a `done` said after it, before the seam has read it,
  grades nothing of its own."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.body.falls = 1
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  assert life._done(ov.Decision(action="idle", reason="", done=task.id)) is False
  # ...no time said for a fall the seam has not read yet (#701's review)
  assert any(ln.endswith(f"said done on {task.id} (open_drawer): not graded, because I fell, "
                         "and that ended it -- it is graded as things stand once I am up")
             for ln in life.thoughts.texts[HISTORY].splitlines())
  life.body.run(life.body.hold_routine(drawer.HOLD_S + 0.1))
  verdict = life.tasks.get(task.id).verdict
  assert not verdict["ok"] and verdict["reason"].startswith("you fell"), verdict


def _fall(life) -> float:
  """The robot falls now, by its body's own count (the stub's is what the
  test sets): when."""
  life.body.falls += 1
  return float(life.data.time)


def _history(life) -> list[str]:
  return life.thoughts.texts[HISTORY].splitlines()


def test_a_fall_while_a_job_is_held_ends_it_there_graded_as_the_world_stands(tmp_path):
  """#701: a fall fails a hand job's try (#544), but the job was graded only
  at a `done`, or failed when its minutes ran out (#602) -- held meanwhile,
  a cupboard stayed booked 985 s past the fall that had failed its job
  (11-12), and a drawer was opened as asked in a try a fall had lost
  (11-14). A fall ends it there: the fall its call, the hold a `done`'s
  grade takes from it, ONE verdict, the fall first and the world after it
  in its grader's words, History told at the fall, and its prop offered
  again. Shown to fail with no `_fall_step`: the job held on."""
  life = _life(tmp_path)
  life.producer = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2       # out as the job asks...
  mujoco.mj_forward(life.model, life.data)
  life.data.time += 30.0
  t = _fall(life)                                              # ...and the robot fell
  life.body.run(life.body.hold_routine(drawer.HOLD_S + 0.1))
  ended = life.tasks.get(task.id)
  assert ended.state == "failed"
  assert ended.verdict["reason"] == ("you fell, and a fall fails the try -- the drawer 20 cm "
                                     "out in 30 s, and let go")
  [grade] = life.grades
  assert grade["t"] == pytest.approx(t, abs=0.01), "its call is the fall"
  assert life.ledger.robots.get(life.root, {}).get("balance", 0) == 0
  lines = _history(life)
  fell = [i for i, ln in enumerate(lines) if ln.endswith(
    f"fell holding {task.id} (open_drawer): a fall fails the try, so the job ends here "
    "-- graded as things stand once I am up, 5 s on at the least")]
  failed = [i for i, ln in enumerate(lines) if ln.endswith(
    f"failed the job {task.id} (open_drawer): {ended.verdict['reason']}")]
  assert len(fell) == len(failed) == 1 and fell[0] < failed[0], lines
  assert task.id not in life._job_watches and not life._fallen
  assert not any(isinstance(h, drawer.Watch) for h in life.body.step_hooks)
  for _ in range(3):
    life.data.time += 600.0
    life._task_step()
  assert any(t.target == "drawer" and t.id != task.id for t in life.tasks.tasks.values())


@pytest.mark.parametrize("when", ["in its hold", "after it"])
def test_a_done_on_a_job_a_fall_ended_is_heard_and_grades_nothing_again(tmp_path, when):
  """...and a `done` said on it later -- the answer that says it in flight
  as the robot fell -- is a `done` on a job not held (#648's line), never a
  second verdict."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0 if when == "in its hold" else drawer.HOLD_S + 0.1))
  assert life._done(ov.Decision(action="idle", reason="", done=task.id)) is False
  assert life._grade_pending == ""
  life.body.run(life.body.hold_routine(drawer.HOLD_S + 0.1))
  assert len(life.grades) == 1 and len(life.verdicts) == 1
  [heard] = [ln for ln in _history(life)
             if f"said done on {task.id} (open_drawer): not graded, because " in ln]
  assert heard.endswith("because I fell at t=0, and that ended it -- it is graded as things "
                        "stand once I am up" if when == "in its hold" else
                        "because it is over -- I took it at t=0, and it failed at t=5"), heard


def test_a_fall_in_a_dones_hold_is_that_grades_and_ends_nothing_twice(tmp_path):
  """A job whose `done` is holding is ending already, and its grade says the
  fall first: a fall then begins no second end, banks nothing twice."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  said: list = []
  life.say_hooks.append(lambda t, msg: said.append(msg))
  t0 = float(life.data.time)

  def fall_in_the_hold():
    if not life.body.falls and float(life.data.time) > t0 + 2.0:
      _fall(life)
  life.body.step_hooks.insert(0, fall_in_the_hold)    # counted as the body steps
  verdict = _grade(life, task.id)
  assert verdict["reason"].startswith("you fell, and a fall fails the try -- "), verdict
  assert len(life.grades) == 1 and not life._fallen
  assert not any("closed already" in m for m in said), said
  assert not any("fell holding" in ln for ln in _history(life))


def test_a_fall_inside_a_run_ends_the_run_and_its_job(tmp_path):
  """Inside a procedure's run as outside one: the run ends at the fall
  (#671) and the job with it -- and a `done` said beside the run, its grade
  waiting on the run's end (#689), grades nothing: the fall's verdict is
  the one."""
  from pluggybot.mission.errand import programmed_errand
  from pluggybot.procedure.steps import Program, Step
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life._grade_pending = task.id                  # as a `done` beside the run leaves it
  t0 = float(life.data.time)

  def fall_at_two_s():
    if not life.body.falls and float(life.data.time) >= t0 + 2.0:
      _fall(life)
  life.body.step_hooks.insert(0, fall_at_two_s)       # counted as the body steps
  run = life.run_errand(programmed_errand(
    Program.single("p", [Step("wait", {"seconds": 10.0})])))["procedure"]
  assert not run["ok"] and run["steps"][0]["fell"], run
  t1 = float(life.data.time)
  assert t1 < t0 + 3.0, "the run ended at the fall"
  said: list = []
  life.say_hooks.append(lambda t, msg: said.append(msg))
  life.body.run(life._grade_routine())           # the loop's pass after the run
  assert f"GRADE {task.id}: a fall ended it -- graded as its hold ends" in said, said
  assert float(life.data.time) == t1 and life._grade_pending == "", "no hold of its own"
  life.body.run(life.body.hold_routine(drawer.HOLD_S + 0.1))
  ended = life.tasks.get(task.id)
  assert ended.state == "failed" and ended.verdict["reason"].startswith("you fell"), ended
  assert len(life.grades) == 1


def test_a_job_a_fall_ended_is_the_falls_past_its_minutes_too(tmp_path):
  """Its minutes running out in the hold, it is the fall's, in the fall's
  words, never "not graded"; a job no fall ended runs out as before (#602's
  tests)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.data.time += 58.0
  _fall(life)
  life.body.run(life.body.hold_routine(3.0))             # 61 s of its 60, in the hold
  life._run_out_held()
  assert life.tasks.get(task.id).state == "active", "the fall's, not the clock's"
  life.body.run(life.body.hold_routine(drawer.HOLD_S))
  verdict = life.tasks.get(task.id).verdict
  assert verdict["reason"].startswith("you fell, and a fall fails the try -- "), verdict


def test_a_fall_a_restart_cut_is_graded_once_after_it(tmp_path):
  """A restart between the fall and its verdict (#345): the fall's call
  rides the save -- its time and its reading -- and its watch is taken up
  with the save's readings as the world comes back, so the job is settled
  once, in its grader's words, the day's look round outlasting the hold or
  not; never "still mine" first and never called again (#701's reviews:
  called again, History said the fall twice and the grade's time was the
  restart's; left to `_resume_jobs`, after the look round, a hold ending
  first failed the job with nothing read)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  assert task.id in life._fallen
  t = life._fallen[task.id][0]
  state, arrays = life.kept_state()
  state = json.loads(json.dumps(state))         # ...as a save writes it
  board = tmp_path / "tasks.json"
  saved = board.read_bytes()
  for look_s, resumed in ((0.5, True), (drawer.HOLD_S + 1.0, False)):
    board.write_bytes(saved)                    # the board as the save left it
    again = _life(tmp_path)                     # ...and a world compiled anew
    again.restore_kept(state, arrays, in_place=True)
    again.body.run(again.body.hold_routine(look_s))   # the day's look round
    again._resume_jobs({"name": "pace", "taskId": task.id})
    again.body.run(again.body.hold_routine(drawer.HOLD_S))
    ended = again.tasks.get(task.id)
    assert ended.state == "failed" and ended.verdict["reason"] == (
      "you fell, and a fall fails the try -- the drawer came out 0 cm, short of the 15 it must")
    [grade] = again.grades
    assert grade["t"] == pytest.approx(t), "its call the fall's"
    assert not again._kept_watches and not again._fallen
    lines = _history(again)
    assert not any("still mine" in ln or "fell holding" in ln for ln in lines), lines
    said = (f"the restart cut short pace; my fall had ended the job {task.id} (open_drawer), "
            "graded as its hold ends" if resumed else
            f"the restart cut short pace; the job {task.id} is no longer mine")
    assert any(ln.endswith(said) for ln in lines), lines


def test_a_fall_in_its_hold_rides_a_restore_into_another_world(tmp_path):
  """...and a restore NOT in place -- another geometry, another map epoch:
  the jobs carried, the watches' readings not -- keeps the fall (#701's
  second review: forgotten, the job came back "still mine", a fresh watch
  read no fall, and a later `done` could pass a try a fall had lost)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  state, arrays = life.kept_state()
  again = _life(tmp_path)
  again.restore_kept(json.loads(json.dumps(state)), arrays, in_place=False)
  again._resume_jobs(None)
  again.body.run(again.body.hold_routine(drawer.HOLD_S + 0.1))
  ended = again.tasks.get(task.id)
  assert ended.state == "failed", ended
  assert ended.verdict["reason"].startswith("you fell, and a fall fails the try"), ended
  assert not any("still mine" in ln for ln in _history(again))


def test_a_falls_hold_lasts_until_the_body_stands_again(tmp_path):
  """The hold is the grade's "stand clear", and a body getting up beside the
  counter cannot: the job ends once both the hold and the get-up are over
  (#701's third review: settled at 5 s, its prop was free to be set out
  beside a get-up that may take 20 s)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.posture = "getting_up"
  life.body.run(life.body.hold_routine(drawer.HOLD_S + 1.0))
  assert life.tasks.get(task.id).state == "active", "settled under a body getting up"
  assert [h["id"] for h in life.tasks.held_context(life.root, 0.0)] == [task.id]
  life.body.posture = "standing"
  life.body.run(life.body.hold_routine(0.1))
  ended = life.tasks.get(task.id)
  assert ended.state == "failed" and ended.verdict["reason"].startswith("you fell"), ended


def test_a_falls_hold_never_ends_while_its_robot_lies_dead(tmp_path):
  """...nor while it lies dead (#701's fourth review): its body by the
  prop until its stand-up, a prop set out then is set out under it -- as a
  run-out waits for the stand-up (#602)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  life.dead = {"t": float(life.data.time), "cause": "stuck"}
  life.body.run(life.body.hold_routine(drawer.HOLD_S + 1.0))
  assert life.tasks.get(task.id).state == "active", "settled under its body, dead"
  life.dead = None                              # ...stood up
  life.body.run(life.body.hold_routine(0.1))
  ended = life.tasks.get(task.id)
  assert ended.state == "failed" and ended.verdict["reason"].startswith("you fell"), ended


def test_a_fallen_job_whose_watch_a_restart_cannot_take_up_is_failed(tmp_path, monkeypatch):
  """...and one whose kept watch the next world cannot take up -- its prop
  not where its watch reads it -- is failed as its hold ends, never held
  for good waiting on it (#701's third review)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  state, arrays = life.kept_state()
  again = _life(tmp_path)
  monkeypatch.setattr(again, "_watch_job", lambda task_id: None)
  again.restore_kept(json.loads(json.dumps(state)), arrays, in_place=True)
  again._resume_jobs(None)
  again.body.run(again.body.hold_routine(drawer.HOLD_S + 0.1))
  ended = again.tasks.get(task.id)
  assert ended.state == "failed" and ended.verdict["reason"].startswith("you fell"), ended
  assert not again._kept_watches and not again._fallen


def test_a_job_taken_again_is_a_try_of_its_own(tmp_path):
  """A job given back and taken again is a new try: no fall of the try
  before it (a fall's hold, its watch) ends it, and its `done` is graded on
  its own (#701's third review)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  life.tasks.release(task.id)
  life._kept_watches[task.id] = {"most": 0.0, "fastest": 0.0, "falls": 1}   # a save's, left
  assert life._claim_task(task.id)
  assert task.id not in life._fallen and life._job_watches[task.id].falls() == 0
  assert task.id not in life._kept_watches
  verdict = _grade(life, task.id)
  assert verdict["reason"] == "the drawer came out 0 cm, short of the 15 it must", verdict


def test_a_watch_left_on_a_job_given_back_ends_nothing_of_the_others(tmp_path):
  """A watch outlives a job given back by up to a sweep: a fall then ends
  nothing the robot no longer holds -- the other robot's take of it least
  of all (#701's second review)."""
  from pluggybot.robot import SECOND
  luca = _life(tmp_path)
  body = StubBody(luca.model, luca.data, handle=SECOND, grid_bounds=(-3.0, -8.0, 30.0, 8.0))
  rowan = HubLifecycle(luca.model, luca.data, realtime=False, world=HUMANOID_HOME, body=body,
                       tasks=luca.tasks, ledger=luca.ledger, overseer=_Mind(), handle=SECOND)
  task = _offer(luca, "open_drawer")
  assert luca._claim_task(task.id)
  luca.tasks.release(task.id)
  assert rowan._claim_task(task.id) and task.id in luca._job_watches
  _fall(luca)
  assert not luca._fell_holding(task.id), "the fall's to settle, a job Rowan holds"
  luca.body.step_hooks[luca.body.step_hooks.index(luca._fall_step)]()
  assert not luca._fallen and not any("fell holding" in ln for ln in _history(luca))
  assert luca.tasks.get(task.id).state == "active"
  assert luca._done(ov.Decision(action="idle", reason="", done=task.id)) is False
  [heard] = [ln for ln in _history(luca) if "said done on" in ln]
  assert "I fell" not in heard and "took it" in heard, heard
  luca.data.time = max(float(luca.data.time), luca._next_task_check) + 1.0
  luca._task_step()                             # ...and the board's next sweep
  assert task.id not in luca._job_watches, "a watch on Rowan's job, saved with Luca's"
  assert task.id in rowan._job_watches


def test_a_job_settled_in_a_falls_hold_is_heard_as_over(tmp_path):
  """A job settled some other way inside a fall's hold (a true death's)
  leaves it: a `done` on it then hears it is over, never that it is being
  graded (#701's second review)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  life._fail_held(life.tasks.get(task.id), "the robot that took it ran out of lives")
  assert not life._fallen
  assert life._done(ov.Decision(action="idle", reason="", done=task.id)) is False
  [heard] = [ln for ln in _history(life)
             if f"said done on {task.id} (open_drawer): not graded, because " in ln]
  assert "because it is over -- I took it at t=0, and it failed at t=1" in heard, heard


def test_a_fall_kept_with_no_save_of_its_call_ends_its_job_after_a_restart(tmp_path):
  """...and a save that kept the watch's fall and not its call (one from
  before #701's review) ends the job at the step after it is taken up --
  the day's look round having stepped the world first."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  state, arrays = life.kept_state()
  del state["fallen"]
  again = _life(tmp_path)
  again.restore_kept(state, arrays, in_place=True)
  again.body.run(again.body.hold_routine(0.5))
  again._resume_jobs(None)
  assert not any("still mine" in ln for ln in _history(again))
  again.body.run(again.body.hold_routine(drawer.HOLD_S + 0.1))
  ended = again.tasks.get(task.id)
  assert ended.state == "failed" and ended.verdict["reason"].startswith("you fell"), ended
  assert len(again.grades) == 1


def test_a_job_a_fall_ended_whose_watch_is_gone_is_failed_never_dropped(tmp_path):
  """#701's review: a fall's hold that found its watch gone returned with
  the job still held, its fall forgotten -- shown held again, and a later
  `done` watched afresh could pass a try a fall had lost."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  life._unwatch_job(task.id)
  life.body.run(life.body.hold_routine(drawer.HOLD_S))
  ended = life.tasks.get(task.id)
  assert ended.state == "failed"
  assert ended.verdict["reason"].startswith("you fell, and a fall fails the try"), ended


def test_a_fall_ends_its_own_robots_job_and_never_the_others(tmp_path):
  """Two robots, one board: Luca's fall ends Luca's job, and Rowan's, held
  beside it, is Rowan's still -- watched, offered to nobody."""
  from pluggybot.robot import SECOND
  luca = _life(tmp_path)
  body = StubBody(luca.model, luca.data, handle=SECOND, grid_bounds=(-3.0, -8.0, 30.0, 8.0))
  rowan = HubLifecycle(luca.model, luca.data, realtime=False, world=HUMANOID_HOME, body=body,
                       tasks=luca.tasks, ledger=luca.ledger, overseer=_Mind(), handle=SECOND)
  mine, theirs = _offer(luca, "open_drawer"), _offer(luca, "open_door")
  assert luca._claim_task(mine.id) and rowan._claim_task(theirs.id)
  _fall(luca)
  luca.body.run(luca.body.hold_routine(drawer.HOLD_S + 0.1))
  rowan.body.run(rowan.body.hold_routine(1.5))
  assert luca.tasks.get(mine.id).state == "failed"
  assert luca.tasks.get(theirs.id).state == "active" and theirs.id in rowan._job_watches
  assert rowan.grades == [] and not rowan._fallen


def test_a_job_a_fall_ended_is_shown_held_no_longer(tmp_path):
  """...and from the fall it is no longer among the jobs it is shown it
  holds (`heldTasks`): the fall ends the run, the robot is asked within the
  hold, and its turn said "the job ends here" in History beside the job
  listed held, with its time left. Shown to fail with `overseer_context`
  listing every held job."""
  from pluggybot.lifecycle import overseer_context, world_facts
  from pluggybot.procedure import library as lib
  life = _life(tmp_path)
  life.overseer.menu = None                      # ...a mind with no eye: no `look`
  life.overseer.library = lib.Library(world_facts(HUMANOID_HOME))
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  assert [h["id"] for h in overseer_context(life)["heldTasks"]] == [task.id]
  _fall(life)
  life.body.run(life.body.hold_routine(1.0))
  assert life.tasks.get(task.id).state == "active", "in the fall's hold"
  assert "heldTasks" not in overseer_context(life)


def test_a_job_called_past_its_minutes_fails(tmp_path):
  """The time a job gives runs from its claim (`minutes`), whatever the
  robot did in between: its `done` waiting on the queue, the grade's call
  came 1 s on."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=0.01)
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  life.body.run(life.body.hold_routine(1.0))
  life.body.run(life._grade_routine())
  verdict = life.tasks.get(task.id).verdict
  # 0.6 s at 0.01 minutes: said "past the 1 s" before #737
  assert not verdict["ok"] and ("called 1.0 s after the start, past the 0.6 s the job allows"
                                in verdict["reason"]), verdict["reason"]


class _Stop(Exception):
  pass


def _day_begun(life) -> None:
  """`begin` and its day routine driven up to the loop's first pass."""
  def stop():
    raise _Stop
  life.at_loop_top.append(stop)
  with pytest.raises(_Stop):
    life.body.run(life.begin(world_config(life.world)["start"]))


def test_a_job_held_past_its_minutes_is_failed_and_its_prop_offered_again(tmp_path):
  """#602: held, a job ended only on a `done` that could only fail it, and
  booked its prop for good -- three held booked all three of the counter's,
  and nothing was offered from 02:22 UTC on 11-03. Past its minutes it is
  failed at its holder's next pass of the day loop, never on the seam:
  freed there, its prop was set out from under the procedure still working
  it. Its watch is unhooked, its holder's History told, and its prop
  offered again."""
  life = _life(tmp_path)
  life.producer = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.data.time += 59.0
  life._run_out_held()
  assert life.tasks.get(task.id).state == "active", "its minute is not yet past"
  life.body.run(life.body.hold_routine(2.0))
  assert life.tasks.get(task.id).state == "active", "the seam runs nothing out"
  life._run_out_held()
  ended = life.tasks.get(task.id)
  assert ended.state == "failed"
  assert ended.verdict["reason"].startswith("not graded within the 60 s the job allows")
  assert task.id not in life._job_watches
  assert not any(isinstance(h, drawer.Watch) for h in life.body.step_hooks)
  assert (f"failed the job {task.id} (open_drawer): not graded within"
          in life.thoughts.texts[HISTORY])
  for _ in range(3):
    life.data.time += 600.0
    life._task_step()
  assert any(t.target == "drawer" and t.id != task.id for t in life.tasks.tasks.values())


def test_the_day_loops_pass_runs_out_a_job_held_past_its_minutes(tmp_path, monkeypatch):
  """...wired into the loop's pass, after the board's sweep, and before
  anything the robot does next."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)

  def a_minute_on():
    assert life.tasks.get(task.id).state == "active", "in time as the day began"
    life.data.time += 61.0

  def stop():
    raise _Stop
  monkeypatch.setattr(life, "_visitor_step", a_minute_on)
  monkeypatch.setattr(life, "_metabolism_step", stop)
  with pytest.raises(_Stop):
    life.body.run(life.begin(world_config(life.world)["start"]))
  assert life.tasks.get(task.id).state == "failed"


def _trip(life, walk_s: float) -> None:
  """A charge trip's walk and charge on the stub, each taking time as a
  real one does (the stub's own take none)."""
  def walk(free=False):
    yield from life.body.hold_routine(walk_s)
    return True

  def charge():
    yield from life.body.hold_routine(1.0)
  life.go_charge_routine = walk
  life.charge_routine = charge


@pytest.mark.parametrize("path", ["loop", "decided"])
def test_on_a_charge_trip_the_seam_runs_a_job_out(tmp_path, path):
  """...and on a charge trip, the loop's or a decided one, away from the
  counter with its hands home, the seam fails it: left to the loop's pass,
  a 74-minute charge held t_0024 and its drawer 69 minutes past its limit
  (#602, read live on 11-03). The trip over, the seam fails nothing."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.data.time += 59.0
  _trip(life, walk_s=2.0)
  if path == "loop":
    life.body.run(life._charge_trip_routine())
  else:
    life._after_decision(ov.Decision(action="charge", reason=""))
  assert life.tasks.get(task.id).state == "failed"
  assert not life._charge_trip and life.state == "CHARGE", "the trip over, its state left"


def test_a_robot_dead_on_a_charge_trip_has_its_jobs_left_for_its_stand_up(tmp_path):
  """...but not while it lies dead there: as anywhere else, its jobs wait
  for the loop's first pass after its stand-up."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.data.time += 59.0
  life._charge_trip = True
  life.dead = {"t": float(life.data.time), "cause": "unminded"}
  life.body.run(life.body.hold_routine(2.0))
  assert life.tasks.get(task.id).state == "active"


def test_a_job_closed_since_its_grade_began_is_banked_nowhere(tmp_path):
  """A job is settled once (#602's review): closed some other way while
  its grade held, it banks no second verdict."""
  from pluggybot.economy.scoring import evaluate
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)

  def close_once():
    if life.tasks.get(task.id).open:
      life.tasks.resolve(task.id, evaluate("open_drawer", {}), t=float(life.data.time))
  life.body.step_hooks.append(close_once)
  _grade(life, task.id)
  assert life.verdicts == [] and life.ledger.robots.get(life.root, {}).get("balance", 0) == 0


@pytest.mark.parametrize("opened", [False, True])
def test_a_done_and_its_grade_are_said_in_history_by_the_jobs_id(tmp_path, opened):
  """#629: a `done` was said to the log alone, and its grade's line named
  the job's kind, never its id -- Rowan took a door's grade, come at a
  `done` 37 s after its take, for another job's. History says the `done`,
  and the grade names its job as a run-out's line does, passed or not."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  if opened:
    life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
    mujoco.mj_forward(life.model, life.data)
  verdict = _grade(life, task.id)
  assert verdict["ok"] is opened, verdict["reason"]
  lines = life.thoughts.texts[HISTORY].splitlines()
  said = [i for i, ln in enumerate(lines) if ln.endswith(
    f"said done on {task.id} (open_drawer): graded now, before this answer's idle")]
  graded = [i for i, ln in enumerate(lines) if ln.endswith(
    f"{'passed' if opened else 'failed'} the job {task.id} (open_drawer): {verdict['reason']}")]
  assert len(said) == len(graded) == 1 and said[0] < graded[0], lines


def _holds(life) -> list:
  """Every hold the body is asked from here on, as (when it began, how
  long, how many grades were in by then): what came before what."""
  asked, hold = [], life.body.hold_routine

  def spy(seconds, *a, **kw):
    asked.append((float(life.data.time), float(seconds), len(life.grades)))
    return hold(seconds, *a, **kw)
  life.body.hold_routine = spy
  return asked


def test_a_done_is_graded_before_its_answers_action(tmp_path):
  """#689: a `done` waited out its answer's action before its grade -- all
  15 said with `idle` from 11-01 to 11-12 were graded a minute after the
  word -- where every hand job says it is graded when you say you are done.
  It is graded at the word, before the answer's idle holds."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  holds = _holds(life)
  t = float(life.data.time)
  life.body.run(life._after_decision_routine(
    ov.Decision(action="idle", reason="", done=task.id)))
  [grade] = life.grades
  assert grade["task"] == task.id and grade["ok"], grade
  assert grade["t"] == pytest.approx(t, abs=1e-3), "graded after the idle, not at the word"
  idle = [h for h in holds if h[1] == life.idle_s]
  assert idle and idle[0][2] == 1, "the idle held before the grade"
  assert life._grade_pending == ""
  assert any(ln.endswith(f"said done on {task.id} (open_drawer): graded now, before this "
                         "answer's idle") for ln in life.thoughts.texts[HISTORY].splitlines())


def test_a_done_beside_a_procedure_is_graded_once_the_run_is_over(tmp_path):
  """...but an answer that runs a procedure is graded once its run is over
  (#689): the run is the work, and graded at the word it would grade the
  work undone."""
  from pluggybot.lifecycle import world_facts
  from pluggybot.procedure import library as lib
  life = _life(tmp_path)
  life.overseer.library = lib.Library(world_facts(HUMANOID_HOME))
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.body.run(life._after_decision_routine(ov.Decision(
    action=ov.PROCEDURE_NEW, reason="", done=task.id,
    define={"name": "pace", "source": "def pace():\n  wait(1)\n"})))
  assert life.grades == [] and life._grade_pending == task.id
  assert len(life.errands) == 1, "the run is queued"
  # ...and History says the `done` once that run begins (#706): one that
  # never begins is not graded, and never promised a grade
  said = f"said done on {task.id} (open_drawer): graded once everything queued has run"
  assert not any(ln.endswith(said) for ln in _history(life))
  life.body.run(life.run_errand_routine(life.errands.pop(0)))
  assert life.grades == [] and life._grade_pending == task.id
  assert sum(ln.endswith(said) for ln in _history(life)) == 1


def test_a_dead_robots_done_is_left_to_the_loops_pass(tmp_path):
  """...and the answer of a robot that died while it thought leaves its
  grade to the loop's pass after the stand-up, as every grade was (#689):
  its answer runs under the stand-up's close, and a grade closed mid-hold
  is lost."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.dead = {"t": float(life.data.time), "cause": "flat"}
  life.body.run(life._after_decision_routine(
    ov.Decision(action="idle", reason="", done=task.id)))
  assert life.grades == [] and life._grade_pending == task.id


#: A define the library refuses (no such verb), and one it takes.
_REFUSED = {"name": "pace", "source": "def pace():\n  hover(1)\n"}
_RUNS = {"name": "pace", "source": "def pace():\n  wait(1)\n"}


def _minded(tmp_path):
  """`_life` with a procedure library, holding a drawer's job."""
  from pluggybot.lifecycle import world_facts
  from pluggybot.procedure import library as lib
  life = _life(tmp_path)
  life.overseer.library = lib.Library(world_facts(HUMANOID_HOME))
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  return life, task


def _answered(life, **answer):
  """An answer acted on, then the day loop's passes until it asks again."""
  life.body.run(life._after_decision_routine(ov.Decision(reason="", **answer)))
  _drained(life)


def _drained(life):
  """The day loop's passes until it would ask again: what the answer queued
  run, then the grade it left pending."""
  while life.errands:
    life.body.run(life.run_errand_routine(life.errands.pop(0)))
  if life._grade_pending:
    life.body.run(life._grade_routine())


def _not_graded(life, task) -> list:
  return [ln for ln in _history(life) if f"said done on {task.id} (open_drawer): not graded" in ln]


@pytest.mark.parametrize("never", ["refused", "arguments", "undefined"])
def test_a_done_beside_a_run_that_never_began_is_not_graded(tmp_path, never):
  """#706 (Ben's call of 11-16): on 11-14 `procedure:new` ran nothing 22
  times, its define refused, and 4 of the 6 answers that said `done` too had
  their jobs graded 4-10 s later on a prop nothing had worked (t_0301,
  t_0316, t_0321, t_0322). A `done` beside a run that never began -- no
  errand built for it, whatever stopped it -- is not graded: one History
  line says so, and the job stays held."""
  life, task = _minded(tmp_path)
  if never == "refused":
    answer = {"action": ov.PROCEDURE_NEW, "define": _REFUSED}
  elif never == "arguments":
    answer = {"action": ov.PROCEDURE_NEW,
              "define": {"name": "pace", "source": "def pace(s):\n  wait(s)\n"}}
  else:
    life.overseer.library.define("pace", _RUNS["source"], t=0.0)
    answer = {"action": ov.PROCEDURE_PREFIX + "pace", "undefine": "pace"}
  _answered(life, done=task.id, **answer)
  assert life.grades == [] and life.verdicts == []
  assert life.tasks.get(task.id).state == "active", "still held"
  assert life._grade_pending == "" and not life._grade_waits
  from pluggybot.lifecycle import NEVER_BEGAN
  [line] = [ln for ln in _history(life) if f"said done on {task.id}" in ln]
  assert line.endswith(f"said done on {task.id} (open_drawer): not graded, because {NEVER_BEGAN}")


def test_a_done_beside_a_run_refused_before_its_first_verb_is_not_graded(tmp_path, monkeypatch):
  """...nor one whose run its own validation refused as it began (the
  `refused` a procedure's event says): no verb was called."""
  from pluggybot.procedure import lang
  life, task = _minded(tmp_path)
  life.body.run(life._after_decision_routine(ov.Decision(
    action=ov.PROCEDURE_NEW, reason="", done=task.id, define=_RUNS)))
  assert len(life.errands) == 1
  monkeypatch.setattr(lang, "validate", lambda proc, facts: ["refused as it began"])
  _drained(life)
  [run] = [r["procedure"] for r in life.errand_results]
  assert run["program"] == "pace" and run["refused"] == ["refused as it began"]
  assert life.grades == [] and life.tasks.get(task.id).state == "active"
  assert len(_not_graded(life, task)) == 1


def test_a_done_waits_on_its_own_run_never_another(tmp_path, monkeypatch):
  """...the run it waits on is the one its answer built: another errand's
  run validated ahead of it -- a preset's after a restart, say -- does not
  end the wait."""
  from pluggybot.lifecycle import errand_from
  from pluggybot.procedure import lang
  life, task = _minded(tmp_path)
  life.overseer.library.define("other", "def other():\n  wait(1)\n", t=0.0)
  life.body.run(life._after_decision_routine(ov.Decision(
    action=ov.PROCEDURE_NEW, reason="", done=task.id, define=_RUNS)))
  life.errands.insert(0, errand_from(ov.Decision(action=ov.PROCEDURE_PREFIX + "other"),
                                     library=life.overseer.library))
  validate = lang.validate
  monkeypatch.setattr(lang, "validate", lambda proc, facts: (
    ["refused as it began"] if proc.name == "pace" else validate(proc, facts)))
  _drained(life)
  assert [r["procedure"]["completed"] for r in life.errand_results] == [1, 0]
  assert life.grades == [] and life.tasks.get(task.id).state == "active"
  assert len(_not_graded(life, task)) == 1


def test_a_done_whose_run_never_began_on_a_job_ended_meanwhile_says_so(tmp_path):
  """...and one whose job ended while it waited -- here failed in the
  other's records' way, by the board -- still has its one History line,
  saying why it is not graded (#648's words)."""
  from pluggybot.economy.scoring import evaluate
  life, task = _minded(tmp_path)
  life.body.run(life._after_decision_routine(ov.Decision(
    action=ov.PROCEDURE_NEW, reason="", done=task.id, define=_REFUSED)))
  life.tasks.resolve(task.id, evaluate("open_drawer", {}), t=float(life.data.time))
  _drained(life)
  [line] = [ln for ln in _history(life) if f"said done on {task.id}" in ln]
  assert f"said done on {task.id} (open_drawer): not graded, because it is over" in line, line


def test_a_done_beside_a_run_that_never_began_with_its_time_up_runs_it_out(tmp_path):
  """...and one whose time is already up is not told it still holds the
  job: it runs out there, its line the run-out's (#602)."""
  life, task = _minded(tmp_path)
  held = life.tasks.get(task.id)
  life.data.time = held.claimed_t + held.limit_s + 1.0
  _answered(life, action=ov.PROCEDURE_NEW, done=task.id, define=_REFUSED)
  ended = life.tasks.get(task.id)
  assert ended.state == "failed" and len(life.grades) == 1
  assert ended.verdict["reason"].startswith(f"not graded within the {held.limit_s:.0f} s")
  assert _not_graded(life, task) == []


def test_a_done_not_graded_leaves_its_job_held_and_its_time_running(tmp_path):
  """...the job stays the robot's, and its time runs on (#602): past it, it
  is failed as any held job is."""
  life, task = _minded(tmp_path)
  _answered(life, action=ov.PROCEDURE_NEW, done=task.id, define=_REFUSED)
  held = life.tasks.get(task.id)
  assert held.state == "active" and held.id in [t.id for t in life.tasks.held_by(life.root)]
  life.data.time = held.claimed_t + held.limit_s + 1.0
  life._run_out_held()
  ended = life.tasks.get(task.id)
  assert ended.state == "failed"
  assert ended.verdict["reason"].startswith(f"not graded within the {held.limit_s:.0f} s")


def test_a_done_said_again_after_one_not_graded_is_graded(tmp_path):
  """...and a `done` said again is graded as any is: here at the word,
  beside an idle (#689), the drawer out."""
  life, task = _minded(tmp_path)
  _answered(life, action=ov.PROCEDURE_NEW, done=task.id, define=_REFUSED)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  _answered(life, action="idle", done=task.id)
  [grade] = life.grades
  assert grade["task"] == task.id and grade["ok"], grade


def test_a_done_beside_a_run_that_began_is_graded_once_it_is_over(tmp_path):
  """...but a `done` beside a run that began is graded once it is over, as
  it was (#689): the run is the work."""
  life, task = _minded(tmp_path)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  _answered(life, action=ov.PROCEDURE_NEW, done=task.id, define=_RUNS)
  [run] = [r["procedure"] for r in life.errand_results]
  assert run["program"] == "pace" and run["completed"] == 1, "the run ran"
  [grade] = life.grades
  assert grade["task"] == task.id and grade["ok"], grade
  assert _not_graded(life, task) == []


def test_a_done_waiting_on_a_run_that_never_began_is_kept_across_a_restart(tmp_path):
  """...and a restart before its run began keeps it waiting (`gradeWaits`):
  the restart drops the errand, so that run never begins -- read back
  without it, the grade was made on the bare prop."""
  life, task = _minded(tmp_path)
  life.body.run(life._after_decision_routine(ov.Decision(
    action=ov.PROCEDURE_NEW, reason="", done=task.id, define=_RUNS)))
  assert life._grade_pending == task.id and len(life.errands) == 1
  state, arrays = life.kept_state()
  again = _life(tmp_path)
  again.restore_kept(json.loads(json.dumps(state)), arrays, in_place=True)
  again._resume_jobs(None)
  assert again.errands == [], "a restart drops an answer's queued run"
  again.body.run(again._grade_routine())
  assert again.grades == [] and again.tasks.get(task.id).state == "active"
  assert len(_not_graded(again, task)) == 1


def test_a_dead_robots_done_beside_a_run_that_never_began_is_not_graded(tmp_path):
  """...and the answer of a robot that died while it thought, its run never
  begun, is not graded at the loop's pass after its stand-up either."""
  life, task = _minded(tmp_path)
  life.dead = {"t": float(life.data.time), "cause": "flat"}
  life.body.run(life._after_decision_routine(ov.Decision(
    action=ov.PROCEDURE_NEW, reason="", done=task.id, define=_REFUSED)))
  assert life.grades == [] and life._grade_pending == task.id
  life.dead = None
  _drained(life)
  assert life.grades == [] and life.tasks.get(task.id).state == "active"
  assert len(_not_graded(life, task)) == 1


def test_the_rule_says_a_done_beside_a_run_that_never_began_is_not_graded():
  """#706: the rule says it, beside #689's words (kept as they were)."""
  flat = " ".join(ov.CHALLENGE_RULE.split())
  assert ("A `done` beside a run that never begins -- a `procedure:new` whose define was "
          "refused, say -- is not graded: the job stays yours, its time runs on, and a "
          "`done` said again is graded.") in flat


def test_each_robot_is_shown_the_jobs_it_holds_and_the_time_each_has_left():
  """#602: a robot was shown an offer and nothing of the job once it took
  it, and spent its recalls finding out when it had ("I'm holding the
  open_chest job but I don't actually know when I took it", Rowan, 11-03).
  While it holds one, its context lists it (`heldTasks`): what it is,
  where, and `leftS`; its own alone, the board being shared -- and the
  rule names what the context carries."""
  from test_two_minds import stub_pair
  from pluggybot.lifecycle import overseer_context
  a, b = stub_pair(tasks=True)
  drawer_job = a.tasks.offer("open_drawer", "drawer", t=0.0)
  door_job = a.tasks.offer("open_door", "door", t=0.0)
  assert "heldTasks" not in overseer_context(a)
  assert a._claim_task(drawer_job.id) and b._claim_task(door_job.id)
  a.data.time += 100.0
  held = ov.model_state(overseer_context(a))["heldTasks"]
  assert [h["id"] for h in held] == [drawer_job.id]
  assert held[0]["leftS"] == 60.0 * HAND_MINUTES - 100.0
  assert held[0]["description"] == drawer_job.description
  assert [h["id"] for h in overseer_context(b)["heldTasks"]] == [door_job.id]
  assert "`heldTasks`" in ov.CHALLENGE_RULE and "`leftS`" in ov.CHALLENGE_RULE


def test_the_rule_says_a_done_is_graded_at_once_unless_its_answer_runs_a_procedure():
  """#689's quiz: told "the grade happens once everything you queued has
  finished", 9 of 12 asked read an answer's idle and its charge as queued
  and the grade as coming after them, where the code grades first; told
  this, 12 of 12 read it right (loop-1113's L2)."""
  flat = " ".join(ov.CHALLENGE_RULE.split())
  assert ("on the same answer as the run if you like, and the grade then waits until that run "
          "has finished; on an answer that runs no procedure, the grade comes at once, before "
          "that answer's action.") in flat
  assert "everything you queued" not in flat


def test_the_rule_says_left_s_is_the_asks_and_the_clock_runs_while_it_thinks():
  """#689: `leftS` is the board's as the turn is built, and the clock runs
  on through the answer: Rowan, asked with 193 s of t_0202 left, answered
  160 s later with 33 s left, its own think uncounted ("my inter-ask gaps
  run ~170 s"), and its run's grade came 5 s past the job's time."""
  flat = " ".join(ov.CHALLENGE_RULE.split())
  assert ("its `leftS` the seconds of that time still left when you were asked; the clock "
          "runs on while you think.") in flat


def test_the_rule_says_a_fall_ends_a_held_challenge_and_each_job_says_it_fails():
  """#701: where the rule says when a job is graded, it says a fall ends it
  -- a fact of the world, the moment of the grade -- and every hand job's
  own sentence says it is done without falling: the two agree."""
  flat = " ".join(ov.CHALLENGE_RULE.split())
  assert ("A claimed challenge stands until you say done, until the time its description "
          "gives is up, or until you fall: a fall while you hold one ends it there, graded as "
          "if you had said done as you fell, and a fall fails it.") in flat
  for kind in HAND_JOBS:
    assert "without falling" in TaskBoard().offer(kind, KINDS[kind].target_kind).description, kind


@pytest.mark.parametrize("left", ["", "CHARGE", "GO_CHARGE"])
def test_a_grade_holding_past_the_limit_is_never_cut_short(tmp_path, left):
  """A grade is its grader's (#602): called at 58 s of 60, the drawer held
  out through the hold that ends at 63 s passes. Run out on the seam, the
  hold was cut, the pass lost and the job banked twice -- after a charge
  too, its state left behind (`done` and `charge` on one answer: #609's
  review)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  life.data.time += 58.0
  if left:
    life.state = left
  verdict = _grade(life, task.id)
  assert verdict["ok"], verdict["reason"]
  assert len(life.verdicts) == 1


def test_a_done_pending_past_the_limit_is_its_grades_to_judge(tmp_path):
  """...and a `done` waiting on the queue is its grade's too, judged in
  the grade's own words at its call, never "not graded"."""
  life = _life(tmp_path)
  task = _offer(life, "open_chest", minutes=1)
  assert life._claim_task(task.id)
  life.data.time += 58.0
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  life.data.time += 5.0
  life._run_out_held()
  assert life.tasks.get(task.id).state == "active"
  life.body.run(life._grade_routine())
  verdict = life.tasks.get(task.id).verdict
  assert "called 63 s after the start, past the 60 s the job allows" in verdict["reason"]


def test_a_start_over_moves_a_held_claim_and_fails_one_out_of_time_as_the_day_begins(tmp_path):
  """A start-over moves a held claim with the clock (#602, the board's
  `rebase`): held 100 s of its 60 when the world began again at 0, it is
  failed as its jobs are taken up, after the day's first line and never
  "still mine" first -- not during the look round that seeds the map,
  before the robot has woken."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.data.time = 100.0
  _offer(life, "open_door")                     # the board's last change, at 100 s
  again = _life(tmp_path)                       # the same board, a world begun at 0
  assert again.tasks.get(task.id).claimed_t == -100.0
  # ...its look round stepping the world, as the body's does (the stub's is none)
  again.body.look_around_routine = lambda: again.body.hold_routine(2.0)
  _day_begun(again)
  assert again.tasks.get(task.id).state == "failed"
  lines = again.thoughts.texts[HISTORY].splitlines()
  woke = next(i for i, ln in enumerate(lines) if "woke up" in ln)
  failed = next(i for i, ln in enumerate(lines) if f"failed the job {task.id}" in ln)
  assert woke < failed and not any("still mine" in ln for ln in lines)


def test_a_start_overs_moved_clock_is_written_as_the_day_begins(tmp_path):
  """...and written down then (`begin`): moved in memory alone, a restart
  that carried on before the board's next change read the old clock back,
  a claim ahead of the clock again, and kept its `simTime` for the next
  start-over to move every claim past its limit by."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.data.time = 100.0
  _offer(life, "open_door")
  again = _life(tmp_path)
  again.begin(world_config(HUMANOID_HOME)["start"])
  kept = TaskBoard(path=str(tmp_path / "tasks.json"), rebase=False)
  assert kept.get(task.id).claimed_t == -100.0 and kept.sim_t == 0.0


def test_a_true_death_ends_the_jobs_its_robot_held_in_its_own_records(tmp_path):
  """A true death fails the jobs its robot held, before its records go
  (#602): kept under its root, one ran out into the next robot's History,
  ledger and event map, whose only inheritance is being told it is not the
  first."""
  from pluggybot.economy.ledger import HEARTS
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  for _ in range(HEARTS - 1):
    life.ledger.lose_heart(life.root)
  life._die("flat", "the pack ran out")
  assert life.true_deaths
  ended = life.tasks.get(task.id)
  assert ended.state == "failed"
  assert ended.verdict["reason"].startswith("the robot that took it ran out of lives")
  assert task.id not in life._job_watches
  assert "failed the job" not in life.thoughts.texts[HISTORY]


def test_a_bare_offer_gives_its_kinds_minutes_and_is_graded_on_them(tmp_path):
  """#580: the time a job gives is its kind's (`TaskKind.minutes`). Kept
  by the cadence alone, an offer made any other way -- `TaskBoard.offer`
  in a script or a test -- read "within  minutes", and its grade fell back
  on the quadruped's 300 s (`lid.TIME_LIMIT_S`). An offer that carries
  none, saved before its kind did, is graded on the kind's."""
  for kind in HAND_JOBS:
    task = TaskBoard().offer(kind, KINDS[kind].target_kind)
    assert f"within {KINDS[kind].minutes} minutes" in task.description, kind
    assert task.params["minutes"] == KINDS[kind].minutes
  assert {KINDS[k].minutes for k in HAND_JOBS
          if k not in ("set_table", "place_cube", "set_table_pass", "set_table_shelf")} \
    == {HAND_MINUTES}
  life = _life(tmp_path)
  task = life.tasks.offer("open_drawer", "drawer", ttl=3000.0, t=float(life.data.time))
  life.tasks.tasks[task.id] = replace(task, params={})
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  verdict = _grade(life, task.id)
  assert verdict["ok"] and verdict["metrics"]["limitS"] == 60.0 * HAND_MINUTES


def test_the_cube_gives_its_own_minutes_and_is_paid_for_them():
  """Offered on the table's terms (#560; Ben, 2026-11-10): three times its reference's slowest
  pass over the tour, and 90 -- the counter's jobs' pay for their 16 minutes -- times its
  minutes over 16, rounded, so a robot choosing by the minute is not steered off it."""
  task = TaskBoard().offer("place_cube", "cube")
  assert task.params["minutes"] == KINDS["place_cube"].minutes == CUBE_MINUTES
  assert f"within {CUBE_MINUTES} minutes" in task.description
  assert default_table()["place_cube"].base == round(90 * CUBE_MINUTES / HAND_MINUTES)


def test_a_restart_puts_back_each_props_newest_draw(tmp_path):
  """The world file carries each prop's compiled draw; an offer that came
  back off the board still means the one it drew -- put back as the day
  begins (`begin` -> `restore_jobs`), before anything reads the prop."""
  life = _life(tmp_path)
  task = _offer(life, "open_door")
  k = task.secret["draw"]
  fresh = _life(tmp_path)                       # the same board, a world compiled anew
  dof = int(fresh.model.jnt_dofadr[fresh.model.joint("door_hinge").id])
  assert fresh.model.dof_damping[dof] == pytest.approx(dd.draw(0).damping)
  fresh.begin((0.0, 0.0, 0.0))
  assert fresh.model.dof_damping[dof] == pytest.approx(dd.draw(k).damping)
  assert fresh._target_activity("door").hinge == dd.draw(k)


def test_the_draw_put_back_is_the_newest_offers_on_the_boards_count(tmp_path):
  """...newest on the board's own count, never by its stamp (#610): an
  offer on the door lapsed before a start-over is stamped on the clock
  before it, ahead of the door's offer since, and by `createdT` every
  restart put the old offer's draw back on the door."""
  life = _life(tmp_path)
  life.data.time = 356_727.0
  old = _offer(life, "open_door")
  life.tasks.expire_due(359_727.0)              # lapsed, on the clock before
  again = _life(tmp_path)                       # the world begun again at 0
  again.data.time = 100.0
  new = _offer(again, "open_door")              # the door's job now
  fresh = _life(tmp_path)                       # ...and the next restart
  assert fresh.tasks.get(old.id).created_t > fresh.tasks.get(new.id).created_t
  assert fresh.tasks.get(new.id).state == "offered"
  fresh.begin((0.0, 0.0, 0.0))
  assert fresh._target_activity("door").hinge == dd.draw(new.secret["draw"])


def test_an_offer_the_saved_world_never_saw_has_its_prop_set_out_again(tmp_path):
  """A crash between an offer and the world's next save brings the world
  back from before the offer set its prop out (#610's review): the drawer
  the last job left out stood out under the standing offer, a job paid for
  nothing. The restore sets out a prop whose newest offer is numbered past
  the board's count the save kept, and leaves one whose offer it saw as it
  was saved -- by the count, never the stamp: the door's offer is stamped on
  the clock before a start-over, ahead of the save's."""
  from pluggybot import continuation
  life = _life(tmp_path)
  slide, hinge = _joint_q(life, "drawer_slide"), _joint_q(life, "door_hinge")
  life.data.time = 357_689.0
  _offer(life, "open_door")                     # seen: offered before the save
  life.data.time = 100.0                        # ...and the world begun again since
  life.data.qpos[slide] = 0.2                   # left out by the last job
  life.data.qpos[hinge] = 0.3                   # the door, worked since its offer
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  _offer(life, "open_drawer")                   # set out shut ... and then the crash
  again = _life(tmp_path, rebase=False)         # the same board, carried on
  again.begin((0.0, 0.0, 0.0))
  got = continuation.restore([again], continuation.read(path))
  assert got["inPlace"] and got["setOut"] == ["drawer"]
  assert again.data.qpos[slide] == 0.0, "set out for the offer it never saw"
  assert again.data.qpos[hinge] == pytest.approx(0.3), "the door as the world was saved"


def test_a_restore_names_only_the_props_it_set_out(tmp_path):
  """...and in a world without the counter, an offer on the drawer past the
  saved count sets nothing out, and is not said to have."""
  from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
  life = stub_life(tasks=TaskBoard(path=str(tmp_path / "tasks.json")))
  life.tasks.offer("open_drawer", "drawer", secret={"draw": 3})
  assert life.set_out_unseen(0) == []


def test_the_hand_jobs_are_offered_only_where_a_mind_writes_procedures():
  """Each prop is a target of its own kind, on the challenges' gate."""
  here = world_targets(HUMANOID_HOME, procedures=True)
  for kind in HAND_JOBS:
    assert KINDS[kind].target_kind in here and KINDS[kind].discharge == "procedure"
  assert not set(storeroom.PROPS) & set(world_targets(HUMANOID_HOME))


def test_each_jobs_row_is_its_own_shown_only_where_it_is_offered():
  """A served job's row is paid and the practice's beside it unpaid, each
  graded by the same evaluator; the humanoid's table shows its jobs' rows,
  and a table naming no world's kinds -- every `shown` row -- none of them,
  nor the practice's."""
  from pluggybot.economy import scoring
  from pluggybot.mind.overseer import world_kinds
  table = default_table()
  from pluggybot.economy.cadence import default_cadence
  offered = {KINDS[k].task for k in default_cadence(HUMANOID_HOME).kinds}
  for job, check in HAND_JOBS.items():
    assert scoring.EVALUATORS[job] is scoring.EVALUATORS[check]
    assert 0 == table[check].base
    # ...paid where a world offers it, and nothing where none does (#682's
    # stage C: the place laid from the pass, its reference read, never offered)
    assert (table[job].base > 0) == (job in offered), job
  served = {r["task"] for r in table.as_context(challenges=True,
                                                kinds=world_kinds(HUMANOID_HOME))}
  assert offered and offered <= set(HAND_JOBS)
  assert served & set(HAND_JOBS) == offered, "a row only where its job is offered"
  assert not set(HAND_JOBS.values()) & served
  unnamed = {r["task"] for r in table.as_context(challenges=True)}
  assert not (set(HAND_JOBS) | set(HAND_JOBS.values())) & unnamed


def test_an_offer_says_where_by_the_areas_terms_and_how_long_it_gives():
  """Its address and directions (#419: every number in them a tag) and the
  minutes it gives from the claim; never a position."""
  from pluggybot.perception.tags import TAG_SIZES
  board = TaskBoard()
  maker = task_producer(board, HUMANOID_HOME, procedures=True)
  offered = list(maker.cadence.kinds)
  # ...each once its reference passed there (cadence.json's note): the table's
  # 24 of 32 and the cube's 28 of 32 over the tour (#560) -- all but the place
  # laid from the pass and from the kitchen's shelf, never offered (#682's
  # stage C, Ben's call)
  assert set(HAND_JOBS) - set(offered) == {"set_table_pass", "set_table_shelf"}
  here = world_targets(HUMANOID_HOME, procedures=True)
  for kind in offered:
    task = maker._offer(kind, here[KINDS[kind].target_kind][0], 0.0)
    assert task.params["address"]["house"] == ("mall" if kind == "set_table" else "facility")
    assert f"within {KINDS[kind].minutes} minutes" in task.description
    assert set(task.secret) == {"draw"}
    for n in re.findall(r"\d+(?:\.\d+)?", task.params["directions"]):
      assert int(float(n)) in TAG_SIZES, f"{kind}: {n} is no tag"


def test_each_jobs_sentence_says_its_graders_own_numbers():
  """The template's numbers are the grader's constants, so neither can
  move without the other."""
  says = {"open_drawer": [f"{100 * drawer.OPEN_M:g} cm", f"{drawer.HOLD_S:g} s"],
          "open_door": [f"{math.degrees(door.OPEN_RAD):g} degrees", f"{door.HOLD_S:g} s"],
          "place_cube": [f"{100 * place.PLACE_M:g} cm", f"{place.HOLD_S:g} s"],
          "open_chest": [f"{math.degrees(lid.OPEN_RAD):.0f} degrees",
                         f"{lid.LOWER_RAD_S:g} rad/s", f"{lid.HOLD_S:g} s"]}
  for kind, words in says.items():
    for w in words:
      assert w in KINDS[kind].template, (kind, w)


def test_a_job_whose_offer_sets_its_things_out_says_when_as_offered():
  """Each offer of a `draws` kind puts its things back (#659). Told nothing,
  Rowan timed a spring that is not there and Luca said done on a drawer it
  believed still out. Read as the robot reads it, through `Task.create`'s
  cut, on the offer's own clock: a clause the template has and the offer has
  lost says nothing, and a date that is not this offer's says something false
  -- a parameter of that name included. ⚠ A kind whose area keeps its things
  put away until the job is taken (`take`: the table's pieces, #775) sets
  them out at the claim, so its sentence says that, and no date: the offer's
  would be false of them. Rewritten for #775: every kind said the offer's."""
  from pluggybot.home import jobs
  drawing = [name for name, kind in KINDS.items() if kind.draws]
  assert set(HAND_JOBS) <= set(drawing)
  at_claim = {name for name in drawing
              if hasattr(jobs.area_of(next(t for t, k in jobs.targets().items()
                                           if k == KINDS[name].target_kind)), "take")}
  assert at_claim == {"set_table", "set_table_pass", "set_table_shelf"}
  for name in drawing:
    for t in (205338.0, 7.0):
      offered = Task.create(name, KINDS[name].target_kind, "t_1",
                            params={"minutes": 999, "offered": 1}, t=t)
      if name in at_claim:
        assert "set out" in offered.description and "when you take this job." \
          in offered.description, name
        assert "t=" not in offered.description and "was offered" not in offered.description, name
      else:
        assert f"at t={t:.0f}, when this job was offered." in offered.description, name


def test_what_a_jobs_sentence_says_of_its_prop_is_what_its_offer_does(tmp_path):
  """The clause is only as true as the world (#659; the table's dishware,
  `test_table_job.py`): each prop of the counter's, left off its rest by an
  earlier job, is back at it once its job is offered."""
  life = _life(tmp_path)
  said = {"open_drawer": ("The drawer was put back shut", "drawer_slide"),
          "open_door": ("The door was put back shut", "door_hinge"),
          "open_chest": ("The lid was put back shut", "chest_hinge"),
          "place_cube": ("The cube was put back beside its mat", "cube_free")}
  for kind, (words, joint) in said.items():
    q = _joint_q(life, joint)
    life.data.qpos[q] = life.model.qpos0[q] + 0.1
    task = _offer(life, kind)
    assert words in task.description, kind
    assert life.data.qpos[q] == life.model.qpos0[q], kind


def test_the_chests_grade_reads_the_catch_its_offer_drew(tmp_path):
  """The lid's grader keeps the catch secret (`catchN`): the grade hands it
  the offer's own draw's, never the compiled world's."""
  life = _life(tmp_path)
  task = _offer(life, "open_chest")
  assert life._claim_task(task.id)
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and "never opened" in verdict["reason"]
  assert "catchN" not in verdict["metrics"]
  assert life._target_activity("chest").lid == ch.draw(task.secret["draw"])


def test_a_jobs_watch_ends_with_its_job_however_it_ends(tmp_path):
  """A watch reads its prop every physics step: one whose job ended some
  other way than its grade -- failed by an admin, a true death -- is
  dropped on the board's next sweep."""
  from pluggybot.economy.scoring import evaluate
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id) and task.id in life._job_watches
  life.tasks.resolve(task.id, evaluate("open_drawer", {}), t=float(life.data.time))
  life.body.run(life.body.hold_routine(1.5))
  assert task.id not in life._job_watches
  assert not any(isinstance(h, drawer.Watch) for h in life.body.step_hooks)


def test_no_offer_lands_on_a_prop_a_robot_is_touching(tmp_path, monkeypatch):
  """An offer sets its prop out, so none lands on one a robot is touching
  (`_worked_props`, the producer's `busy`): set out from under a hand, a
  door was clamped shut round it, and a drawer left out was a job paid for
  nothing. Let go, it is offered, and set out shut."""
  life = _life(tmp_path)
  life.producer = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  slide = _joint_q(life, "drawer_slide")
  life.data.qpos[slide] = 0.2                      # a hand on its knob, left out
  held = {"drawer"}
  monkeypatch.setattr(storeroom, "touched_by", lambda model, data, prop, geoms: prop in held)

  def offered() -> set:
    for _ in range(4):
      life.data.time += 600.0
      life._task_step()
    return {t.target for t in life.tasks.tasks.values()}
  assert offered() == {"door", "chest"}
  assert life.data.qpos[slide] == 0.2, "left as the hand has it"
  held.clear()
  assert "drawer" in offered() and life.data.qpos[slide] == 0.0


def test_a_jobs_readings_ride_a_restart(tmp_path):
  """A restart is a continuation (#345): a job's watch carries on from what
  it read before it -- a lid it saw opened and lowered is still one that
  was opened (a fresh watch failed it, "the lid never opened"), and a fall
  it saw still counts against it."""
  life = _life(tmp_path)
  task = _offer(life, "open_chest")
  assert life._claim_task(task.id)
  watch = life._job_watches[task.id]
  hinge = _joint_q(life, "chest_hinge")
  life.data.qpos[hinge] = 1.2
  watch()
  life.data.qpos[hinge] = 0.0
  life.body.falls = 1
  watch()
  state, arrays = life.kept_state()
  again = _life(tmp_path)                       # the same board, a world compiled anew
  again.restore_kept(state, arrays, in_place=True)
  again._resume_jobs(None)
  carried = again._job_watches[task.id]
  assert carried.top == pytest.approx(1.2)
  assert carried.falls() == 1, "a restart's body counts its falls from its own"


def test_a_job_with_nothing_to_watch_is_settled_never_left_standing(tmp_path, monkeypatch):
  """A job graded in a world without its prop -- a body changed under it --
  is failed through the same door (a missing measurement is no pass),
  never left active, holding its prop's offers off for good."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life._unwatch_job(task.id)
  monkeypatch.setattr(life, "_watch_job", lambda task_id: None)
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and verdict["reason"].startswith("this world has no drawer")
  assert life.tasks.get(task.id).state == "failed"


def test_an_offer_shuts_and_bolts_a_door_left_open(tmp_path):
  """A prop's activity is begun again IN PLACE (`Door.renew`): a door left
  open with its latch tripped is set out shut, its latch bolted round it --
  the hinge's range in the model -- and its activity says so."""
  life = _life(tmp_path)
  act = life._target_activity("door")
  hinge = life.model.joint("door_hinge").id
  q = int(life.model.jnt_qposadr[hinge])
  act.bolted = False
  act._limit(life.model)
  life.data.qpos[q] = 1.0
  _offer(life, "open_door")
  assert life._target_activity("door") is act, "renewed in place, never swapped"
  assert act.bolted and life.data.qpos[q] == 0.0
  assert life.model.jnt_range[hinge][1] == pytest.approx(dd.SHUT_RAD)
