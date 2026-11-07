"""The humanoids' first jobs (issue #559, stage 3): the hands' four checks on
the storeroom's counter (`home.storeroom`), offered, set out, watched from
the claim and graded after `done` -- each rule pinned on a stub whose world
is the counter alone (`StubBody`), nothing flown. The references flown from
the station are `scripts/solve.py --feature drawer|door|cube|chest`'s.
"""

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
from pluggybot.economy.tasks import HAND_JOBS, HAND_MINUTES, KINDS, TaskBoard
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
  body's own count -- fails a drawer that ends out (#544's rule)."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  life.body.falls = 1
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and "fell" in verdict["reason"]


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
  assert not verdict["ok"] and "called 1 s after the start" in verdict["reason"]


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
    f"said done on {task.id} (open_drawer): graded once everything queued has run")]
  graded = [i for i, ln in enumerate(lines) if ln.endswith(
    f"{'passed' if opened else 'failed'} the job {task.id} (open_drawer): {verdict['reason']}")]
  assert len(said) == len(graded) == 1 and said[0] < graded[0], lines


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
  assert {KINDS[k].minutes for k in HAND_JOBS if k != "set_table"} == {HAND_MINUTES}
  life = _life(tmp_path)
  task = life.tasks.offer("open_drawer", "drawer", ttl=3000.0, t=float(life.data.time))
  life.tasks.tasks[task.id] = replace(task, params={})
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  mujoco.mj_forward(life.model, life.data)
  verdict = _grade(life, task.id)
  assert verdict["ok"] and verdict["metrics"]["limitS"] == 60.0 * HAND_MINUTES


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
  for job, check in HAND_JOBS.items():
    assert scoring.EVALUATORS[job] is scoring.EVALUATORS[check]
    assert table[job].base > 0 == table[check].base
  from pluggybot.economy.cadence import default_cadence
  served = {r["task"] for r in table.as_context(challenges=True,
                                                kinds=world_kinds(HUMANOID_HOME))}
  offered = {KINDS[k].task for k in default_cadence(HUMANOID_HOME).kinds}
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
  # ...each once its reference passed there (cadence.json's note): the cube's
  # set it down off the mat, 0 of 8, and the table's laid 13 of 32 (#560)
  assert set(offered) == set(HAND_JOBS) - {"place_cube", "set_table"}
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
