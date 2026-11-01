"""The humanoids' first jobs (issue #559, stage 3): the hands' four checks on
the storeroom's counter (`home.storeroom`), offered, set out, watched from
the claim and graded after `done` -- each rule pinned on a stub whose world
is the counter alone (`StubBody`), nothing flown. The references flown from
the station are `scripts/solve.py --feature drawer|door|cube|chest`'s.
"""

import math
import re

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
from pluggybot.economy.tasks import HAND_JOBS, KINDS, TaskBoard
from pluggybot.home import storeroom
from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, task_producer, world_targets


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


def _counter_world():
  """The stub's floor with the storeroom's counter on it, compiled."""
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.texturedir = "models/"
  storeroom.attach(spec)
  model = spec.compile()
  return model, mujoco.MjData(model)


def _life(tmp_path):
  """A humanoid world's lifecycle on a stub at the counter: its board, its
  ledger, a mind, the counter's activities on its seam."""
  model, data = _counter_world()
  body = StubBody(model, data, grid_bounds=(-3.0, -8.0, 30.0, 8.0))
  life = HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME, body=body,
                      tasks=TaskBoard(path=str(tmp_path / "tasks.json")),
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
  from pluggybot.mind import overseer as ov
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
  assert life._counter_activity("drawer").runners == want
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
  robot did in between."""
  life = _life(tmp_path)
  task = _offer(life, "open_drawer", minutes=0.01)
  assert life._claim_task(task.id)
  life.data.qpos[_joint_q(life, "drawer_slide")] = 0.2
  life.body.run(life.body.hold_routine(1.0))
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and "the job allows" in verdict["reason"]


def test_a_restart_puts_back_each_props_newest_draw(tmp_path):
  """The world file carries each prop's compiled draw; an offer that came
  back off the board still means the one it drew -- put back as the day
  begins (`begin` -> `restore_counter`), before anything reads the prop."""
  life = _life(tmp_path)
  task = _offer(life, "open_door")
  k = task.secret["draw"]
  fresh = _life(tmp_path)                       # the same board, a world compiled anew
  dof = int(fresh.model.jnt_dofadr[fresh.model.joint("door_hinge").id])
  assert fresh.model.dof_damping[dof] == pytest.approx(dd.draw(0).damping)
  fresh.begin((0.0, 0.0, 0.0))
  assert fresh.model.dof_damping[dof] == pytest.approx(dd.draw(k).damping)
  assert fresh._counter_activity("door").hinge == dd.draw(k)


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
  # set it down off the mat, 0 of 8
  assert set(offered) == set(HAND_JOBS) - {"place_cube"}
  for kind in offered:
    params, secret = maker._build(kind, KINDS[kind].target_kind)
    assert params["address"]["house"] == "facility" and params["minutes"] > 0
    assert set(secret) == {"draw"}
    for n in re.findall(r"\d+(?:\.\d+)?", params["directions"]):
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
  assert life._counter_activity("chest").lid == ch.draw(task.secret["draw"])


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
  act = life._counter_activity("door")
  hinge = life.model.joint("door_hinge").id
  q = int(life.model.jnt_qposadr[hinge])
  act.bolted = False
  act._limit(life.model)
  life.data.qpos[q] = 1.0
  _offer(life, "open_door")
  assert life._counter_activity("door") is act, "renewed in place, never swapped"
  assert act.bolted and life.data.qpos[q] == 0.0
  assert life.model.jnt_range[hinge][1] == pytest.approx(dd.SHUT_RAD)
