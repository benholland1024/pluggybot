"""The tower is OFFERED (issue #207): the first task kind with no errand
behind it, on the `autonomous` arm.

What these pin, each without a mission (docs/Testing.md):

  1. `stack_tower` is a real kind -- evaluator, reward row, a cadence slot
     on home -- and it is offered exactly where a procedure can be written:
     `world_targets` names the `challenge` target on the autonomous arm and
     not on `guarded`, so the control's offered set is unchanged.
  2. Claiming it queues NOTHING: the robot's own procedure is the attempt.
     The scripted claim skips it as it skips a question.
  3. `done` is paperwork that sets the grade pending; the loop grades once
     the queue is empty, through `scoring.evaluate` -- one verdict, banked,
     and the task resolved off it. The grader's own tests
     (test_challenge_stack.py) hold the criteria; these reach them through
     the offer with the hold on the seam:
       - a tower standing at 10 s passes and pays;
       - one that falls at 9 s fails ("it fell");
       - the robot's chassis touching a block DURING the hold fails, even
         though the tower stands at the end (criterion 5, the seam's own).
"""

import mujoco

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.challenge import stack
from pluggybot.economy.cadence import Cadence, TaskProducer, default_cadence
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import KINDS, TaskBoard
from pluggybot.lifecycle import HubLifecycle, world_config, world_targets
from pluggybot.mind import overseer as ov
from pluggybot.robot import world_spec


class _Mind:
  """What the lifecycle reads off an overseer on this path: a library
  (the mark of the autonomous arm), and the map the seam consults."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  workshop = None

  def __init__(self, library=object()):
    self.library = library
    self.decisions = []
    self.menu = ov.Menu.for_world("room_hub")


def _life(tmp_path, mind=None):
  """A room_hub lifecycle with the three blocks added through MjSpec (the
  challenge's own props), a board, and a ledger with a balance."""
  cfg = world_config("room_hub")
  spec = stack.add_blocks(world_spec(cfg["model"]))
  model = spec.compile()
  data = mujoco.MjData(model)
  ledger = Ledger(path=str(tmp_path / "ledger.json"))
  board = TaskBoard(path=str(tmp_path / "tasks.json"))
  life = HubLifecycle(model, data, realtime=False, world="room_hub",
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      spec=spec, errand=False, ledger=ledger, tasks=board,
                      # the arm the tower is offered on: the claim rail is
                      # off, as it is there (a 1.0 Wh demo cell could never
                      # fund a 2.7 Wh estimate through `Task.claimable`)
                      autonomous=True,
                      overseer=mind if mind is not None else _Mind())
  for _ in range(200):
    mujoco.mj_step(model, data)
  return life


def _claimed_tower(life):
  """An offered tower, taken by this robot: the state `done` is judged
  against."""
  task = life.tasks.offer("stack_tower", "workshop", t=float(life.data.time))
  assert task is not None
  assert life._claim_task(task.id)
  return life.tasks.get(task.id)


def _tower(life, offsets=((0, 0), (0, 0), (0, 0)), xy=(1.10, 0.30)):
  x, y = xy
  for i, (name, (dx, dy)) in enumerate(zip(stack.BLOCKS, offsets)):
    x, y = x + dx, y + dy
    stack.place(life.model, life.data, name,
                (x, y, stack.BLOCK_HALF + i * (stack.PITCH_M + 0.0005)))
  mujoco.mj_forward(life.model, life.data)


def _grade(life, task_id):
  events = []
  life.tasks.on_event.append(events.append)
  life._done(ov.Decision(action="idle", reason="", done=task_id))
  assert life._grade_pending == task_id
  tick.run(life.mission.swap, life._grade_routine())
  return events


# ---- 1. the kind, and where it is offered ----------------------------------


def test_the_tower_is_a_real_kind_discharged_by_a_procedure():
  kind = KINDS["stack_tower"]
  assert kind.discharge == "procedure" and kind.task == "stack"
  assert kind.target_kind == "challenge"
  # every other kind is what every kind used to be -- but the real-stake
  # task (issue #228), whose claim is the act, and the bench (#227), the
  # second challenge
  assert all(k.discharge == "errand" for n, k in KINDS.items()
             if n not in ("stack_tower", "take_points", "find_mass"))
  assert "stack_tower" in default_cadence("home").kinds
  assert "stack_tower" not in default_cadence("room_hub").kinds


def test_the_offer_exists_on_the_autonomous_arm_and_not_on_guarded():
  """The gate is the target seam, not a new field: a challenge's target is
  named only where a procedure can be written, so `guarded`'s offered set
  is byte-for-byte what it was (its committed series stay comparable)."""
  book = lc.board_book("home")
  guarded = world_targets("home", book)
  autonomous = world_targets("home", book, procedures=True)
  assert "challenge" not in guarded
  assert autonomous["challenge"] == ["workshop"]
  # (`cage` and `bench` ride the same gate, issues #226 and #227)
  assert {k: v for k, v in autonomous.items()
          if k not in ("challenge", "cage", "bench")} == guarded
  # ...and the producer follows the targets: on `guarded` the rotation
  # simply has no tower in it.
  beat = default_cadence("home")
  off = TaskProducer(TaskBoard(), beat, guarded)
  on = TaskProducer(TaskBoard(), beat, autonomous)
  assert "stack_tower" not in off.kinds and "stack_tower" in on.kinds
  # room_hub has no blocks and names no tower on either arm
  assert "challenge" not in world_targets("room_hub", None, procedures=True)


def test_the_producer_puts_the_tower_up_with_the_room_as_its_target():
  beat = Cadence._build("home", {"firstAtS": 0.0, "everyS": 1.0, "initial": 0,
                                  "kinds": {"stack_tower": {}}}, None)
  board = TaskBoard()
  producer = TaskProducer(board, beat, {"challenge": ["workshop"]})
  producer.tick(1.0, pack_wh=8.0)
  [task] = board.offered()
  assert task.kind == "stack_tower" and task.target == "workshop"
  assert "workshop" in task.description and "say you are done" in task.description
  assert task.reward(board.table)["base"] == 40


# ---- 2. the claim queues nothing ------------------------------------------


def test_claiming_the_tower_queues_no_errand(tmp_path):
  life = _life(tmp_path)
  task = _claimed_tower(life)
  assert task.state == "active" and task.claimed_by == life.root
  assert life.errands == []
  assert lc.errand_for_task(task, "room_hub") is None


def test_a_mind_without_a_library_cannot_claim_it(tmp_path):
  life = _life(tmp_path, mind=_Mind(library=None))
  task = life.tasks.offer("stack_tower", "workshop")
  assert not life._claim_task(task.id)
  assert life.tasks.get(task.id).state == "offered"


def test_the_scripted_claim_skips_a_challenge(tmp_path):
  """Like a question: a job for a mind, left offered rather than attempted
  by something that cannot write the procedure."""
  life = _life(tmp_path)
  life.tasks.offer("stack_tower", "workshop")
  assert not life._claim_next_task()
  assert [t.state for t in life.tasks.tasks.values()] == ["offered"]


# ---- 3. done, and the grade with its hold ---------------------------------


def test_done_is_refused_for_a_task_this_robot_does_not_hold(tmp_path):
  life = _life(tmp_path)
  task = life.tasks.offer("stack_tower", "workshop")    # offered, not claimed
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  assert life._grade_pending == ""
  life._done(ov.Decision(action="idle", reason="", done="t_nope"))
  assert life._grade_pending == ""


def test_a_standing_tower_passes_the_hold_and_is_paid(tmp_path):
  life = _life(tmp_path)
  task = _claimed_tower(life)
  _tower(life)
  t0 = float(life.data.time)
  events = _grade(life, task.id)
  assert float(life.data.time) - t0 >= stack.HOLD_S
  closed = life.tasks.get(task.id)
  assert closed.state == "done", closed.verdict
  assert closed.verdict["ok"] and closed.points >= 20
  assert life.ledger.balance() == closed.points
  [grade] = life.grades
  assert grade["ok"] and grade["touchedDuringHold"] == []
  assert any(e.get("type") == "task_resolved" for e in events)
  assert life._grade_pending == ""


def test_a_tower_that_falls_during_the_hold_fails(tmp_path):
  """The existing grader's overhung tower, reached through the offer: it
  passes every geometric check at the call and is on the floor inside a
  second. Fails, resolves `failed`, pays nothing."""
  life = _life(tmp_path)
  task = _claimed_tower(life)
  _tower(life, offsets=((0, 0), (0.009, 0), (0.009, 0)))
  assert stack.measure(life.model, life.data)["layers"] == 3   # the premise
  _grade(life, task.id)
  closed = life.tasks.get(task.id)
  assert closed.state == "failed" and "it fell" in closed.verdict["reason"]
  assert life.ledger.balance() == 0


def test_a_block_against_the_chassis_is_read_off_the_contact_list(tmp_path):
  """The per-step reading behind criterion 5: `foreign_contacts` names the
  robot's geoms touching a block, and names nothing for a tower on open
  floor. (A tower leaning on the chassis at BOTH snapshots is already
  criterion 4's "holding the tower up"; the seam's reading is for the
  touch that comes and goes between them.)"""
  life = _life(tmp_path)
  _tower(life)
  assert stack.foreign_contacts(life.model, life.data) == set()
  cx, cy = life.data.xpos[life.mission.swap.chassis_bid][:2]
  half_x = float(life.model.geom_size[life.model.geom(
    life.mission.swap.handle.el("chassis")).id][0])
  _tower(life, xy=(float(cx) + half_x + stack.BLOCK_HALF - 0.0005, float(cy)))
  for _ in range(50):
    mujoco.mj_step(life.model, life.data)
  # the caster sits proud of the chassis front at floor height, so it is
  # what a floor-standing block meets first: the robot, by any of its names
  touched = stack.foreign_contacts(life.model, life.data)
  assert touched and touched <= {"chassis", "caster"}, touched


def test_a_touch_during_the_hold_fails_a_tower_that_stands_at_both_ends(
    tmp_path, monkeypatch):
  """Criterion 5, the seam's own: the snapshots at the call and after the
  hold both read a free-standing tower, and a hand on it in between fails
  it anyway -- the loop reads what touches a block EVERY step. Pinned by
  making the reading say "chassis" for one second in the middle of the
  hold, so the two snapshots the grader used to be judged on alone are
  both clean."""
  life = _life(tmp_path)
  task = _claimed_tower(life)
  _tower(life)
  real = stack.foreign_contacts
  t0 = float(life.data.time)

  def brushed(model, data):
    now = float(data.time) - t0
    return {"chassis"} if 4.0 <= now <= 5.0 else real(model, data)
  monkeypatch.setattr(stack, "foreign_contacts", brushed)
  _grade(life, task.id)
  closed = life.tasks.get(task.id)
  assert closed.state == "failed"
  assert "chassis touched the tower during the hold" in closed.verdict["reason"]
  assert closed.verdict["metrics"]["freeStanding"] is True
  assert closed.verdict["metrics"]["freeStandingAtDone"] is True
  assert life.grades[-1]["touchedDuringHold"] == ["chassis"]
  assert life.ledger.balance() == 0


def test_the_grade_waits_for_the_queue_and_runs_before_the_mind(tmp_path):
  """`done` on the same answer as a queued procedure: the loop's branch
  order is queue, grade, mind. Read off the day loop's source rather than
  flown, on `test_tick`'s terms: the rule is one line of ordering."""
  import inspect
  src = inspect.getsource(HubLifecycle._day_routine)
  queue = src.index("yield from self.run_errand_routine(self.errands.pop(0))")
  grade = src.index("elif self._grade_pending:")
  mind = src.index("yield from self._arbitrate_routine()")
  assert queue < grade < mind


# ---- the mind's side: `done`, the rule, the table -------------------------


def test_done_rides_the_library_slot_and_guarded_never_sees_it():
  """`done` is offered exactly where a procedure can be written: in the
  schema and parsed with a library, absent and dropped without one -- so
  `guarded`'s schema and prefix are what they were."""
  from dataclasses import replace
  menu = replace(ov.Menu.for_world("home"), procedures=True)
  with_library = menu.schema(procedures=("stack",))
  assert with_library["properties"]["done"] == {"type": "string"}
  assert "done" in with_library["required"]
  without = menu.schema()
  assert "done" not in without["properties"] and "done" not in without["required"]
  d = menu.validate({"action": "procedure:stack", "reason": "r", "done": "t_0007"},
                    procedures=("stack",))
  assert d.done == "t_0007" and d.as_dict()["done"] == "t_0007"
  guarded = ov.Menu.for_world("home")
  dropped = guarded.validate({"action": "idle", "reason": "r", "done": "t_0007"})
  assert dropped.done == "" and "done" not in dropped.as_dict()


def test_the_challenge_rule_names_no_charging_and_the_table_shows_the_row_on_autonomous_only():
  """EVENT_MAP_RULE's fence, kept: nothing in the rule demonstrates the
  measured action. And the reward table the prompt carries gains the
  challenge rows only where a challenge can be attempted, so the control's
  cached prefix is byte-identical to the flown one."""
  from pluggybot.economy import scoring
  for word in ("charge", "battery", "rack"):
    assert word not in ov.CHALLENGE_RULE, word
  assert "done" in ov.CHALLENGE_RULE and "stand clear" in ov.CHALLENGE_RULE
  table = scoring.default_table()
  guarded = [r["task"] for r in table.as_context()]
  autonomous = [r["task"] for r in table.as_context(challenges=True)]
  assert "stack" not in guarded and "stack" in autonomous
  assert set(guarded) < set(autonomous)
