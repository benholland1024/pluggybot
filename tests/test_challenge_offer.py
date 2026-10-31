"""A challenge is OFFERED (issue #207): a task kind with no errand behind it,
offered to a mind -- the tower first, the humanoids' hand jobs now (#559).

What these pin, each without a mission (docs/Testing.md):

  1. A hand job is a real kind -- evaluator, reward row -- and it is
     offered exactly where a procedure can be written: `world_targets`
     names its prop where there is a mind and not without one.
  2. Claiming it queues NOTHING: the robot's own procedure is the attempt.
     The claim of the loop with no mind skips it.
  3. `done` is paperwork that sets the grade pending, refused for a job
     this robot does not hold; the loop grades once the queue is empty and
     before the mind is asked again. The grade itself, its hold and its
     verdict, is `test_hand_jobs.py`'s.
"""

from pluggybot import lifecycle as lc
from pluggybot.economy.cadence import Cadence, TaskProducer
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import HAND_JOBS, KINDS, TaskBoard
from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, world_targets
from pluggybot.mind import overseer as ov
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


class _Mind:
  """What the lifecycle reads off an overseer on this path: a library
  (the mark of a mind), and the map the seam consults."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None

  def __init__(self, library=object()):
    self.library = library
    self.decisions = []
    self.menu = ov.Menu.for_world(HUMANOID_HOME)


def _claiming_life(tmp_path, mind=None):
  """A robot for a CLAIM, which is the board's and the ledger's
  bookkeeping: on a stub body, with no prop to grade (issue #380)."""
  return stub_life(ledger=Ledger(path=str(tmp_path / "ledger.json")),
                   tasks=TaskBoard(path=str(tmp_path / "tasks.json")),
                   overseer=mind if mind is not None else _Mind())


def _claimed_drawer(life):
  """An offered drawer job, taken by this robot: the state `done` is judged
  against."""
  task = life.tasks.offer("open_drawer", "drawer", t=float(life.data.time))
  assert task is not None
  assert life._claim_task(task.id)
  return life.tasks.get(task.id)


# ---- 1. the kind, and where it is offered ----------------------------------


def test_a_hand_job_is_a_real_kind_discharged_by_a_procedure():
  kind = KINDS["open_drawer"]
  assert kind.discharge == "procedure" and kind.task == "open_drawer"
  assert kind.target_kind == "drawer"
  # every kind is one the robot writes the procedure for -- but the
  # real-stake task (issue #228), whose claim is the act
  assert {n for n, k in KINDS.items() if k.discharge != "procedure"} == {"take_points"}
  assert set(HAND_JOBS) <= set(KINDS)


def test_the_offer_exists_where_there_is_a_mind_and_not_without_one(monkeypatch):
  """The gate is the target seam, not a new field: a challenge's target is
  named only where a procedure can be written. Asked of the house with its
  counter's props set out -- and without them, nothing names one."""
  book = lc.board_book(HUMANOID_HOME)
  mindless = world_targets(HUMANOID_HOME, book)
  minded = world_targets(HUMANOID_HOME, book, procedures=True)
  assert "drawer" not in mindless
  assert minded["drawer"] == ["drawer"]
  real = lc.world_config
  monkeypatch.setattr(lc, "world_config", lambda world: {
    k: v for k, v in real(world).items() if k != "counter"})
  assert "drawer" not in world_targets(HUMANOID_HOME, book, procedures=True)
  monkeypatch.setattr(lc, "world_config", real)
  props = {KINDS[k].target_kind for k in HAND_JOBS}
  assert {k: v for k, v in minded.items() if k not in props} == mindless
  # ...and the producer follows the targets: with no mind the rotation
  # simply has no hand job in it.
  beat = Cadence._build("test", {"kinds": {"open_drawer": {}, "open_door": {}}}, None)
  off = TaskProducer(TaskBoard(), beat, mindless)
  on = TaskProducer(TaskBoard(), beat, minded)
  assert off.kinds == () and set(on.kinds) == {"open_drawer", "open_door"}


def test_the_producer_puts_the_job_up_with_its_prop_as_its_target():
  beat = Cadence._build("home", {"firstAtS": 0.0, "everyS": 1.0, "initial": 0,
                                  "kinds": {"open_drawer": {"params": {"minutes": 16}}}},
                        None)
  board = TaskBoard()
  producer = TaskProducer(board, beat, {"drawer": ["drawer"]})
  producer.tick(1.0, pack_wh=KINDS["open_drawer"].estimate_wh + 1.0)
  [task] = board.offered()
  assert task.kind == "open_drawer" and task.target == "drawer"
  assert "drawer" in task.description and "say you are done" in task.description
  assert task.reward(board.table)["base"] == 90


# ---- 2. the claim queues nothing ------------------------------------------


def test_claiming_a_hand_job_queues_no_errand(tmp_path):
  life = _claiming_life(tmp_path)
  task = _claimed_drawer(life)
  assert task.state == "active" and task.claimed_by == life.root
  assert life.errands == []
  assert lc.errand_for_task(task, HUMANOID_HOME) is None


def test_a_mind_without_a_library_cannot_claim_it(tmp_path):
  life = _claiming_life(tmp_path, mind=_Mind(library=None))
  task = life.tasks.offer("open_drawer", "drawer")
  assert not life._claim_task(task.id)
  assert life.tasks.get(task.id).state == "offered"


def test_the_scripted_claim_skips_a_challenge(tmp_path):
  """A job for a mind, left offered rather than attempted by something that
  cannot write the procedure."""
  life = _claiming_life(tmp_path)
  life.tasks.offer("open_drawer", "drawer")
  assert not life._claim_next_task()
  assert [t.state for t in life.tasks.tasks.values()] == ["offered"]


# ---- 3. done, and when the grade runs --------------------------------------


def test_done_is_refused_for_a_task_this_robot_does_not_hold(tmp_path):
  life = _claiming_life(tmp_path)
  task = life.tasks.offer("open_drawer", "drawer")       # offered, not claimed
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  assert life._grade_pending == ""
  life._done(ov.Decision(action="idle", reason="", done="t_nope"))
  assert life._grade_pending == ""
  # ...and one it holds is pending its grade
  held = _claimed_drawer(life)
  life._done(ov.Decision(action="idle", reason="", done=held.id))
  assert life._grade_pending == held.id


def test_the_grade_waits_for_the_queue_and_runs_before_the_mind(tmp_path):
  """`done` on the same answer as a queued procedure: the loop's branch
  order is queue, grade, mind. Read off the day loop's source rather than
  flown, on `test_tick`'s terms: the rule is one line of ordering."""
  import inspect
  src = inspect.getsource(HubLifecycle._day_routine)
  queue = src.index("self.run_errand_routine(self.errands.pop(0))")
  grade = src.index("elif self._grade_pending:")
  mind = src.index("yield from self._arbitrate_routine()")
  assert queue < grade < mind


# ---- the mind's side: `done`, the rule, the table -------------------------


def test_done_rides_the_library_slot_and_a_menu_without_one_never_sees_it():
  """`done` is offered exactly where a procedure can be written: in the
  schema and parsed with a library, absent and dropped without one."""
  from dataclasses import replace
  menu = replace(ov.Menu.for_world(HUMANOID_HOME), procedures=True)
  with_library = menu.schema(procedures=("drawer",))
  assert with_library["properties"]["done"] == {"type": "string"}
  assert "done" in with_library["required"]
  without = menu.schema()
  assert "done" not in without["properties"] and "done" not in without["required"]
  d = menu.validate({"action": "procedure:drawer", "reason": "r", "done": "t_0007"},
                    procedures=("drawer",))
  assert d.done == "t_0007" and d.as_dict()["done"] == "t_0007"
  bare = ov.Menu.for_world(HUMANOID_HOME)
  dropped = bare.validate({"action": "idle", "reason": "r", "done": "t_0007"})
  assert dropped.done == "" and "done" not in dropped.as_dict()


def test_the_challenge_rule_names_no_charging_and_the_table_shows_the_row_with_a_library_only():
  """EVENT_MAP_RULE's fence, kept: nothing in the rule demonstrates the
  measured action. And the reward table the prompt carries gains the
  challenge rows only where a challenge can be attempted."""
  from pluggybot.economy import scoring
  for word in ("charge", "battery", "rack"):
    assert word not in ov.CHALLENGE_RULE, word
  assert "done" in ov.CHALLENGE_RULE and "stand clear" in ov.CHALLENGE_RULE
  table = scoring.default_table()
  bare = [r["task"] for r in table.as_context()]
  with_library = [r["task"] for r in table.as_context(challenges=True)]
  assert "program" not in bare and "program" in with_library
  assert set(bare) < set(with_library)
