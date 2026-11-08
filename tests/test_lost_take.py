"""A take that did not happen says so, and why (issue #648).

Two robots share one board and an answer takes 10-90 s to think, so an offer
can be gone by the time the answer that took it lands. Said to the log alone,
the loser's History read `chose take_task (t_0122): ...` and nothing after it:
Luca wrote "t_0122 is mine", ran its procedure on Rowan's job and stood where
Rowan's grip had to go. A `done` on a job the robot does not hold was heard
the same way, by nobody. Bookkeeping, so all of it is on the stub.
"""

from dataclasses import replace

import pytest

from pluggybot.body import StubBody
from pluggybot.lifecycle import HUMANOID_HOME, task_board
from pluggybot.mind import overseer as ov

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_two_minds import stub_pair


def _history(life) -> list[str]:
  return [ln for ln in life.thoughts.read("History.md").splitlines() if ln.strip()]


def _said(life, words: str) -> list[str]:
  return [ln for ln in _history(life) if words in ln]


def _pair():
  return stub_pair(tasks=True, names=("Luca", "Rowan"))


def _board():
  return task_board(world=HUMANOID_HOME)


def _offer(life, t: float = 185000.0, **kw):
  return life.tasks.offer("open_drawer", "drawer", t=t, **kw)


def _take(life, task, t: float) -> str:
  """An answer's `take_task` landing at `t`, down the route a real one takes."""
  life.data.time = t
  return life._after_decision(ov.Decision(action="take_task", task=task.id, reason="mine"))


def _done(life, task_id: str) -> None:
  life._done(ov.Decision(action="idle", reason="", done=task_id))


def _ended(life, task, state: str, t: float):
  """The job as it stands once it is over: how it got there is the grade's."""
  life.tasks.tasks[task.id] = ended = replace(life.tasks.get(task.id), state=state, resolved_t=t)
  return ended


def test_a_take_the_other_robot_won_says_who_took_it():
  """Rowan's answer lands first and Luca's 62 s later: t_0122's race, whose
  loser's History stopped at its own `chose`."""
  luca, rowan = _pair()
  task = _offer(luca)
  assert _take(rowan, task, 185073.0) == ""
  assert _take(luca, task, 185135.0) == "unclaimable"
  assert luca.tasks.held_by(luca.root) == []
  assert [t.id for t in luca.tasks.held_by(rowan.root)] == [task.id]
  lines = _history(luca)
  lost = [i for i, ln in enumerate(lines) if "could not take" in ln]
  assert len(lost) == 1, lines
  assert lines[lost[0]].endswith(f"could not take {task.id} (open_drawer): "
                                 "Rowan took it at t=185073")
  assert "chose take_task" in lines[lost[0] - 1], "...said after the `chose` it answers"
  assert _said(rowan, "could not take") == [], "and the winner has nothing to hear"


def test_a_done_on_the_other_robots_job_says_who_took_it():
  """Graded by nobody, as #648's `done` on a job Rowan held was: the line
  says that and why, where it read as `said done` or as nothing."""
  luca, rowan = _pair()
  task = _offer(luca)
  assert _take(rowan, task, 185073.0) == ""
  luca.data.time = 185160.0
  _done(luca, task.id)
  assert luca._grade_pending == "" and rowan._grade_pending == ""
  assert _said(luca, f"said done on {task.id} (open_drawer): not graded, because "
                     "Rowan took it at t=185073")
  assert luca.tasks.get(task.id).state == "active", "and nothing happened to Rowan's job"


def test_a_done_on_its_own_job_is_not_refused():
  luca, _ = _pair()
  task = _offer(luca)
  assert _take(luca, task, 185073.0) == ""
  _done(luca, task.id)
  assert luca._grade_pending == task.id
  assert _said(luca, "not graded") == [] and _said(luca, "could not take") == []


# ---- every way the board can say no, in words that are true of it -----------


def test_an_offer_that_lapsed_says_when_swept_or_not():
  """Past its deadline it may still read `offered` until the sweep."""
  luca, _ = _pair()
  task = _offer(luca, t=100.0, ttl=50.0)
  assert _take(luca, task, 200.0) == "unclaimable"
  assert _said(luca, f"could not take {task.id} (open_drawer): it lapsed at t=150")
  swept = _offer(luca, t=100.0, ttl=50.0)
  luca.tasks.expire_due(200.0)
  _done(luca, swept.id)
  assert _said(luca, f"said done on {swept.id} (open_drawer): not graded, because "
                     "it lapsed at t=150")


def test_a_job_the_board_does_not_have_is_named_alone():
  luca, _ = _pair()
  assert luca._claim_task("t_gone") is False
  assert _said(luca, "could not take t_gone: the board has no such job")
  _done(luca, "t_nope")
  assert _said(luca, "said done on t_nope: not graded, because the board has no such job")


def test_a_done_on_an_offer_nobody_took_says_so():
  """Luca's t_0137, 11-07: `done` on the offer the same answer declined."""
  luca, _ = _pair()
  task = _offer(luca)
  _done(luca, task.id)
  assert _said(luca, f"said done on {task.id} (open_drawer): not graded, because "
                     "nobody has taken it")


@pytest.mark.parametrize("state, how", [("done", "passed"), ("failed", "failed")])
def test_a_job_that_is_over_says_who_took_it_and_how_it_ended(state, how):
  luca, rowan = _pair()
  task = _offer(luca)
  assert _take(rowan, task, 185073.0) == ""
  _ended(luca, task, state, 185200.0)
  _done(luca, task.id)
  assert _said(luca, f"because Rowan took it at t=185073, and it {how} at t=185200")
  _done(rowan, task.id)
  assert _said(rowan, f"because I took it at t=185073, and it {how} at t=185200")


def test_a_job_nobody_took_that_ended_does_not_blame_anybody():
  luca, _ = _pair()
  task = _offer(luca)
  _ended(luca, task, "failed", 185200.0)
  _done(luca, task.id)
  assert _said(luca, "because nobody took it, and it failed at t=185200")


def test_a_holder_not_in_this_world_is_no_name_at_all():
  """A root is the board's word for a robot, never the robot's."""
  luca, _ = _pair()
  task = _offer(luca)
  luca.tasks.claim(task.id, robot="r9_pluggybot", t=185073.0)
  _done(luca, task.id)
  assert _said(luca, "because another robot took it at t=185073")
  assert _said(luca, "r9_pluggybot") == []


def test_what_the_pack_cannot_fund_is_said_as_that():
  """The one gate about the robot, and the scripted arm's alone: with a mind
  `claim_budget_wh` is None, and offered but refused would be nobody's fault."""
  life = stub_life(HUMANOID_HOME, body=StubBody(), tasks=_board())
  task = life.tasks.offer("open_drawer", "drawer", t=0.0)
  life.tasks.tasks[task.id] = replace(task, estimate_wh=life.claim_budget_wh + 1.0)
  assert life._claim_task(task.id) is False
  assert _said(life, f"could not take {task.id} (open_drawer): "
                     "it costs more than the pack has left")


def test_a_job_done_to_a_robot_says_why_it_cannot_be_taken_here():
  """An act's own gates (#228), down the same door: the offer naming the
  robot that took it, naming nobody here, and a mind that may not act."""
  luca, _ = _pair()
  amount = {"amount": 5}
  mine = luca.tasks.offer("take_points", "Luca", params=amount, ttl=100.0, t=0.0)
  assert luca._claim_task(mine.id) is False
  assert _said(luca, f"could not take {mine.id} (take_points): it names Luca, and that is me")
  nobody = luca.tasks.offer("take_points", "Ada", params=amount, ttl=100.0, t=0.0)
  assert luca._claim_task(nobody.id) is False
  assert _said(luca, f"could not take {nobody.id} (take_points): it names Ada, and that is "
                     "nobody here")
  rowans = luca.tasks.offer("take_points", "Rowan", params=amount, ttl=100.0, t=0.0)
  luca.overseer._acts = lambda: None
  assert luca._claim_task(rowans.id) is False
  assert _said(luca, f"could not take {rowans.id} (take_points): nothing here may act on Rowan")
  assert not luca.tasks.held_by(luca.root)


def test_a_mind_that_cannot_write_a_procedure_hears_that_too():
  """A claim's other refusals go through the same door."""
  class Bare:
    event_map = pending = interrupt_pending = spend = None
    can_escalate = False
    library = None
    decisions = []
    menu = ov.Menu.for_world(HUMANOID_HOME)
  life = stub_life(HUMANOID_HOME, body=StubBody(), tasks=_board(), overseer=Bare())
  task = life.tasks.offer("open_drawer", "drawer", t=0.0)
  assert life._claim_task(task.id) is False
  assert _said(life, f"could not take {task.id} (open_drawer): "
                     "it takes a procedure, and nothing here can write one")
