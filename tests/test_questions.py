"""Guards for the whiteboard question (issue #22).

The first task kind with a RIGHT ANSWER, and four claims worth breaking a
build over:

  1. THE ARITHMETIC IS THE MIND'S. Code never computes an answer, and a robot
     with no mind cannot take the job -- the scripted rotation leaves a
     question standing and it lapses.
  2. GRADING READS THE BOARD. A caller that reports a perfect drawing over a
     board carrying something else scores nothing.
  3. NOTHING SECRET REACHES THE WIRE OR THE MODEL. The right answer lives in
     `Task.secret`, is written to the state file and is published nowhere.
  4. A WRONG ANSWER PAYS NOTHING, however neatly it is written.

There is a fifth thing here that is not a guard but a MEASUREMENT worth
pinning: `test_the_ink_cannot_tell_a_six_from_an_eight`. It exists to stop a
future reader "improving" the grader into handwriting recognition, which is
the design this issue started from and which the pen cannot support.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from pluggybot.hub import lifecycle as lc
from pluggybot.hub import questions as q
from pluggybot.hub import scoring, strokes
from pluggybot.hub.boards import BoardBook, BoardRecord, decimate
from pluggybot.hub.drawing import Envelope
from pluggybot.hub.errand import Errand
from pluggybot.hub.ledger import Ledger
from pluggybot.hub.overseer import Menu, scripted
from pluggybot.hub.tasks import KINDS, TaskBoard

TABLE = scoring.default_table()
BANK = q.default_bank()


def board(**kw) -> TaskBoard:
  return TaskBoard(clock=lambda: "2026-08-23T00:00:00", **kw)


def question_task(b: TaskBoard, ask="2 + 3", answer="5", target="whiteboard_a"):
  task = b.offer("whiteboard_answer", target, params={"question": ask},
                 secret={"answer": answer}, t=0.0)
  assert task is not None
  return task


def ink_for(answer: str, board_name="whiteboard_a") -> BoardBook:
  """A board carrying exactly the glyphs of `answer`, decimated as the pen's
  own trace would have been."""
  book = BoardBook([BoardRecord(board_name, (0.11, 0.2))],
                   clock=lambda: "2026-08-23T00:00:00")
  for stroke in q.answer_strokes(answer):
    book.stroke(board_name, "answer", decimate(list(stroke)))
  return book


def fake_life(book=None, tasks=None, time=10.0):
  return SimpleNamespace(boards=book, tasks=tasks,
                         data=SimpleNamespace(time=time))


def measure(book, task, board_name="whiteboard_a", **result):
  """What `run_errand` would hand the evaluator for a finished answer."""
  errand = Errand(name=f"draw:{board_name}", module="module_pen", station_y=0.0,
                  use_at=(0, 0), task="answer", task_id=task.id,
                  detail={"board": board_name, "strokes": 1})
  life = fake_life(book, tasks=_Board(task))
  report = {"drew": True, "strokes": 1, "strokes_drawn": 1,
            "shape_rms_mm": 0.7, "inked_fraction": 0.99,
            "travel_ink_fraction": 0.0, **result}
  return scoring.score_errand(life, errand, report,
                              before={"board": board_name, "strokes": 0,
                                      "clears": 0, "lines": 0}, table=TABLE)


class _Board:
  """Just enough task board for a sampler: `get(id)`."""

  def __init__(self, *tasks):
    self.tasks = {t.id: t for t in tasks}

  def get(self, task_id):
    return self.tasks.get(task_id)


# ---- the bank is DATA ---------------------------------------------------------


def test_the_shipped_bank_asks_things_this_pen_can_answer():
  """Every question has an answer the robot could physically write. A bank
  entry answering "SEVEN" would look like a robot that is bad at arithmetic
  rather than one that was asked something it cannot draw."""
  assert len(BANK) >= 4
  envelope = Envelope(-0.055, 0.055, -0.1, 0.1)
  for question in BANK.questions:
    assert question.ask.strip() and question.answer
    assert q.clean_answer(question.answer) == question.answer
    figure = strokes.program("answer", text=question.answer)
    assert figure.fits(envelope), \
      f"{question.id}: {question.answer!r} does not fit the pen's reach"


def test_a_new_question_needs_no_release(tmp_path, monkeypatch):
  """The acceptance criterion: question templates are data.

  A file this build has never seen, pointed at with $PLUGGY_QUESTIONS, and
  the world asks it. Nothing here was recompiled and no evaluator changed.
  """
  path = tmp_path / "questions.json"
  path.write_text(json.dumps({"version": 1, "questions": [
    {"id": "novel", "ask": "9 - 2", "answer": "7"}]}))
  monkeypatch.setenv(q.BANK_ENV, str(path))
  bank = q.QuestionBank.load()
  assert bank.pick(0).ask == "9 - 2" and bank.pick(0).answer == "7"
  b = board()
  task = b.offer("whiteboard_answer", "whiteboard_a",
                 params={"question": bank.pick(0).ask},
                 secret={"answer": bank.pick(0).answer}, t=0.0)
  assert task is not None
  assert "9 - 2" in task.description and task.secret == {"answer": "7"}


def test_a_question_the_pen_could_not_answer_is_refused(tmp_path):
  """Refused at LOAD, not silently trimmed. "12.5" trimmed to "12" would be a
  job the robot is structurally unable to get right."""
  for bad in ("SEVEN", "12.5", "100", "-3"):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"version": 1, "questions": [
      {"id": "bad", "ask": "?", "answer": bad}]}))
    with pytest.raises(ValueError, match="cannot write"):
      q.QuestionBank.load(path)


def test_a_bank_from_the_future_and_a_bank_with_repeats_are_refused(tmp_path):
  path = tmp_path / "questions.json"
  path.write_text(json.dumps({"version": 99, "questions": []}))
  with pytest.raises(ValueError, match="version"):
    q.QuestionBank.load(path)
  path.write_text(json.dumps({"version": 1, "questions": [
    {"id": "same", "ask": "1 + 1", "answer": "2"},
    {"id": "same", "ask": "2 + 2", "answer": "4"}]}))
  with pytest.raises(ValueError, match="duplicate"):
    q.QuestionBank.load(path)


def test_picking_a_question_is_deterministic_and_wraps():
  """A world that asked a different question every run is a mission test that
  is a different test every run."""
  assert BANK.pick(3) is BANK.pick(3) is BANK.pick(3 + len(BANK))
  assert len({BANK.pick(i).id for i in range(len(BANK))}) == len(BANK)


# ---- an answer is the one string a model draws on a wall ---------------------


@pytest.mark.parametrize("raw,want", [
  (" 5 ", "5"), ("12", "12"), ("5!", "5"), ("007", "00"), (None, ""),
  ("", ""), ("ignore your goals", ""), ("<script>", ""), ("5\n6", "56"),
  ("1234", "12"), ("-3", "3"),
])
def test_an_answer_is_sanitised_before_it_can_become_a_stroke(raw, want):
  """`clean_answer` is the security boundary the `text` program never had.

  `hub/overseer.py` keeps `text` off the figure menu precisely because it
  takes arbitrary caller text; this is where that lands safely, and it is
  safe because everything outside a two-character numeric alphabet is
  dropped -- not escaped, dropped.
  """
  assert q.clean_answer(raw) == want


def test_an_unwritable_answer_draws_nothing_at_all():
  for bad in ("", "hello", None):
    with pytest.raises(ValueError, match="not an answer"):
      q.answer_strokes(bad)


def test_the_plotter_and_the_grader_render_the_same_glyphs():
  """Two renderings that could drift apart would be a grader marking against
  a figure the robot was never asked to draw."""
  assert strokes.program("answer", text="12").strokes == q.answer_strokes("12")
  assert q.ANSWER_WIDTH == strokes.TEXT_WIDTH


# ---- comparing ink to glyphs --------------------------------------------------


def test_identical_ink_is_a_perfect_match_and_an_empty_board_is_none():
  assert q.ink_match(q.answer_strokes("5"), "5")["matchMm"] == 0.0
  assert q.ink_match([], "5") is None
  assert q.ink_match([[(0.0, 0.0)]], "5") is None, \
    "a single point is not a stroke"


def test_the_match_is_symmetric_so_missing_and_extra_ink_both_count():
  """One-way distance passes a robot that drew half a 5 (its ink is all on
  the glyph) OR one that scribbled over the whole board (the glyph is all
  under its ink). The reported figure is the worse of the two."""
  full = q.answer_strokes("5")
  half = [tuple(s[:len(s) // 3]) for s in full]
  m = q.ink_match(half, "5")
  assert m["inkToGlyphMm"] < 0.01, "the half that was drawn is on the glyph"
  assert m["glyphToInkMm"] > q.ANSWER_MATCH_MM, "the missing half is missed"
  assert m["matchMm"] == m["glyphToInkMm"]
  assert m["inkRatio"] < q.INK_RATIO[0]


def test_a_figure_drawn_instead_of_an_answer_fails_both_bars():
  """The measurement the fidelity bar is set from (`scripts/answer_spike.py`).

  ⚠ 5.05 mm with the real pen, NOT the 20 mm two such different figures
  suggest: a busy figure covers the glyph it is standing in for, so the
  glyph-to-ink direction reads 1.5 mm and only ink-to-glyph notices. The
  first version of this bar was 8.0 mm, set from synthetic renderings, and
  the robot figure went straight through it.
  """
  m = q.ink_match(strokes.program("robot").strokes, "5")
  assert m["matchMm"] > q.ANSWER_MATCH_MM
  assert m["glyphToInkMm"] < q.ANSWER_MATCH_MM, \
    "if this ever fails, the one-way distance would have been enough"
  assert m["inkRatio"] > q.INK_RATIO[1]


def test_the_ink_cannot_tell_a_six_from_an_eight():
  """NOT a guard -- the measurement that decides the whole design.

  A correctly drawn answer sits ~1.2 mm from its own ideal (measured, real
  pen). A 6 and an 8 sit closer together than that. So a grader that decided
  CORRECTNESS by reading the board would fail correct drawings and pass wrong
  ones, roughly at random, on exactly the pairs arithmetic produces -- which
  is why correctness is decided against the answer the mind committed to and
  the ink is only ever a fidelity check.

  If this test ever fails because the glyphs moved apart, the design could be
  revisited. It must not be deleted to make room for one that assumes it.
  """
  confusable = q.ink_match(q.answer_strokes("6"), "8")["matchMm"]
  assert confusable < 3.0, (
    f"a 6 and an 8 are {confusable:.2f} mm apart, against ~1.2 mm of real "
    f"drawing error -- still inside the noise")
  assert confusable < q.ANSWER_MATCH_MM, \
    "the fidelity bar cannot be tightened into handwriting recognition"


# ---- the evaluator ------------------------------------------------------------


def test_a_right_answer_written_legibly_pays():
  b = board()
  task = b.claim(question_task(b).id, t=1.0, answer="5")
  v = measure(ink_for("5"), task)
  assert v.ok and v.points > 0
  assert v.metrics["correct"] and v.metrics["wrote"] == "5"
  assert v.metrics["matchMm"] < q.ANSWER_MATCH_MM
  assert "correct" in v.reason


def test_a_wrong_answer_pays_nothing_however_neatly_it_is_written():
  """The failable outcome the task kind exists for, and the explicit design
  call issue #22 asks for: NO partial credit for a legible wrong answer.

  Beautiful handwriting scales the bonus on a right answer; it cannot buy a
  wrong one. A consolation payout for showing up is the gradient that teaches
  a robot to attempt the cheapest task it can fail at.
  """
  b = board()
  task = b.claim(question_task(b).id, t=1.0, answer="6")
  v = measure(ink_for("6"), task)
  assert not v.ok and v.points == 0
  assert v.metrics["wrote"] == "6" and not v.metrics["correct"]
  assert v.metrics["matchMm"] < q.ANSWER_MATCH_MM, \
    "the 6 was drawn beautifully, and it still pays nothing"
  assert "wrong" in v.reason


def test_the_right_answer_on_a_board_showing_something_else_fails():
  """The ink is not decoration. A right answer whose board carries a house
  is a job that did not get done, whatever was committed to."""
  b = board()
  task = b.claim(question_task(b).id, t=1.0, answer="5")
  book = BoardBook([BoardRecord("whiteboard_a", (0.11, 0.2))],
                   clock=lambda: "2026-08-23T00:00:00")
  for stroke in strokes.program("robot").strokes:
    book.stroke("whiteboard_a", "answer", decimate(list(stroke)))
  v = measure(book, task)
  assert not v.ok and v.points == 0
  assert "does not show it" in v.reason or "not what got drawn" in v.reason


@pytest.mark.parametrize("missing", ["wrote", "expected"])
def test_a_missing_measurement_is_not_a_passing_one(missing):
  """The census's lesson, arriving at a new task: an absence that compared
  EQUAL would score a question nobody answered as correct."""
  m = {"wrote": "5", "expected": "5", "question": "2 + 3",
       "board": "whiteboard_a", "strokes": 1, "strokesInked": 1,
       "matchMm": 1.0, "inkRatio": 0.9}
  assert scoring.evaluate("answer", m, table=TABLE).ok
  m[missing] = ""
  v = scoring.evaluate("answer", m, table=TABLE)
  assert not v.ok and v.points == 0


def test_an_answer_with_no_ink_behind_it_scores_nothing():
  b = board()
  task = b.claim(question_task(b).id, t=1.0, answer="5")
  blank = BoardBook([BoardRecord("whiteboard_a", (0.11, 0.2))],
                    clock=lambda: "2026-08-23T00:00:00")
  v = measure(blank, task)
  assert not v.ok and v.points == 0 and "no ink" in v.reason


def test_a_lying_caller_cannot_score():
  """Acceptance: grading reads the BOARD BOOK, not the requested program.

  The errand reports a flawless drawing of the right answer. The board says
  otherwise, twice over -- it is blank, and then it carries a different
  figure. Neither pays. Only ink in the shape of the committed answer does.
  """
  b = board()
  task = b.claim(question_task(b).id, t=1.0, answer="5")
  perfect = {"strokes": 1, "strokes_drawn": 1, "shape_rms_mm": 0.1,
             "inked_fraction": 1.0, "travel_ink_fraction": 0.0,
             # ...including a straight-out lie about the grader's own metrics
             "matchMm": 0.0, "inkRatio": 1.0, "wrote": "5", "expected": "5"}
  blank = BoardBook([BoardRecord("whiteboard_a", (0.11, 0.2))],
                    clock=lambda: "2026-08-23T00:00:00")
  assert not measure(blank, task, **perfect).ok, \
    "a claim about a blank board paid out"
  # ...and ink that is real but is not the answer. This is the case the
  # report's own `matchMm` would rescue: the caller says the board is a
  # perfect 5, the board is a 12, and the grader has to believe the board.
  assert not measure(ink_for("12"), task, **perfect).ok, \
    "the caller's own account of the ink was taken at face value"
  assert measure(ink_for("5"), task, **perfect).ok, \
    "the same report over real ink must still score"


def test_the_right_answer_never_leaves_the_evaluator():
  """The stream reaches the site AND the robot's own context. A question that
  published its answer would arrive pre-solved next time."""
  b = board()
  task = b.claim(question_task(b).id, t=1.0, answer="6")
  v = measure(ink_for("6"), task)
  assert v.metrics["expected"] == "5", "the evaluator did compare against it"
  assert "expected" not in v.public_metrics()
  assert "5" not in v.reason
  ledger = Ledger(table=TABLE, clock=lambda: "2026-08-23T00:00:00")
  entry = ledger.award(v, t=1.0)
  assert "expected" not in json.dumps(entry)
  assert "expected" not in json.dumps(v.as_dict())


# ---- a question is a job for a MIND -------------------------------------------


def test_a_question_cannot_be_claimed_without_an_answer():
  b = board()
  task = question_task(b)
  assert task.needs_answer
  assert b.claim(task.id, t=1.0) is None
  assert b.claim(task.id, t=1.0, answer="hello") is None, \
    "a sanitised-to-nothing answer is no answer"
  assert b[task.id].state == "offered", "a refused claim must not fail the job"
  claimed = b.claim(task.id, t=1.0, answer="5")
  assert claimed.state == "claimed" and claimed.answer == "5"


def test_the_commitment_is_frozen_at_claim_time():
  """Correctness is decided against this, so a commitment that could be
  edited once the ink was down would not be a commitment."""
  b = board()
  task = question_task(b)
  b.claim(task.id, t=1.0, answer="6")
  assert b.claim(task.id, t=2.0, answer="5") is None, "a second bite"
  assert b[task.id].answer == "6"
  errand = lc.errand_for_task(b[task.id], "home", lc.board_book("home"))
  assert errand is not None
  # The errand is handed the GLYPHS and nothing else: no question, no right
  # answer, no `secret`, nothing it could be wrong about.
  assert set(errand.detail) == {"board", "figure", "strokes", "ink_m"}
  assert errand.detail["figure"] == "answer"
  assert errand.task_id == task.id and errand.task == "answer"


def test_a_robot_with_no_mind_leaves_the_question_standing():
  """Rule 1, at both places a job can be taken without an LLM.

  The scripted rotation has no arithmetic to offer, and the two ways code
  could get an answer are worse than not taking the job: reading it out of
  the bank is the sim marking its own homework, and guessing puts a confident
  wrong number on a wall.
  """
  menu = Menu.for_world("home", lc.board_book("home"))
  b = board()
  question = question_task(b)
  state = {"offeredTasks": [t.as_context(1.0, 5.0) for t in b.offered()],
           "tasksThisMission": [], "decisions": 0}
  assert scripted(menu, state, "test").action != "take_task", \
    "the scripted policy took on a question it cannot answer"
  # ...and the loop's own claim branch skips it rather than failing it.
  life = SimpleNamespace(tasks=b, spendable_wh=5.0,
                         data=SimpleNamespace(time=1.0))
  assert not lc.HubLifecycle._claim_next_task(life)
  assert b[question.id].state == "offered"
  # An ordinary job on the same board is still taken -- it is the QUESTION
  # that is skipped, not the task branch that is broken.
  b.offer("draw_figure", "whiteboard_a", params={"program": "house"}, t=0.0)
  state["offeredTasks"] = [t.as_context(1.0, 5.0) for t in b.offered()]
  assert scripted(menu, state, "test").action == "take_task"


def test_the_overseer_is_told_a_job_asks_something_and_must_answer_it():
  """`needsAnswer` on the offer, `answer` in the schema, and a `take_task`
  without one refused -- a claim missing its answer would be refused by the
  board a moment later, and half a decision costs a whole turn."""
  b = board()
  task = question_task(b)
  assert task.as_context(1.0, 5.0)["needsAnswer"] is True
  menu = Menu.for_world("home", lc.board_book("home"))
  assert "answer" in menu.schema()["required"]
  raw = {"action": "take_task", "task": task.id, "reason": "I know this one"}
  with pytest.raises(ValueError, match="asks a question"):
    menu.validate(raw, offered=(task.id,), answering=(task.id,))
  good = menu.validate({**raw, "answer": " 5 "}, offered=(task.id,),
                       answering=(task.id,))
  assert good.action == "take_task" and good.answer == "5"
  assert good.as_dict()["answer"] == "5"
  # ...and an answer attached to anything else is dropped: it is the one
  # string a model draws on a wall, and it may only ride on the job that
  # asked for it.
  idle = menu.validate({"action": "idle", "reason": "", "answer": "7"})
  assert idle.answer == ""


def test_the_answer_survives_a_restart_but_never_reaches_the_wire(tmp_path):
  """An offer that came back with no right answer behind it could never be
  graded. The state file is not the wire -- it sits in /var/lib/pluggybot
  beside the reward table, which also decides what things are worth."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = question_task(b)
  b.claim(task.id, t=1.0, answer="5")
  saved = json.loads(path.read_text())
  assert saved["tasks"][0]["secret"] == {"answer": "5"}
  back = TaskBoard(path=path)[task.id]
  assert back.secret == {"answer": "5"} and back.answer == "5"
  # ...and none of the three published shapes carries either.
  for shape in (back.as_dict(), back.snapshot(TABLE), back.as_context(2.0, 5.0)):
    assert "secret" not in shape and "expected" not in json.dumps(shape)


def test_the_kind_is_scoreable_and_prices_nothing_itself():
  kind = KINDS["whiteboard_answer"]
  assert kind.task in scoring.EVALUATORS and kind.task in TABLE
  assert kind.needs_answer and kind.target_kind == "board"
  # The issue sketched "Worth 2 PluggyPoints. Draw the answer to..." -- and a
  # description that quoted a price would go stale the first time the table
  # was re-tuned, with the stale figure being the half a person reads.
  b = board()
  task = question_task(b)
  assert "point" not in task.description.lower()
  assert task.reward(TABLE)["base"] == TABLE["answer"].base
  assert not any(f.name in ("base", "bonus", "points_in")
                 for f in dataclasses.fields(task))


def test_the_question_rotates_across_restarts():
  """A deployed robot works through the bank instead of being asked the same
  thing every morning. The question is picked off the BOARD's own sequence
  number, which is what survives in the state file -- so the rotation is a
  property of the WORLD's history, not of a counter that resets on boot.

  ⚠ The picking moved out of `lifecycle.seed_tasks` and into
  `cadence.TaskProducer` with issue #23, and it moved because that placeholder
  put a starter set up once and nothing ever asked again. What did NOT move is
  which counter decides -- keep this test pointed at the seq, not at the
  producer, and it goes on being true wherever the picking lives.
  """
  book = lc.board_book("home")
  asked = []
  for seq in range(3):
    b = board()
    b.seq = seq * 10                      # ...as a board with a past would be
    lc.task_producer(b, "home", book).seed(0.0)
    asked += [t.params["question"] for t in b.tasks.values()
              if t.kind == "whiteboard_answer"]
  assert len(set(asked)) == 3, f"the world asked {asked} three mornings running"


def test_the_home_world_asks_a_question():
  """End of the wiring: the world puts one up on its own."""
  book = lc.board_book("home")
  b = board()
  lc.task_producer(b, "home", book).seed(0.0)
  asked = [t for t in b.tasks.values() if t.kind == "whiteboard_answer"]
  assert len(asked) == 1
  task = asked[0]
  assert task.target in book.names
  assert task.params["question"] == BANK.pick(0).ask
  assert task.secret == {"answer": BANK.pick(0).answer}
  assert task.params["question"] in task.description


# ---- the whole way round ------------------------------------------------------


class Arithmetician:
  """An overseer client that reads the offers and answers the question.

  A stand-in for the real Haiku call, and deliberately a stand-in for a MIND
  rather than a shortcut: it sees exactly what the model sees -- the volatile
  user turn, with `offeredTasks` in it -- and answers from the question text,
  looking the sum up in the bank the way a model would know it. It has no
  access to `Task.secret`, which is the only way it could cheat.

  ⚠ Not a fixture with a hardcoded id: the ids change with what the seed put
  up and with which offers have lapsed, so a test that named one would be
  testing its own bookkeeping.
  """

  def __init__(self) -> None:
    self.messages = self
    self.answered: list[tuple[str, str]] = []
    self.asks = {question.ask: question.answer for question in BANK.questions}

  def create(self, **kwargs):
    state = json.loads(kwargs["messages"][0]["content"]
                       .split("\n\n", 1)[1].rsplit("\n\n", 1)[0])
    for offer in state.get("offeredTasks", ()):
      if offer.get("needsAnswer") and offer.get("claimable"):
        ask = offer["description"].split(": ", 1)[-1]
        answer = self.asks.get(ask, "")
        self.answered.append((offer["id"], answer))
        return _FakeResponse({"action": "take_task", "task": offer["id"],
                              "answer": answer, "reason": f"{ask} is {answer}",
                              "board": "", "program": "", "zone": "",
                              "note": "", "respond_to": "", "outcome": "",
                              "reply": ""})
    return _FakeResponse({"action": "explore", "reason": "nothing to answer",
                          "board": "", "program": "", "zone": "", "note": "",
                          "respond_to": "", "outcome": "", "reply": "",
                          "task": "", "answer": ""})


class _FakeResponse:
  def __init__(self, payload):
    self.content = [SimpleNamespace(type="text", text=json.dumps(payload))]
    self.usage = SimpleNamespace(input_tokens=1200, output_tokens=60,
                                 cache_read_input_tokens=0,
                                 cache_creation_input_tokens=0)


@pytest.mark.slow
def test_a_question_is_asked_answered_and_graded_twice_unattended():
  """THE acceptance test for issue #22: the whole way round, twice, on the
  world the site serves.

  A real home mission -- explore, battery arbitration, fetch the pen, drive to
  the board, erase, write, stow, score, bank, close the offer -- with a mind
  that answers arithmetic and no human anywhere in it. Then again, because
  the second time is a different test: the pen starts from where the first
  cycle left it (issue #10, and the reason `--cycles 2` exists), the board
  already has yesterday's answer on it, and the evaluator has to score the
  ink THIS errand laid down rather than what was already there.

  ⚠ The two offers are put up DIRECTLY rather than through `seed_tasks`, and
  they are the only work on the board. The seed's mixture is about giving a
  recording an expiry and a drawing to show; paying two more errands' worth
  of mission time to re-confirm those here would buy nothing, and a question
  that lost a race with a house would make this test flaky rather than
  informative.

  ⚠ AND IT RUNS ON THE HOUSE'S OWN 1.1 Wh CELL, so a CHARGE CYCLE lands
  between the two questions -- an answer errand costs ~0.74 Wh here, so two
  of them cannot fit in one pack and the arbitration loop has to go to the
  rack in the middle. That is not incidental: this test is the first thing in
  the repo to fetch the SAME tool twice across a charge, and the first time
  it ran it found a defect nothing else could (`tests/test_rack_belief.py`) --
  the charge trip drives behind the rack, the free-space sum it takes the
  rack's FACING from stops being conditioned, and the bay standoff computed
  from a 20 deg-rotated rack sat a metre from the bay. Give this test a
  bigger battery and that whole path stops being exercised.
  """
  import mujoco
  from pluggybot.hub.journal import Journal
  from pluggybot.hub.mission import MissionAborted
  from pluggybot.hub.overseer import Overseer

  cfg = lc.world_config("home")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  book = lc.board_book("home")
  tasks = board()
  mind = Arithmetician()
  boss = Overseer(Menu.for_world("home", book), client=mind)
  ledger = Ledger(table=TABLE, clock=lambda: "2026-08-23T00:00:00")
  life = lc.HubLifecycle(
    model, data, realtime=False, world="home", battery_wh=cfg["battery_wh"],
    rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
    low_battery_wh=cfg["low_battery_wh"], boards=book, ledger=ledger,
    tasks=tasks, overseer=boss, journal=Journal(), errand=False)
  # Two DIFFERENT questions, on the same board, both standing until taken.
  board_name = next(iter(book.names))
  asked = [tasks.offer("whiteboard_answer", board_name,
                       params={"question": BANK.pick(i).ask},
                       secret={"answer": BANK.pick(i).answer}, t=0.0)
           for i in (0, 1)]
  assert all(asked) and asked[0].secret != asked[1].secret

  # Stop the moment the second verdict lands. Without it the mission runs its
  # whole budget deciding what to do with an afternoon it has no work left in,
  # which is honest robot behaviour and 300 s of nothing in the suite.
  def stop_when_both_are_graded():
    if all(tasks[t.id].state in ("done", "failed") for t in asked):
      raise MissionAborted

  life.mission.step_hooks.append(stop_when_both_are_graded)
  life.run(cfg["start"], max_sim_time=900.0, explore_budget=10.0)

  done = [tasks[t.id] for t in asked if tasks[t.id].state != "offered"]
  assert len(done) == 2, (
    f"{len(done)} of 2 questions were attempted -- "
    f"{[tasks[t.id].state for t in asked]}")
  for task in done:
    assert task.state == "done", f"{task.description}: {task.verdict}"
    assert task.answer == task.secret["answer"], "the mind got it wrong"
    assert task.points > 0
    verdict = task.verdict
    assert verdict["ok"] and verdict["metrics"]["correct"]
    assert verdict["metrics"]["matchMm"] <= q.ANSWER_MATCH_MM
    # ...and what closed the task is what PAID for the errand: one evaluation,
    # two consumers.
    assert any(e["reason"] == verdict["reason"] and e["task"] == "answer"
               for e in ledger.entries())
    # The right answer is never on the wire, even when it is also the ink.
    assert "expected" not in json.dumps(task.snapshot(TABLE))
  # The second answer was scored on the second answer's ink. Without the
  # erase, or with `_errand_lines` reading the whole board, the second
  # verdict would be measured against both answers at once.
  assert {t.answer for t in done} == {t.secret["answer"] for t in asked}
  assert len(mind.answered) >= 2


def test_answer_is_not_a_figure_anyone_may_ask_for():
  """`text` is off the overseer's figure menu because it takes arbitrary
  caller text; `answer` is where that text landed, so it is off it too.

  Not a style rule: chosen as a `draw` program it would take its default and
  write a lone "0" on a wall for no reason, and the string it writes is only
  safe because a question task put it through `clean_answer` first.
  """
  menu = Menu.for_world("home", lc.board_book("home"))
  assert "answer" in strokes.PROGRAMS
  assert "answer" not in menu.programs and "text" not in menu.programs
  assert set(menu.programs) == set(strokes.LIBRARY) | {"square", "circle"}
  with pytest.raises(ValueError, match="unknown program"):
    menu.validate({"action": "draw", "program": "answer", "board": "",
                   "zone": "", "reason": "", "note": ""})
