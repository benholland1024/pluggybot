"""Every fall a row, and a fall nothing else told a line (issue #707): a
`fall` event at every fall the body's own watch counts (`Body.falls`), sent
as its get-up ends -- where it fell and what it was doing (`at`, a death's
shape), how far the get-up moved the body and the belief, how it ended and
what told the robot -- and ONE History line where nothing else did: no run's
verb (#671), no hand job (#701), no step aside, no death. Each rule on the
stub, nothing flown.
"""

import math

import pytest

from pluggybot import lifecycle as lc
from pluggybot.lifecycle import FELL_LINE, STOOD_AGAIN_S, Stuck, world_facts
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import HISTORY
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import lang
from pluggybot.telemetry.protocol import FALL_ENDINGS, FALL_TOLD
from test_body import stub_life


def _rows(life) -> list[dict]:
  """The `fall` rows `life` sends, as the wire takes them."""
  rows: list[dict] = []
  life.on_event.append(lambda m: rows.append(m) if m.get("type") == "fall" else None)
  return rows


def _hold(life, seconds: float) -> None:
  life.body.run(life.body.hold_routine(seconds))


def _fall(life) -> float:
  """Its body's watch counts a fall and its get-up takes the legs, as the
  humanoid's do in one step: when."""
  life.body.falls += 1
  life.body.posture = "getting_up"
  return float(life.data.time)


def _history(life) -> list[str]:
  return life.thoughts.texts[HISTORY].splitlines()


def _fell_lines(life) -> list[str]:
  """History's lines in `FELL_LINE`'s words, whatever their seconds."""
  head = FELL_LINE.split("{")[0]
  return [ln for ln in _history(life) if head in ln]


def _vocabulary(row: dict) -> None:
  assert row["ended"] in FALL_ENDINGS and (row["told"] is None or row["told"] in FALL_TOLD), row


def test_a_fall_counted_is_one_row_sent_as_its_get_up_ends():
  """Where it fell, the truth beside the belief and what it was doing (a
  death's `at`), and as its get-up stands it how far that moved the body and
  the belief, the belief's error then and how long it took -- once, however
  long it stands after. Shown to fail with no `_fall_row_step`: no row."""
  life = stub_life()
  rows = _rows(life)
  life.state = "EXPLORE"
  life.body.x, life.body.y = 2.0, 1.0
  _hold(life, 0.5)
  t0 = _fall(life)
  truth = [2.0, 1.0, 0.0]
  life.body.true_pose = lambda: tuple(truth)
  _hold(life, 1.0)
  truth[:] = [3.2, 1.5, math.radians(30.0)]      # ...the get-up carried the body
  assert rows == [], "no row while its get-up has its legs"
  _hold(life, 2.0)
  life.body.posture = "standing"
  _hold(life, 1.0)
  [row] = rows
  _vocabulary(row)
  assert row["robot"] == life.root and row["falls"] == 1 and row["ended"] == "stood"
  assert row["fellT"] == pytest.approx(t0, abs=0.01)
  assert row["seconds"] == pytest.approx(3.0, abs=0.01) and row["t"] == pytest.approx(t0 + 3.0, abs=0.01)
  at = row["at"]
  assert at["state"] == "EXPLORE" and at["errand"] is None and at["step"] is None
  assert at["pose"] == {"x": 2.0, "y": 1.0, "yawDeg": 0.0} == at["believed"]
  assert row["after"] == {"pose": {"x": 3.2, "y": 1.5, "yawDeg": 30.0},
                          "believed": {"x": 2.0, "y": 1.0, "yawDeg": 0.0}}
  assert row["movedM"] == pytest.approx(math.hypot(1.2, 0.5), abs=1e-3)
  assert row["turnedDeg"] == pytest.approx(30.0, abs=0.1)
  assert row["beliefMovedM"] == 0.0 and row["beliefTurnedDeg"] == 0.0
  assert row["errorM"] == pytest.approx(math.hypot(1.2, 0.5), abs=1e-3)
  assert row["headingDeg"] == pytest.approx(-30.0, abs=0.1)
  assert row["struggle"] == 1 and row["told"] == "line"
  _hold(life, 10.0)
  assert len(rows) == 1 and life.fall_rows == rows


def test_a_fall_between_runs_is_one_history_line_and_its_narration():
  """The robot's own watch and its own clock, nothing of the truth: told
  nothing, a robot tipped on a charge trip or at its station and carried by
  its get-up read nothing of it (#707: Rowan, 11-14 17:32). Shown to fail
  with no `_fall_row_step`: no line."""
  life = stub_life()
  said: list = []
  life.say_hooks.append(lambda t, msg: said.append(msg))
  _fall(life)
  _hold(life, 4.0)
  assert _fell_lines(life) == [], "nothing said while it is down"
  life.body.posture = "standing"
  _hold(life, 1.0)
  line = FELL_LINE.format(seconds=4.0)
  assert line == "I fell over and got up again 4 s later"
  assert [ln for ln in _history(life) if ln.endswith(line)] == _fell_lines(life)
  assert len(_fell_lines(life)) == 1
  assert said.count(f"FELL: {line}") == 1


def test_a_fall_inside_a_runs_verb_is_the_verbs_words_and_no_second_line():
  """A run a fall ends says it in its own line (#671's `FELL`): the row is
  the run's and its verb's, and no line of the fall's own."""
  life = stub_life()
  rows = _rows(life)
  t0 = float(life.data.time)

  def fall_at_two_s():
    if not life.body.falls and float(life.data.time) >= t0 + 2.0:
      _fall(life)
  life.body.step_hooks.insert(0, fall_at_two_s)
  proc = lang.compile_procedure("def lap():\n    wait(10)\n", world_facts(life.world))
  life.run_errand(programmed_errand(proc, name="lap"))
  _hold(life, 3.0)
  life.body.posture = "standing"
  _hold(life, 1.0)
  [row] = rows
  _vocabulary(row)
  assert row["told"] == "run" and row["ended"] == "stood"
  assert row["at"]["state"] == "PROCEDURE" and row["at"]["errand"]["name"] == "lap"
  assert {k: row["at"]["step"][k] for k in ("procedure", "line", "verb")} == {
    "procedure": "lap", "line": 2, "verb": "wait"}
  assert any("the procedure lap did not finish" in ln and "you fell" in ln
             for ln in _history(life)), _history(life)
  assert _fell_lines(life) == []


def test_two_falls_in_one_struggle_are_two_rows_and_two_lines_as_the_body_counts_them():
  """A get-up's second fall under `STOOD_AGAIN_S` after it stood is one
  struggle on the `stuck` clock, and two falls by the body's watch: a row
  and a line each, the row saying which of the struggle's falls it was, as
  that clock counts it (`_down_since`, the death seam's: two clocks parted
  at their edges); a fall after it stood longer begins a struggle of its
  own."""
  life = stub_life()
  life.body.rights_itself, life.body.stuck_after_s = True, 60.0   # ...its clock running
  rows = _rows(life)
  for down, up in ((2.0, 1.0), (3.0, STOOD_AGAIN_S + 1.0), (2.0, 1.0)):
    _fall(life)
    _hold(life, down)
    life.body.posture = "standing"
    _hold(life, up)
  assert [(r["falls"], r["struggle"]) for r in rows] == [(1, 1), (2, 2), (3, 1)]
  assert [ln.split("] ", 1)[-1] for ln in _fell_lines(life)] == [
    FELL_LINE.format(seconds=s) for s in (2.0, 3.0, 2.0)]


@pytest.mark.parametrize("since, struggle", [(0.0, 1), (1.0, 2)])
def test_a_fall_the_stuck_clock_stamped_as_it_came_begins_a_struggle_of_its_own(since, struggle):
  """The death seam stamps a fall's down-start in the step it comes, before
  the row's hook reads it: a clock begun THEN is this fall's own, and only
  one already running is the struggle it joins."""
  life = stub_life()
  t = float(life.data.time)
  life._down_since = t - since
  life._struggle = 1
  life.body.falls += 1
  life._fall_read()
  assert life._fall_open["struggle"] == struggle


def test_a_fall_holding_a_job_is_the_jobs_line_alone(tmp_path):
  """A hand job a fall ends says so as it falls (#701), its grade the fall
  first: the fall's own line would read as a second fall."""
  from test_hand_jobs import _life, _offer
  life = _life(tmp_path)
  rows = _rows(life)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  _fall(life)
  _hold(life, 2.0)
  life.body.posture = "standing"
  _hold(life, 6.0)
  [row] = rows
  assert row["told"] == "job" and row["ended"] == "stood"
  assert sum("fell holding" in ln for ln in _history(life)) == 1
  assert _fell_lines(life) == []


def test_a_fall_in_a_dones_hold_is_its_grades_and_no_line(tmp_path):
  """...and a `done` holding: its grade says the fall first (#701)."""
  from test_hand_jobs import _life, _offer
  life = _life(tmp_path)
  rows = _rows(life)
  task = _offer(life, "open_drawer")
  assert life._claim_task(task.id)
  t0 = float(life.data.time)

  def fall_in_the_hold():
    if not life.body.falls and float(life.data.time) > t0 + 2.0:
      _fall(life)
    elif life.body.falls and float(life.data.time) > t0 + 3.0:
      life.body.posture = "standing"
  life.body.step_hooks.insert(0, fall_in_the_hold)
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  life.body.run(life._grade_routine())
  assert life.tasks.get(task.id).verdict["reason"].startswith("you fell")
  [row] = rows
  assert row["told"] == "job" and _fell_lines(life) == []


@pytest.mark.parametrize("why", ["fell", "its own walk"])
def test_a_fall_stepping_aside_is_the_asides_line_alone_where_it_says_so(why):
  """A step aside a fall ends says "I fell over" in its own line
  (`ASIDE_ENDED`), as it says it: one a walk of its own ended first says
  nothing of it, and the fall says itself."""
  life = stub_life()
  rows = _rows(life)
  life.body.making_way = "r2_pluggybot"
  _fall(life)
  _hold(life, 0.01)
  life.body.making_way = None                    # ...ended the step after, as `_aside_ready` does
  life.body.asides, life.body.last_aside = 1, {
    "by": "r2_pluggybot", "from": (0.0, 0.0), "to": (0.5, 0.0), "at": (0.1, 0.0),
    "seconds": 1.0, "why": why}
  _hold(life, 2.0)
  life.body.posture = "standing"
  _hold(life, 1.0)
  [row] = rows
  said = [ln for ln in _history(life) if "set off to step aside" in ln]
  assert len(said) == 1 and said[0].endswith(lc.ASIDE_ENDED[why]), said
  if why == "fell":
    assert row["told"] == "aside" and _fell_lines(life) == []
  else:
    assert row["told"] == "line" and len(_fell_lines(life)) == 1


def test_a_fall_in_a_house_programs_verb_is_the_falls_own_line():
  """A #58 program leaves no line of its run (its task's verdict is its
  word, "step 1 failed"): only a run whose line says the fall tells it, and
  this one is the fall's own."""
  from pluggybot.procedure.steps import Program, Step
  life = stub_life()
  rows = _rows(life)
  t0 = float(life.data.time)

  def fall_at_two_s():
    if not life.body.falls and float(life.data.time) >= t0 + 2.0:
      _fall(life)
  life.body.step_hooks.insert(0, fall_at_two_s)
  run = life.run_errand(programmed_errand(
    Program.single("p", [Step("wait", {"seconds": 10.0})])))["procedure"]
  assert run["steps"][-1]["fell"]
  _hold(life, 2.0)
  life.body.posture = "standing"
  _hold(life, 1.0)
  [row] = rows
  assert row["told"] == "line" and row["at"]["errand"]["name"] == "program:p"
  assert len(_fell_lines(life)) == 1


def test_a_fall_its_robot_dies_in_is_the_deaths_line_alone():
  """A get-up that never stands is the `stuck` death, said in its own words:
  the row ends there, the death's cause on it."""
  life = stub_life(mortal=True)
  rows = _rows(life)
  _fall(life)
  _hold(life, 1.0)
  life._die("stuck", "fell and could not get up in 20 s (80 deg from upright)")
  _hold(life, 1.0)
  life.body.posture = "standing"
  _hold(life, 1.0)
  [row] = rows
  _vocabulary(row)
  assert (row["ended"], row["cause"], row["told"]) == ("died", "stuck", "death")
  assert _fell_lines(life) == []


@pytest.mark.parametrize("how", ["unfinished", "stuck", "reset", "reset at once"])
def test_a_get_up_the_run_or_a_person_ends_first_is_a_row_still(how):
  """Every fall the body counts is a row, its get-up ended by the end of a
  run nothing carries on, by the run its get-up would have died in where
  nothing dies (`Stuck`), or by a reset that put it at its start -- one in
  the step the fall was counted too, before the row's hook read it: the
  warp's, never "got up again 0 s later"."""
  life = stub_life()
  rows = _rows(life)
  life.begin((0.0, 0.0, 0.0))
  _fall(life)
  if how != "reset at once":
    _hold(life, 1.0)
  if how == "unfinished":
    summary = life.end()
    assert summary["falls"] == rows
  elif how == "stuck":
    life._down_since = float(life.data.time) - 1.0
    with pytest.raises(Stuck):
      life._end_stuck("fell and could not get up in 2 s (80 deg from upright)")
  else:
    life.stand_up("an admin", auto=False)
    life.body.posture = "standing"               # ...as the warp leaves it
    _hold(life, 1.0)
  [row] = rows
  _vocabulary(row)
  assert row["ended"] == how.split()[0] and row["told"] is None and _fell_lines(life) == []


def test_a_restored_body_brings_its_count_back_never_a_row():
  """The body's count rides a restart (its kept state); read afresh with it,
  never as falls."""
  life = stub_life()
  rows = _rows(life)
  state, arrays = life.kept_state()
  life.body.falls = 3                            # ...as the body's restore puts it
  life.restore_kept(state, arrays, in_place=True)
  _hold(life, 1.0)
  assert rows == []
  _fall(life)
  life.body.posture = "standing"
  _hold(life, 1.0)
  assert [r["falls"] for r in rows] == [4]


@pytest.mark.parametrize("in_place", [True, False])
def test_a_get_up_a_run_ends_in_is_that_runs_row_and_the_next_runs_line(in_place):
  """A run's end saves the world mid get-up (`Keeper.finish`): the fall's
  row is the run's that saw it, `unfinished` as it left it -- a save may
  never carry it on (a start-over, a stop while paused) -- and the next run
  that carries the get-up on writes its line as it stands, and no second
  row; one that cannot put the body back writes neither. Shown to fail with
  no `fallOpen` kept: no line in the next run (#707's review)."""
  life = stub_life()
  life.continuing = True
  life.begin((0.0, 0.0, 0.0))
  rows = _rows(life)
  t0 = _fall(life)
  _hold(life, 1.0)
  summary = life.end()
  [row] = rows
  assert row["ended"] == "unfinished" and row["seconds"] == pytest.approx(1.0, abs=0.01)
  assert summary["falls"] == rows and _fell_lines(life) == []
  state, arrays = life.kept_state()
  nxt = stub_life()
  nxt_rows = _rows(nxt)
  nxt.data.time = float(life.data.time)
  nxt.body.falls, nxt.body.posture = 1, "getting_up"      # ...as the body's restore puts it
  nxt.restore_kept(state, arrays, in_place=in_place)
  _hold(nxt, 2.0)
  nxt.body.posture = "standing"
  _hold(nxt, 1.0)
  assert nxt_rows == [], "one row a fall"
  lines = [ln.split("] ", 1)[-1] for ln in _fell_lines(nxt)]
  assert lines == ([FELL_LINE.format(seconds=float(nxt.data.time) - 1.0 - t0)]
                   if in_place else [])


def test_a_get_up_stood_as_the_run_ends_is_a_stand():
  """A get-up that stood in the run's last step, before the row's hook read
  it, stood: never `unfinished` (#707's second review)."""
  life = stub_life()
  life.begin((0.0, 0.0, 0.0))
  rows = _rows(life)
  _fall(life)
  _hold(life, 2.0)
  life.body.posture = "standing"
  life.end()
  [row] = rows
  assert row["ended"] == "stood" and row["told"] == "line" and len(_fell_lines(life)) == 1


@pytest.mark.parametrize("cause, told", [("stuck", "death"), ("flat", None)])
def test_a_death_tells_a_fall_only_where_its_line_says_it(cause, told):
  """A `stuck` death's line says the fall ("fell and could not get up");
  a flat pack's, or an `unminded` one's, says nothing of it (#707's second
  review)."""
  life = stub_life(mortal=True)
  rows = _rows(life)
  _fall(life)
  _hold(life, 1.0)
  life._die(cause, "why")
  _hold(life, 0.1)
  [row] = rows
  assert (row["ended"], row["cause"], row["told"]) == ("died", cause, told)


def test_a_struggle_restored_from_a_save_before_it_counts_on():
  """A save from before the row kept no count of its struggle: a stuck
  clock it brings back running is a struggle of at least one fall."""
  life = stub_life()
  state, arrays = life.kept_state()
  del state["fallStruggle"]
  state["downSince"] = float(life.data.time) - 2.0
  life.restore_kept(state, arrays, in_place=True)
  life.body.falls += 1
  life._fall_read()
  assert life._fall_open["struggle"] == 2


def test_a_get_up_carried_on_that_a_restore_moves_clear_ends_with_no_line(tmp_path):
  """The world's hand that puts a restored robot clear of the other
  (`unclip`) ends a get-up the next run carried on, as a reset does: a
  warp, never "got up again" (#707's second review)."""
  from pluggybot import continuation
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.pair import build_pair
  starts = (world_config(HUMANOID_HOME)["start"], world_config(HUMANOID_HOME)["start2"])
  b = starts[1]
  lives = build_pair(HUMANOID_HOME, thoughts_root=str(tmp_path / "before"))
  lives[0].body.start_at(b[0] + 0.05, b[1], b[2])
  lives[1].body.start_at(*b)
  m = lives[1].body.mission
  m.falls, m.legs = 1, "getup"                    # ...the second robot mid get-up
  lives[1]._falls_rowed = 1
  lives[1]._fall_open = {"falls": 1, "fellT": 0.0, "at": {}, "truth": list(b),
                         "belief": list(b), "struggle": 1, "dead": False, "told": None,
                         "rowed": True}
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture(lives, lives[0].world_fingerprint), path)
  snap = continuation.read(path)
  back = build_pair(HUMANOID_HOME, resume=snap, thoughts_root=str(tmp_path / "after"))
  rows: list = []
  for life, start in zip(back, starts):
    life.begin(start)
    life.on_event.append(lambda e: rows.append(e) if e.get("type") == "fall" else None)
  assert continuation.restore(back, snap)["unclipped"] == [back[1].root]
  back[1].body.run(back[1].body.hold_routine(0.01))
  assert rows == [] and _fell_lines(back[1]) == []


def test_a_pairs_recording_takes_the_row_its_runs_end_sends(tmp_path, monkeypatch):
  """A pair's recorder closes after both runs' ends (`run_pair`): one ended
  mid get-up sends its row there, which a recorder closed first lost
  (#707's third review)."""
  from pluggybot import pair
  from test_two_minds import stub_pair
  lives = stub_pair(tmp_path / "t", str(tmp_path / "ledger.json"))
  got: list = []

  class Recorder:
    closed = False

    def emit(self, message):
      got.append((self.closed, message))

    def close(self):
      self.closed = True

  rec = Recorder()

  def recording(ls, path):
    for life in ls:
      life.on_event.append(rec.emit)
    return rec
  monkeypatch.setattr(pair, "record_pair", recording)
  a = lives[0]

  def falls_at_one_s():
    if not a.body.falls and float(a.data.time) >= 1.0:
      _fall(a)
  a.body.step_hooks.insert(0, falls_at_one_s)
  pair.run_pair(lives, max_sim_time=3.0, explore_budget=0.0, record="pair.jsonl.gz")
  assert [(closed, m["ended"]) for closed, m in got if m.get("type") == "fall"] == [
    (False, "unfinished")]
