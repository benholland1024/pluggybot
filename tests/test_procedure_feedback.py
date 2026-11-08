"""The development loop a robot needs to finish a challenge (issue #264):
the one History line a procedure leaves, the one-answer replacement the
prompt promises, a procedure run on the answer that defines it, what a
failed walk says, and a failed step's trace -- each rule pinned in
milliseconds. (The sentence a failed pick was said in, and the fork checked
before a fetch, went with the quadruped's tools: #559.)
"""

import itertools
import math
import re
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot import navigator as nav
from pluggybot import tick
from pluggybot.lifecycle import HubLifecycle, errand_from
from pluggybot.mind.overseer import Decision, FIELD_INDEX, HANDS_PROCEDURE_HEAD
from pluggybot.procedure import library as lib
from pluggybot.procedure import steps as st
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

WORLD = "home_humanoid"


# ---- 1. one History line per procedure the robot wrote ---------------------


def test_a_procedure_cut_short_tells_the_robot_the_line_and_the_reason():
  """A walk that never arrives, on a procedure's first line: the robot is
  told where and why -- on the History it reads, and as `failedReason` on
  the wire."""
  life = stub_life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  life.drive_why = lambda x, y, record=None: "no route there on my map"
  events = []
  life.on_event.append(events.append)
  L = lib.Library(lc.world_facts(WORLD))
  L.define("walk", 'def walk():\n  drive_to(1, 1)\n  wait(1)\n')
  life.run_errand(errand_from(Decision(action="procedure:walk"), library=L))
  history = life.thoughts.read("History.md")
  assert ("the procedure walk did not finish -- it stopped at line 2, drive_to: "
          "did not arrive at (1, 1): no route there on my map, at (0.0, 0.0) "
          "(0 steps had run)") in history
  cut = next(e for e in events if e.get("type") == "procedure"
             and e.get("outcome") == "aborted")
  assert cut["failedLine"] == 2
  assert cut["failedReason"].startswith("did not arrive at (1, 1)")


def test_a_call_refused_at_run_time_is_named_once():
  """A computed tag is checked when it is computed, and History names the
  failed step's call before its reason -- a reason that named it too read
  `line 3, pick: pick(tag=99) ...` (#463)."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("big", "def big():\n  t = 99\n  find(t, 1, 1)\n")
  life.run_errand(errand_from(Decision(action="procedure:big"), library=L))
  assert ("it stopped at line 3, find: tag 99 names no place this world has "
          "(places: 35 to 46, 60 to 63, 68 to 75, 93) (0 steps had run)") in life.thoughts.read("History.md")


def test_an_argument_the_run_cannot_compute_is_named_by_its_call():
  """A fault computing a call's argument said `it stopped at an expression:
  division by zero`, no line, no call -- though the step knew both (#463)."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("u", "def u():\n  t = 0\n  wait(1 / t)\n")
  life.run_errand(errand_from(Decision(action="procedure:u"), library=L))
  assert "it stopped at line 3, wait: division by zero" in life.thoughts.read("History.md")


def test_a_local_is_said_with_every_digit():
  """`:g` kept six digits: a readout of 1234567 reached History as
  `1.23457e+06`, beside a refusal that quoted its tag whole (#463)."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("big", "def big():\n  t = 1234567\n  u = 2.5\n  wait(0.1)\n")
  life.run_errand(errand_from(Decision(action="procedure:big"), library=L))
  assert "it ended with t = 1234567, u = 2.5" in life.thoughts.read("History.md")


def test_a_procedure_that_finished_says_so():
  """A robot must know its procedure ENDED to say `done` after it: round 2
  of #264's probes set `done` before the run, and the grade found one block."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("tidy", 'def tidy():\n  wait(1)\n  face(0.5)\n')
  life.run_errand(errand_from(Decision(action="procedure:tidy"), library=L))
  history = life.thoughts.read("History.md")
  assert re.search(r"ran the procedure tidy to its end \(2 steps\)$", history, re.M)


def test_tests_that_read_false_are_said_and_their_run_is_no_error():
  """#629, Rowan's `chest_band2` (t=140003, 11-03): a walk, a turn, then
  80 `if measure(...)` that each read false and said why -- to the log
  alone. History said "to its end (2 steps)", and the run's verdict
  "2/82 steps, stopped (error)": Rowan concluded a measure costs almost
  no steps and that sensing there was flaky. The robot reads the calls
  made, how many tests read false, and why; the run reached its end; and
  the wire counts them too."""
  life = stub_life()
  events = []
  life.on_event.append(events.append)
  L = lib.Library(lc.world_facts(WORLD))
  L.define("band", "def band():\n  drive_to(1.55, 2.35)\n  face(0.5)\n"
                   "  for i in range(2):\n    for j in range(5):\n"
                   "      if measure(0.95 + 0.1 * j, 1.35):\n        s = 1\n")
  life.run_errand(errand_from(Decision(action="procedure:band"), library=L))
  history = life.thoughts.read("History.md")
  assert ("ran the procedure band to its end (12 steps; 10 tests read false -- line 6, "
          "measure: no flat top seen round (0.95, 1.35) from where you stand) -- it ended "
          "with i = 1, j = 4") in history
  assert "program: band: ran to its end, 12 steps; 10 tests read false" in history
  assert "stopped (error)" not in history
  ran = next(e for e in events if e.get("type") == "procedure" and e.get("outcome") == "ran")
  assert ran["completed"] == ran["total"] == 12 and ran["testsFalse"] == 10


def test_tests_that_read_false_are_said_line_by_line_within_historys_cap():
  """Each line that read false once, with its first reason and, where
  several lines did, how many times -- a fault in a tested call's
  arithmetic among them -- in the order they first read false; and as
  many as fit inside History's cap, its own stamp included, the rest
  counted (#629). A failed step's reason keeps its place: the tests say
  only what fits beside it."""
  from pluggybot.mind.thoughts import MAX_LINE_CHARS
  room = MAX_LINE_CHARS - lc.HISTORY_STAMP_ROOM
  def read(line, verb, reason, **at):
    return {"verb": verb, "line": line, "ok": False, "tested": True, "reason": reason, **at}
  steps = [read(4, "measure", "no flat top seen round (1, 0) from where you stand"),
           read(4, "measure", "no flat top seen round (2, 0) from where you stand"),
           {"i": 2, "verb": "wait", "line": 5, "ok": True},
           read(8, "expr", "division by zero", **{"in": "grip"}),
           read(4, "measure", "no flat top seen round (3, 0) from where you stand")]
  [line] = lc.procedure_outcome("p", {"ok": True, "completed": 4, "total": 4, "steps": steps})
  assert line == ("ran the procedure p to its end (4 steps; 4 tests read false -- line 4, "
                  "measure (3 times): no flat top seen round (1, 0) from where you stand; "
                  "line 8, an expression (in grip): division by zero)")
  # ...past the room, the lines left over counted, and the whole within it
  long = [read(10 + k, "reach", "your right hand stopped short on its way: " + "x" * 60)
          for k in range(9)]
  [line] = lc.procedure_outcome("p", {"ok": True, "completed": 9, "total": 9, "steps": long})
  assert len(line) <= room and line.endswith(" more))"), line
  assert line.startswith("ran the procedure p to its end (9 steps; 9 tests read false -- line 10, ")
  life = stub_life()
  life.data.time = 123456.0
  life._remember(line)
  assert life.thoughts.read("History.md").splitlines()[-1].endswith(line)
  # ...and beside a failure whose reason leaves little room, the count
  # alone; none at all, the reason whole
  def cut(k):
    failed = {"i": 9, "verb": "pull", "line": 30, "ok": False, "reason": "y" * k}
    return lc.procedure_outcome("p", {"ok": False, "completed": 9, "total": 10,
                                      "failedAt": 9, "steps": long + [failed]})[0]
  assert cut(280).endswith("y" * 280 + " (9 steps had run; 9 tests read false)")
  assert len(cut(280)) <= room
  assert cut(310).endswith("y" * 310 + " (9 steps had run)")


def test_what_a_stop_condition_read_is_said_beside_its_run():
  """#631, Luca's drawer (11-03 on): `reach(..., until="right.fx",
  above=3)` stopped 0.18 s in, its hand by its side, and History said only
  that the run reached its end -- `reach.stopped` read 1, Luca read every
  stop as its hand meeting the drawer's face, and shut its jaws on air
  thirteen times. Each line's stop condition is said as practice's report
  says it -- what it read, past which number, how far into the move -- and
  one that never had a reading says so, after the tests that read false; a
  move that ended otherwise says nothing of its condition."""
  reach = {"verb": "reach", "line": 3, "ok": True, "why": "until"}
  steps = [{"i": 0, **reach, "until": "right.fx read 3.626 (above 3) 0.18 s in"},
           {"i": 1, "verb": "close_hand", "line": 4, "ok": True},
           {"i": 2, "verb": "pull", "line": 6, "ok": True, "why": "distance",
            "until": "watch.z above 1.05 never had a reading: tag 53 has not been seen "
                     "since `watch` began"},
           {"i": 3, "verb": "move_hand", "line": 7, "ok": True, "why": "arrived"},
           {"i": 4, **reach, "until": "right.fx read 3.9 (above 3) 0.21 s in"},
           {"verb": "measure", "line": 8, "ok": False, "tested": True,
            "reason": "no flat top seen round (1, 0) from where you stand"}]
  [line] = lc.procedure_outcome("drawer_run3", {"ok": True, "completed": 5, "total": 5,
                                                "steps": steps})
  assert line == ("ran the procedure drawer_run3 to its end (5 steps; 1 test read false -- "
                  "line 8, measure: no flat top seen round (1, 0) from where you stand; 3 stop "
                  "conditions said -- line 3, reach (2 times): right.fx read 3.626 (above 3) "
                  "0.18 s in; line 6, pull: watch.z above 1.05 never had a reading: tag 53 has "
                  "not been seen since `watch` began)")
  # ...a run cut short says them too, and none where no move had a condition
  [line] = lc.procedure_outcome("p", {"ok": False, "completed": 1, "total": 2, "failedAt": 1,
                                      "steps": steps[:1] + [{"i": 1, "verb": "close_hand",
                                                             "line": 4, "ok": False,
                                                             "reason": "nothing held"}]})
  assert line.endswith("(1 step had run; 1 stop condition said -- line 3, reach: right.fx "
                       "read 3.626 (above 3) 0.18 s in)"), line
  [line] = lc.procedure_outcome("p", {"ok": True, "completed": 1, "total": 1,
                                      "steps": [steps[3]]})
  assert line == "ran the procedure p to its end (1 step)"


def test_a_procedure_out_of_budget_says_first_that_it_did_not_finish():
  """MEASURED (ladder B, 2026-09-24): the line said "(4/4 steps) -- it ran
  out of the time its budget gave it", the model read "all four steps
  landed", said `done`, and the grade found two blocks of three. The count
  is of calls MADE, so a stopped run never shows it as a fraction."""
  run = {"ok": False, "completed": 4, "total": 4, "stopped": "budget", "seconds": 324.2,
         "steps": [{"i": 3, "verb": "pick", "line": 6, "ok": True}]}
  [line] = lc.procedure_outcome("stack_tower", run)
  assert line == ("the procedure stack_tower did not finish -- it ran out of the time "
                  "its budget gave it (324 s) after line 6 (4 steps had run)")
  assert "/" not in line


def test_a_long_reason_never_costs_a_run_its_readout():
  """History cuts a line at 400 characters without a mark, its own
  "[t=NNNNs] " stamp INSIDE the 400, and a failed pick's reason in front of
  a weighing's locals cut off `mass =` -- the one number the bench grade
  needed recorded. Written through History at a clock with a long stamp,
  across every length where the old split kept "mass = 0" of
  "mass = 0.207812", the number arrives whole or not at all. A fault inside
  the run is named, never quoted (the exception is the log's), and keeps no
  false count."""
  life = stub_life()
  life.data.time = 123456.0
  for k in range(120, 420):
    run = {"ok": False, "completed": 7, "failedAt": 6, "stopped": None,
           "steps": [{"i": 6, "verb": "pick", "line": 12, "ok": False, "reason": "x" * k}],
           "locals": {"f23": 7.36057, "f0": 6.40608, "mass": 0.207812}}
    for line in lc.procedure_outcome("weigh", run):
      life._remember(line)
    last = life.thoughts.read("History.md").splitlines()[-1]
    assert "mass = 0.207812" in last, (k, last[-60:])
  # twelve locals with long names: whole values only, and it says so
  names = {f"reading_of_the_lift_force_number_{i:02d}": 1.2345 + i for i in range(12)}
  run = {"ok": True, "completed": 3, "locals": names}
  lines = lc.procedure_outcome("weigh", run)
  assert len(lines) == 2 and lines[1].endswith(" more)")
  for item in lines[1].split(" ended with ")[1].rsplit(" (+", 1)[0].split(", "):
    key, value = item.split(" = ")
    assert names[key] == pytest.approx(float(value), abs=1e-4)
  [line] = lc.procedure_outcome("weigh", {"ok": False, "completed": 0,
                                          "error": "KeyError: 'axis'"})
  assert line == "the procedure weigh did not finish -- it stopped on a fault inside the run"


def test_every_name_a_run_assigned_is_said_or_counted():
  """#658, on the live `fe07629`: History showed the first twelve names a
  run assigned and `(+N more)` counted only what the line's width left out
  of those twelve, so a 24-name run read as complete -- 97 of 167 runs, and
  what went was the last names assigned, what each run was written to find.
  The line's width is the only limit: sixteen short names are all said, as
  they were assigned, and whatever the run's own line left beside them,
  every name is said whole or counted, and a count means the line was full."""
  from pluggybot.mind.thoughts import MAX_LINE_CHARS
  room = MAX_LINE_CHARS - lc.HISTORY_STAMP_ROOM
  sixteen = {f"v{i}": i + 0.5 for i in range(16)}
  assert lc.procedure_outcome("p", {"ok": True, "completed": 2, "locals": sixteen}) == [
    "ran the procedure p to its end (2 steps) -- it ended with "
    + ", ".join(f"v{i} = {i + 0.5}" for i in range(16))]
  # ...at every width of name, or a full line's slack hides a count with no room
  for case in itertools.product((1, 12, 13, 16, 24, 32, 60, 200), (0, 150, 340), range(12)):
    count, why, pad = case
    names = {f"name{'_' * pad}{i:03d}": 1.234567 + i for i in range(count)}
    run = {"ok": not why, "completed": 2, "locals": names}
    if why:
      run |= {"failedAt": 1, "steps": [{"i": 1, "verb": "pull", "line": 3, "ok": False,
                                        "reason": "x" * why}]}
    lines = lc.procedure_outcome("p", run)
    assert len(lines[-1]) <= room, case
    tail = lines[-1].split(" ended with ", 1)[1]
    more = re.search(r" \(\+(\d+) more\)$", tail)
    items = (tail[:more.start()] if more else tail).split(", ")
    assert len(items) + (int(more[1]) if more else 0) == count, case
    assert [i.split(" = ")[0] for i in items] == list(names)[:len(items)], case
    assert all(float(i.split(" = ")[1]) == names[i.split(" = ")[0]] for i in items), case
    if more:
      said = len(items)
      name = list(names)[said]
      after = len(f" (+{count - said - 1} more)") if said + 1 < count else 0
      kept = len(lines[-1]) - len(more[0])
      assert kept + len(f", {name} = {st.number_text(names[name])}") + after > room, case


def test_a_run_of_two_dozen_names_reaches_history_whole():
  """Rowan's `door_fast` (t=212582, 11-08): 24 names in the order its source
  assigns them. History kept the first twelve and said nothing, the run's
  own final angle (`ang`), `xl`, `xl2` and `prot` were gone, and a `recall`
  gives back only what the line said. Off the robot's language to the record
  it reads, with a clock as long as the deployed one's and no line cut."""
  from pluggybot.mind.thoughts import cut_mark, MAX_LINE_CHARS
  values = {"got": 1.0, "n": 3.0, "lx": 0.429388, "ly": -0.042355, "lz": 0.864542,
            "d": 0.434169, "ox": 0.0, "oy": 0.05, "oz": -0.02, "g": 1.0, "r": 1.0,
            "moved": 0.7281, "rounds": 2.0, "pf": 0.51, "xw": -1.0, "xl": 0.755119,
            "nx": 0.999549, "ny": -0.030046, "ang": 1.721775, "stick": -1.755119,
            "prot": 0.061257, "mx": 1.054539, "my": -0.046634, "xl2": 0.757498}
  life = stub_life()
  life.data.time = 212582.108
  L = lib.Library(lc.world_facts(WORLD))
  L.define("door_fast", "def door_fast():\n"
           + "".join(f"  {k} = {v}\n" for k, v in values.items()) + "  wait(0.1)\n")
  life.run_errand(errand_from(Decision(action="procedure:door_fast"), library=L))
  history = life.thoughts.read("History.md")
  for k, v in values.items():
    assert f"{k} = {st.number_text(v)}" in history, k
  assert "more)" not in history and cut_mark(MAX_LINE_CHARS) not in history


# ---- 2. a replacement is one answer, as the prompt says --------------------


def test_one_answer_replaces_a_procedure_and_the_prompt_says_it_can():
  """The lifecycle has always applied `undefine` before `define`; the robot
  was told "there is no replace -- undefine, then define" and read it as two
  turns. The claim and the behaviour are pinned together."""
  assert "no replace" not in HANDS_PROCEDURE_HEAD
  assert "on\nthe same answer" in HANDS_PROCEDURE_HEAD or "same answer" in HANDS_PROCEDURE_HEAD
  assert any(name == "undefine" and "same answer" in text
             for name, _, _, text in FIELD_INDEX)
  life = SimpleNamespace(data=SimpleNamespace(time=0.0), world=WORLD,
                         root="pluggybot",
                         _say=lambda *a, **k: None, _remember=lambda *a, **k: None,
                         _emit=lambda e: None)
  L = lib.Library(lc.world_facts(WORLD), cap=2)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("b", "def b():\n  wait(1)\n")          # full
  life.overseer = SimpleNamespace(library=L)
  HubLifecycle._define(life, Decision(action="idle", undefine="a",
                                      define={"name": "a", "source": "def a():\n  wait(2)\n"}))
  assert "wait(2)" in L.entries["a"].source and len(L.entries) == 2


def test_a_refused_define_says_the_undefine_goes_on_the_same_answer():
  L = lib.Library(lc.world_facts(WORLD), cap=1)
  L.define("a", "def a():\n  wait(1)\n")
  with pytest.raises(lib.LibraryRefused) as e:
    L.define("a", "def a():\n  wait(2)\n")
  said = " ".join(e.value.reasons)
  assert "same answer" in said and "there is no replace" not in said
  assert "(it holds 1)" in said                     # never "2 of 1"


# ---- 3. a procedure is run on the answer that defines it -------------------


def _menu():
  from dataclasses import replace
  from pluggybot.mind.overseer import Menu
  from pluggybot.lifecycle import board_book
  return replace(Menu.for_world(WORLD, board_book(WORLD)), procedures=True)


def test_the_answer_that_defines_a_procedure_can_name_it_even_in_an_empty_library():
  """The enum is built from the library BEFORE the answer: a procedure
  defined in it could never be named in it, and the decoder substituted
  one that could -- Luca's weighing ran `count_blocks` and it blamed
  itself. `procedure:new` is the answer's own define, and never an order."""
  from pluggybot.mind.overseer import PROCEDURE_NEW, standing_order
  menu = _menu()
  schema = menu.schema(procedures=())
  assert PROCEDURE_NEW in schema["properties"]["action"]["enum"]
  assert PROCEDURE_NEW not in menu.orderable(())
  with pytest.raises(ValueError, match="an order has no answer"):
    standing_order(PROCEDURE_NEW, menu)


def test_procedure_new_needs_a_define_on_the_same_answer():
  """...and the refusal says how to run one that is already there: ladder B
  on the bench lost two turns in a row to `procedure:new` with no define,
  meaning "run my weigh again"."""
  from pluggybot.mind.overseer import PROCEDURE_NEW
  menu = _menu()
  with pytest.raises(ValueError, match="defines none"):
    menu.validate({"action": PROCEDURE_NEW, "reason": "x"}, procedures=())
  with pytest.raises(ValueError, match=r"name it \(procedure:weigh\)"):
    menu.validate({"action": PROCEDURE_NEW, "reason": "x"}, procedures=("weigh",))
  menu.validate({"action": PROCEDURE_NEW, "reason": "x",
                 "define": {"name": "weigh", "source": "def weigh():\n  wait(1)\n"}},
                procedures=())


def _deciding_life(library):
  life = stub_life()
  life.overseer = SimpleNamespace(library=library, event_map=None)
  events = []
  life.on_event.append(events.append)
  return life, events


def test_procedure_new_runs_what_the_same_answer_defined_and_not_another():
  from pluggybot.mind.overseer import PROCEDURE_NEW
  L = lib.Library(lc.world_facts(WORLD))
  L.define("old", "def old():\n  wait(1)\n  face(0.5)\n")
  life, events = _deciding_life(L)
  decision = Decision(action=PROCEDURE_NEW, reason="weigh it now",
                      define={"name": "weigh", "source": "def weigh():\n  wait(1)\n"})
  life.body.run(life._after_decision_routine(decision))
  queued = [e.program.name for e in life.errands if e.program is not None]
  assert queued == ["weigh"]                 # queued for the loop, and not `old`


def test_procedure_new_runs_nothing_when_the_define_was_refused():
  from pluggybot.mind.overseer import PROCEDURE_NEW
  L = lib.Library(lc.world_facts(WORLD))
  L.define("old", "def old():\n  wait(1)\n")
  life, events = _deciding_life(L)
  decision = Decision(action=PROCEDURE_NEW, reason="try",
                      define={"name": "bad", "source": "def bad():\n  import os\n"})
  life.body.run(life._after_decision_routine(decision))
  assert not [e for e in life.errands if e.program is not None]
  assert "ran nothing: `procedure:new`" in life.thoughts.read("History.md")


def test_a_refused_define_is_written_where_the_robot_reads():
  """Narrated and put on the wire, a refusal reached everyone but the robot
  that made it -- and both deployed libraries were full, so "an `undefine`
  on the same answer makes room" was said to nobody who could act on it."""
  life, _ = _deciding_life(lib.Library(lc.world_facts(WORLD)))
  life._define(Decision(action="idle", reason="x",
                        define={"name": "bad", "source": "def bad():\n  import os\n"}))
  assert "could not write the procedure bad:" in life.thoughts.read("History.md")


def test_a_one_answer_rewrite_that_is_refused_keeps_the_procedure_it_had():
  """Undefine and define of one name is a swap: applied in order, a refused
  define had already deleted the procedure it was meant to improve."""
  L = lib.Library(lc.world_facts(WORLD))
  L.define("weigh", "def weigh():\n  wait(1)\n")
  life, _ = _deciding_life(L)
  life._define(Decision(action="idle", reason="x", undefine="weigh",
                        define={"name": "weigh", "source": "def weigh():\n  import os\n"}))
  assert "weigh" in L.runnable() and "wait(1)" in L.entries["weigh"].source
  history = life.thoughts.read("History.md")
  assert "could not rewrite the procedure weigh -- the one I had is kept:" in history
  assert "forgot the procedure weigh" not in history
  life._define(Decision(action="idle", reason="x", undefine="weigh",
                        define={"name": "weigh", "source": "def weigh():\n  wait(2)\n"}))
  assert "wait(2)" in L.entries["weigh"].source and life._defined_now == "weigh"


def test_new_is_not_a_name_a_procedure_may_take():
  L = lib.Library(lc.world_facts(WORLD))
  with pytest.raises(lib.LibraryRefused, match="procedure:new"):
    L.define("new", "def new():\n  wait(1)\n")


# ---- 4. a failed drive says what it read ------------------------------------


def test_a_drive_that_did_not_arrive_says_where_it_stopped_and_why():
  """...and WHY (issue #350): "stopped 9.1 m short of (22, 3)" was a route
  the planner could not make, read by Rowan as the pack running short."""
  asked = []
  life = SimpleNamespace(
    body=SimpleNamespace(go_to_routine=lambda x, y, timeout: tick.result(False),
                         in_sight=lambda x, y: True, pose=(1.0, 2.0, 0.0)),
    drive_why=lambda x, y: (asked.append((x, y)), "the drive gave up (why)")[1])
  verdict = tick.run(SimpleNamespace(step=lambda *a: None),
                     st._drive_to(life, {"x": 4.0, "y": 6.0}))
  assert verdict["reason"] == "did not arrive at (4, 6): the drive gave up (why), at (1.0, 2.0)"
  assert verdict["why"] == "the drive gave up (why)" and asked == [(4.0, 6.0)]


# ---- 5. a failed step leaves its trace in the log --------------------------


def test_the_trace_measures_the_belief_against_the_true_pose():
  """A walk's evidence says how far the belief had drifted from the truth
  when it gave up (a find's trace, a drift row): the belief minus the true
  pose, mm and degrees, the heading's difference wrapped."""
  me = SimpleNamespace(true_pose=lambda: (1.0, 1.0, 3.1), pose=(1.012, 0.995, -3.1))
  dx, dy, dyaw = nav.Navigator.truth_error(me)
  assert (dx, dy) == pytest.approx((12.0, -5.0))
  assert dyaw == pytest.approx(round(math.degrees(2 * math.pi - 6.2), 2))   # not -355


def test_a_failed_step_puts_its_trace_in_the_log_and_never_the_status():
  """`_say`'s message is the robot's status line on the site -- a sentence
  it could say; a step's trace is evidence, `detail`, the log's alone."""
  life = stub_life()
  life.body.find_tag_routine = lambda tag, near, patience, stop=None: tick.result({
    "tag": tag, "found": False, "why": "out of time", "seconds": 12.0,
    "trail": [{"to": "address", "goal": (1.0, 1.0), "why": "gave up", "shortM": 0.4,
               "seconds": 9.0}], "err": (3.0, -12.0, 0.4)})
  said = []
  life.say_hooks.append(lambda t, msg, *a: said.append(msg))
  L = lib.Library(lc.world_facts(WORLD))
  L.define("look", "def look():\n  find(35, 1, 1)\n")
  life.run_errand(errand_from(Decision(action="procedure:look"), library=L))
  assert any("belief off +3,-12 mm" in line for line in life.log)
  assert any("PROCEDURE look failed at step 1" in msg for msg in said)
  assert not any("belief off" in msg for msg in said)


# ---- review of #336: the library's edges -----------------------------------


def test_undefine_can_name_an_entry_that_cannot_run(tmp_path):
  """An entry marked not runnable (after a restart, or a retired tool) was
  outside the `undefine` enum -- the one thing that could free its slot --
  while the refusal told the robot to undefine it. And a procedure saved as
  `new` before the token existed loads as one that cannot run, so
  `procedure:new` keeps its one meaning and the slot can still be freed."""
  (tmp_path / f"new{lib.SUFFIX}").write_text("def new():\n  wait(1)\n")
  (tmp_path / f"broken{lib.SUFFIX}").write_text("def broken():\n  fetch('module_gone')\n")
  (tmp_path / f"ok{lib.SUFFIX}").write_text("def ok():\n  wait(1)\n")
  L = lib.Library(lc.world_facts(WORLD), root=tmp_path)
  assert set(L.names()) == {"new", "broken", "ok"} and L.runnable() == ("ok",)
  schema = _menu().schema(procedures=L.runnable(), keeps=L.names())
  props = schema["properties"]
  assert {"new", "broken", "ok"} <= set(props["undefine"]["enum"])
  assert "procedure:broken" not in props["action"]["enum"]
  assert props["action"]["enum"].count("procedure:new") == 1


def test_making_room_for_a_refused_procedure_keeps_the_one_it_would_have_replaced():
  """The refusal for a full library advises an `undefine` on the same answer
  -- and both deployed libraries were full. Applied in order, a refused
  define then cost the working procedure it was making room for (second
  review of #336); the undefine now waits for a define that goes through.
  An answer with no undefine is a plain refusal, never "kept"."""
  L = lib.Library(lc.world_facts(WORLD), cap=2)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("c", "def c():\n  wait(1)\n")
  life, _ = _deciding_life(L)
  life._define(Decision(action="idle", reason="x", undefine="a",
                        define={"name": "b", "source": "def b():\n  import os\n"}))
  history = life.thoughts.read("History.md")
  assert set(L.names()) == {"a", "c"}
  assert "could not write the procedure b, so a is kept too" in history
  assert "forgot the procedure a" not in history
  life._define(Decision(action="idle", reason="x", undefine="a",
                        define={"name": "b", "source": "def b():\n  wait(2)\n"}))
  assert set(L.names()) == {"b", "c"}                    # the room was made
  life._define(Decision(action="idle", reason="x",
                        define={"name": "", "source": "def x():\n  wait(1)\n"}))
  assert "could not write the procedure :" in life.thoughts.read("History.md")
  assert "is kept" not in life.thoughts.read("History.md").splitlines()[-1]
